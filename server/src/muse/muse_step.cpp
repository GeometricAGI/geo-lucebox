// Muse-Glimmer KV cache + step driver (prefill and decode).
//
// The cache is per-layer K/V with two shapes: full-attention layers hold
// `max_ctx` rows; sliding-window layers hold a RING indexed by
// (absolute position % swa_ring). The ring is what makes long context cheap
// on this family — 39 of 52 layers are SWA — and it is also the part that is
// easy to get subtly wrong, so the mask is built from the ring's actual
// occupancy rather than from an assumed contiguous span.
//
// Ring rows and the attention WINDOW are separate numbers: the ring is
// window + one chunk of headroom. With ring == window a chunked prefill
// evicts rows that queries in the same chunk still need (every write in a
// graph lands before any read), which mis-attends silently rather than
// failing — measured on an 8-token probe with a 4-row ring: chunked and
// stepwise prefill diverged by rms 2.11 with different argmax.

#include "muse_internal.h"
#include "internal.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <cstdio>
#include <cstring>
#include <vector>

namespace dflash::common {

bool create_muse_cache(ggml_backend_t backend, const MuseWeights & w,
                       int max_ctx, MuseCache & out, int max_chunk) {
    if (max_ctx <= 0) { set_last_error("muse cache: max_ctx <= 0"); return false; }
    if (max_chunk <= 0) { set_last_error("muse cache: max_chunk <= 0"); return false; }

    out.max_ctx = max_ctx;
    out.n_layer = w.n_layer;
    // A window >= max_ctx degenerates to full attention; allocating max_ctx
    // rows then keeps the ring arithmetic a no-op instead of a special case.
    if (w.sliding_window > 0 && w.sliding_window < max_ctx) {
        out.swa_window = w.sliding_window;
        // Headroom of one chunk so a chunk's writes cannot evict the history
        // its own queries read. Clamped to max_ctx, at which point the ring is
        // simply the whole context.
        out.swa_ring = std::min(max_ctx, w.sliding_window + max_chunk);
    } else {
        out.swa_window = max_ctx;
        out.swa_ring   = max_ctx;
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(w.n_layer * 2 + 16);
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) { set_last_error("muse cache: ggml_init"); return false; }

    out.k.assign((size_t)w.n_layer, nullptr);
    out.v.assign((size_t)w.n_layer, nullptr);
    for (int il = 0; il < w.n_layer; ++il) {
        const int rows = muse_is_swa_layer(w, il) ? out.swa_ring : max_ctx;
        out.k[(size_t)il] = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F16,
                                               w.head_dim, rows, w.n_head_kv);
        out.v[(size_t)il] = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F16,
                                               w.head_dim, rows, w.n_head_kv);
    }
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        set_last_error("muse cache: backend alloc failed");
        ggml_free(out.ctx); out.ctx = nullptr;
        return false;
    }
    for (int il = 0; il < w.n_layer; ++il) {
        ggml_backend_tensor_memset(out.k[(size_t)il], 0, 0,
                                   ggml_nbytes(out.k[(size_t)il]));
        ggml_backend_tensor_memset(out.v[(size_t)il], 0, 0,
                                   ggml_nbytes(out.v[(size_t)il]));
    }

    size_t bytes = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        bytes += ggml_nbytes(out.k[(size_t)il]) + ggml_nbytes(out.v[(size_t)il]);
    }
    std::printf("[muse-cache] max_ctx=%d swa_window=%d swa_ring=%d "
                "max_chunk=%d layers=%d kv=%.2f GB\n",
                max_ctx, out.swa_window, out.swa_ring, out.max_chunk(),
                w.n_layer, bytes / 1e9);
    std::fflush(stdout);
    return true;
}

void free_muse_cache(MuseCache & c) {
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    if (c.ctx) { ggml_free(c.ctx); c.ctx = nullptr; }
    c.k.clear(); c.v.clear();
}

// Ring visibility predicate, exposed (not static) so the wrap arithmetic can
// be unit-tested without a model: an end-to-end wrap needs a prompt longer
// than the 2048-token window, which no unit test wants to pay for.
//
// Slot `s` holds the largest absolute position p with p % ring_rows == s and
// p < total. The query at absolute position `q_abs` may attend to it unless
// it is in the future or has fallen out of the window.
bool muse_swa_slot_visible(int total, int q_abs, int slot, int ring_rows,
                           int window) {
    if (slot < 0 || slot >= ring_rows || total <= 0) return false;
    // Occupancy uses the RING size; visibility uses the WINDOW. Conflating
    // the two is the bug this signature exists to prevent.
    const int p = slot + ((total - 1 - slot) / ring_rows) * ring_rows;
    if (p < 0 || p >= total) return false;
    if (p > q_abs) return false;               // not written yet
    if (p <= q_abs - window) return false;     // outside the attention window
    return true;
}

namespace {

// Causal mask over a flat [0, kv_len) span: query i (absolute kv_start+i)
// attends to every key at absolute position <= its own.
void fill_full_mask(std::vector<ggml_fp16_t> & m, int kv_pad, int q_pad,
                    int kv_start, int n_tokens) {
    std::fill(m.begin(), m.end(), ggml_fp32_to_fp16(-INFINITY));
    for (int i = 0; i < n_tokens; ++i) {
        const int q_abs = kv_start + i;
        const int upper = std::min(q_abs, kv_pad - 1);
        for (int k = 0; k <= upper; ++k) {
            m[(size_t)i * kv_pad + k] = ggml_fp32_to_fp16(0.0f);
        }
    }
    (void)q_pad;
}

// Ring mask. Slot s currently holds the largest absolute position p with
// p % ring_rows == s and p < total_written. A query at absolute position q may
// attend to it when it is neither in the future (p > q) nor outside the
// window (p <= q - window). Deriving the slot's occupant instead of
// assuming a contiguous block is what makes wrap-around correct: mid-chunk
// the ring holds a rotated view, and a "start..end" span would either mask
// live rows or expose stale ones.
void fill_swa_mask(std::vector<ggml_fp16_t> & m, int kv_pad, int q_pad,
                   int kv_start, int n_tokens, int ring_rows, int window) {
    std::fill(m.begin(), m.end(), ggml_fp32_to_fp16(-INFINITY));
    const int total = kv_start + n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        const int q_abs = kv_start + i;
        for (int s = 0; s < ring_rows && s < kv_pad; ++s) {
            if (muse_swa_slot_visible(total, q_abs, s, ring_rows, window)) {
                m[(size_t)i * kv_pad + s] = ggml_fp32_to_fp16(0.0f);
            }
        }
    }
    (void)q_pad;
}

}  // namespace

bool muse_step(ggml_backend_t backend, const MuseWeights & w, MuseCache & cache,
               const float * embed, int n_tokens, int kv_start,
               std::vector<float> & out_logits) {
    if (n_tokens <= 0) { set_last_error("muse_step: n_tokens <= 0"); return false; }
    if (kv_start + n_tokens > cache.max_ctx) {
        set_last_error("muse_step: kv_start+n_tokens exceeds max_ctx");
        return false;
    }
    // A chunk larger than the ring's headroom clobbers its own K/V: every
    // write in the graph lands before attention reads the cache, so the
    // earliest tokens of the chunk get overwritten by its latest ones and the
    // sliding-window layers attend to the wrong keys. Nothing crashes — the
    // prefill silently degrades — so refuse instead.
    if (cache.swa_ring < cache.max_ctx && n_tokens > cache.max_chunk()) {
        set_last_error("muse_step: chunk of " + std::to_string(n_tokens) +
                       " tokens exceeds this cache's headroom (" +
                       std::to_string(cache.max_chunk()) + " = ring " +
                       std::to_string(cache.swa_ring) + " - window " +
                       std::to_string(cache.swa_window) +
                       "); split the prefill or build the cache with a larger "
                       "max_chunk");
        return false;
    }

    const size_t arena = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() +
                         (size_t)32 * 1024 * 1024;
    static thread_local std::vector<uint8_t> g_arena;
    if (g_arena.size() < arena) g_arena.resize(arena);
    ggml_init_params ip{};
    ip.mem_size   = arena;
    ip.mem_buffer = g_arena.data();
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(inp);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(positions);

    // Two masks: one over the full span, one over the SWA ring. Both are
    // padded to the FA kernels' expectations (kv to 256, queries to 64).
    const int full_kv_len = std::min((kv_start + n_tokens + 255) & ~255, cache.max_ctx);
    const int swa_kv_len  = std::min((std::min(kv_start + n_tokens, cache.swa_ring)
                                      + 255) & ~255, cache.swa_ring);
    const int q_pad = (n_tokens + 63) & ~63;
    ggml_tensor * mask_full = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, full_kv_len, q_pad);
    ggml_set_input(mask_full);
    ggml_tensor * mask_swa = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, swa_kv_len, q_pad);
    ggml_set_input(mask_swa);

    // Per-layer cache row indices for the set_rows append: absolute for full
    // layers, ring slots for SWA layers.
    ggml_tensor * kvi_full = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_input(kvi_full);
    ggml_tensor * kvi_swa = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_input(kvi_swa);

    ggml_tensor * cur = build_muse_inp_norm(ctx, w, inp);
    for (int il = 0; il < w.n_layer; ++il) {
        const bool swa = muse_is_swa_layer(w, il);
        cur = build_muse_layer(ctx, gf, w,
                               cache.k[(size_t)il], cache.v[(size_t)il], il, cur,
                               positions, swa ? mask_swa : mask_full,
                               swa ? kvi_swa : kvi_full,
                               kv_start, n_tokens);
    }
    cur = build_muse_head(ctx, w, cur);
    ggml_build_forward_expand(gf, cur);

    static thread_local ggml_gallocr_t alloc = nullptr;
    if (!alloc) {
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        set_last_error("muse_step: graph alloc failed");
        return false;
    }

    ggml_backend_tensor_set(inp, embed, 0, ggml_nbytes(inp));

    std::vector<int32_t> pos((size_t)n_tokens);
    std::vector<int64_t> idx_full((size_t)n_tokens), idx_swa((size_t)n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        const int abs = kv_start + i;
        pos[(size_t)i]      = abs;
        idx_full[(size_t)i] = abs;
        idx_swa[(size_t)i]  = abs % cache.swa_ring;
    }
    ggml_backend_tensor_set(positions, pos.data(), 0, ggml_nbytes(positions));
    ggml_backend_tensor_set(kvi_full, idx_full.data(), 0, ggml_nbytes(kvi_full));
    ggml_backend_tensor_set(kvi_swa,  idx_swa.data(),  0, ggml_nbytes(kvi_swa));

    std::vector<ggml_fp16_t> mf((size_t)full_kv_len * q_pad);
    fill_full_mask(mf, full_kv_len, q_pad, kv_start, n_tokens);
    ggml_backend_tensor_set(mask_full, mf.data(), 0, ggml_nbytes(mask_full));

    std::vector<ggml_fp16_t> ms((size_t)swa_kv_len * q_pad);
    fill_swa_mask(ms, swa_kv_len, q_pad, kv_start, n_tokens, cache.swa_ring,
                  cache.swa_window);
    ggml_backend_tensor_set(mask_swa, ms.data(), 0, ggml_nbytes(mask_swa));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        set_last_error("muse_step: graph compute failed");
        return false;
    }

    out_logits.resize((size_t)w.n_vocab);
    ggml_backend_tensor_get(cur, out_logits.data(),
                            (size_t)(n_tokens - 1) * w.n_vocab * sizeof(float),
                            out_logits.size() * sizeof(float));
    return true;
}

}  // namespace dflash::common
