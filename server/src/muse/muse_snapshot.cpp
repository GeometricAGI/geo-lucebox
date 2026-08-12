// KV snapshot / restore for speculative decode.
//
// Speculative decode runs the target over draft tokens that may be REJECTED,
// so the KV writes those tokens made have to be undoable. The generic
// DFlashTarget contract calls snapshot_kv() before the speculative forward and
// restore_kv() when tokens are rejected.
//
// Two facts about this cache make the cheap version correct, and both are
// load-bearing:
//
//   1. FULL-attention layers need NO saved data. Their row index IS the
//      absolute position, so a speculative write at position p lands in row p,
//      which no earlier position occupies. Rolling back only has to move the
//      cursor; the stale rows are simply never read again (the mask is built
//      from kv_start) and are overwritten when the positions are re-run.
//
//   2. SWA layers DO need saved data. Their row is `p % swa_ring`, so a
//      speculative write at p clobbers whatever position p - swa_ring left
//      there. That row is still live history: the ring mask derives each
//      slot's occupant as "largest q < total with q % S == slot", so after a
//      rollback the mask would compute the OLD occupant while the ring
//      physically holds the speculative FUTURE one. Nothing crashes; attention
//      silently reads the wrong keys.
//
// So the snapshot saves exactly the SWA rows the speculative forward is about
// to overwrite — at most `n` rows per SWA layer, as one or two contiguous
// spans (contiguous because the slots are consecutive modulo the ring). That
// is O(speculation depth), not O(context): for 39 SWA layers at depth 16 it is
// ~640 KB against ~110 MB for copying the whole cache.
//
// The copies go through a single ggml graph rather than per-row backend calls.
// A row is `head_dim` elements but rows are the MIDDLE dimension of
// [head_dim, rows, n_head_kv], so a per-row-per-head loop would be thousands of
// ~256-byte transfers per speculation round and cost more than the decode step
// it is meant to accelerate. A span of rows is contiguous within a head, so
// ggml_view_3d expresses the whole thing in ~4 nodes per layer.

#include "muse_internal.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <string>
#include <vector>

namespace dflash::common {

namespace {

// The (at most two) contiguous ring spans covering slots for absolute
// positions [base, base+n). Two only when the range wraps the ring end.
struct RingSpans {
    int off0, len0;   // first span: ring slot `off0`, `len0` rows
    int off1, len1;   // wrapped remainder (len1 == 0 when there is none)
};

RingSpans ring_spans(int base, int n, int ring) {
    RingSpans s{};
    s.off0 = base % ring;
    s.len0 = std::min(n, ring - s.off0);
    s.off1 = 0;
    s.len1 = n - s.len0;
    return s;
}

// Copy `len` rows starting at row `src_row` of a [head_dim, rows, n_head_kv]
// cache tensor into row `dst_row` of a same-shaped snapshot tensor (or the
// reverse when `to_snapshot` is false). All heads move together: the head
// stride is carried by the view, so this is one node, not one node per head.
void append_row_copy(ggml_context * ctx, ggml_cgraph * gf,
                     ggml_tensor * cache_t, ggml_tensor * snap_t,
                     int src_row, int dst_row, int len, bool to_snapshot) {
    if (len <= 0) return;
    const int64_t head_dim = cache_t->ne[0];
    const int64_t n_kv     = cache_t->ne[2];

    ggml_tensor * cv = ggml_view_3d(ctx, cache_t, head_dim, len, n_kv,
                                    cache_t->nb[1], cache_t->nb[2],
                                    (size_t)src_row * cache_t->nb[1]);
    ggml_tensor * sv = ggml_view_3d(ctx, snap_t, head_dim, len, n_kv,
                                    snap_t->nb[1], snap_t->nb[2],
                                    (size_t)dst_row * snap_t->nb[1]);
    ggml_build_forward_expand(gf, to_snapshot ? ggml_cpy(ctx, cv, sv)
                                              : ggml_cpy(ctx, sv, cv));
}

// Build and run the save/restore graph for every SWA layer.
bool run_snapshot_graph(ggml_backend_t backend, const MuseWeights & w,
                        MuseCache & cache, MuseKvSnapshot & snap,
                        bool to_snapshot) {
    const RingSpans sp = ring_spans(snap.base_pos, snap.n, cache.swa_ring);

    // 4 copy nodes per layer worst case (2 spans x K/V), each needing its own
    // views; the overhead figure is deliberately generous.
    const size_t arena = ggml_tensor_overhead() * (size_t)(w.n_layer * 24 + 64) +
                         ggml_graph_overhead_custom(4096, false) + (size_t)1024 * 1024;
    std::vector<uint8_t> mem(arena);
    ggml_init_params ip{};
    ip.mem_size   = arena;
    ip.mem_buffer = mem.data();
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { set_last_error("muse snapshot: ggml_init"); return false; }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);

    for (int il = 0; il < w.n_layer; ++il) {
        if (!muse_is_swa_layer(w, il)) continue;      // full layers: cursor only
        ggml_tensor * kc = cache.k[(size_t)il];
        ggml_tensor * vc = cache.v[(size_t)il];
        ggml_tensor * ks = snap.k[(size_t)il];
        ggml_tensor * vs = snap.v[(size_t)il];
        if (!kc || !ks) continue;

        append_row_copy(ctx, gf, kc, ks, sp.off0, 0,       sp.len0, to_snapshot);
        append_row_copy(ctx, gf, vc, vs, sp.off0, 0,       sp.len0, to_snapshot);
        append_row_copy(ctx, gf, kc, ks, sp.off1, sp.len0, sp.len1, to_snapshot);
        append_row_copy(ctx, gf, vc, vs, sp.off1, sp.len0, sp.len1, to_snapshot);
    }

    bool ok = true;
    if (ggml_graph_n_nodes(gf) > 0) {
        static thread_local ggml_gallocr_t alloc = nullptr;
        if (!alloc) alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            set_last_error("muse snapshot: graph alloc failed");
            ok = false;
        } else if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            set_last_error("muse snapshot: graph compute failed");
            ok = false;
        }
    }
    ggml_free(ctx);
    return ok;
}

}  // namespace

bool create_muse_kv_snapshot(ggml_backend_t backend, const MuseWeights & w,
                             const MuseCache & cache, int max_depth,
                             MuseKvSnapshot & out) {
    free_muse_kv_snapshot(out);
    if (max_depth <= 0) {
        set_last_error("muse snapshot: max_depth must be positive");
        return false;
    }
    // A speculation deeper than the ring would alias itself WITHIN one forward,
    // and no saved copy could undo that. Refuse rather than silently corrupt.
    if (max_depth > cache.swa_ring) {
        set_last_error("muse snapshot: max_depth " + std::to_string(max_depth) +
                       " exceeds swa_ring " + std::to_string(cache.swa_ring));
        return false;
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(w.n_layer * 2 + 16);
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) { set_last_error("muse snapshot: ggml_init"); return false; }

    out.k.assign((size_t)w.n_layer, nullptr);
    out.v.assign((size_t)w.n_layer, nullptr);
    for (int il = 0; il < w.n_layer; ++il) {
        if (!muse_is_swa_layer(w, il)) continue;   // full layers need no storage
        out.k[(size_t)il] = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F16,
                                               w.head_dim, max_depth, w.n_head_kv);
        out.v[(size_t)il] = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F16,
                                               w.head_dim, max_depth, w.n_head_kv);
    }
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        set_last_error("muse snapshot: backend alloc failed");
        ggml_free(out.ctx); out.ctx = nullptr;
        return false;
    }
    out.max_depth = max_depth;
    out.valid = false;
    return true;
}

void free_muse_kv_snapshot(MuseKvSnapshot & s) {
    if (s.buf) { ggml_backend_buffer_free(s.buf); s.buf = nullptr; }
    if (s.ctx) { ggml_free(s.ctx); s.ctx = nullptr; }
    s.k.clear(); s.v.clear();
    s.max_depth = 0; s.base_pos = -1; s.n = 0; s.valid = false;
}

bool muse_kv_snapshot_save(ggml_backend_t backend, const MuseWeights & w,
                           MuseCache & cache, int base_pos, int n,
                           MuseKvSnapshot & snap) {
    if (!snap.ctx) {
        set_last_error("muse snapshot: not allocated");
        return false;
    }
    if (n <= 0 || n > snap.max_depth) {
        set_last_error("muse snapshot: depth " + std::to_string(n) +
                       " outside [1, " + std::to_string(snap.max_depth) + "]");
        return false;
    }
    snap.base_pos = base_pos;
    snap.n        = n;
    snap.valid    = false;
    if (!run_snapshot_graph(backend, w, cache, snap, /*to_snapshot=*/true)) {
        return false;
    }
    snap.valid = true;
    return true;
}

bool muse_kv_snapshot_restore(ggml_backend_t backend, const MuseWeights & w,
                              MuseCache & cache, MuseKvSnapshot & snap) {
    if (!snap.valid) {
        set_last_error("muse snapshot: restore without a live save");
        return false;
    }
    if (!run_snapshot_graph(backend, w, cache, snap, /*to_snapshot=*/false)) {
        return false;
    }
    // One-shot: a second restore from the same save would write rows that the
    // caller has since legitimately re-filled.
    snap.valid = false;
    return true;
}

}  // namespace dflash::common
