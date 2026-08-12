// Numerical parity for the muse-glimmer forward graph.
//
// Runs lucebox's graph (loader + muse_graph.cpp builders) over a real
// artifact for a fixed token sequence on the CPU backend, and compares the
// resulting logits against a reference dump produced by llama.cpp — the
// implementation the geo-quant artifacts were gated on. Reading a graph
// against a reference and *believing* it is how the ATEM renderer shipped
// three bugs earlier in this branch; this is the equivalent check for the
// forward pass.
//
//   MUSE_GGUF=/path/model.gguf            the artifact (required)
//   MUSE_REF_LOGITS=/path/logits.bin      float32 [n_vocab] for the LAST
//                                         token of MUSE_PROMPT_IDS (required)
//   MUSE_PROMPT_IDS=1,2,3                 token ids (default: a short fixed
//                                         sequence)
//
// Skips (77) when either input is absent so CI stays green.
//
// Tolerance is CALIBRATED, not guessed. Logits are compared after the tanh
// softcap so they live in [-20, 20], and the reference implementation is not
// self-consistent to better than that: llama.cpp's own CUDA vs CPU logits on
// this artifact and prompt differ by rms 0.107 / max 0.522 (measured
// 2026-08-12; the model is heavily quantized — Q2_K embeddings and head — so
// dequant and reduction order matter). A threshold below that noise floor
// fails a correct port. The bar here is rms <= 0.25 (~2.3x the measured
// reference self-noise), plus exact argmax agreement and top-8 set overlap,
// which is what actually protects sampling behaviour.
//
// For scale: the errors this catches are far larger. Using NeoX rope instead
// of NORMAL (the gemma4-shaped mistake) measured rms 1.08 / max 4.63 while
// leaving argmax intact on a short prompt.

#include "muse_internal.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace dflash::common;

static std::vector<int32_t> parse_ids(const char * s) {
    std::vector<int32_t> ids;
    if (!s || !*s) return ids;
    const char * p = s;
    while (*p) {
        char * end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) break;
        ids.push_back((int32_t)v);
        p = (*end == ',') ? end + 1 : end;
        if (!*end) break;
    }
    return ids;
}

int main() {
    const char * path = std::getenv("MUSE_GGUF");
    const char * ref  = std::getenv("MUSE_REF_LOGITS");
    if (!path || !*path || !ref || !*ref) {
        std::printf("SKIP: set MUSE_GGUF and MUSE_REF_LOGITS "
                    "(reference logits from llama.cpp)\n");
        return 77;
    }

    std::vector<int32_t> ids = parse_ids(std::getenv("MUSE_PROMPT_IDS"));
    if (ids.empty()) ids = {200000, 3923, 374, 220, 17, 10, 17, 30};

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { std::fprintf(stderr, "FAIL: cpu backend init\n"); return 1; }

    MuseWeights w;
    if (!load_muse_gguf(path, backend, w)) {
        std::fprintf(stderr, "FAIL: load: %s\n", dflash27b_last_error());
        return 1;
    }

    const int n_tokens = (int)ids.size();
    const int n_ctx    = 256;   // ample for the fixed probe sequence

    // KV cache: one K and one V per layer, [head_dim, n_ctx, n_head_kv].
    // SWA layers would normally ring at `sliding_window`, but the probe is
    // far shorter than the 2048 window, so a flat cache is exact here.
    std::vector<ggml_tensor *> cache_k(w.n_layer), cache_v(w.n_layer);
    ggml_init_params cp{};
    cp.mem_size = ggml_tensor_overhead() * (size_t)(w.n_layer * 2 + 16);
    cp.no_alloc = true;
    ggml_context * cctx = ggml_init(cp);
    for (int il = 0; il < w.n_layer; ++il) {
        cache_k[il] = ggml_new_tensor_3d(cctx, GGML_TYPE_F16, w.head_dim,
                                         n_ctx, w.n_head_kv);
        cache_v[il] = ggml_new_tensor_3d(cctx, GGML_TYPE_F16, w.head_dim,
                                         n_ctx, w.n_head_kv);
    }
    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, backend);
    if (!cbuf) { std::fprintf(stderr, "FAIL: kv alloc\n"); return 1; }
    for (int il = 0; il < w.n_layer; ++il) {
        ggml_backend_tensor_memset(cache_k[il], 0, 0, ggml_nbytes(cache_k[il]));
        ggml_backend_tensor_memset(cache_v[il], 0, 0, ggml_nbytes(cache_v[il]));
    }

    // Build the graph.
    const size_t arena = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() +
                         (size_t)16 * 1024 * 1024;
    std::vector<uint8_t> abuf(arena);
    ggml_init_params ip{};
    ip.mem_size   = arena;
    ip.mem_buffer = abuf.data();
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

    ggml_tensor * inp_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(inp_embd);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(positions);
    // FA mask is [kv_len_padded, n_tokens_padded]; built causal below.
    const int kv_pad = (n_tokens + 255) & ~255;
    // FA on CPU wants the query rows padded to a multiple of 64.
    const int q_pad  = (n_tokens + 63) & ~63;
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_pad, q_pad);
    ggml_set_input(mask);

    ggml_tensor * cur = build_muse_inp_norm(ctx, w, inp_embd);
    for (int il = 0; il < w.n_layer; ++il) {
        cur = build_muse_layer(ctx, gf, w, cache_k[il], cache_v[il], il, cur,
                               positions, mask, /*kv_idx=*/nullptr,
                               /*kv_start=*/0, n_tokens);
    }
    cur = build_muse_head(ctx, w, cur);
    ggml_build_forward_expand(gf, cur);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        std::fprintf(stderr, "FAIL: graph alloc\n"); return 1;
    }

    // Inputs: embeddings via the CPU embedder, positions, causal mask.
    std::vector<float> embd((size_t)w.n_embd * n_tokens);
    if (!w.embedder.embed(ids.data(), n_tokens, embd.data())) {
        std::fprintf(stderr, "FAIL: embedder.embed\n");
        return 1;
    }
    ggml_backend_tensor_set(inp_embd, embd.data(), 0, ggml_nbytes(inp_embd));

    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; ++i) pos[(size_t)i] = i;
    ggml_backend_tensor_set(positions, pos.data(), 0, ggml_nbytes(positions));

    std::vector<ggml_fp16_t> mbuf((size_t)kv_pad * q_pad,
                                  ggml_fp32_to_fp16(-INFINITY));
    for (int q = 0; q < n_tokens; ++q) {
        for (int k = 0; k <= q; ++k) {
            mbuf[(size_t)q * kv_pad + k] = ggml_fp32_to_fp16(0.0f);
        }
    }
    ggml_backend_tensor_set(mask, mbuf.data(), 0, ggml_nbytes(mask));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: graph compute\n"); return 1;
    }

    // Logits for the LAST token.
    std::vector<float> got((size_t)w.n_vocab);
    ggml_backend_tensor_get(cur, got.data(),
                            (size_t)(n_tokens - 1) * w.n_vocab * sizeof(float),
                            got.size() * sizeof(float));

    std::vector<float> want((size_t)w.n_vocab);
    {
        std::ifstream f(ref, std::ios::binary);
        if (!f) { std::fprintf(stderr, "FAIL: cannot read %s\n", ref); return 1; }
        f.read((char *)want.data(), (std::streamsize)(want.size() * sizeof(float)));
        if (f.gcount() != (std::streamsize)(want.size() * sizeof(float))) {
            std::fprintf(stderr,
                         "FAIL: %s holds %lld bytes, expected %zu (n_vocab=%d f32)\n",
                         ref, (long long)f.gcount(), want.size() * sizeof(float),
                         w.n_vocab);
            return 1;
        }
    }

    int argmax_got = 0, argmax_want = 0;
    double max_abs = 0.0, sum_sq = 0.0;
    for (int i = 0; i < w.n_vocab; ++i) {
        if (got[(size_t)i]  > got[(size_t)argmax_got])   argmax_got = i;
        if (want[(size_t)i] > want[(size_t)argmax_want]) argmax_want = i;
        const double d = std::fabs((double)got[(size_t)i] - (double)want[(size_t)i]);
        if (d > max_abs) max_abs = d;
        sum_sq += d * d;
    }
    const double rms = std::sqrt(sum_sq / (double)w.n_vocab);

    // Top-8 set overlap: the distribution head is what sampling sees, and a
    // graph error that preserves argmax can still churn the rest of it.
    auto top_k = [&](const std::vector<float> & v, int k) {
        std::vector<int> idx((size_t)v.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return v[(size_t)a] > v[(size_t)b]; });
        idx.resize((size_t)k);
        std::sort(idx.begin(), idx.end());
        return idx;
    };
    const int kTop = 8;
    const std::vector<int> tg = top_k(got, kTop), tw = top_k(want, kTop);
    std::vector<int> inter;
    std::set_intersection(tg.begin(), tg.end(), tw.begin(), tw.end(),
                          std::back_inserter(inter));
    const int overlap = (int)inter.size();

    std::printf("muse graph parity: argmax got=%d want=%d | top%d overlap=%d/%d "
                "| max|d|=%.4f rms=%.5f (n_vocab=%d, %d tokens)\n",
                argmax_got, argmax_want, kTop, overlap, kTop, max_abs, rms,
                w.n_vocab, n_tokens);

    int fails = 0;
    if (argmax_got != argmax_want) {
        std::fprintf(stderr, "FAIL: argmax disagrees (got %d, want %d)\n",
                     argmax_got, argmax_want);
        ++fails;
    }
    // See the header note: 0.25 is ~2.3x llama.cpp's own measured
    // cross-backend self-noise on this artifact (rms 0.107), and ~8x below
    // the rope-type error this harness caught (rms 1.08).
    if (rms > 0.25) {
        std::fprintf(stderr, "FAIL: logit RMS %.5f exceeds 0.25\n", rms);
        ++fails;
    }
    if (overlap < kTop - 1) {
        std::fprintf(stderr, "FAIL: top-%d overlap %d/%d\n", kTop, overlap, kTop);
        ++fails;
    }

    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(cbuf);
    ggml_free(cctx);
    free_muse_weights(w);
    ggml_backend_free(backend);

    if (fails == 0) { std::printf("test_muse_graph_parity: OK\n"); return 0; }
    return 1;
}
