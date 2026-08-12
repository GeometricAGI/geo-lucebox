// Incremental-decode equivalence for the muse step driver.
//
// The single-forward parity test (test_muse_graph_parity) validates the graph
// but says nothing about the KV cache: it prefills once and reads logits. The
// serving path instead prefills a prompt and then decodes one token at a
// time, which exercises the cache append, the SWA ring slot arithmetic, the
// per-step masks and the position stream. An off-by-one there does not crash
// — it produces a plausible but different continuation.
//
// So this compares two ways of reaching the SAME final state:
//
//   A: one prefill of all N prompt tokens          -> logits for token N-1
//   B: prefill of N-1 tokens, then a 1-token step  -> logits for token N-1
//
// A and B must agree: same argmax, negligible RMS. They share a backend, so
// unlike the cross-implementation parity test there is no reduction-order
// excuse — a real divergence here is a cache or mask bug.
//
// With MUSE_REF_LOGITS also set, path B is additionally checked against the
// llama.cpp reference, which ties the incremental path to the external
// implementation rather than only to path A.
//
//   MUSE_GGUF=/path/model.gguf     artifact (required)
//   MUSE_PROMPT_IDS=…              prompt token ids (required, >= 2)
//   MUSE_REF_LOGITS=/path.bin      optional llama.cpp logits for token N-1
//   MUSE_MAX_CTX=…                 cache size (default 512)

#include "muse_internal.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

static int argmax_of(const std::vector<float> & v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i) if (v[i] > v[(size_t)best]) best = (int)i;
    return best;
}

static double rms_between(const std::vector<float> & a,
                          const std::vector<float> & b, double & max_abs) {
    double sum_sq = 0.0;
    max_abs = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > max_abs) max_abs = d;
        sum_sq += d * d;
    }
    return std::sqrt(sum_sq / (double)a.size());
}

// Ring-wrap arithmetic, model-free. An end-to-end wrap needs a prompt longer
// than the 2048-token window; this pins the slot math that makes it correct.
static int test_swa_ring_wrap() {
    int fails = 0;
    auto expect = [&](bool got, bool want, const char * what) {
        if (got != want) {
            std::fprintf(stderr, "FAIL: swa ring: %s (got %d want %d)\n",
                         what, (int)got, (int)want);
            ++fails;
        }
    };
    const int S = 4;

    // Before any wrap (total=3): slots 0..2 hold positions 0..2, slot 3 empty.
    expect(muse_swa_slot_visible(3, 2, 0, S), true,  "t=3 q=2 slot0 (pos 0)");
    expect(muse_swa_slot_visible(3, 2, 2, S), true,  "t=3 q=2 slot2 (pos 2)");
    expect(muse_swa_slot_visible(3, 2, 3, S), false, "t=3 q=2 slot3 (empty)");
    expect(muse_swa_slot_visible(3, 1, 2, S), false, "t=3 q=1 slot2 (future)");

    // After wrap (total=6): slot0 -> pos 4, slot1 -> pos 5, slot2 -> pos 2,
    // slot3 -> pos 3. A query at pos 5 sees the window [2,5] — all four.
    expect(muse_swa_slot_visible(6, 5, 0, S), true, "t=6 q=5 slot0 (pos 4)");
    expect(muse_swa_slot_visible(6, 5, 1, S), true, "t=6 q=5 slot1 (pos 5)");
    expect(muse_swa_slot_visible(6, 5, 2, S), true, "t=6 q=5 slot2 (pos 2)");
    expect(muse_swa_slot_visible(6, 5, 3, S), true, "t=6 q=5 slot3 (pos 3)");

    // A query at pos 4 must NOT see slot1 (pos 5 is in its future) — the case
    // a naive contiguous-span mask gets wrong mid-chunk.
    expect(muse_swa_slot_visible(6, 4, 1, S), false, "t=6 q=4 slot1 (future)");
    // ...and pos 1 has been overwritten by pos 5, so nothing exposes it.
    expect(muse_swa_slot_visible(6, 4, 0, S), true,  "t=6 q=4 slot0 (pos 4)");

    // Window edge: at total=8, q=7, slot3 holds pos 7-... check eviction.
    // slot0 -> 4, slot1 -> 5, slot2 -> 6, slot3 -> 7; window [4,7].
    expect(muse_swa_slot_visible(8, 7, 0, S), true, "t=8 q=7 slot0 (pos 4)");
    expect(muse_swa_slot_visible(8, 4, 2, S), false, "t=8 q=4 slot2 (future 6)");
    return fails;
}

int main() {
    const char * path = std::getenv("MUSE_GGUF");
    const std::vector<int32_t> prompt = parse_ids(std::getenv("MUSE_PROMPT_IDS"));
    if (!path || !*path || prompt.size() < 2) {
        // The ring arithmetic needs no artifact, so check it before skipping.
        if (test_swa_ring_wrap() != 0) return 1;
        std::printf("swa ring wrap: OK\n");
        std::printf("SKIP: set MUSE_GGUF and MUSE_PROMPT_IDS (>= 2 ids) for "
                    "the prefill/incremental check\n");
        return 77;
    }
    const int max_ctx = std::getenv("MUSE_MAX_CTX")
                            ? std::atoi(std::getenv("MUSE_MAX_CTX")) : 512;
    const char * ref = std::getenv("MUSE_REF_LOGITS");

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { std::fprintf(stderr, "FAIL: cpu backend\n"); return 1; }

    MuseWeights w;
    if (!load_muse_gguf(path, backend, w)) {
        std::fprintf(stderr, "FAIL: load: %s\n", dflash27b_last_error());
        return 1;
    }

    const int n = (int)prompt.size();
    std::vector<float> embd((size_t)w.n_embd * n);
    if (!w.embedder.embed(prompt.data(), n, embd.data())) {
        std::fprintf(stderr, "FAIL: embed\n"); return 1;
    }

    // ── Path A: single prefill ─────────────────────────────────────────
    std::vector<float> logits_a;
    {
        MuseCache cache;
        if (!create_muse_cache(backend, w, max_ctx, cache)) {
            std::fprintf(stderr, "FAIL: cache A: %s\n", dflash27b_last_error());
            return 1;
        }
        if (!muse_step(backend, w, cache, embd.data(), n, 0, logits_a)) {
            std::fprintf(stderr, "FAIL: prefill A: %s\n", dflash27b_last_error());
            return 1;
        }
        free_muse_cache(cache);
    }

    // ── Path B: prefill n-1, then one decode step ──────────────────────
    std::vector<float> logits_b;
    {
        MuseCache cache;
        if (!create_muse_cache(backend, w, max_ctx, cache)) {
            std::fprintf(stderr, "FAIL: cache B: %s\n", dflash27b_last_error());
            return 1;
        }
        std::vector<float> tmp;
        if (!muse_step(backend, w, cache, embd.data(), n - 1, 0, tmp)) {
            std::fprintf(stderr, "FAIL: prefill B: %s\n", dflash27b_last_error());
            return 1;
        }
        const float * last = embd.data() + (size_t)(n - 1) * w.n_embd;
        if (!muse_step(backend, w, cache, last, 1, n - 1, logits_b)) {
            std::fprintf(stderr, "FAIL: decode B: %s\n", dflash27b_last_error());
            return 1;
        }
        free_muse_cache(cache);
    }

    int fails = test_swa_ring_wrap();
    double max_ab = 0.0;
    const double rms_ab = rms_between(logits_a, logits_b, max_ab);
    const int am_a = argmax_of(logits_a), am_b = argmax_of(logits_b);
    std::printf("prefill-vs-incremental: argmax %d/%d | max|d|=%.5f rms=%.6f\n",
                am_a, am_b, max_ab, rms_ab);
    if (am_a != am_b) {
        std::fprintf(stderr, "FAIL: argmax differs between prefill and "
                             "incremental decode\n");
        ++fails;
    }
    // Same weights, same backend, same arithmetic — the only difference is
    // how the KV rows got there. 0.01 is far above f16 cache round-trip noise
    // and far below anything a mask or position bug produces.
    if (rms_ab > 0.01) {
        std::fprintf(stderr, "FAIL: prefill/incremental RMS %.6f exceeds 0.01 "
                             "— KV append, ring slot or mask mismatch\n", rms_ab);
        ++fails;
    }

    if (ref && *ref) {
        std::vector<float> want((size_t)w.n_vocab);
        std::ifstream f(ref, std::ios::binary);
        if (!f) { std::fprintf(stderr, "FAIL: cannot read %s\n", ref); return 1; }
        f.read((char *)want.data(), (std::streamsize)(want.size() * sizeof(float)));
        double max_br = 0.0;
        const double rms_br = rms_between(logits_b, want, max_br);
        std::printf("incremental-vs-llama.cpp: argmax %d/%d | max|d|=%.4f "
                    "rms=%.5f\n", am_b, argmax_of(want), max_br, rms_br);
        if (am_b != argmax_of(want)) {
            std::fprintf(stderr, "FAIL: incremental argmax != reference\n");
            ++fails;
        }
        // Same calibrated bar as test_muse_graph_parity (llama.cpp's own
        // cross-backend self-noise on this artifact is rms 0.107).
        if (rms_br > 0.25) {
            std::fprintf(stderr, "FAIL: incremental vs reference RMS %.5f "
                                 "exceeds 0.25\n", rms_br);
            ++fails;
        }
    }

    free_muse_weights(w);
    ggml_backend_free(backend);

    if (fails == 0) { std::printf("test_muse_generate: OK\n"); return 0; }
    return 1;
}
