// muse_verify_batch + DFlash feature capture.
//
// Two properties, both of which fail silently rather than loudly if wrong:
//
// 1. VERIFY EQUIVALENCE. Speculative decode accepts a draft chain up to the
//    first position where the target disagrees, so verify_batch's per-position
//    argmax has to equal what stepping those same tokens one at a time would
//    produce. If position t's argmax is off, spec decode still emits fluent
//    text — just not the text greedy decoding would have produced, which
//    destroys the "output-identical" guarantee that is the whole point.
//
//    Compared against the SEQUENTIAL path, not against itself: a batched
//    forward that reads its own future keys (a mask bug) agrees with a second
//    batched forward perfectly and only disagrees with one-at-a-time decode.
//
// 2. CAPTURE COVERAGE. The ring must be written at every capture layer for
//    every position in the batch. A capture that silently writes nothing leaves
//    the drafter cross-attending to zeros: it still produces tokens, they are
//    just poor, so acceptance rate quietly collapses instead of anything
//    failing. Checked by zeroing the ring, running a forward, and requiring
//    each capture row-block to be non-zero — plus a control confirming that
//    NON-capture row blocks would have been caught.
//
//   MUSE_GGUF=/path/model.gguf   artifact (required; exits 77 when absent)
//   MUSE_SPEC=…                  batch length to verify (default 8)

#include "muse_internal.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace dflash::common;

namespace {

int g_fails = 0;

int env_int(const char * k, int d) {
    const char * v = getenv(k);
    return (v && *v) ? atoi(v) : d;
}

int argmax_of(const std::vector<float> & v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i) if (v[i] > v[(size_t)best]) best = (int)i;
    return best;
}

}  // namespace

int main() {
    const char * path = getenv("MUSE_GGUF");
    if (!path || !*path) {
        std::fprintf(stderr, "test_muse_verify_capture: set MUSE_GGUF\n");
        return 77;
    }
    const int spec = env_int("MUSE_SPEC", 8);

    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { std::fprintf(stderr, "GPU init failed\n"); return 1; }

    MuseWeights w;
    if (!load_muse_gguf(path, be, w)) {
        std::fprintf(stderr, "load failed: %s\n", dflash27b_last_error());
        return 1;
    }

    const int n_prompt = 24;
    const int max_ctx  = 512;

    std::vector<int32_t> prompt((size_t)n_prompt);
    for (int i = 0; i < n_prompt; ++i) prompt[(size_t)i] = 1000 + i;
    std::vector<int32_t> chain((size_t)spec);
    for (int i = 0; i < spec; ++i) chain[(size_t)i] = 5000 + i * 7;

    std::vector<float> embd, logits;
    auto embed_n = [&](const int32_t * ids, int n) {
        embd.resize((size_t)w.n_embd * n);
        return w.embedder.embed(ids, n, embd.data());
    };

    // ── 1. Sequential reference: step the chain one token at a time ──────────
    std::vector<int32_t> seq_argmax((size_t)spec);
    {
        MuseCache c;
        if (!create_muse_cache(be, w, max_ctx, c, n_prompt)) {
            std::fprintf(stderr, "cache failed: %s\n", dflash27b_last_error());
            return 1;
        }
        if (!embed_n(prompt.data(), n_prompt) ||
            !muse_step(be, w, c, embd.data(), n_prompt, 0, logits)) {
            std::fprintf(stderr, "prefill failed: %s\n", dflash27b_last_error());
            return 1;
        }
        for (int t = 0; t < spec; ++t) {
            if (!embed_n(&chain[(size_t)t], 1) ||
                !muse_step(be, w, c, embd.data(), 1, n_prompt + t, logits)) {
                std::fprintf(stderr, "sequential step %d failed\n", t);
                return 1;
            }
            seq_argmax[(size_t)t] = argmax_of(logits);
        }
        free_muse_cache(c);
    }

    // ── 2. Batched verify over the same chain, with capture active ───────────
    MuseCache c;
    if (!create_muse_cache(be, w, max_ctx, c, n_prompt)) {
        std::fprintf(stderr, "cache failed: %s\n", dflash27b_last_error());
        return 1;
    }
    const std::vector<int> cap_ids = muse_default_capture_layers(w, 5);
    if ((int)cap_ids.size() != 5) {
        std::fprintf(stderr, "FAIL: default capture layers gave %zu, want 5\n",
                     cap_ids.size());
        ++g_fails;
    }
    std::printf("capture layers:");
    for (int il : cap_ids) std::printf(" %d", il);
    std::printf("  (of %d)\n", w.n_layer);

    if (!create_muse_target_feat(be, w, c, cap_ids, max_ctx)) {
        std::fprintf(stderr, "target_feat alloc failed: %s\n", dflash27b_last_error());
        return 1;
    }

    if (!embed_n(prompt.data(), n_prompt) ||
        !muse_step(be, w, c, embd.data(), n_prompt, 0, logits)) {
        std::fprintf(stderr, "prefill(2) failed: %s\n", dflash27b_last_error());
        return 1;
    }

    // Zero the ring so "was it written" is unambiguous for the batch positions.
    ggml_backend_tensor_memset(c.target_feat, 0, 0, ggml_nbytes(c.target_feat));

    std::vector<int32_t> batch_argmax;
    if (!embed_n(chain.data(), spec) ||
        !muse_verify_batch(be, w, c, embd.data(), spec, n_prompt, batch_argmax)) {
        std::fprintf(stderr, "verify_batch failed: %s\n", dflash27b_last_error());
        return 1;
    }

    if (batch_argmax != seq_argmax) {
        std::fprintf(stderr, "FAIL: verify_batch argmax != sequential\n");
        for (int t = 0; t < spec; ++t) {
            if (batch_argmax[(size_t)t] != seq_argmax[(size_t)t]) {
                std::fprintf(stderr, "  pos %d: batch %d vs sequential %d\n",
                             t, batch_argmax[(size_t)t], seq_argmax[(size_t)t]);
            }
        }
        ++g_fails;
    } else {
        std::printf("verify equivalence: %d/%d positions match one-at-a-time "
                    "decode\n", spec, spec);
    }

    // ── 3. Capture coverage ─────────────────────────────────────────────────
    const int hidden = w.n_embd;
    const int fc_in  = (int)cap_ids.size() * hidden;
    std::vector<float> col((size_t)fc_in);
    int empty_blocks = 0, checked = 0;
    for (int t = 0; t < spec; ++t) {
        const int slot = (n_prompt + t) % c.target_feat_cap;
        ggml_backend_tensor_get(c.target_feat, col.data(),
                                (size_t)slot * c.target_feat->nb[1],
                                col.size() * sizeof(float));
        for (int k = 0; k < (int)cap_ids.size(); ++k) {
            bool nonzero = false;
            for (int i = 0; i < hidden && !nonzero; ++i) {
                if (col[(size_t)k * hidden + i] != 0.0f) nonzero = true;
            }
            ++checked;
            if (!nonzero) ++empty_blocks;
        }
    }
    if (empty_blocks) {
        std::fprintf(stderr, "FAIL: %d/%d (capture-layer, position) blocks are "
                             "all-zero — capture did not write them\n",
                     empty_blocks, checked);
        ++g_fails;
    } else {
        std::printf("capture coverage: all %d (layer, position) blocks written\n",
                    checked);
    }

    // Control: a column BEYOND the batch was never captured, so it must still be
    // zero. Without this, a capture that floods the whole ring would pass above.
    {
        const int slot = (n_prompt + spec + 1) % c.target_feat_cap;
        ggml_backend_tensor_get(c.target_feat, col.data(),
                                (size_t)slot * c.target_feat->nb[1],
                                col.size() * sizeof(float));
        bool nonzero = false;
        for (float v : col) if (v != 0.0f) { nonzero = true; break; }
        if (nonzero) {
            std::fprintf(stderr, "FAIL (control): an uncaptured ring column is "
                                 "non-zero — capture is writing out of range\n");
            ++g_fails;
        } else {
            std::printf("control: uncaptured columns untouched\n");
        }
    }

    free_muse_cache(c);
    free_muse_weights(w);
    ggml_backend_free(be);

    if (g_fails == 0) { std::printf("test_muse_verify_capture: OK\n"); return 0; }
    std::fprintf(stderr, "test_muse_verify_capture: %d failure(s)\n", g_fails);
    return 1;
}
