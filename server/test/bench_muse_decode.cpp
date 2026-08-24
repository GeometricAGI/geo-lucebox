// Muse-Glimmer decode-throughput bench.
//
// Why this exists: the only decode number we had came from the 122-item golden
// suite (~80 min/artifact), which is far too slow to iterate a kernel against
// and mixes sampling, HTTP and tokenizer cost into the figure. This drives the
// step driver directly — prefill once, then time N single-token steps — so a
// kernel change can be A/B'd in seconds against the SAME artifact the gate ran.
//
// It deliberately does NOT sample: the token fed back is a fixed id. Decode
// cost is token-independent (the same graph runs for any id) and greedy argmax
// over a 202048-vocab logit vector on the host would otherwise show up in the
// measurement as if it were model cost.
//
//   MUSE_GGUF=/path/model.gguf   artifact (required; exits 77 when absent)
//   MUSE_N_PROMPT=…              prefill length in tokens (default 128)
//   MUSE_N_DECODE=…              timed decode steps (default 128)
//   MUSE_WARMUP=…                untimed decode steps first (default 16)
//   MUSE_MAX_CTX=…               cache size (default 2048)
//   MUSE_FA_WINDOW=…             --fa-window equivalent: cap how far back the
//                                FULL-attention layers look (0 = unlimited,
//                                the default). Exposed here so the flag can be
//                                A/B'd for both throughput AND output effect
//                                without standing a server up.
//
// The warmup is not politeness: the first steps pay lazy kernel module loads
// and allocator growth, which at these step counts would dominate.

#include "muse_internal.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"
// `ggml_backend_cuda_init` is the HIP entry point too (the HIP build compiles
// the same ggml-cuda sources), so this bench builds unchanged on both.
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

int env_int(const char * k, int dflt) {
    const char * v = getenv(k);
    return (v && *v) ? atoi(v) : dflt;
}

// Median is reported alongside the mean because a single step that lands on an
// allocator hiccup or a clock-throttle event skews the mean by more than the
// kernel deltas we are trying to resolve.
double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

}  // namespace

int main() {
    const char * path = getenv("MUSE_GGUF");
    if (!path || !*path) {
        std::fprintf(stderr, "bench_muse_decode: set MUSE_GGUF=/path/model.gguf\n");
        return 77;
    }
    const int n_prompt = std::max(1, env_int("MUSE_N_PROMPT", 128));
    const int n_decode = std::max(1, env_int("MUSE_N_DECODE", 128));
    const int n_warmup = std::max(0, env_int("MUSE_WARMUP", 16));
    const int max_ctx  = std::max(n_prompt + n_decode + n_warmup + 8,
                                  env_int("MUSE_MAX_CTX", 2048));

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "bench_muse_decode: GPU backend init failed\n");
        return 1;
    }

    MuseWeights w;
    const double t_load = now_s();
    if (!load_muse_gguf(path, backend, w)) {
        std::fprintf(stderr, "bench_muse_decode: load failed: %s\n",
                     dflash27b_last_error());
        return 1;
    }
    const double load_s = now_s() - t_load;

    MuseCache cache;
    if (!create_muse_cache(backend, w, max_ctx, cache, n_prompt)) {
        std::fprintf(stderr, "bench_muse_decode: cache failed: %s\n",
                     dflash27b_last_error());
        return 1;
    }
    cache.fa_window = env_int("MUSE_FA_WINDOW", 0);
    if (cache.fa_window < 0 || cache.fa_window > cache.max_ctx) {
        std::fprintf(stderr, "bench_muse_decode: MUSE_FA_WINDOW %d out of range "
                             "for ctx %d\n", cache.fa_window, cache.max_ctx);
        return 1;
    }

    // Any in-vocab ids will do; decode cost does not depend on which.
    std::vector<int32_t> ids((size_t)n_prompt);
    for (int i = 0; i < n_prompt; ++i) ids[(size_t)i] = 1000 + (i % 4096);

    std::vector<float> embd, logits;
    auto embed = [&](const int32_t * p, int n) -> bool {
        embd.resize((size_t)w.n_embd * n);
        return w.embedder.embed(p, n, embd.data());
    };

    // ── Prefill ────────────────────────────────────────────────────────
    const double t_pre = now_s();
    if (!embed(ids.data(), n_prompt) ||
        !muse_step(backend, w, cache, embd.data(), n_prompt, 0, logits)) {
        std::fprintf(stderr, "bench_muse_decode: prefill failed: %s\n",
                     dflash27b_last_error());
        return 1;
    }
    ggml_backend_synchronize(backend);
    const double prefill_s = now_s() - t_pre;

    // ── Decode ─────────────────────────────────────────────────────────
    int kv = n_prompt;
    const int32_t tok = 1234;
    std::vector<double> step_ms;
    step_ms.reserve((size_t)n_decode);

    for (int i = 0; i < n_warmup + n_decode; ++i) {
        const bool timed = (i >= n_warmup);
        const double t0 = timed ? now_s() : 0.0;
        if (!embed(&tok, 1) ||
            !muse_step(backend, w, cache, embd.data(), 1, kv, logits)) {
            std::fprintf(stderr, "bench_muse_decode: decode failed at %d: %s\n",
                         i, dflash27b_last_error());
            return 1;
        }
        // muse_step's compute is async on the backend stream; without this the
        // per-step timings would measure enqueue rate, not execution.
        ggml_backend_synchronize(backend);
        if (timed) step_ms.push_back((now_s() - t0) * 1e3);
        kv += 1;
    }

    double sum = 0.0;
    for (double m : step_ms) sum += m;
    const double mean_ms = sum / (double)step_ms.size();
    const double med_ms  = median_of(step_ms);

    std::printf("model            %s\n", path);
    std::printf("load             %.1f s\n", load_s);
    std::printf("prefill          %d tok in %.3f s  (%.1f tok/s)\n",
                n_prompt, prefill_s, n_prompt / prefill_s);
    std::printf("fa_window        %d%s\n", cache.fa_window,
                cache.fa_window ? " (full-attn layers windowed)" : " (unlimited)");
    std::printf("decode           %d steps (warmup %d)\n", n_decode, n_warmup);
    std::printf("  mean           %.3f ms/tok  (%.2f tok/s)\n",
                mean_ms, 1e3 / mean_ms);
    std::printf("  median         %.3f ms/tok  (%.2f tok/s)\n",
                med_ms, 1e3 / med_ms);
    std::printf("  min/max        %.3f / %.3f ms\n",
                *std::min_element(step_ms.begin(), step_ms.end()),
                *std::max_element(step_ms.begin(), step_ms.end()));
    std::fflush(stdout);

    // MUSE_DUMP_LOGITS lets a kernel change be A/B'd for BIT-exactness, not just
    // for speed: dump the final step's logits before and after, and diff the
    // files. A matvec rewrite that claims to preserve the summation order has to
    // produce identical bytes — "close enough" logits silently flip greedy tokens,
    // which is exactly the failure the golden gate is too coarse to localise.
    if (const char * dump = getenv("MUSE_DUMP_LOGITS")) {
        FILE * f = fopen(dump, "wb");
        if (!f) {
            std::fprintf(stderr, "bench_muse_decode: cannot write %s\n", dump);
            return 1;
        }
        fwrite(logits.data(), sizeof(float), logits.size(), f);
        fclose(f);
        std::printf("logits           %zu floats -> %s\n", logits.size(), dump);
    }

    free_muse_cache(cache);
    free_muse_weights(w);
    ggml_backend_free(backend);
    return 0;
}
