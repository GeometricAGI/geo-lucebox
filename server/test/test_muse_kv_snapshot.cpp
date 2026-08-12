// KV snapshot/restore round-trip for speculative-decode rollback.
//
// The invariant: running a speculative forward and then restoring must leave
// the cache indistinguishable from never having run it. "Indistinguishable"
// means a subsequent step produces BIT-IDENTICAL logits, not close ones — a
// rollback that leaks a few wrong keys shifts logits slightly and flips greedy
// tokens occasionally, which is the worst possible failure mode to debug.
//
// MEASURED FINDING, and the reason this test is shaped the way it is.
//
// A first version checked the invariant through logits and its negative control
// FAILED: skipping the restore entirely changed nothing. That is not a broken
// test, it is a real property of this cache.
//
// A speculative span at positions [P+1, P+D] writes ring slots (P+i) % S. For
// the probe at P to READ one of those slots, the slot's pre-speculation occupant
// (position P+i-S) must be inside P's window, i.e. P+i-S >= P-window+1, i.e.
//     i >= S - window  ==  the ring HEADROOM.
// And muse_step already REFUSES any forward longer than that headroom (it would
// clobber its own history mid-chunk). So every speculative span that can
// actually run clobbers only rows that are already outside every future query's
// window — the data restore is redundant in the current sizing regime.
//
// That makes a logits-level negative control impossible to satisfy, so the
// PRIMARY test here checks the save/restore MECHANISM directly on cache bytes,
// where the teeth are unambiguous. The logits round-trip is kept as an
// integration check, with its expected-vacuous status stated rather than
// dressed up as proof.
//
// The mechanism is kept (rather than reduced to a cursor restore) because it is
// the correct implementation for the general case: it stops being redundant the
// moment ring sizing, chunk policy or speculation depth changes, and the guard
// that makes the fast path safe lives in muse_step, not here.
//
//   MUSE_GGUF=/path/model.gguf   artifact (required; exits 77 when absent)
//   MUSE_DEPTH=…                 speculation depth to roll back (default 16)

#include "muse_internal.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace dflash::common;

namespace {

int g_fails = 0;

int env_int(const char * k, int d) {
    const char * v = getenv(k);
    return (v && *v) ? atoi(v) : d;
}

// Prefill `n` tokens in `chunk`-sized pieces, exactly as the serving path does.
bool prefill(ggml_backend_t be, const MuseWeights & w, MuseCache & c,
             const std::vector<int32_t> & ids, int chunk,
             std::vector<float> & logits) {
    std::vector<float> embd;
    const int n = (int)ids.size();
    for (int done = 0; done < n; ) {
        const int take = std::min(chunk, n - done);
        embd.resize((size_t)w.n_embd * take);
        if (!w.embedder.embed(ids.data() + done, take, embd.data())) return false;
        if (!muse_step(be, w, c, embd.data(), take, done, logits)) return false;
        done += take;
    }
    return true;
}

bool step_one(ggml_backend_t be, const MuseWeights & w, MuseCache & c,
              int32_t tok, int kv_start, std::vector<float> & logits) {
    std::vector<float> embd((size_t)w.n_embd);
    if (!w.embedder.embed(&tok, 1, embd.data())) return false;
    return muse_step(be, w, c, embd.data(), 1, kv_start, logits);
}

}  // namespace

int main() {
    const char * path = getenv("MUSE_GGUF");
    if (!path || !*path) {
        std::fprintf(stderr, "test_muse_kv_snapshot: set MUSE_GGUF\n");
        return 77;
    }
    const int depth = env_int("MUSE_DEPTH", 16);

    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { std::fprintf(stderr, "GPU init failed\n"); return 1; }

    MuseWeights w;
    if (!load_muse_gguf(path, be, w)) {
        std::fprintf(stderr, "load failed: %s\n", dflash27b_last_error());
        return 1;
    }

    // Force ring < max_ctx so the ring genuinely wraps: create_muse_cache only
    // takes the ring branch when sliding_window < max_ctx, and sizes the ring
    // as window + chunk.
    const int chunk   = 64;
    const int max_ctx = w.sliding_window + 512;
    MuseCache cache;
    if (!create_muse_cache(be, w, max_ctx, cache, chunk)) {
        std::fprintf(stderr, "cache failed: %s\n", dflash27b_last_error());
        return 1;
    }
    if (cache.swa_ring >= cache.max_ctx) {
        std::fprintf(stderr,
                     "SETUP FAIL: ring %d >= max_ctx %d, so the ring cannot wrap "
                     "and this test would pass vacuously\n",
                     cache.swa_ring, cache.max_ctx);
        return 1;
    }

    // Long enough to wrap the ring, leaving room for the speculative span.
    const int n_prompt = cache.swa_ring + 32;
    if (n_prompt + depth + 4 > cache.max_ctx) {
        std::fprintf(stderr, "SETUP FAIL: prompt %d + depth %d exceeds ctx %d\n",
                     n_prompt, depth, cache.max_ctx);
        return 1;
    }
    std::printf("ring=%d max_ctx=%d prompt=%d depth=%d (ring wraps: %s)\n",
                cache.swa_ring, cache.max_ctx, n_prompt, depth,
                n_prompt > cache.swa_ring ? "yes" : "NO");

    std::vector<int32_t> ids((size_t)n_prompt);
    for (int i = 0; i < n_prompt; ++i) ids[(size_t)i] = 1000 + (i % 4096);

    std::vector<float> logits;
    if (!prefill(be, w, cache, ids, chunk, logits)) {
        std::fprintf(stderr, "prefill failed: %s\n", dflash27b_last_error());
        return 1;
    }

    // Reference: the logits a step at `n_prompt` produces on an unpolluted cache.
    const int32_t probe_tok = 4242;
    std::vector<float> ref;
    if (!step_one(be, w, cache, probe_tok, n_prompt, ref)) {
        std::fprintf(stderr, "reference step failed: %s\n", dflash27b_last_error());
        return 1;
    }

    MuseKvSnapshot snap;
    if (!create_muse_kv_snapshot(be, w, cache, depth, snap)) {
        std::fprintf(stderr, "snapshot alloc failed: %s\n", dflash27b_last_error());
        return 1;
    }

    // Speculative span sits immediately after the probe position, exactly as a
    // draft batch would.
    const int spec_base = n_prompt + 1;
    std::vector<int32_t> spec((size_t)depth);
    for (int i = 0; i < depth; ++i) spec[(size_t)i] = 7777 + i;
    std::vector<float> embd((size_t)w.n_embd * depth);

    auto run_spec = [&]() -> bool {
        std::vector<float> throwaway;
        if (!w.embedder.embed(spec.data(), depth, embd.data())) return false;
        return muse_step(be, w, cache, embd.data(), depth, spec_base, throwaway);
    };

    // ── Positive: save -> speculate -> restore -> step must be bit-identical ──
    if (!muse_kv_snapshot_save(be, w, cache, spec_base, depth, snap)) {
        std::fprintf(stderr, "save failed: %s\n", dflash27b_last_error());
        return 1;
    }
    if (!run_spec()) {
        std::fprintf(stderr, "speculative forward failed: %s\n", dflash27b_last_error());
        return 1;
    }
    if (!muse_kv_snapshot_restore(be, w, cache, snap)) {
        std::fprintf(stderr, "restore failed: %s\n", dflash27b_last_error());
        return 1;
    }
    std::vector<float> after;
    if (!step_one(be, w, cache, probe_tok, n_prompt, after)) {
        std::fprintf(stderr, "post-restore step failed: %s\n", dflash27b_last_error());
        return 1;
    }

    if (after.size() != ref.size() ||
        std::memcmp(after.data(), ref.data(), ref.size() * sizeof(float)) != 0) {
        double worst = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            worst = std::max(worst, (double)std::fabs(after[i] - ref[i]));
        }
        std::fprintf(stderr,
                     "FAIL: restore did not reproduce the cache — logits differ "
                     "(max|d| = %.6g, want exactly 0)\n", worst);
        ++g_fails;
    } else {
        std::printf("integration: logits BIT-IDENTICAL after restore (%zu floats)"
                    " [expected to hold with or without the data restore in this"
                    " sizing regime — see the header]\n", ref.size());
    }

    // ── Mechanism test: save -> scribble -> restore, on cache BYTES ─────────
    // This is where the teeth are. Pick a wrapped SWA layer, record the exact
    // rows the snapshot covers, overwrite them with a known pattern, restore,
    // and require the original bytes back. It does not care whether anything
    // ever reads those rows, so it cannot go vacuous the way the logits check
    // does.
    int swa_il = -1;
    for (int il = 0; il < w.n_layer; ++il) {
        if (muse_is_swa_layer(w, il)) { swa_il = il; break; }
    }
    if (swa_il < 0) {
        std::fprintf(stderr, "FAIL: no SWA layer to test the mechanism on\n");
        ++g_fails;
    } else {
        ggml_tensor * kt = cache.k[(size_t)swa_il];
        const int slot0 = spec_base % cache.swa_ring;
        const int span  = std::min(depth, cache.swa_ring - slot0);  // first span only
        const size_t row_elems = (size_t)w.head_dim;
        const size_t bytes = row_elems * (size_t)span * sizeof(uint16_t);

        std::vector<uint16_t> before(row_elems * (size_t)span);
        std::vector<uint16_t> probe(row_elems * (size_t)span);
        std::vector<uint16_t> poison(row_elems * (size_t)span, 0x3C00 /* fp16 1.0 */);

        // Head 0 only; the copy moves all heads together, so head 0 witnessing
        // the round-trip is sufficient and keeps the offsets readable.
        const size_t off = (size_t)slot0 * kt->nb[1];

        ggml_backend_tensor_get(kt, before.data(), off, bytes);
        if (!muse_kv_snapshot_save(be, w, cache, spec_base, depth, snap)) {
            std::fprintf(stderr, "mechanism save failed: %s\n", dflash27b_last_error());
            return 1;
        }
        ggml_backend_tensor_set(kt, poison.data(), off, bytes);
        ggml_backend_tensor_get(kt, probe.data(), off, bytes);
        if (std::memcmp(probe.data(), poison.data(), bytes) != 0) {
            std::fprintf(stderr, "FAIL: could not scribble the cache; the "
                                 "mechanism test would be vacuous\n");
            ++g_fails;
        }
        if (!muse_kv_snapshot_restore(be, w, cache, snap)) {
            std::fprintf(stderr, "mechanism restore failed: %s\n", dflash27b_last_error());
            return 1;
        }
        ggml_backend_tensor_get(kt, probe.data(), off, bytes);
        if (std::memcmp(probe.data(), before.data(), bytes) != 0) {
            size_t bad = 0;
            for (size_t i = 0; i < probe.size(); ++i) if (probe[i] != before[i]) ++bad;
            std::fprintf(stderr, "FAIL: restore did not put the rows back "
                                 "(%zu/%zu fp16 differ)\n", bad, probe.size());
            ++g_fails;
        } else {
            std::printf("mechanism: %d rows on layer %d scribbled and restored "
                        "byte-exactly\n", span, swa_il);
        }

        // Second restore must be refused: the save is one-shot, and re-applying
        // it would overwrite rows the caller has legitimately re-filled.
        if (muse_kv_snapshot_restore(be, w, cache, snap)) {
            std::fprintf(stderr, "FAIL: a consumed snapshot restored twice\n");
            ++g_fails;
        } else {
            std::printf("one-shot: second restore correctly refused\n");
        }
    }

    free_muse_kv_snapshot(snap);
    free_muse_cache(cache);
    free_muse_weights(w);
    ggml_backend_free(be);

    if (g_fails == 0) { std::printf("test_muse_kv_snapshot: OK\n"); return 0; }
    std::fprintf(stderr, "test_muse_kv_snapshot: %d failure(s)\n", g_fails);
    return 1;
}
