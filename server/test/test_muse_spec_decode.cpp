// Speculative decode must be TOKEN-IDENTICAL to greedy autoregressive decode.
//
// That is the whole contract. A spec loop with a subtly wrong acceptance rule,
// a stale feature ring or an unrolled-back SWA row still emits fluent text —
// just not the text the model would have produced — so "it looks fine" proves
// nothing here and speed proves nothing either.
//
// Four checks, layered so that no one of them can pass vacuously:
//
//  1. The spec path actually RAN and accepted draft tokens. Identity between
//     two AR runs is trivially true, so without this the headline assertion
//     could pass because spec decode quietly fell back to AR.
//  2. Spec output == AR output, token for token.
//  3. NEGATIVE CONTROL — could check 2 have failed? The two streams are
//     compared against a stream from a DIFFERENT prompt, which must differ. A
//     prompt that ends in an immediate EOS makes every stream a 1-token match,
//     and that is a passing test that measures nothing.
//  4. NEGATIVE CONTROL on the mechanism — re-run with the drafter's hidden
//     states double-normed before the lm_head (the target's out_norm applied
//     on top of the drafter's own), so the drafter proposes worse tokens.
//     Acceptance rate must MOVE (it is a real knob on a real path) while the
//     output stays identical (verification, not the drafter, decides what is
//     emitted). If acceptance does not move, the projection is not on the path
//     we think it is; if the output moves, the drafter is steering the result.
//
// AR is a genuinely different code path (do_ar_decode: one token per forward,
// no drafter, no rollback), reached with GenerateRequest::force_ar_decode —
// not a second run of the spec loop, which would agree with itself.
//
//   MUSE_GGUF=/path/muse.gguf          target artifact  (required; else 77)
//   MUSE_DRAFT_GGUF=/path/dflash.gguf  vendor drafter   (required; else 77)
//   MUSE_SPEC_N=64                     tokens to generate (default 64)
//   MUSE_SPEC_PROMPT_LEN=32            prompt length (default 32)

#include "muse_backend.h"
#include "common/model_backend.h"
#include "server/tokenizer.h"
#include "internal.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

int g_fails = 0;

int env_int(const char * k, int d) {
    const char * v = getenv(k);
    return (v && *v) ? atoi(v) : d;
}

void fail(const std::string & msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
    ++g_fails;
}

// Real text, tokenized with the model's own vocab. The invariant would hold on
// arbitrary ids too, but the ACCEPTANCE RATE would not mean anything: random
// ids drive the model into a degenerate short loop, which is neither the
// distribution the drafter was trained on nor what anyone serves. Measured
// side by side on this artifact, a synthetic-id prompt reports 12.5%
// acceptance and a natural-language one reports far more — so a number taken
// from synthetic ids would understate the feature by a wide margin.
//
// Falls back to synthetic ids if the vocab cannot be read, and says so, rather
// than reporting a number whose provenance is unclear.
std::vector<int32_t> make_prompt(const Tokenizer * tok, const std::string & text,
                                 int fallback_len, int32_t fallback_base) {
    if (tok) {
        std::vector<int32_t> ids = tok->encode(text);
        if (!ids.empty()) return ids;
    }
    std::vector<int32_t> p((size_t)fallback_len);
    for (int i = 0; i < fallback_len; ++i) {
        p[(size_t)i] = fallback_base + (i * 131) % 4096;
    }
    return p;
}

struct RunOut {
    std::vector<int32_t> tokens;
    // Target top-2 logit margin at each committed position (spec runs only).
    std::vector<float>   margins;
    float accept_rate = 0.0f;
    bool  spec_ran    = false;
    double decode_s   = 0.0;
};

bool run(MuseBackend & be, const std::vector<int32_t> & prompt, int n_gen,
         bool force_ar, RunOut & out) {
    GenerateRequest req;
    req.prompt          = prompt;
    req.n_gen           = n_gen;
    req.do_sample       = false;
    req.force_ar_decode = force_ar;

    DaemonIO io;              // stream_fd < 0: emit() is a no-op sink
    be.spec_collect_margins(force_ar ? nullptr : &out.margins);
    GenerateResult r = be.generate(req, io);
    be.spec_collect_margins(nullptr);
    if (!r.ok()) {
        fail(std::string(force_ar ? "AR" : "spec") + " generate failed: " +
             std::string(r.error_code()) + " " + std::string(r.error_detail()));
        return false;
    }
    out.tokens      = r.tokens;
    out.accept_rate = r.accept_rate;
    out.spec_ran    = r.spec_decode_ran;
    out.decode_s    = r.decode_s;
    return true;
}

void print_head(const char * label, const std::vector<int32_t> & t) {
    std::printf("%s (%zu tokens):", label, t.size());
    for (size_t i = 0; i < t.size() && i < 12; ++i) std::printf(" %d", t[i]);
    if (t.size() > 12) std::printf(" ...");
    std::printf("\n");
}

int first_divergence(const std::vector<int32_t> & a,
                     const std::vector<int32_t> & b) {
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) return (int)i;
    return a.size() == b.size() ? -1 : (int)n;
}

}  // namespace

int main() {
    const char * target_path = getenv("MUSE_GGUF");
    const char * draft_path  = getenv("MUSE_DRAFT_GGUF");
    if (!target_path || !*target_path) {
        std::fprintf(stderr, "test_muse_spec_decode: set MUSE_GGUF\n");
        return 77;
    }
    if (!draft_path || !*draft_path) {
        std::fprintf(stderr, "test_muse_spec_decode: set MUSE_DRAFT_GGUF\n");
        return 77;
    }
    const int n_gen      = env_int("MUSE_SPEC_N", 64);
    const int prompt_len = env_int("MUSE_SPEC_PROMPT_LEN", 32);

    MuseBackendConfig cfg;
    cfg.model_path = target_path;
    cfg.draft_path = draft_path;
    cfg.device.gpu = 0;
    cfg.device.max_ctx = 4096;
    cfg.chunk      = 512;

    MuseBackend be(cfg);
    if (!be.init()) {
        std::fprintf(stderr, "backend init failed\n");
        return 1;
    }

    Tokenizer tok;
    const bool have_tok = tok.load_from_gguf(target_path);
    if (!have_tok) {
        std::fprintf(stderr, "NOTE: vocab unavailable; falling back to "
                             "synthetic token ids. The invariant still holds, "
                             "but the acceptance rate below is not "
                             "representative.\n");
    }
    // Two prompts with different jobs. The IDENTITY prompt continues a rigid
    // pattern, so the target's top-2 margins stay far outside the batch-vs-
    // sequential drift band for a long run and the token-identity assertion
    // actually covers tokens. The NATURAL prompt is ordinary prose — its
    // margins fall inside the drift band within a few tokens (measured:
    // position 1 at margin 0.23), which makes it useless for identity but
    // representative for the acceptance rate.
    // Numeric counting rather than words: the continuation is as close to
    // forced as this model gets, which keeps the top-2 margins far outside the
    // drift band for a long run. Word-counting was tried first and its margins
    // fell inside the band by position 19 on gfx1201, where the kernels are
    // noisiest — not wrong, just too little coverage to prove anything.
    const std::vector<int32_t> prompt_identity = make_prompt(
        have_tok ? &tok : nullptr,
        "Continue this sequence, one number per line, nothing else.\n"
        "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n13\n14\n",
        prompt_len, 1000);
    const std::vector<int32_t> prompt = make_prompt(
        have_tok ? &tok : nullptr,
        "The sliding-window attention cache stores keys and values in a ring "
        "buffer indexed by absolute position modulo the ring size. This is "
        "cheap for long context, but it means a speculative write can clobber "
        "history that is still live, so",
        prompt_len, 1000);
    const std::vector<int32_t> prompt2 = make_prompt(
        have_tok ? &tok : nullptr,
        "In the spring of that year the harbour froze over for the first time "
        "in living memory, and the fishing boats sat locked in the ice while "
        "their owners walked out across it to argue about",
        prompt_len, 7000);
    std::printf("prompts: identity %zu / natural %zu tokens (%s)\n",
                prompt_identity.size(), prompt.size(),
                have_tok ? "real text" : "synthetic ids");

    // The headline invariant — with the one exception no implementation can
    // remove. ggml runs a matrix-VECTOR kernel at one token and a GEMM at
    // many, so a 16-token verify forward and 16 single-token forwards
    // accumulate in different orders. Where the target's own top-2 margin is
    // wider than that drift, the two paths must agree exactly; where it is
    // narrower, which token wins is kernel-scheduling luck, not a property
    // spec decode can be held to.
    //
    // So: identity is required up to the first committed position whose
    // margin is inside the drift band. Diverging at a wide-margin position is
    // a real bug and fails. `min_coverage` guards the check against becoming
    // vacuous — a prompt whose margins collapse immediately proves nothing.
    //
    // The threshold has to clear the drift on the WORST platform this runs on,
    // and that is not a constant to guess: test_muse_verify_capture measures it
    // per run and reports max drift 0.42 on H200 (CUDA), 0.56 on gfx1151 and
    // 0.77 on gfx1201 (HIP) for the same batch — the AMD kernels drift ~1.8x
    // further than CUDA, which a 0.5 constant calibrated on H200 alone did not
    // cover. 1.0 clears all three and still sits well under the 1.3-2.7
    // separation of a confident prediction, so a real accept-rule / rollback /
    // KV bug fails loudly. Raise it from a measurement, never by feel.
    const float kTieMargin = 1.0f;
    auto first_tie_of = [&](const RunOut & r) {
        for (size_t i = 0; i < r.margins.size(); ++i) {
            if (r.margins[i] < kTieMargin) return (int)i;
        }
        return (int)r.margins.size();
    };
    auto check_identity = [&](const char * label, const RunOut & ar_ref,
                              const RunOut & sp_run, int min_coverage) {
        const int first_tie = first_tie_of(sp_run);
        const int diverge   = first_divergence(sp_run.tokens, ar_ref.tokens);
        if (diverge >= 0 && diverge < first_tie) {
            std::fprintf(stderr,
                "FAIL [%s]: speculative output differs from greedy AR at "
                "position %d, where the target's own top-2 margin was %.4g "
                "(>= %.3g). That is not a tie-break — the accept rule, the "
                "rollback or the KV state is wrong. (spec %d vs AR %d)\n",
                label, diverge,
                diverge < (int)sp_run.margins.size()
                    ? sp_run.margins[(size_t)diverge] : 0.0f,
                kTieMargin,
                diverge < (int)sp_run.tokens.size()
                    ? sp_run.tokens[(size_t)diverge] : -1,
                diverge < (int)ar_ref.tokens.size()
                    ? ar_ref.tokens[(size_t)diverge] : -1);
            ++g_fails;
            return;
        }
        if (diverge < 0) {
            std::printf("invariant [%s]: %zu/%zu tokens identical to greedy AR "
                        "(no divergence at all)\n", label,
                        sp_run.tokens.size(), ar_ref.tokens.size());
        } else {
            std::printf("invariant [%s]: identical to greedy AR up to the first "
                        "in-drift-band margin (position %d, margin %.4g); "
                        "diverged at %d\n", label, first_tie,
                        first_tie < (int)sp_run.margins.size()
                            ? sp_run.margins[(size_t)first_tie] : 0.0f,
                        diverge);
        }
        const int covered = diverge < 0 ? (int)sp_run.tokens.size() : first_tie;
        if (covered < min_coverage) {
            fail(std::string(label) + ": the identity check covered only " +
                 std::to_string(covered) + " tokens (want >= " +
                 std::to_string(min_coverage) + ") — margins collapsed too "
                 "early for this to prove anything");
        }
    };

    // ── Identity on the rigid prompt: margins stay wide, so this covers a
    //    long stretch and a real accept/rollback/KV bug cannot hide in the
    //    drift band.
    {
        RunOut ar_id, sp_id;
        if (!run(be, prompt_identity, n_gen, /*force_ar=*/true, ar_id)) return 1;
        if (!run(be, prompt_identity, n_gen, /*force_ar=*/false, sp_id)) return 1;
        print_head("AR  [identity]", ar_id.tokens);
        print_head("spec[identity]", sp_id.tokens);
        if (!sp_id.spec_ran) {
            fail("spec run did not report spec_decode_ran — it fell back to "
                 "AR, so the identity check proves nothing");
        }
        check_identity("identity-prompt", ar_id, sp_id, /*min_coverage=*/24);
    }

    // ── Natural prose: identity holds only up to the drift band (measured to
    //    collapse within a few tokens), so this prompt's job is the
    //    representative acceptance rate and the controls.
    RunOut ar;
    if (!run(be, prompt, n_gen, /*force_ar=*/true, ar)) return 1;
    if (ar.spec_ran) fail("force_ar_decode still reported spec_decode_ran");
    RunOut sp;
    if (!run(be, prompt, n_gen, /*force_ar=*/false, sp)) return 1;
    print_head("AR  [natural]", ar.tokens);
    print_head("spec[natural]", sp.tokens);
    if (sp.accept_rate <= 0.0f) {
        fail("acceptance rate is 0 on the natural prompt: the drafter "
             "contributed no accepted tokens");
    }
    check_identity("natural-prompt", ar, sp, /*min_coverage=*/1);

    // Throughput has to be measured with the instrumentation OFF. Collecting
    // margins forces verify_batch to keep its full [16 x 202048] f32 logits
    // (13 MB per round) and adds a host-side top-2 scan over all of it, which
    // on gfx1201 turned a real ~0.6x into a reported 0.32x. Every number below
    // comes from this uninstrumented pair.
    RunOut ar_t, sp_t;
    {
        GenerateRequest req;
        req.prompt = prompt; req.n_gen = n_gen; req.do_sample = false;
        DaemonIO io;
        be.spec_collect_margins(nullptr);
        req.force_ar_decode = true;
        GenerateResult r1 = be.generate(req, io);
        req.force_ar_decode = false;
        GenerateResult r2 = be.generate(req, io);
        if (!r1.ok() || !r2.ok()) { fail("uninstrumented timing pass failed"); }
        ar_t.decode_s = r1.decode_s; ar_t.tokens = r1.tokens;
        sp_t.decode_s = r2.decode_s; sp_t.tokens = r2.tokens;
        sp_t.accept_rate = r2.accept_rate;
    }

    // Control: the comparison above must be capable of failing. Two streams
    // that are both one EOS token long match trivially.
    if (ar.tokens.size() < 8) {
        fail("AR produced " + std::to_string(ar.tokens.size()) +
             " tokens — too few for the identity check to mean anything "
             "(raise MUSE_SPEC_N or pick a prompt that does not stop at once)");
    }
    {
        RunOut other;
        if (!run(be, prompt2, n_gen, /*force_ar=*/true, other)) return 1;
        if (other.tokens == ar.tokens) {
            fail("control: a DIFFERENT prompt produced the same token stream, "
                 "so token equality is not discriminating here");
        } else {
            std::printf("control: a different prompt diverges at position %d — "
                        "the equality check can fail\n",
                        first_divergence(other.tokens, ar.tokens));
        }
    }

    // 4. Control: perturb the drafter, not the verifier. Double-norming the
    //    drafter's hidden states before the lm_head must degrade the proposals
    //    (acceptance moves) without changing a single emitted token.
    {
        if (!be.spec_set_project_out_norm(true)) {
            fail("control: no drafter to perturb");
        } else {
            RunOut perturbed;
            if (!run(be, prompt, n_gen, /*force_ar=*/false, perturbed)) return 1;
            be.spec_set_project_out_norm(false);

            std::printf("control: double-normed projection -> acceptance %.1f%% "
                        "(vs %.1f%%)\n",
                        100.0 * perturbed.accept_rate, 100.0 * sp.accept_rate);
            if (perturbed.accept_rate == sp.accept_rate) {
                fail("control: applying out_norm changed the acceptance rate by "
                     "exactly nothing — the lm_head projection is not on the "
                     "path this test believes it is");
            }
            // Same tie rule as the headline check: a worse drafter changes
            // which positions land in which batch, so it hits the target's
            // near-ties in different places. What must NOT happen is a
            // divergence at a position the target was confident about.
            int p_tie = (int)perturbed.margins.size();
            for (size_t i = 0; i < perturbed.margins.size(); ++i) {
                if (perturbed.margins[i] < kTieMargin) { p_tie = (int)i; break; }
            }
            const int p_div = first_divergence(perturbed.tokens, ar.tokens);
            if (p_div >= 0 && p_div < p_tie) {
                std::fprintf(stderr,
                    "FAIL: control: a WORSE drafter changed the output at "
                    "position %d, where the target's margin was %.4g — "
                    "verification is not governing what is emitted\n",
                    p_div,
                    p_div < (int)perturbed.margins.size()
                        ? perturbed.margins[(size_t)p_div] : 0.0f);
                ++g_fails;
            } else {
                std::printf("control: output unchanged under a worse drafter up "
                            "to the first tie (%d tokens) — verification "
                            "governs, the drafter only accelerates\n", p_tie);
            }
        }
    }

    std::printf("\nuninstrumented: acceptance %.1f%% over %d tokens; decode "
                "%.3f s spec vs %.3f s AR = %.2fx (%.1f vs %.1f tok/s)\n",
                100.0 * sp_t.accept_rate, (int)sp_t.tokens.size(),
                sp_t.decode_s, ar_t.decode_s,
                sp_t.decode_s > 0.0 ? ar_t.decode_s / sp_t.decode_s : 0.0,
                sp_t.decode_s > 0.0 ? sp_t.tokens.size() / sp_t.decode_s : 0.0,
                ar_t.decode_s > 0.0 ? ar_t.tokens.size() / ar_t.decode_s : 0.0);

    be.shutdown();

    if (g_fails == 0) { std::printf("test_muse_spec_decode: OK\n"); return 0; }
    std::fprintf(stderr, "test_muse_spec_decode: %d failure(s)\n", g_fails);
    return 1;
}
