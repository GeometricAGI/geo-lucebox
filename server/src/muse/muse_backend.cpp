// Muse-Glimmer ModelBackend implementation: chunked prefill, AR decode, and
// greedy-chain DFlash speculative decode (see do_spec_decode).

#include "muse_backend.h"
#include "common/sampler.h"
#include "common/dflash_draft_graph.h"
#include "common/step_graph.h"
#include "internal.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace dflash::common {

namespace {

double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

}  // namespace

MuseBackend::MuseBackend(const MuseBackendConfig & cfg) : cfg_(cfg) {}

MuseBackend::~MuseBackend() { shutdown(); }

void MuseBackend::shutdown() {
    free_decode_draft();
    free_muse_cache(cache_);
    if (loaded_) { free_muse_weights(w_); loaded_ = false; }
    if (backend_) { ggml_backend_free(backend_); backend_ = nullptr; }
}

bool MuseBackend::init() {
    // Opt this model into the batched mix-qtype (MMQ) path. muse's 105/106
    // tensors are dense and go through ggml_mul_mat, which is where MMQ
    // applies; measured 103/122 on the golden suite at 81.0 tok/s with it on,
    // against 101/122 at 43.5 with it off. It is enabled here rather than as a
    // process-wide default because that result is muse's alone — models whose
    // mix tensors are MoE experts reach them through ggml_mul_mat_id and were
    // never measured to benefit. An explicit DFLASH_MIX_MMQ wins over this.
    if (!ggml_cuda_mix_mmq_env_pinned()) {
        ggml_cuda_set_mix_mmq_enabled(true);
    }
    // `ggml_backend_cuda_init` is the HIP entry point too — the HIP build
    // compiles the same ggml-cuda sources, so this line is backend-agnostic
    // exactly as it is for the other families.
    backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[muse] GPU backend init failed (gpu=%d)\n",
                     cfg_.device.gpu);
        return false;
    }
    if (!cfg_.model_path) {
        std::fprintf(stderr, "[muse] no model path\n");
        return false;
    }
    if (!load_muse_gguf(cfg_.model_path, backend_, w_)) {
        std::fprintf(stderr, "[muse] load failed: %s\n", dflash27b_last_error());
        return false;
    }
    loaded_ = true;

    const int max_ctx = cfg_.device.max_ctx > 0 ? cfg_.device.max_ctx : 8192;
    if (!create_muse_cache(backend_, w_, max_ctx, cache_, cfg_.chunk)) {
        std::fprintf(stderr, "[muse] cache failed: %s\n", dflash27b_last_error());
        return false;
    }

    // Reject rather than silently clamp: a window wider than the context is a
    // misunderstanding of the flag, and quietly treating it as "unlimited"
    // would hide that. Negative is nonsense.
    if (cfg_.fa_window < 0 || cfg_.fa_window > cache_.max_ctx) {
        std::fprintf(stderr, "[muse] --fa-window %d is out of range for ctx %d "
                             "(0 = unlimited)\n", cfg_.fa_window, cache_.max_ctx);
        return false;
    }
    cache_.fa_window = cfg_.fa_window;
    if (cache_.fa_window > 0) {
        // Loud on purpose. This model keeps its global context in 13
        // full-attention layers; windowing them can drop the system prompt and
        // tool definitions out of view, which is exactly how tool calling
        // breaks. It has not been scored on the golden suite at any non-zero
        // value, so the honest thing is to say so at startup.
        std::fprintf(stderr,
                     "[muse] WARNING --fa-window %d limits the %d full-attention "
                     "layers to the last %d positions. Untested for quality on "
                     "this model: run the golden suite before trusting it.\n",
                     cache_.fa_window,
                     (int)std::count(w_.swa_layers.begin(), w_.swa_layers.end(), false),
                     cache_.fa_window);
    }

    // A missing or unloadable drafter is not fatal: the AR path is complete on
    // its own, so the daemon comes up serving without spec decode rather than
    // refusing to start.
    if (cfg_.draft_path && *cfg_.draft_path && !load_decode_draft()) {
        std::fprintf(stderr, "[muse] draft unavailable (%s); speculative "
                             "decode disabled\n", dflash27b_last_error());
        free_decode_draft();
    }
    return true;
}

// ── Speculative decode: draft load ─────────────────────────────────────

bool MuseBackend::load_decode_draft() {
    const int draft_gpu = cfg_.draft_gpu >= 0 ? cfg_.draft_gpu : cfg_.device.gpu;
    draft_backend_ = ggml_backend_cuda_init(draft_gpu);
    if (!draft_backend_) {
        std::fprintf(stderr, "[muse] draft GPU init failed (gpu=%d)\n", draft_gpu);
        return false;
    }
    if (!load_draft_gguf(cfg_.draft_path, draft_backend_, dw_, nullptr)) {
        std::fprintf(stderr, "[muse] draft load failed: %s\n",
                     dflash27b_last_error());
        return false;
    }

    // The drafter's `fc` input width fixes how many target layers it expects
    // to cross-attend to, and it must divide the target's hidden size. A
    // mismatch here means the drafter was built for a different target; it
    // would still run and still emit tokens, just against features whose
    // layout it never saw, so check rather than trust the file name.
    const int fc_in = (int)dw_.fc->ne[0];
    if (w_.n_embd == 0 || fc_in % w_.n_embd != 0) {
        std::fprintf(stderr, "[muse] draft fc_in=%d is not a multiple of the "
                             "target n_embd=%d — wrong drafter for this model\n",
                     fc_in, w_.n_embd);
        return false;
    }
    const int n_capture = fc_in / w_.n_embd;
    const int draft_hidden = (int)dw_.fc->ne[1];
    if (draft_hidden != dw_.n_embd) {
        std::printf("[muse] draft: n_embd %d -> %d (from fc weight)\n",
                    dw_.n_embd, draft_hidden);
        dw_.n_embd = draft_hidden;
    }
    if (dw_.block_size <= 1) {
        std::fprintf(stderr, "[muse] draft declares block_size=%d; needs >= 2 "
                             "to speculate\n", dw_.block_size);
        return false;
    }

    // Capture layers come from the drafter, not from a default. The vendor
    // file names [2,14,26,38,50]; muse_default_capture_layers would give
    // 1/13/25/37/49, which is close enough to look right and wrong enough to
    // cost acceptance rate with nothing failing.
    std::vector<int> capture_ids = dw_.capture_layer_ids;
    if ((int)capture_ids.size() != n_capture) {
        if (!capture_ids.empty()) {
            std::fprintf(stderr, "[muse] draft declares %zu capture layers but "
                                 "fc_in implies %d; ignoring the declaration\n",
                         capture_ids.size(), n_capture);
        }
        capture_ids = muse_default_capture_layers(w_, n_capture);
        std::fprintf(stderr, "[muse] draft names no capture layers; falling "
                             "back to evenly-spaced defaults (unverified for "
                             "this drafter)\n");
    }
    if ((int)capture_ids.size() != n_capture) {
        std::fprintf(stderr, "[muse] cannot resolve %d capture layers\n", n_capture);
        return false;
    }
    // A/B knob: the drafter's `dflash.target_layers` does not say whether its
    // indices are 0-based layer OUTPUTS (assumed), 1-based, or layer inputs.
    // An off-by-one still produces features a drafter can partially use, so it
    // costs acceptance without failing anything — this shifts the ids to
    // measure which convention the drafter was trained on.
    if (const char * s = std::getenv("MUSE_CAPTURE_SHIFT")) {
        const int shift = atoi(s);
        for (int & il : capture_ids) {
            il = std::min(std::max(il + shift, 0), w_.n_layer - 1);
        }
        std::fprintf(stderr, "[muse] MUSE_CAPTURE_SHIFT=%d applied\n", shift);
    }
    dw_.n_target_layers = n_capture;

    // The drafter's own SWA metadata is already applied by the loader; only
    // report it, do not reassert it.
    std::printf("[muse] draft loaded: blocks=%d block_size=%d fc_in=%d "
                "draft_hidden=%d capture=%d mask_tok=%d\n",
                dw_.n_layer, dw_.block_size, fc_in, draft_hidden, n_capture,
                dw_.mask_token_id);
    std::printf("[muse] capture layers:");
    for (int il : capture_ids) std::printf(" %d", il);
    std::printf(" (of %d)\n", w_.n_layer);

    // Target-side capture ring, then the draft-side mirror it feeds.
    const int feat_cap = std::min(cache_.max_ctx, std::max(1, cfg_.draft_ctx_max));
    if (!create_muse_target_feat(backend_, w_, cache_, capture_ids, feat_cap)) {
        std::fprintf(stderr, "[muse] target_feat alloc failed: %s\n",
                     dflash27b_last_error());
        return false;
    }
    if (!draft_feature_mirror_init(feature_mirror_, draft_backend_, draft_gpu,
                                   cfg_.device.gpu, feat_cap, n_capture,
                                   w_.n_embd)) {
        std::fprintf(stderr, "[muse] feature mirror init failed\n");
        return false;
    }

    dflash_target_ = new MuseDFlashTarget(w_, cache_, backend_,
                                          dw_.mask_token_id, capture_ids);
    // The snapshot must cover the deepest verify span, which is one draft
    // block. muse_kv_snapshot refuses a depth beyond the ring, so a chunk
    // smaller than block_size fails here rather than corrupting the ring.
    if (!dflash_target_->init(dw_.block_size)) {
        std::fprintf(stderr, "[muse] rollback snapshot alloc failed: %s\n",
                     dflash27b_last_error());
        return false;
    }

    // A/B knob for the noise-embedding convention (see
    // MuseDFlashTarget::embed_tokens). Raw is the reference behaviour and the
    // default; this turns the RMS-normed variant on to re-measure it.
    if (const char * v = std::getenv("MUSE_DRAFT_NORM_EMBED")) {
        if (*v && *v != '0') {
            dflash_target_->set_norm_noise_embed(true);
            std::fprintf(stderr, "[muse] MUSE_DRAFT_NORM_EMBED: RMS-norming the "
                                 "drafter's noise embeddings\n");
        }
    }

    std::printf("[muse] spec-decode ready: depth=%d mirror_cap=%d\n",
                dw_.block_size, feat_cap);
    // Honest expectation-setting, same spirit as the --fa-window warning, and
    // platform-specific because the measurement is: acceptance is the same
    // everywhere (~0.15-0.24) but the batched verify is not. A 16-token verify
    // costs ~1.9 AR steps on H200, ~4.4 on gfx1201, ~6.6 on gfx1151, so
    // speculation pays on CUDA (1.05-1.45x) and LOSES on both AMD parts
    // (0.28-0.76x). A user who passes --draft expects a speedup; on HIP they
    // will get the opposite, so say so rather than let them discover it.
#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    std::fprintf(stderr,
        "[muse] WARNING speculative decode is output-verified but is a "
        "SLOWDOWN on the AMD parts measured so far: 0.49-0.76x vs AR on "
        "gfx1201, 0.28-0.46x on gfx1151 (acceptance is fine; the 16-token "
        "verify costs 4.4-6.6 AR steps). Prefer plain AR decode here.\n");
#else
    std::fprintf(stderr,
        "[muse] speculative decode: output-verified; measured 1.05-1.45x vs AR "
        "on H200 chat traffic (vendor's llama.cpp reference reaches 3.1x — "
        "verify-cost headroom remains). Watch the [muse-spec] line per "
        "request.\n");
#endif
    std::fflush(stdout);
    return true;
}

void MuseBackend::free_decode_draft() {
    delete dflash_target_;
    dflash_target_ = nullptr;
    draft_feature_mirror_free(feature_mirror_);
    free_muse_target_feat(cache_);
    free_draft_weights(dw_);
    if (draft_backend_) {
        ggml_backend_free(draft_backend_);
        draft_backend_ = nullptr;
    }
}

void MuseBackend::print_ready_banner() const {
    std::printf("[muse-daemon] ready model=%s layers=%d ctx=%d chunk=%d "
                "swa=%d/%d window=%d\n",
                cfg_.model_path ? cfg_.model_path : "?", w_.n_layer,
                cache_.max_ctx, cfg_.chunk,
                (int)std::count(w_.swa_layers.begin(), w_.swa_layers.end(), true),
                w_.n_layer, cache_.swa_window);
    std::fflush(stdout);
}

bool MuseBackend::park(ParkTarget target) {
    (void)target;
    // Parking frees weights/KV to hand the GPU to another process. Nothing
    // here reloads them yet, so claiming success would strand the daemon in a
    // state it cannot leave.
    std::fprintf(stderr, "[muse] park is not implemented for this family\n");
    return false;
}

bool MuseBackend::unpark(ParkTarget target) {
    (void)target;
    std::fprintf(stderr, "[muse] unpark is not implemented for this family\n");
    return false;
}

bool MuseBackend::snapshot_save(int slot) {
    (void)slot;
    std::fprintf(stderr, "[muse] snapshots are not implemented for this "
                         "family\n");
    return false;
}

GenerateResult MuseBackend::restore_and_generate_impl(
        int slot, const GenerateRequest & req, const DaemonIO & io) {
    (void)req; (void)io;
    GenerateResult r;
    r.fail(GenerateErrorCode::InvalidSnapshotSlot,
           "muse: snapshots are not implemented (slot " + std::to_string(slot) +
           ")");
    return r;
}

bool MuseBackend::handle_compress(const std::string & line,
                                  const DaemonIO & io) {
    (void)line; (void)io;
    std::fprintf(stderr, "[muse] compress is not implemented for this family\n");
    return false;
}

GenerateResult MuseBackend::generate_impl(const GenerateRequest & req,
                                          const DaemonIO & io) {
    GenerateResult result;

    if (parked_) {
        result.fail(GenerateErrorCode::ModelParked, "muse: model is parked");
        return result;
    }
    if (req.prompt.empty()) {
        result.fail(GenerateErrorCode::PrefillFailed, "muse: empty prompt");
        return result;
    }
    const int n_prompt = (int)req.prompt.size();
    if (n_prompt + req.n_gen > cache_.max_ctx) {
        result.fail(GenerateErrorCode::ContextOverflow,
                    "muse: prompt " + std::to_string(n_prompt) + " + gen " +
                    std::to_string(req.n_gen) + " exceeds ctx " +
                    std::to_string(cache_.max_ctx));
        return result;
    }

    std::vector<float> embd, logits;
    auto embed_ids = [&](const int32_t * ids, int n) -> bool {
        embd.resize((size_t)w_.n_embd * n);
        return w_.embedder.embed(ids, n, embd.data());
    };

    // ── Prefill, chunked ───────────────────────────────────────────────
    // The chunk is bounded by the cache's headroom, not only by throughput:
    // a larger chunk would evict rows the same chunk still reads.
    const double t_pre = now_s();
    const int chunk = std::max(1, std::min(cfg_.chunk, cache_.max_chunk()));
    for (int done = 0; done < n_prompt; ) {
        if (io.is_cancelled()) {
            result.fail(GenerateErrorCode::PrefillFailed, "muse: cancelled");
            return result;
        }
        const int take = std::min(chunk, n_prompt - done);
        if (!embed_ids(req.prompt.data() + done, take)) {
            result.fail(GenerateErrorCode::PrefillFailed, "muse: embed failed");
            return result;
        }
        if (!muse_step(backend_, w_, cache_, embd.data(), take, done, logits)) {
            result.fail(GenerateErrorCode::PrefillFailed,
                        std::string("muse: ") + dflash27b_last_error());
            return result;
        }
        // Mirror this chunk's captured features onto the draft GPU. The
        // drafter cross-attends to them, so a chunk that is forwarded but not
        // mirrored leaves the drafter reading zeros for that span — it still
        // produces tokens, they are just poor, and only the acceptance rate
        // shows it.
        if (cache_.target_feat && feature_mirror_.target_feat &&
            !draft_feature_mirror_sync_range(cache_.target_feat,
                                             cache_.target_feat_cap,
                                             feature_mirror_, done, take)) {
            result.fail(GenerateErrorCode::PrefillFailed,
                        "muse: feature mirror sync failed");
            return result;
        }
        done += take;
    }
    result.prefill_s = now_s() - t_pre;

    // ── Decode ─────────────────────────────────────────────────────────
    // `logits` holds the prefill's last-position logits; both decode paths
    // start from the token they imply.
    const double t_dec = now_s();
    bool visible_emitted = false;

    // Greedy chain speculative decode reproduces greedy AR output exactly, so
    // it is only usable when the request IS greedy. Any sampler that processes
    // logits (temperature, top-p, penalties) would need sampled verify, which
    // is not implemented here — falling back to AR is the honest answer rather
    // than silently sampling from the draft.
    const bool greedy = !(req.do_sample && req.sampler.needs_logit_processing());
    const bool use_spec = greedy && !req.force_ar_decode && spec_decode_ready();

    if (use_spec) {
        int32_t seed = 0;
        for (int v = 1; v < w_.n_vocab; ++v) {
            if (logits[(size_t)v] > logits[(size_t)seed]) seed = v;
        }
        result.spec_decode_ran = true;
        if (!do_spec_decode(n_prompt, req.n_gen, seed, result.tokens, io,
                            &visible_emitted, &result.accept_rate,
                            commit_margins_)) {
            result.fail(GenerateErrorCode::DecodeFailed,
                        std::string("muse spec: ") + dflash27b_last_error());
            return result;
        }
    } else {
        int32_t seed;
        std::vector<int32_t> history = req.prompt;
        if (greedy) {
            seed = 0;
            for (int v = 1; v < w_.n_vocab; ++v) {
                if (logits[(size_t)v] > logits[(size_t)seed]) seed = v;
            }
        } else {
            seed = sample_logits(logits.data(), w_.n_vocab, req.sampler,
                                 history, rng_);
        }
        if (!do_ar_decode(n_prompt, req.n_gen, seed, req, result.tokens, io,
                          &visible_emitted, result)) {
            return result;   // do_ar_decode filled result.error
        }
    }

    io.emit(-1);
    result.decode_s = now_s() - t_dec;
    result.empty_visible_output = !visible_emitted;
    result.succeed();
    return result;
}

// ── Autoregressive decode ──────────────────────────────────────────────

bool MuseBackend::do_ar_decode(int committed, int n_gen, int32_t seed_tok,
                               const GenerateRequest & req,
                               std::vector<int32_t> & out_tokens,
                               const DaemonIO & io,
                               bool * visible_emitted,
                               GenerateResult & result) {
    std::vector<float> embd, logits;
    std::vector<int32_t> history = req.prompt;
    int32_t tok = seed_tok;
    int kv = committed;

    for (int i = 0; i < n_gen; ++i) {
        out_tokens.push_back(tok);
        history.push_back(tok);

        // <|eot|> (and eos) end the TURN. <|eom|> does NOT: it closes one
        // SEGMENT and the model continues with the next — the reasoning
        // channel is followed by the user-addressed answer, and chained tool
        // calls are separated the same way. Stopping on <|eom|> truncated
        // every reply at the end of its reasoning and returned an empty
        // answer with finish_reason=stop (observed live before this fix).
        // Structural tokens never count as visible output.
        const bool is_stop = (tok == w_.eos_id) || (tok == w_.eos_chat_id);
        const bool is_structural = is_stop || (tok == w_.eom_id);
        if (!is_structural) *visible_emitted = true;

        io.emit(tok);
        if (req.on_token && !req.on_token(tok)) break;
        if (io.is_cancelled()) break;
        if (is_stop) break;
        if (i + 1 >= n_gen) break;

        if (kv + 1 > cache_.max_ctx) {
            result.fail(GenerateErrorCode::ContextOverflow,
                        "muse: context exhausted during decode");
            return false;
        }
        embd.resize((size_t)w_.n_embd);
        if (!w_.embedder.embed(&tok, 1, embd.data())) {
            result.fail(GenerateErrorCode::DecodeFailed, "muse: embed failed");
            return false;
        }
        if (!muse_step(backend_, w_, cache_, embd.data(), 1, kv, logits)) {
            result.fail(GenerateErrorCode::DecodeFailed,
                        std::string("muse: ") + dflash27b_last_error());
            return false;
        }
        kv += 1;

        // Keep the capture ring current so a later spec-decode request over
        // the same conversation sees real features rather than stale ones.
        if (cache_.target_feat && feature_mirror_.target_feat) {
            draft_feature_mirror_sync_range(cache_.target_feat,
                                            cache_.target_feat_cap,
                                            feature_mirror_, kv - 1, 1);
        }

        if (req.do_sample && req.sampler.needs_logit_processing()) {
            tok = sample_logits(logits.data(), w_.n_vocab, req.sampler, history,
                                rng_);
        } else {
            tok = 0;
            for (int v = 1; v < w_.n_vocab; ++v) {
                if (logits[(size_t)v] > logits[(size_t)tok]) tok = v;
            }
        }
    }
    return true;
}

// ── Speculative decode ─────────────────────────────────────────────────
//
// Greedy chain speculation. The invariant is that this emits exactly what
// do_ar_decode would emit greedily — `test_muse_spec_decode` asserts it
// token-for-token, because a spec loop that is merely fluent is broken.
//
// Round shape, with D = the drafter's block size:
//
//   noise      = [seed, MASK x (D-1)]              (drafter input)
//   draft[0]   = seed                              (the target's own last
//                                                   prediction, so correct
//                                                   by construction)
//   draft[1..] = argmax(lm_head(drafter hidden))
//   verify     = target forward over draft at [committed, committed+D)
//   accept     draft[i+1] while it equals the target's argmax at i
//   bonus      = the target's argmax at the first rejected position
//
// Everything after the accepted prefix wrote KV it must give back. On the
// full-attention layers that is free (row index is the absolute position, so
// the stale rows are never read and get overwritten in place when those
// positions are legitimately reached). On the SWA layers it is not: the row is
// `p % swa_ring`, so a rejected write clobbers live history — which is what
// rollback_to undoes, restoring only the rejected suffix.
bool MuseBackend::do_spec_decode(int committed, int n_gen, int32_t seed_tok,
                                 std::vector<int32_t> & out_tokens,
                                 const DaemonIO & io,
                                 bool * visible_emitted,
                                 float * accept_rate_out,
                                 std::vector<float> * commit_margins) {
    MuseDFlashTarget * target = dflash_target_;
    const int hidden = w_.n_embd;
    const int q_len  = dw_.block_size;

    StepGraph draft_sg;
    std::vector<float>   noise_embed((size_t)hidden * q_len);
    std::vector<int32_t> noise_ids((size_t)q_len);
    std::vector<int32_t> draft_tok((size_t)q_len);
    std::vector<int32_t> target_tok;
    std::vector<int32_t> pos_q((size_t)q_len), pos_k;
    std::vector<float>   local_hidden;

    // Top-2 margin of one logits row. Only used when a caller asked for
    // margins, so this vocab-wide scan is off the serving path.
    auto top2_margin = [&](const float * row) {
        float b0 = -1e30f, b1 = -1e30f;
        for (int v = 0; v < w_.n_vocab; ++v) {
            const float x = row[(size_t)v];
            if (x > b0) { b1 = b0; b0 = x; } else if (x > b1) { b1 = x; }
        }
        return b0 - b1;
    };
    std::vector<float> round_logits;
    // Margin of the row that produced the token about to be committed at
    // offset 0. On the first round that row is prefill's, which both decode
    // paths compute the same way, so nothing there can diverge.
    float seed_margin = std::numeric_limits<float>::infinity();

    int32_t last_tok = seed_tok;
    int n_generated = 0, n_rounds = 0, n_accept_sum = 0, n_draft_pos = 0;
    bool ok = true;
    double t_draft = 0.0, t_verify = 0.0;   // where a round's time goes

    auto fail = [&](const char * what) {
        set_last_error(std::string(what) + ": " + dflash27b_last_error());
        ok = false;
    };

    while (n_generated < n_gen && ok) {
        const int budget = n_gen - n_generated;

        // A verify batch is q_len tokens plus, when a bonus token follows, one
        // more. Both must fit the context.
        if (committed + q_len + 1 > cache_.max_ctx) break;

        // 1. Drafter input: the realized token, then masks.
        noise_ids[0] = last_tok;
        for (int i = 1; i < q_len; ++i) noise_ids[(size_t)i] = target->mask_token_id();
        if (!target->embed_tokens(noise_ids.data(), q_len, noise_embed.data())) {
            fail("muse-spec: noise embed"); break;
        }

        // 2. Draft forward over the mirrored target features.
        const double t_d0 = now_s();
        const int ring_cap  = feature_mirror_.cap;
        const int draft_ctx = std::min(committed, ring_cap);
        if (draft_ctx <= 0) break;          // nothing to cross-attend to yet
        int mirror_slot0 = 0;
        const bool use_mirror_view = draft_feature_mirror_can_view(
            feature_mirror_, committed, draft_ctx, mirror_slot0);
        if (!build_draft_step(draft_sg, dw_, /*lm_head=*/nullptr, draft_backend_,
                              draft_ctx, use_mirror_view ? &feature_mirror_ : nullptr,
                              committed, ring_cap)) {
            fail("muse-spec: draft graph"); break;
        }
        if (!use_mirror_view &&
            !copy_feature_ring_range_to_tensor(feature_mirror_,
                                               draft_sg.target_hidden_cat,
                                               committed - draft_ctx, draft_ctx)) {
            fail("muse-spec: feature copy"); break;
        }
        ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed.data(), 0,
                                sizeof(float) * noise_embed.size());
        pos_k.resize((size_t)draft_ctx + q_len);
        for (int i = 0; i < q_len; ++i) pos_q[(size_t)i] = draft_ctx + i;
        for (int i = 0; i < draft_ctx + q_len; ++i) pos_k[(size_t)i] = i;
        ggml_backend_tensor_set(draft_sg.positions, pos_q.data(), 0,
                                sizeof(int32_t) * pos_q.size());
        ggml_backend_tensor_set(draft_sg.positions_k, pos_k.data(), 0,
                                sizeof(int32_t) * pos_k.size());
        if (ggml_backend_graph_compute(draft_backend_, draft_sg.gf) !=
                GGML_STATUS_SUCCESS) {
            set_last_error("muse-spec: draft compute failed");
            ok = false; break;
        }
        local_hidden.resize((size_t)hidden * q_len);
        ggml_backend_tensor_get(draft_sg.hidden_states, local_hidden.data(), 0,
                                sizeof(float) * local_hidden.size());

        // 3. Draft hidden -> token ids through the target's lm_head.
        if (!target->project_hidden_to_tokens(local_hidden.data(), q_len,
                                              draft_tok)) {
            fail("muse-spec: projection"); break;
        }
        draft_tok[0] = last_tok;
        t_draft += now_s() - t_d0;

        // 4. Verify. snapshot_kv() arms the row-scoped save; verify_batch
        //    takes it once the span is known.
        const double t_v0 = now_s();
        if (!target->snapshot_kv()) { fail("muse-spec: snapshot"); break; }
        int verify_last = -1;
        if (!target->verify_batch(draft_tok, committed, verify_last, &target_tok)) {
            fail("muse-spec: verify"); break;
        }
        t_verify += now_s() - t_v0;

        // Debug: dump the raw proposal/verify pair per round. The alignment
        // between draft slot i+1 and verify row i is exactly the kind of thing
        // that is wrong silently, and staring at the two arrays is the fastest
        // way to see a systematic shift.
        if (std::getenv("MUSE_SPEC_DEBUG")) {
            std::fprintf(stderr, "[muse-spec-dbg] draft :");
            for (int i = 0; i < q_len; ++i) std::fprintf(stderr, " %6d", draft_tok[(size_t)i]);
            std::fprintf(stderr, "\n[muse-spec-dbg] target:");
            for (int i = 0; i < q_len; ++i) std::fprintf(stderr, " %6d", target_tok[(size_t)i]);
            std::fprintf(stderr, "\n");
        }

        // 5. Longest matching prefix. draft_tok[0] is accepted by
        //    construction: it IS the target's prediction for this position.
        int accept_n = 1;
        for (int i = 0; i < q_len - 1; ++i) {
            if (draft_tok[(size_t)i + 1] == target_tok[(size_t)i]) ++accept_n;
            else break;
        }
        // The successor of the last accepted position, from the target's own
        // logits — free to EMIT, but NOT forwarded: it becomes slot 0 of the
        // NEXT round's verify batch, which writes its KV and features then.
        // The first version of this loop gave the bonus its own 1-token
        // forward, which billed every round a full extra decode step and put
        // spec decode at 0.7x AR; folding it into the next batch is the
        // llama.cpp convention and removes that cost. On loop exit the pending
        // token has been emitted and nothing after it needs its KV — the next
        // request re-prefills from scratch.
        const int pending_tok = (accept_n < q_len)
                                    ? target_tok[(size_t)accept_n - 1]
                                    : verify_last;

        // 5b. Per-committed-token margins, for callers that need to tell a
        //     broken accept rule from a coin-flip on a tied distribution. The
        //     token emitted at offset i was predicted by the verify row at
        //     i-1; offset 0 was predicted by the row that produced `last_tok`,
        //     which belongs to the previous round — or, on the first round, to
        //     prefill, which both decode paths run identically.
        std::vector<float> round_margins;
        if (commit_margins) {
            if (!target->read_verify_logits(q_len, round_logits)) {
                set_last_error("muse-spec: verify logits unavailable");
                ok = false; break;
            }
            round_margins.push_back(seed_margin);
            for (int i = 1; i < accept_n; ++i) {
                round_margins.push_back(top2_margin(
                    round_logits.data() + (size_t)(i - 1) * w_.n_vocab));
            }
            // pending_tok's margin: the row that predicted it.
            seed_margin = top2_margin(
                round_logits.data() + (size_t)(accept_n - 1) * w_.n_vocab);
        }

        // 6. Undo the rejected suffix's SWA writes, keeping the accepted
        //    prefix. Without this the ring holds speculative future keys while
        //    the mask computes the old occupant — silently wrong attention,
        //    nothing fails.
        if (!target->rollback_to(committed, accept_n)) {
            fail("muse-spec: rollback"); break;
        }

        // 7. Mirror the accepted positions' features for the next round.
        //    (pending_tok has no features yet; the next verify writes them,
        //    and the drafter's context window ends at `committed` either way.)
        if (cache_.target_feat && feature_mirror_.target_feat &&
            !draft_feature_mirror_sync_range(cache_.target_feat,
                                             cache_.target_feat_cap,
                                             feature_mirror_, committed,
                                             accept_n)) {
            set_last_error("muse-spec: feature mirror sync failed");
            ok = false; break;
        }

        // 8. Emit the accepted slots — [last_tok, draft_tok[1..accept_n-1]].
        //    pending_tok is emitted as slot 0 of the next round.
        bool hit_stop = false;
        int emitted = 0;
        const int emit_n = std::min(accept_n, budget);
        for (int i = 0; i < emit_n; ++i) {
            const int tok = draft_tok[(size_t)i];
            out_tokens.push_back(tok);
            ++emitted;
            const bool is_stop = (tok == w_.eos_id) || (tok == w_.eos_chat_id);
            if (!is_stop && tok != w_.eom_id) *visible_emitted = true;
            io.emit(tok);
            if (io.is_cancelled()) break;
            if (is_stop) { hit_stop = true; break; }
        }
        if (commit_margins) {
            round_margins.resize((size_t)emitted, seed_margin);
            commit_margins->insert(commit_margins->end(),
                                   round_margins.begin(), round_margins.end());
        }
        committed    += accept_n;   // KV frontier: accepted positions only
        n_generated  += emitted;
        n_accept_sum += std::min(accept_n, emitted);
        n_draft_pos  += q_len;
        ++n_rounds;
        last_tok = pending_tok;
        if (io.is_cancelled() || hit_stop) break;

        // A pending stop token needs no verify round of its own: emit and end.
        if (n_generated < n_gen &&
            (last_tok == w_.eos_id || last_tok == w_.eos_chat_id)) {
            out_tokens.push_back(last_tok);
            ++n_generated;
            io.emit(last_tok);
            if (commit_margins) commit_margins->push_back(seed_margin);
            break;
        }
    }

    const double accept_frac =
        n_draft_pos > 0 ? (double)n_accept_sum / (double)n_draft_pos : 0.0;
    std::fprintf(stderr, "[muse-spec] tokens=%d rounds=%d accepted=%d/%d "
                 "(%.1f%%) avg_commit=%.2f draft=%.0fms verify=%.0fms\n",
                 n_generated, n_rounds, n_accept_sum, n_draft_pos,
                 100.0 * accept_frac,
                 n_rounds > 0 ? (double)n_generated / (double)n_rounds : 0.0,
                 1e3 * t_draft, 1e3 * t_verify);
    if (accept_rate_out) *accept_rate_out = (float)accept_frac;

    step_graph_destroy(draft_sg);
    return ok;
}

}  // namespace dflash::common
