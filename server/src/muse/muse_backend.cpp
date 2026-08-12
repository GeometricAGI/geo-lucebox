// Muse-Glimmer ModelBackend implementation: chunked prefill + AR decode.

#include "muse_backend.h"
#include "common/sampler.h"
#include "internal.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
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
    free_muse_cache(cache_);
    if (loaded_) { free_muse_weights(w_); loaded_ = false; }
    if (backend_) { ggml_backend_free(backend_); backend_ = nullptr; }
}

bool MuseBackend::init() {
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
    return true;
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
        done += take;
    }
    result.prefill_s = now_s() - t_pre;

    // ── Decode ─────────────────────────────────────────────────────────
    const double t_dec = now_s();
    int kv = n_prompt;
    std::vector<int32_t> history = req.prompt;
    bool visible_emitted = false;

    for (int i = 0; i < req.n_gen; ++i) {
        int tok;
        if (req.do_sample && req.sampler.needs_logit_processing()) {
            tok = sample_logits(logits.data(), w_.n_vocab, req.sampler, history,
                                rng_);
        } else {
            tok = 0;
            for (int v = 1; v < w_.n_vocab; ++v) {
                if (logits[(size_t)v] > logits[(size_t)tok]) tok = v;
            }
        }

        result.tokens.push_back(tok);
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
        if (!is_structural) visible_emitted = true;

        io.emit(tok);
        if (req.on_token && !req.on_token(tok)) break;
        if (io.is_cancelled()) break;
        if (is_stop) break;
        if (i + 1 >= req.n_gen) break;

        if (kv + 1 > cache_.max_ctx) {
            result.fail(GenerateErrorCode::ContextOverflow,
                        "muse: context exhausted during decode");
            return result;
        }
        if (!embed_ids(&tok, 1)) {
            result.fail(GenerateErrorCode::DecodeFailed, "muse: embed failed");
            return result;
        }
        if (!muse_step(backend_, w_, cache_, embd.data(), 1, kv, logits)) {
            result.fail(GenerateErrorCode::DecodeFailed,
                        std::string("muse: ") + dflash27b_last_error());
            return result;
        }
        kv += 1;
    }
    io.emit(-1);
    result.decode_s = now_s() - t_dec;
    result.empty_visible_output = !visible_emitted;
    result.succeed();
    return result;
}

}  // namespace dflash::common
