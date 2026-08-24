// MuseDFlashTarget — see the header for what differs from the gemma4 adapter.

#include "muse_dflash_target.h"
#include "internal.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace dflash::common {

MuseDFlashTarget::MuseDFlashTarget(MuseWeights & w,
                                   MuseCache & cache,
                                   ggml_backend_t backend,
                                   int mask_token_id,
                                   std::vector<int> capture_ids)
    : w_(w), cache_(cache), backend_(backend),
      mask_token_id_(mask_token_id), capture_ids_(std::move(capture_ids)) {}

MuseDFlashTarget::~MuseDFlashTarget() {
    free_muse_kv_snapshot(snap_);
}

bool MuseDFlashTarget::init(int max_depth) {
    if (mask_token_id_ < 0 || mask_token_id_ >= w_.n_vocab) {
        std::fprintf(stderr, "[muse-dflash] mask token %d outside vocab %d\n",
                     mask_token_id_, w_.n_vocab);
        return false;
    }
    if (capture_ids_.empty()) {
        std::fprintf(stderr, "[muse-dflash] no capture layers\n");
        return false;
    }
    // The bonus-token forward is a 1-token verify, so depth 1 must fit too;
    // max_depth is the largest span, not the only one.
    return create_muse_kv_snapshot(backend_, w_, cache_, max_depth, snap_);
}

bool MuseDFlashTarget::verify_batch(const std::vector<int32_t> & tokens,
                                    int base_pos,
                                    int & last_tok,
                                    std::vector<int32_t> * all_argmax,
                                    bool capture_ssm_intermediates) {
    (void)capture_ssm_intermediates;   // dense attention: no recurrent state
    const int n_tokens = (int)tokens.size();
    if (n_tokens <= 0) {
        set_last_error("muse verify_batch: empty token list");
        return false;
    }

    // Embed. NO sqrt(n_embd) scale — see the header.
    std::vector<float> embed((size_t)n_tokens * w_.n_embd);
    if (!w_.embedder.embed(tokens.data(), n_tokens, embed.data())) {
        set_last_error("muse verify_batch: embed failed");
        return false;
    }

    // Take the rollback save now, not in snapshot_kv(): this is the first
    // point at which the span the forward is about to overwrite is known.
    if (pending_save_) {
        if (!muse_kv_snapshot_save(backend_, w_, cache_, base_pos, n_tokens,
                                   snap_)) {
            return false;
        }
        pending_save_ = false;
    }

    // Full logits only when someone will read them (read_verify_logits /
    // margin diagnostics) — the block is 13 MB per round at this vocab, and
    // the argmax-only path reads back n_tokens ints instead.
    std::vector<int32_t> argmax_buf;
    if (!muse_verify_batch(backend_, w_, cache_, embed.data(), n_tokens,
                           base_pos, argmax_buf,
                           keep_verify_logits_ ? &verify_logits_ : nullptr)) {
        return false;
    }
    verify_n_tokens_ = keep_verify_logits_ ? n_tokens : 0;

    last_tok = argmax_buf[(size_t)n_tokens - 1];
    if (all_argmax) *all_argmax = std::move(argmax_buf);
    return true;
}

bool MuseDFlashTarget::read_verify_logits(int n_tokens,
                                          std::vector<float> & out) {
    if (n_tokens <= 0 || n_tokens > verify_n_tokens_) return false;
    if (verify_logits_.size() < (size_t)n_tokens * w_.n_vocab) return false;
    out.assign(verify_logits_.begin(),
               verify_logits_.begin() + (size_t)n_tokens * w_.n_vocab);
    return true;
}

bool MuseDFlashTarget::snapshot_kv() {
    if (!snap_.ctx) {
        set_last_error("muse snapshot_kv: init() was not called");
        return false;
    }
    pending_save_ = true;
    return true;
}

bool MuseDFlashTarget::restore_kv() {
    return muse_kv_snapshot_restore(backend_, w_, cache_, snap_, /*from_row=*/0);
}

bool MuseDFlashTarget::rollback_to(int base_pos, int commit_n) {
    if (!snap_.valid) {
        set_last_error("muse rollback_to: no live save");
        return false;
    }
    // A rollback against a different span than the one saved would restore
    // the wrong ring rows — and would do it quietly, since every row involved
    // holds plausible K/V either way.
    if (base_pos != snap_.base_pos) {
        set_last_error("muse rollback_to: base_pos " + std::to_string(base_pos) +
                       " does not match the saved span at " +
                       std::to_string(snap_.base_pos));
        return false;
    }
    return muse_kv_snapshot_restore(backend_, w_, cache_, snap_, commit_n);
}

bool MuseDFlashTarget::is_eos(int token) const {
    // <|eom|> is deliberately absent: it ends a SEGMENT (the reasoning
    // channel, a chained tool call) and the model continues afterwards.
    // Treating it as EOS truncates every reply at the end of its reasoning.
    return token == w_.eos_id || token == w_.eos_chat_id;
}

bool MuseDFlashTarget::embed_tokens(const int32_t * tokens, int n,
                                    float * out) const {
    if (!w_.embedder.embed(tokens, n, out)) return false;
    if (!norm_noise_embed_) return true;
    // The reference implementation (llama.cpp models/dflash.cpp) feeds the
    // drafter RAW embedding rows — `ggml_get_rows(tok_embd, tokens)`, no norm,
    // no scale — so raw is the default here. The RMS-norm variant (muse's own
    // entry convention, build_muse_inp_norm) measured within noise of it
    // (9.4% vs 9.1% acceptance — the draft graph's first op is a
    // scale-invariant RMS norm, so only the residual path sees the
    // difference), and matching the implementation the drafter was trained
    // against beats matching the target.
    const int d = w_.n_embd;
    for (int t = 0; t < n; ++t) {
        float * row = out + (size_t)t * d;
        double sum = 0.0;
        for (int i = 0; i < d; ++i) sum += (double)row[i] * (double)row[i];
        const float inv = 1.0f / std::sqrt((float)(sum / (double)d) + w_.norm_eps);
        for (int i = 0; i < d; ++i) row[i] *= inv;
    }
    return true;
}

bool MuseDFlashTarget::project_hidden_to_tokens(
        const float * hidden, int n_tokens,
        std::vector<int32_t> & tokens_out) {
    return muse_project_hidden(backend_, w_, hidden, n_tokens, tokens_out,
                               apply_out_norm_);
}

}  // namespace dflash::common
