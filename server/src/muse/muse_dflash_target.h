// MuseDFlashTarget — DFlashTarget adapter for muse-glimmer.
//
// Wraps MuseWeights / MuseCache / muse_verify_batch behind the generic
// interface in common/dflash_target.h so the vendor DFlash drafter can drive
// speculative verification.
//
// Three things differ from the gemma4 adapter next door, and each of them is
// silent when wrong (fluent output, collapsed acceptance rate, or wrong text
// that still reads well):
//
//   1. NO sqrt(n_embd) embedding scale. gemma4 scales its embeddings and so
//      must feed the drafter scaled ones; muse's entry is an unweighted RMS
//      norm with no scale at all (build_muse_inp_norm), so scaling here would
//      hand the drafter inputs the target never sees.
//   2. The rollback is ROW-SCOPED, not a whole-cache copy. gemma4 duplicates
//      every K/V tensor each round (~110 MB); muse saves only the SWA ring
//      rows the speculative span will clobber (~640 KB at depth 16), and
//      restores only the REJECTED suffix so accepted KV survives.
//      muse_kv_snapshot_save needs base_pos and n, which the interface's
//      argument-free snapshot_kv() does not carry — so snapshot_kv() arms the
//      save and verify_batch performs it once those are known.
//   3. `mask_token_id` and the capture layer ids come from the DRAFTER's GGUF,
//      not from a constant. The muse drafter declares mask 201818 and capture
//      layers [2,14,26,38,50]; gemma4's 4 and evenly-spaced defaults are both
//      wrong here and neither failure is loud.

#pragma once

#include "common/dflash_target.h"
#include "muse_internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

namespace dflash::common {

class MuseDFlashTarget : public DFlashTarget {
public:
    // Non-owning references — the caller owns the weights, cache and backend.
    // `mask_token_id` and `capture_ids` come from the loaded drafter.
    MuseDFlashTarget(MuseWeights & w,
                     MuseCache & cache,
                     ggml_backend_t backend,
                     int mask_token_id,
                     std::vector<int> capture_ids);

    ~MuseDFlashTarget() override;

    // Allocate the rollback snapshot for speculation depths up to
    // `max_depth`. Separate from the constructor because it can fail (the
    // depth must fit inside the SWA ring).
    bool init(int max_depth);

    // ── DFlashTarget interface ──────────────────────────────────────

    bool verify_batch(const std::vector<int32_t> & tokens,
                      int base_pos,
                      int & last_tok,
                      std::vector<int32_t> * all_argmax = nullptr,
                      bool capture_ssm_intermediates = false) override;

    bool read_verify_logits(int n_tokens, std::vector<float> & out) override;

    bool snapshot_kv() override;
    bool restore_kv() override;

    // muse has no recurrent state, so "restore SSM intermediates" is
    // vacuously satisfied and rollback_to reduces to exactly what this family
    // needs: keep the accepted prefix's KV, undo the rejected suffix's writes
    // to the SWA ring, and leave the cursor at base_pos + commit_n. The
    // full-attention layers need no data restored at all — their row index is
    // the absolute position, so a rejected write never lands on a row an
    // earlier position occupies (see muse_snapshot.cpp).
    bool supports_fast_rollback() const override { return true; }
    bool rollback_to(int base_pos, int commit_n) override;

    bool is_eos(int token) const override;

    bool embed_tokens(const int32_t * tokens, int n,
                      float * out) const override;

    bool project_hidden_to_tokens(const float * hidden,
                                  int n_tokens,
                                  std::vector<int32_t> & tokens_out) override;

    int hidden_size() const override { return w_.n_embd; }
    int mask_token_id() const override { return mask_token_id_; }
    const std::vector<int> & capture_layer_ids() const override {
        return capture_ids_;
    }

    // Test/diagnostic hook: run project_hidden_to_tokens with the target's
    // out_norm applied. The drafter emits already-normed hidden states, so
    // this should be the WORSE of the two; it exists so that claim can be
    // measured against acceptance rate rather than asserted.
    void set_apply_out_norm(bool on) { apply_out_norm_ = on; }

    // Whether the drafter's noise embeddings get muse's entry RMS norm. See
    // embed_tokens for why they should; exposed so the choice is measurable.
    void set_norm_noise_embed(bool on) { norm_noise_embed_ = on; }

    // Whether verify_batch keeps its full [n_tokens x n_vocab] logits for
    // read_verify_logits. Off by default: the block is 13 MB per round at this
    // vocab and nothing on the greedy serving path reads it — only the
    // margin diagnostics (spec_collect_margins) and any future sampled-verify
    // path need it.
    void set_keep_verify_logits(bool on) { keep_verify_logits_ = on; }

private:
    MuseWeights &  w_;
    MuseCache &    cache_;
    ggml_backend_t backend_;
    int            mask_token_id_ = -1;
    std::vector<int> capture_ids_;
    bool           apply_out_norm_ = false;
    bool           norm_noise_embed_ = false;
    bool           keep_verify_logits_ = false;

    // Row-scoped rollback state. `pending_save_` is set by snapshot_kv() and
    // consumed by the next verify_batch, which is the first point at which
    // the span (base_pos, n) is known.
    MuseKvSnapshot snap_;
    bool           pending_save_ = false;

    // Logits of the most recent verify_batch, kept for read_verify_logits.
    std::vector<float> verify_logits_;
    int                verify_n_tokens_ = 0;
};

}  // namespace dflash::common
