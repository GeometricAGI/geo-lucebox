#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dflash {
namespace deepseek4 {

// Level-2 thinking-budget force-close, as applied to one sampled token.
//
// Header-only and free of backend state so the rule can be tested without a model or a GPU;
// the AR decode loop is the only caller.
//
// Contract: when the remaining window falls to the reserved reply budget, steer the stream
// into `close_ids` one token per step, then STOP INTERVENING so the model spends the rest of
// the window on a visible answer. Overriding the sampled token (rather than appending tokens
// out of band) keeps every emitted token on the normal decode path, so the next step's forward
// pass sees it and KV state stays consistent.
//
// The previous DeepSeek4 implementation pushed the whole close sequence and broke out of the
// loop instead. Two consequences, both measured on DeepSeek-V4-Flash-0731: no forward ever ran
// over the injected tokens, and generation ended immediately, so the reserved reply budget was
// never usable. Completions came to exactly `thinking_ceiling + close_ids.size()` tokens for
// sequences of length 1, 3 and 23 -- zero tokens of answer every time -- while finish_reason
// still reported "stop", which a client cannot distinguish from a real completion. One item
// returned 22,706 characters of reasoning and 1 character of answer. qwen35_backend already
// used the override-and-continue shape this restores.
//
//   close_ids        the close sequence; empty disables the hook entirely
//   remaining        n_gen - generated, i.e. window left INCLUDING this token
//   hard_limit       reserved reply budget that triggers the close
//   sampled          the token the sampler chose
//   started          [in,out] false until the close sequence begins
//   inject_pos       [in,out] index of the next close token to emit
//   forced_close     [out] set true on the step the hook first fires; never cleared here
//
// Returns the token that should actually be emitted.
inline int32_t budget_hook_apply(const std::vector<int32_t> & close_ids,
                                 int remaining,
                                 int hard_limit,
                                 int32_t sampled,
                                 bool & started,
                                 std::size_t & inject_pos,
                                 bool & forced_close) {
    if (close_ids.empty()) {
        return sampled;                       // hook disabled
    }
    if (started) {
        if (inject_pos < close_ids.size()) {
            return close_ids[inject_pos++];   // continue the sequence
        }
        return sampled;                       // sequence done: the model answers freely
    }
    if (remaining > hard_limit) {
        return sampled;                       // still inside the thinking window
    }
    started = true;
    inject_pos = 1;
    forced_close = true;
    // If the model happens to have reached the boundary already emitting close[0], consume
    // that token as the start of the sequence rather than overriding it with the same value.
    return close_ids.front();
}

// Rank of `token` within the top `max_rank` logits, 1-based; 0 when it is not in the top
// `max_rank`. Partial selection only -- no full sort of a 129k-entry vocabulary per step.
// Mirrors token_rank_in_top() in antirez/ds4 ds4_eval.c.
inline int token_rank_in_top(const float * logits, int n_vocab, int32_t token, int max_rank) {
    if (!logits || token < 0 || token >= n_vocab || max_rank <= 0) return 0;
    const float target = logits[token];
    int better = 0;
    for (int i = 0; i < n_vocab; ++i) {
        if (i == token) continue;
        // Ties count as "better" so the rank is pessimistic; a tie must not smuggle a token
        // into the top-k it did not clearly earn.
        if (logits[i] >= target) {
            if (++better >= max_rank) return 0;   // already outside the window
        }
    }
    return better + 1;
}

// Two-stage close, matching antirez/ds4 ds4_eval.c (soft_limit_reply_budget=1024,
// hard_limit_reply_budget=512, soft_limit_think_close_rank=3).
//
// Our server previously implemented ONLY the hard stage. The soft stage is not a loop
// preventer -- it fires far too late for that -- it is a graceful exit: when the reply window
// is close and the model ALREADY has </think> among its top few candidates, take it at a
// natural boundary instead of jamming the tag in mid-sentence 512 tokens later. Upstream's
// comment: "it accepts the model's own desire to end thinking when </think> is already near
// the top of the distribution."
//
// Why this may matter more for a quantized model than for the reference: on the bf16 anchor,
// P(</think>) at a genuine stop point is 0.999 at rank 1 (measured, H100 teacher-forced), so
// the hard stage alone suffices. Quantization that blunts that spike can leave the token at
// rank 2-3 -- wanting to stop but not decisively -- which the top-3 test catches and an
// argmax-only path does not.
//
// DELIBERATE DEVIATION from upstream, and the reason for it. ds4_eval.c guards the soft stage
// with `think_close_tokens.len == 1`, so a multi-token close disables it. Our close sequence is
// three tokens -- 201,128822,271, i.e. "\n</think>\n\n" -- because the model card supplies a
// thinking_terminator_hint (that scaffold is worth 65% vs 47% on its own, so we are not giving
// it up). Under upstream's guard our soft stage would simply never run.
//
// Instead of dropping the stage we probe `soft_probe_token` -- the real </think> token -- which
// is the one that answers "does the model want to stop?". Ranking close_ids[0] would be
// meaningless here: it is a newline, and newlines are common at any point in the stream. When
// the caller passes -1 the probe falls back to close_ids[0], which reproduces upstream exactly
// for a single-token close.
//
//   logits/n_vocab   current step distribution (may be null to disable the soft stage)
//   soft_limit       remaining-window threshold at which the soft test starts
//   close_rank       accept the close when its rank is 1..close_rank
//   soft_probe_token token whose rank is tested; -1 => close_ids.front()
//   soft_closed      [out] set true when the SOFT stage fired (distinct from forced_close)
inline int32_t budget_hook_apply_2stage(const std::vector<int32_t> & close_ids,
                                        int remaining,
                                        int soft_limit,
                                        int hard_limit,
                                        int close_rank,
                                        const float * logits,
                                        int n_vocab,
                                        int32_t soft_probe_token,
                                        int32_t sampled,
                                        bool & started,
                                        std::size_t & inject_pos,
                                        bool & forced_close,
                                        bool & soft_closed) {
    if (close_ids.empty()) return sampled;
    if (started) {
        if (inject_pos < close_ids.size()) return close_ids[inject_pos++];
        return sampled;
    }
    // Hard stage first: it is unconditional and must win if both would fire on the same step.
    if (remaining <= hard_limit) {
        started = true;
        inject_pos = 1;
        forced_close = true;
        return close_ids.front();
    }
    // Soft stage: inside the soft window, when the model already ranks the think-close token
    // highly. Emits the FULL close sequence from the top (inject_pos = 1), so a hinted
    // multi-token close still lands intact -- the probe only decides WHETHER to close.
    if (logits && n_vocab > 0 && remaining <= soft_limit && close_rank > 0) {
        const int32_t probe =
            (soft_probe_token >= 0) ? soft_probe_token : close_ids.front();
        const int rank = token_rank_in_top(logits, n_vocab, probe, close_rank);
        if (rank > 0) {
            started = true;
            inject_pos = 1;
            soft_closed = true;
            return close_ids.front();
        }
    }
    return sampled;
}

}  // namespace deepseek4
}  // namespace dflash
