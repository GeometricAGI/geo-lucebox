// Muse-Glimmer forward graph (per-layer block + head).
//
// Semantics are a port of llama.cpp's `src/models/muse-glimmer.cpp`, which is
// the implementation the geo-quant artifacts were gated against (muse-v4
// 103-106 / 122 on the agentic suite). Several details here are NOT
// inferable from the tensor names and are wrong under the "it looks like
// gemma" prior — each is called out where it lands:
//
//   1. the input embeddings get an UNWEIGHTED RMS norm before layer 0;
//   2. post-attention / post-FFN norms use eps 1e-8, not the model's
//      f_norm_rms_eps (1e-5) — they are separate constants;
//   3. RoPE runs on the SWA layers ONLY; full-attention layers are NoPE;
//   4. the attention gate is projected from the PRE-attention normed hidden
//      state, sigmoid'd, and multiplied into the attention output BEFORE wo;
//   5. kq_scale is the standard 1/sqrt(head_dim) — unlike gemma4, which uses
//      1.0 because its Q/K norms absorb the scale. Muse's q_norm absorbs a
//      qk_scale_factor at conversion time but the SDPA scale stays standard;
//   6. the head is lm_head -> scale by logit_scale -> tanh softcap.
//
// The layer block is written against a plain KV cache view so it can be
// driven by the daemon path and by a standalone parity harness; the harness
// is what verifies (5) and (2), which no amount of reading catches.

#include "muse_internal.h"
#include "internal.h"
#include "common/ggml_graph_precision.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace dflash::common {

namespace {

// Post-attention / post-FFN norm epsilon. Deliberately NOT w.norm_eps: the
// reference uses 1e-8 for these two norms and the model's f_norm_rms_eps
// (1e-5 on the shipped artifact) everywhere else. Using one value for both
// shifts every layer's residual slightly — fluent output, wrong logits.
constexpr float kMusePostNormEps = 1e-8f;

// RMS norm with a learned weight. The muse converter folds gemma-style
// `weight + 1` at conversion time, so this is a plain multiply.
ggml_tensor * muse_rms_norm_mul(ggml_context * ctx, ggml_tensor * x,
                                ggml_tensor * weight, float eps) {
    x = rms_norm_input_f32(ctx, x);
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    if (!weight) return n;
    return ggml_mul(ctx, n, graph_tensor_f32(ctx, weight));
}

}  // namespace

// Attention block for one layer, including the muse attention gate.
//
// `cur` is the PRE-attention normed hidden state [n_embd, n_tokens]; the gate
// is projected from it (not from the attention output), which is why it is
// taken as an explicit argument rather than recomputed downstream.
ggml_tensor * build_muse_attn_block(
    ggml_context *        ctx,
    ggml_cgraph *         gf,
    const MuseWeights &   w,
    const MuseLayer &     L,
    ggml_tensor *         cache_k,
    ggml_tensor *         cache_v,
    int                   il,
    ggml_tensor *         cur,
    ggml_tensor *         positions,
    ggml_tensor *         attn_mask,
    ggml_tensor *         kv_idx,
    int                   kv_start,
    int                   n_tokens)
{
    const int head_dim  = w.head_dim;
    const int n_head    = w.n_head;
    const int n_head_kv = w.n_head_kv;
    const int q_dim     = n_head * head_dim;
    const bool is_swa   = muse_is_swa_layer(w, il);

    // (4) The gate is a projection of the layer's normed INPUT. Computing it
    // from the attention output instead would still train-shaped-run and
    // produce plausible text.
    ggml_tensor * gate = ggml_mul_mat(ctx, L.attn_gate, cur);

    ggml_tensor * Qcur = ggml_mul_mat(ctx, L.wq, cur);
    ggml_tensor * Kcur = ggml_mul_mat(ctx, L.wk, cur);
    ggml_tensor * Vcur = ggml_mul_mat(ctx, L.wv, cur);

    Qcur = ggml_reshape_3d(ctx, Qcur, head_dim, n_head,    n_tokens);
    Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_head_kv, n_tokens);

    // Q/K norms use the model eps. q_norm carries the qk_scale_factor folded
    // in at conversion; k_norm is identity (ones) on the shipped artifact but
    // is applied unconditionally so a future artifact that uses it works.
    Qcur = muse_rms_norm_mul(ctx, Qcur, L.q_norm, w.norm_eps);
    Kcur = muse_rms_norm_mul(ctx, Kcur, L.k_norm, w.norm_eps);

    // (3) RoPE on sliding-window layers only; full-attention layers are NoPE.
    // Applying RoPE everywhere is the single easiest way to get this family
    // subtly wrong: short prompts still look fine.
    if (is_swa) {
        // NORMAL (interleaved-pair) rope, NOT NeoX. llama.cpp maps
        // LLM_ARCH_MUSE_GLIMMER to LLAMA_ROPE_TYPE_NORM; gemma4 next door
        // uses NeoX, so copying that neighbour is the natural mistake — and
        // it costs ~1.1 RMS on the logits while leaving argmax intact on
        // short prompts (measured: the parity harness caught exactly this).
        Qcur = ggml_rope_ext(ctx, Qcur, positions, nullptr,
                             head_dim, GGML_ROPE_TYPE_NORMAL,
                             0, w.rope_freq_base, 1.0f,
                             0.0f, 1.0f, 32.0f, 1.0f);
        Kcur = ggml_rope_ext(ctx, Kcur, positions, nullptr,
                             head_dim, GGML_ROPE_TYPE_NORMAL,
                             0, w.rope_freq_base, 1.0f,
                             0.0f, 1.0f, 32.0f, 1.0f);
    }

    // Append K/V to the cache. Row indices arrive as a graph INPUT so the
    // node properties stay identical step to step (CUDA-graph replay), the
    // same contract the gemma4 path uses; SWA layers get ring rows.
    const int cache_len = (int)cache_k->ne[1];
    ggml_tensor * Kcur_T = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));
    ggml_tensor * Vcur_T = ggml_cont(ctx, ggml_permute(ctx, Vcur, 0, 2, 1, 3));
    if (kv_idx) {
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_k, Kcur_T, kv_idx));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_v, Vcur_T, kv_idx));
    } else {
        const int write_pos = is_swa ? (kv_start % cache_len) : kv_start;
        ggml_tensor * k_slot = ggml_view_3d(ctx, cache_k,
            head_dim, n_tokens, n_head_kv,
            cache_k->nb[1], cache_k->nb[2], cache_k->nb[1] * (size_t)write_pos);
        ggml_tensor * v_slot = ggml_view_3d(ctx, cache_v,
            head_dim, n_tokens, n_head_kv,
            cache_v->nb[1], cache_v->nb[2], cache_v->nb[1] * (size_t)write_pos);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur_T, k_slot));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur_T, v_slot));
    }

    const int kv_len_raw = is_swa ? std::min(kv_start + n_tokens, cache_len)
                                  : (kv_start + n_tokens);
    const int kv_len = std::min((kv_len_raw + 255) & ~255, cache_len);

    ggml_tensor * Qfa = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));
    ggml_tensor * Kfa = ggml_view_3d(ctx, cache_k, head_dim, kv_len, n_head_kv,
                                     cache_k->nb[1], cache_k->nb[2], 0);
    ggml_tensor * Vfa = ggml_view_3d(ctx, cache_v, head_dim, kv_len, n_head_kv,
                                     cache_v->nb[1], cache_v->nb[2], 0);

    // (5) Standard SDPA scale. gemma4 passes 1.0 here because its per-head
    // norms absorb the scaling; muse does not — copying gemma4's 1.0 would
    // flatten every attention distribution in the model.
    const float kq_scale = 1.0f / std::sqrt((float)head_dim);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, attn_mask,
                                             kq_scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx, attn, q_dim, n_tokens);

    // (4) sigmoid gate applied BEFORE the output projection.
    attn = ggml_mul(ctx, attn, ggml_sigmoid(ctx, gate));

    return ggml_mul_mat(ctx, L.wo, attn);
}

// One full layer: attention block + post-norm + residual, FFN + post-norm +
// residual. Returns the layer output.
ggml_tensor * build_muse_layer(
    ggml_context *      ctx,
    ggml_cgraph *       gf,
    const MuseWeights & w,
    ggml_tensor *       cache_k,
    ggml_tensor *       cache_v,
    int                 il,
    ggml_tensor *       inpL,
    ggml_tensor *       positions,
    ggml_tensor *       attn_mask,
    ggml_tensor *       kv_idx,
    int                 kv_start,
    int                 n_tokens,
    const MuseCache *   feat_cache,
    int                 capture_idx)
{
    const MuseLayer & L = w.layers[(size_t)il];

    ggml_tensor * cur = muse_rms_norm_mul(ctx, inpL, L.attn_norm, w.norm_eps);
    cur = build_muse_attn_block(ctx, gf, w, L, cache_k, cache_v, il, cur,
                                positions, attn_mask, kv_idx, kv_start,
                                n_tokens);

    // (2) post-attention norm at 1e-8, then residual.
    cur = muse_rms_norm_mul(ctx, cur, L.attn_post_norm, kMusePostNormEps);
    ggml_tensor * ffn_inp = ggml_add(ctx, cur, inpL);

    cur = muse_rms_norm_mul(ctx, ffn_inp, L.ffn_norm, w.norm_eps);

    // SwiGLU: down( silu(gate(x)) * up(x) ). Note SiLU, not gemma4's GELU.
    ggml_tensor * g = ggml_mul_mat(ctx, L.ffn_gate, cur);
    ggml_tensor * u = ggml_mul_mat(ctx, L.ffn_up,   cur);
    cur = ggml_mul_mat(ctx, L.ffn_down, ggml_mul(ctx, ggml_silu(ctx, g), u));

    // (2) post-FFN norm at 1e-8, then residual.
    cur = muse_rms_norm_mul(ctx, cur, L.ffn_post_norm, kMusePostNormEps);
    cur = ggml_add(ctx, cur, ffn_inp);

    // DFlash feature capture: stash this layer's OUTPUT hidden state into the
    // ring so the drafter can cross-attend to it. Writes into
    // target_feat[capture_idx * n_embd .. +n_embd, slot] for each position, in
    // one or two spans depending on whether the range wraps the ring.
    //
    // Placed here, after the residual, so what the drafter sees is exactly the
    // layer's output — the same tensor the next layer consumes. Capturing the
    // pre-residual value would be a different representation and the drafter was
    // not trained on it.
    if (capture_idx >= 0 && feat_cache && feat_cache->target_feat) {
        ggml_tensor * feat = feat_cache->target_feat;
        const int    hidden     = w.n_embd;
        const int    cap        = feat_cache->target_feat_cap;
        const size_t elt        = ggml_element_size(feat);
        const size_t col_stride = feat->nb[1];
        const int    slot0      = kv_start % cap;
        const int    pre_n      = std::min(n_tokens, cap - slot0);
        const int    post_n     = n_tokens - pre_n;
        const size_t row_off    = (size_t)capture_idx * hidden * elt;

        ggml_tensor * src2d = ggml_reshape_2d(ctx, cur, hidden, n_tokens);
        {
            ggml_tensor * dst = ggml_view_2d(ctx, feat, hidden, pre_n, col_stride,
                                             (size_t)slot0 * col_stride + row_off);
            ggml_tensor * src = ggml_view_2d(ctx, src2d, hidden, pre_n,
                                             src2d->nb[1], 0);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, src, dst));
        }
        if (post_n > 0) {
            ggml_tensor * dst = ggml_view_2d(ctx, feat, hidden, post_n,
                                             col_stride, row_off);
            ggml_tensor * src = ggml_view_2d(ctx, src2d, hidden, post_n,
                                             src2d->nb[1],
                                             (size_t)pre_n * src2d->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, src, dst));
        }
    }
    return cur;
}

// Evenly spaced over the interior layers. Layer 0's output is barely past the
// embedding and the last layer's output is what the head already sees, so both
// carry little the drafter cannot get elsewhere.
std::vector<int> muse_default_capture_layers(const MuseWeights & w, int n) {
    std::vector<int> ids;
    if (n <= 0 || w.n_layer <= 2) return ids;
    ids.reserve((size_t)n);
    const int lo = 1, hi = w.n_layer - 2;          // inclusive interior range
    for (int k = 0; k < n; ++k) {
        const int idx = (n == 1) ? (lo + hi) / 2
                                 : lo + (int)((int64_t)k * (hi - lo) / (n - 1));
        ids.push_back(idx);
    }
    return ids;
}

// Embedding entry: (1) an UNWEIGHTED RMS norm over the token embeddings.
// There is no learned weight and no sqrt(n_embd) scale here — gemma's
// embedding scaling does not apply to this family.
ggml_tensor * build_muse_inp_norm(ggml_context * ctx, const MuseWeights & w,
                                  ggml_tensor * inp_embd) {
    return muse_rms_norm_mul(ctx, inp_embd, nullptr, w.norm_eps);
}

// Output head: final norm -> lm_head -> logit scale -> tanh softcap.
// The softcap is skipped when the model does not declare one (0.0).
ggml_tensor * build_muse_head(ggml_context * ctx, const MuseWeights & w,
                              ggml_tensor * cur) {
    cur = muse_rms_norm_mul(ctx, cur, w.out_norm, w.norm_eps);
    cur = ggml_mul_mat(ctx, w.output, cur);
    if (w.logit_scale != 0.0f) {
        cur = ggml_scale(ctx, cur, w.logit_scale);
    }
    if (w.final_logit_softcap != 0.0f) {
        // (6) tanh softcap, exactly the gemma3 formulation the reference
        // borrows: scale down, tanh, scale back up.
        cur = ggml_scale(ctx, cur, 1.0f / w.final_logit_softcap);
        cur = ggml_tanh(ctx, cur);
        cur = ggml_scale(ctx, cur, w.final_logit_softcap);
    }
    return cur;
}

}  // namespace dflash::common
