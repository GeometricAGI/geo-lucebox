// Muse-Glimmer target structs for the dflash daemon.
//
// Architecture summary (from the artifact's GGUF metadata; see
// server/docs/MUSE_GLIMMER.md):
//   - 52-layer DENSE decoder — no experts, no per-layer embeddings, no KV
//     sharing. The gemma4 family is the closest shape but carries all three;
//     this one deliberately does not.
//   - Interleaved sliding-window attention: `sliding_window_pattern` is a
//     bool array (period 8: three SWA layers then one full-attention layer).
//   - GQA 32 query heads / 2 KV heads, head_dim 128 from key_length.
//   - Q/K RMS norms per head; the upstream model's are parameter-free, so
//     the converter writes them as F32 constant vectors — they are real
//     tensors here either way.
//   - An extra per-layer ATTENTION GATE (`attn_gate`, [n_embd, n_head*head_dim])
//     applied to the attention branch. gemma4 has no analogue; this is the
//     one genuinely new node in the family.
//   - Final logit softcapping (20.0) and a final logit scale.
//   - Untied lm_head (`output.weight` is present and distinct from
//     `token_embd.weight`).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "internal.h"  // CpuEmbedder
#include "common/layer_split_utils.h"

namespace dflash::common {

struct MuseLayer {
    // Pre-attention norm
    ggml_tensor * attn_norm      = nullptr;  // [n_embd]

    // Attention projections
    ggml_tensor * wq             = nullptr;  // [n_embd, n_head * head_dim]
    ggml_tensor * wk             = nullptr;  // [n_embd, n_head_kv * head_dim]
    ggml_tensor * wv             = nullptr;  // [n_embd, n_head_kv * head_dim]
    ggml_tensor * wo             = nullptr;  // [n_head * head_dim, n_embd]
    ggml_tensor * q_norm         = nullptr;  // [head_dim]
    ggml_tensor * k_norm         = nullptr;  // [head_dim]

    // Attention output gate — the muse-specific node. Same shape as wq, so
    // it projects the layer input to the attention's head-flattened width
    // and gates the attention branch before wo.
    ggml_tensor * attn_gate      = nullptr;  // [n_embd, n_head * head_dim]

    // Post-attention norm
    ggml_tensor * attn_post_norm = nullptr;  // [n_embd]

    // Dense FFN (every layer)
    ggml_tensor * ffn_norm       = nullptr;  // [n_embd]
    ggml_tensor * ffn_gate       = nullptr;  // [n_embd, n_ff]
    ggml_tensor * ffn_up         = nullptr;  // [n_embd, n_ff]
    ggml_tensor * ffn_down       = nullptr;  // [n_ff, n_embd]
    ggml_tensor * ffn_post_norm  = nullptr;  // [n_embd]
};

struct MuseWeights {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    // Global tensors
    ggml_tensor * tok_embd = nullptr;  // [n_embd, n_vocab]
    ggml_tensor * out_norm = nullptr;  // [n_embd]
    ggml_tensor * output   = nullptr;  // [n_embd, n_vocab] — untied lm_head

    std::vector<MuseLayer> layers;

    CpuEmbedder embedder;

    // Architecture metadata
    int n_layer  = 0;
    int n_head   = 0;
    int n_head_kv = 0;
    int head_dim = 128;
    int n_embd   = 0;
    int n_ff     = 0;
    int n_vocab  = 0;
    int n_ctx_train = 0;

    // Interleaved SWA. `swa_layers[il] == true` → the layer attends within
    // `sliding_window`; false → full attention over the context.
    int  sliding_window = 0;
    std::vector<bool> swa_layers;

    // RoPE
    float rope_freq_base = 500000.0f;

    // Output head
    float final_logit_softcap = 0.0f;
    float logit_scale         = 0.0f;   // 0 → no scaling

    // Tokenizer
    int32_t bos_id      = -1;
    int32_t eos_id      = -1;
    int32_t eos_chat_id = -1;   // <|eot|>
    int32_t eom_id      = -1;   // <|eom|> — ends a turn that CONTINUES the
                                // same role (reasoning channel, chained tool
                                // calls). Must be sampleable: suppressing it
                                // breaks ATEM tool calling outright.

    float norm_eps = 1e-5f;
};

inline bool muse_is_swa_layer(const MuseWeights & w, int il) {
    return il < (int)w.swa_layers.size() && w.swa_layers[il];
}

// Load a muse-glimmer GGUF. The partial form honours a layer range for the
// layer-split path; the full form loads everything.
bool load_muse_gguf(const std::string & path,
                    ggml_backend_t backend,
                    MuseWeights & out);

bool load_muse_gguf_partial(const std::string & path,
                            ggml_backend_t backend,
                            const TargetLoadPlan & plan,
                            MuseWeights & out);

void free_muse_weights(MuseWeights & w);

}  // namespace dflash::common
