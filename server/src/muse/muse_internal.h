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
#include <map>
#include <set>
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

// ── Graph builders (muse_graph.cpp) ────────────────────────────────────
// `cur` for the attention block is the PRE-attention normed hidden state:
// the attention gate is projected from it. `kv_idx` (I32, n_tokens) selects
// cache rows for the set_rows append; pass nullptr for the offset-view path.

ggml_tensor * build_muse_attn_block(ggml_context * ctx, ggml_cgraph * gf,
                                    const MuseWeights & w, const MuseLayer & L,
                                    ggml_tensor * cache_k, ggml_tensor * cache_v,
                                    int il, ggml_tensor * cur,
                                    ggml_tensor * positions,
                                    ggml_tensor * attn_mask,
                                    ggml_tensor * kv_idx,
                                    int kv_start, int n_tokens);

ggml_tensor * build_muse_layer(ggml_context * ctx, ggml_cgraph * gf,
                               const MuseWeights & w,
                               ggml_tensor * cache_k, ggml_tensor * cache_v,
                               int il, ggml_tensor * inpL,
                               ggml_tensor * positions, ggml_tensor * attn_mask,
                               ggml_tensor * kv_idx,
                               int kv_start, int n_tokens);

ggml_tensor * build_muse_inp_norm(ggml_context * ctx, const MuseWeights & w,
                                  ggml_tensor * inp_embd);

ggml_tensor * build_muse_head(ggml_context * ctx, const MuseWeights & w,
                              ggml_tensor * cur);

// ── dmix2 sidecar (muse_dmix2.cpp) ─────────────────────────────────────
// Out-of-band codebooks for the qtype-105/106 tensors of the low-bpw
// artifact. Parsed from the `geoquant.dmix2.sidecar` GGUF KV and handed to
// the CUDA/HIP mix registry shared with the DeepSeek4 line.

struct MuseDmix2Entry {
    std::string name;
    int  qtype = 0;            // 105 or 106
    uint8_t mode = 0;          // 0 fixed levels, 1 learned codebook
    std::vector<uint16_t> codebook;   // C*K bf16 bit patterns, row-major
};

// Parse (and strictly validate) a dmix2 blob.
bool muse_parse_dmix2_sidecar(const uint8_t * blob, size_t len,
                              std::vector<MuseDmix2Entry> & out);

// Register the resident mix tensors with the device registry.
//   `resident`    — name -> tensor for the 105/106 tensors THIS load bound.
//   `all_in_file` — names of every 105/106 tensor in the file.
// Every resident tensor must have an entry (a missing one would decode
// against fixed levels), and every entry must name a mix tensor of the file
// (anything else is sidecar drift). Entries for tensors outside a partial
// load's range are expected and ignored.
bool muse_register_dmix2(const std::vector<MuseDmix2Entry> & entries,
                         const std::map<std::string, ggml_tensor *> & resident,
                         const std::set<std::string> & all_in_file);

// ── KV cache + step driver (muse_step.cpp) ─────────────────────────────

struct MuseCache {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;   // per layer
    int max_ctx    = 0;   // rows on full-attention layers
    // Sliding-window layers keep TWO distinct numbers. The window is the
    // model's attention span; the ring is how many rows are allocated, and it
    // must exceed the window by at least one chunk. With ring == window a
    // chunked prefill evicts rows that queries in the SAME chunk still need
    // (every write in a graph lands before any read), which mis-attends
    // silently instead of failing — measured on an 8-token probe with a
    // 4-row ring: chunked vs stepwise prefill diverged by rms 2.11 with a
    // different argmax.
    int swa_window = 0;   // attention span (from the model)
    // `--fa-window`: cap how far back the FULL-attention layers look during
    // decode. 0 = unlimited (default), which leaves the mask bit-identical to
    // the unwindowed path. The SWA layers are untouched -- they already carry
    // the model's own window.
    //
    // WARNING, and the reason this defaults off: muse-glimmer has only 13
    // full-attention layers and they are what give it global context. A finite
    // window there drops the system prompt and tool definitions out of view,
    // which is exactly how tool calling breaks. Treat any non-zero value as
    // needing a golden-suite run before it is trusted, not as a free speedup.
    int fa_window  = 0;
    int swa_ring   = 0;   // allocated ring rows = window + headroom
    int n_layer    = 0;

    // Largest chunk muse_step will accept. When the ring spans the whole
    // context the rows never rotate, so nothing can clobber itself and the
    // limit is just the context.
    int max_chunk() const {
        return swa_ring >= max_ctx ? max_ctx : swa_ring - swa_window;
    }
};

// True when the SWA ring slot is visible to a query at `q_abs`, given that
// `total` tokens have been written. Occupancy uses `ring_rows`, visibility
// uses `window` — they are different numbers. Exposed for unit testing.
bool muse_swa_slot_visible(int total, int q_abs, int slot, int ring_rows,
                           int window);

// `max_chunk` is the largest prefill chunk that will be used; the SWA ring is
// sized to window + max_chunk so a chunk never evicts its own history.
bool create_muse_cache(ggml_backend_t backend, const MuseWeights & w,
                       int max_ctx, MuseCache & out, int max_chunk = 512);
void free_muse_cache(MuseCache & c);

// One forward step over `n_tokens` embeddings starting at absolute position
// `kv_start`. Returns the logits of the LAST token. Prefill is the same call
// with n_tokens > 1.
bool muse_step(ggml_backend_t backend, const MuseWeights & w, MuseCache & cache,
               const float * embed, int n_tokens, int kv_start,
               std::vector<float> & out_logits);

}  // namespace dflash::common
