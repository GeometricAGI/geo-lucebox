// muse-glimmer loader against a REAL artifact.
//
// Point the test at a muse-glimmer GGUF with MUSE_GGUF=/path/to/model.gguf;
// it skips (exit 77) when unset, so CI stays green on machines without the
// 11–17 GB artifact while developers and the release box get the real check.
//
// What it pins is the stuff that is silently wrong if the loader guesses:
// the hparams the graph will size buffers from, the interleaved-SWA pattern
// (a bool array whose period is 8 while n_layer is 52 — read it as a
// per-layer array and 44 layers get the wrong attention span), the untied
// lm_head, the per-layer attention gate that has no gemma4 analogue, and the
// <|eom|> id the ATEM sampler must be able to emit.

#include "muse_internal.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace dflash::common;

static int g_fails = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", (msg)); ++g_fails; } \
    } while (0)

#define CHECK_EQ_I(got, want, what)                                            \
    do {                                                                       \
        const long long g_ = (long long)(got), w_ = (long long)(want);         \
        if (g_ != w_) {                                                        \
            std::fprintf(stderr, "FAIL: %s = %lld, expected %lld\n",           \
                         (what), g_, w_);                                      \
            ++g_fails;                                                         \
        }                                                                      \
    } while (0)

int main() {
    const char * path = std::getenv("MUSE_GGUF");
    if (!path || !*path) {
        std::printf("SKIP: set MUSE_GGUF=/path/to/muse-glimmer.gguf\n");
        return 77;
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "FAIL: ggml_backend_cpu_init\n");
        return 1;
    }

    // Layer 0 only: the whole point is to parse metadata and bind tensors
    // without paying for a 17 GB residency in a unit test.
    TargetLoadPlan plan;
    plan.layer_begin = 0;
    plan.layer_end   = 1;
    plan.load_output = true;

    MuseWeights w;
    if (!load_muse_gguf_partial(path, backend, plan, w)) {
        std::fprintf(stderr, "FAIL: load_muse_gguf_partial: %s\n",
                     dflash27b_last_error());
        ggml_backend_free(backend);
        return 1;
    }

    // ── hparams ────────────────────────────────────────────────────────
    CHECK_EQ_I(w.n_layer,   52,     "n_layer");
    CHECK_EQ_I(w.n_embd,    6656,   "n_embd");
    CHECK_EQ_I(w.n_ff,      19968,  "n_ff");
    CHECK_EQ_I(w.n_head,    32,     "n_head");
    CHECK_EQ_I(w.n_head_kv, 2,      "n_head_kv");
    CHECK_EQ_I(w.head_dim,  128,    "head_dim");
    CHECK_EQ_I(w.n_vocab,   202048, "n_vocab");
    CHECK_EQ_I(w.sliding_window, 2048, "sliding_window");
    CHECK(w.final_logit_softcap == 20.0f, "final_logit_softcap == 20.0");
    CHECK(w.logit_scale > 0.19f && w.logit_scale < 0.20f,
          "logit_scale ~= 0.1961");
    CHECK(w.rope_freq_base == 500000.0f, "rope_freq_base == 500000");
    CHECK(w.norm_eps > 0.0f && w.norm_eps < 1e-4f, "norm_eps ~= 1e-5");

    // ── interleaved SWA ────────────────────────────────────────────────
    // The artifact writes an 8-long pattern [T,T,T,F,T,T,T,F] that CYCLES:
    // three sliding-window layers then one full-attention layer. Reading it
    // as a 52-long array (or ignoring it) silently gives most layers the
    // wrong span, which degrades long-context behaviour without crashing.
    CHECK_EQ_I(w.swa_layers.size(), 52, "swa_layers.size()");
    int n_full = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        const bool want_swa = (il % 4) != 3;
        if (muse_is_swa_layer(w, il) != want_swa) {
            std::fprintf(stderr,
                         "FAIL: layer %d swa=%d, expected %d (period-4 pattern)\n",
                         il, (int)muse_is_swa_layer(w, il), (int)want_swa);
            ++g_fails;
            break;
        }
        if (!want_swa) ++n_full;
    }
    CHECK_EQ_I(n_full, 13, "full-attention layer count (52/4)");

    // ── tokenizer ids the ATEM path depends on ─────────────────────────
    CHECK(w.bos_id >= 0, "bos_id resolved");
    CHECK(w.eos_id >= 0, "eos_id resolved");
    CHECK(w.eos_chat_id >= 0, "eot id resolved");
    // Not a KV key — the loader finds it by token text. Without it the
    // sampler cannot terminate a reasoning turn or a chained tool call.
    CHECK(w.eom_id >= 0, "<|eom|> id resolved from the token list");
    CHECK(w.eom_id != w.eos_chat_id, "<|eom|> is distinct from <|eot|>");

    // ── global tensors ─────────────────────────────────────────────────
    CHECK(w.tok_embd != nullptr, "token_embd bound");
    CHECK(w.out_norm != nullptr, "output_norm bound");
    CHECK(w.output   != nullptr, "output (lm_head) bound");
    // Untied: a tied head would make these the same tensor, and the model
    // would still emit plausible text.
    CHECK(w.output != w.tok_embd, "lm_head is UNTIED from token_embd");
    if (w.tok_embd) {
        CHECK_EQ_I(w.tok_embd->ne[0], w.n_embd,  "token_embd.ne[0]");
        CHECK_EQ_I(w.tok_embd->ne[1], w.n_vocab, "token_embd.ne[1]");
    }

    // ── layer 0 shapes ─────────────────────────────────────────────────
    const auto & L = w.layers[0];
    const int64_t q_width  = (int64_t)w.n_head * w.head_dim;      // 4096
    const int64_t kv_width = (int64_t)w.n_head_kv * w.head_dim;   // 256
    CHECK(L.wq && L.wk && L.wv && L.wo, "attention projections bound");
    if (L.wq) { CHECK_EQ_I(L.wq->ne[0], w.n_embd, "wq.ne[0]");
                CHECK_EQ_I(L.wq->ne[1], q_width,  "wq.ne[1]"); }
    if (L.wk) { CHECK_EQ_I(L.wk->ne[1], kv_width, "wk.ne[1] (GQA 2 kv heads)"); }
    if (L.wv) { CHECK_EQ_I(L.wv->ne[1], kv_width, "wv.ne[1] (GQA 2 kv heads)"); }
    if (L.wo) { CHECK_EQ_I(L.wo->ne[0], q_width,  "wo.ne[0]");
                CHECK_EQ_I(L.wo->ne[1], w.n_embd, "wo.ne[1]"); }

    // The muse-specific node: same shape as wq, so it gates the attention
    // branch at head-flattened width.
    CHECK(L.attn_gate != nullptr, "attn_gate bound");
    if (L.attn_gate) {
        CHECK_EQ_I(L.attn_gate->ne[0], w.n_embd, "attn_gate.ne[0]");
        CHECK_EQ_I(L.attn_gate->ne[1], q_width,  "attn_gate.ne[1]");
    }

    // Q/K norms are per-head-dim (parameter-free upstream, written as F32).
    CHECK(L.q_norm && L.k_norm, "q_norm/k_norm bound");
    if (L.q_norm) CHECK_EQ_I(L.q_norm->ne[0], w.head_dim, "q_norm.ne[0]");
    if (L.k_norm) CHECK_EQ_I(L.k_norm->ne[0], w.head_dim, "k_norm.ne[0]");

    CHECK(L.attn_norm && L.attn_post_norm, "pre/post attention norms bound");
    CHECK(L.ffn_norm && L.ffn_post_norm,   "pre/post ffn norms bound");
    CHECK(L.ffn_gate && L.ffn_up && L.ffn_down, "dense FFN bound");
    if (L.ffn_gate) CHECK_EQ_I(L.ffn_gate->ne[1], w.n_ff, "ffn_gate.ne[1]");
    if (L.ffn_down) CHECK_EQ_I(L.ffn_down->ne[0], w.n_ff, "ffn_down.ne[0]");

    free_muse_weights(w);
    ggml_backend_free(backend);

    if (g_fails == 0) {
        std::printf("test_muse_loader: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_muse_loader: %d failure(s)\n", g_fails);
    return 1;
}
