// Loads Muse-Glimmer from a GGUF file. Follows the gemma4 loader's structure
// (mmap → backend buffer → per-layer tensor binding, tok_embd left on the CPU
// for CpuEmbedder), minus the MoE / per-layer-embedding / KV-sharing machinery
// this family does not have, plus the per-layer attention gate it does.
//
// Tensor naming (from the geo-quant exporter; verified against the shipped
// artifacts):
//   token_embd.weight, output_norm.weight, output.weight   (lm_head UNTIED)
//   blk.<i>.attn_norm.weight, attn_q.weight, attn_k.weight, attn_v.weight,
//   attn_output.weight, attn_q_norm.weight, attn_k_norm.weight,
//   attn_gate.weight, post_attention_norm.weight,
//   ffn_norm.weight, ffn_gate.weight, ffn_up.weight, ffn_down.weight,
//   post_ffw_norm.weight

#include "muse_internal.h"
#include "internal.h"
#include "dflash27b.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash::common {

namespace {

struct MuseMmap {
    void *  addr = nullptr;
    size_t  len  = 0;
#if defined(_WIN32)
    HANDLE  hFile = INVALID_HANDLE_VALUE;
    HANDLE  hMap  = nullptr;
#else
    int     fd   = -1;
#endif

    bool open_ro(const std::string & path, std::string & err) {
#if defined(_WIN32)
        hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) {
            err = "CreateFileA: " + path + ": error " + std::to_string(GetLastError());
            return false;
        }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile, &sz)) {
            err = "GetFileSizeEx: error " + std::to_string(GetLastError());
            return false;
        }
        len = (size_t)sz.QuadPart;
        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) {
            err = "CreateFileMappingA: error " + std::to_string(GetLastError());
            return false;
        }
        addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!addr) {
            err = "MapViewOfFile: error " + std::to_string(GetLastError());
            return false;
        }
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "open: " + path + " " + strerror(errno); return false; }
        struct stat st;
        if (fstat(fd, &st) < 0) { err = "fstat"; ::close(fd); fd = -1; return false; }
        len = (size_t)st.st_size;
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) { err = "mmap"; addr = nullptr; ::close(fd); fd = -1; return false; }
#endif
        return true;
    }

    void close_map() {
#if defined(_WIN32)
        if (addr) { UnmapViewOfFile(addr); addr = nullptr; }
        if (hMap) { CloseHandle(hMap); hMap = nullptr; }
        if (hFile != INVALID_HANDLE_VALUE) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; }
#else
        if (addr) { ::munmap(addr, len); addr = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
#endif
    }

    ~MuseMmap() { close_map(); }
};

uint32_t get_u32_or(gguf_context * g, const char * key, uint32_t def) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0) return def;
        return ((const uint32_t *)gguf_get_arr_data(g, id))[0];
    }
    return gguf_get_val_u32(g, id);
}

float get_f32_or(gguf_context * g, const char * key, float def) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0) return def;
        return ((const float *)gguf_get_arr_data(g, id))[0];
    }
    return gguf_get_val_f32(g, id);
}

size_t align_up_size(size_t x, size_t a) {
    if (a == 0) return x;
    const size_t r = x % a;
    return r == 0 ? x : x + (a - r);
}

bool parse_block_tensor_name(const char * name, int & layer_id) {
    const char prefix[] = "blk.";
    const size_t prefix_len = sizeof(prefix) - 1;
    if (std::strncmp(name, prefix, prefix_len) != 0) return false;
    const char * p = name + prefix_len;
    if (*p < '0' || *p > '9') return false;
    char * end = nullptr;
    const long v = std::strtol(p, &end, 10);
    if (!end || *end != '.' || v < 0 || v > INT32_MAX) return false;
    layer_id = (int)v;
    return true;
}

bool should_load_muse_tensor(const char * name, const TargetLoadPlan & plan) {
    if (std::strcmp(name, "token_embd.weight") == 0 ||
        std::strcmp(name, "output_norm.weight") == 0 ||
        std::strcmp(name, "output.weight") == 0) {
        return plan.load_output;
    }
    int layer_id = -1;
    if (parse_block_tensor_name(name, layer_id)) {
        return layer_id >= plan.layer_begin && layer_id < plan.layer_end;
    }
    return false;
}

struct MuseTensorAlloc {
    ggml_tensor * tensor = nullptr;
    size_t file_offset   = 0;
    size_t file_size     = 0;
    size_t buffer_offset = 0;
};

// The interleaved-SWA pattern. The artifact writes a BOOL array whose period
// need not equal n_layer (the shipped one is 8 long for 52 layers), so it
// cycles. `true` = sliding window, `false` = full attention.
void read_swa_pattern(gguf_context * gctx, uint32_t n_layer,
                      std::vector<bool> & swa) {
    swa.assign(n_layer, false);
    const int64_t pat_id =
        gguf_find_key(gctx, "muse-glimmer.attention.sliding_window_pattern");
    if (pat_id < 0 || gguf_get_kv_type(gctx, pat_id) != GGUF_TYPE_ARRAY) {
        return;
    }
    const size_t n = gguf_get_arr_n(gctx, pat_id);
    if (n == 0) return;
    const auto arr_type = gguf_get_arr_type(gctx, pat_id);
    const void * data = gguf_get_arr_data(gctx, pat_id);
    for (uint32_t il = 0; il < n_layer; ++il) {
        const size_t idx = il % n;
        if (arr_type == GGUF_TYPE_BOOL || arr_type == GGUF_TYPE_UINT8 ||
            arr_type == GGUF_TYPE_INT8) {
            swa[il] = ((const uint8_t *)data)[idx] != 0;
        } else if (arr_type == GGUF_TYPE_INT32 || arr_type == GGUF_TYPE_UINT32) {
            swa[il] = ((const uint32_t *)data)[idx] != 0;
        }
    }
}

// Resolve a special token id by name from the tokenizer token list. The muse
// converter writes bos/eos/eot ids as KV, but not <|eom|> — and <|eom|> is
// load-bearing for ATEM (it terminates the reasoning channel and chained tool
// calls). Looking it up here keeps the sampler from having to guess.
int32_t find_token_id(gguf_context * gctx, const char * text) {
    const int64_t toks_id = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (toks_id < 0 || gguf_get_kv_type(gctx, toks_id) != GGUF_TYPE_ARRAY) {
        return -1;
    }
    const size_t n = gguf_get_arr_n(gctx, toks_id);
    for (size_t i = 0; i < n; ++i) {
        const char * t = gguf_get_arr_str(gctx, toks_id, i);
        if (t && std::strcmp(t, text) == 0) return (int32_t)i;
    }
    return -1;
}

}  // namespace

bool load_muse_gguf(const std::string & path,
                    ggml_backend_t backend,
                    MuseWeights & out) {
    TargetLoadPlan plan;
    return load_muse_gguf_partial(path, backend, plan, out);
}

bool load_muse_gguf_partial(const std::string & path,
                            ggml_backend_t backend,
                            const TargetLoadPlan & plan_in,
                            MuseWeights & out) {
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) { set_last_error("gguf_init failed: " + path); return false; }

    {
        const int64_t aid = gguf_find_key(gctx, "general.architecture");
        if (aid < 0) {
            set_last_error("missing general.architecture");
            gguf_free(gctx); return false;
        }
        const char * arch = gguf_get_val_str(gctx, aid);
        if (std::string(arch) != "muse-glimmer") {
            set_last_error(std::string("unexpected arch: ") + arch +
                           " (expected muse-glimmer)");
            gguf_free(gctx); return false;
        }
    }

    const uint32_t n_layer   = get_u32_or(gctx, "muse-glimmer.block_count", 0);
    const uint32_t n_embd    = get_u32_or(gctx, "muse-glimmer.embedding_length", 0);
    const uint32_t n_ff      = get_u32_or(gctx, "muse-glimmer.feed_forward_length", 0);
    const uint32_t n_head    = get_u32_or(gctx, "muse-glimmer.attention.head_count", 0);
    const uint32_t n_head_kv = get_u32_or(gctx, "muse-glimmer.attention.head_count_kv", 0);
    const uint32_t key_len   = get_u32_or(gctx, "muse-glimmer.attention.key_length", 128);
    const uint32_t val_len   = get_u32_or(gctx, "muse-glimmer.attention.value_length", key_len);
    const uint32_t sliding   = get_u32_or(gctx, "muse-glimmer.attention.sliding_window", 0);
    const uint32_t n_ctx     = get_u32_or(gctx, "muse-glimmer.context_length", 0);

    uint32_t n_vocab = get_u32_or(gctx, "muse-glimmer.vocab_size", 0);
    if (n_vocab == 0) {
        if (ggml_tensor * te = ggml_get_tensor(meta_ctx, "token_embd.weight")) {
            n_vocab = (uint32_t)te->ne[1];
        }
    }

    const float rope_base = get_f32_or(gctx, "muse-glimmer.rope.freq_base", 500000.0f);
    const float norm_eps  =
        get_f32_or(gctx, "muse-glimmer.attention.layer_norm_rms_epsilon", 1e-5f);
    const float softcap   =
        get_f32_or(gctx, "muse-glimmer.final_logit_softcapping", 0.0f);
    const float logit_scale = get_f32_or(gctx, "muse-glimmer.logit_scale", 0.0f);

    if (n_layer == 0 || n_embd == 0 || n_head == 0 || n_head_kv == 0 ||
        n_vocab == 0) {
        set_last_error("muse: missing essential hparams");
        gguf_free(gctx); return false;
    }
    // K and V head widths are separate KV keys, and the KV cache below sizes
    // both from head_dim. Refusing here beats silently mis-sizing V.
    if (val_len != key_len) {
        set_last_error("muse: key_length " + std::to_string(key_len) +
                       " != value_length " + std::to_string(val_len) +
                       " — asymmetric head widths are not supported");
        gguf_free(gctx); return false;
    }
    if (n_head % n_head_kv != 0) {
        set_last_error("muse: head_count " + std::to_string(n_head) +
                       " is not a multiple of head_count_kv " +
                       std::to_string(n_head_kv));
        gguf_free(gctx); return false;
    }

    out.ctx     = meta_ctx;
    out.backend = backend;
    out.n_layer     = (int)n_layer;
    out.n_head      = (int)n_head;
    out.n_head_kv   = (int)n_head_kv;
    out.head_dim    = (int)key_len;
    out.n_embd      = (int)n_embd;
    out.n_ff        = (int)n_ff;
    out.n_vocab     = (int)n_vocab;
    out.n_ctx_train = (int)n_ctx;
    out.sliding_window      = (int)sliding;
    out.rope_freq_base      = rope_base;
    out.norm_eps            = norm_eps;
    out.final_logit_softcap = softcap;
    out.logit_scale         = logit_scale;

    read_swa_pattern(gctx, n_layer, out.swa_layers);

    const uint32_t miss = 0xFFFFFFFFu;
    out.bos_id      = (int32_t)get_u32_or(gctx, "tokenizer.ggml.bos_token_id", miss);
    out.eos_id      = (int32_t)get_u32_or(gctx, "tokenizer.ggml.eos_token_id", miss);
    out.eos_chat_id = (int32_t)get_u32_or(gctx, "tokenizer.ggml.eot_token_id", miss);
    if (out.bos_id      == (int32_t)miss) out.bos_id      = -1;
    if (out.eos_id      == (int32_t)miss) out.eos_id      = -1;
    if (out.eos_chat_id == (int32_t)miss) out.eos_chat_id = -1;
    out.eom_id = find_token_id(gctx, "<|eom|>");

    {
        int n_swa = 0;
        for (uint32_t il = 0; il < n_layer; ++il) if (out.swa_layers[il]) ++n_swa;
        std::printf("[muse-loader] n_layer=%u n_embd=%u n_ff=%u head_dim=%u "
                    "n_head=%u n_head_kv=%u vocab=%u\n",
                    n_layer, n_embd, n_ff, key_len, n_head, n_head_kv, n_vocab);
        std::printf("[muse-loader] swa_layers=%d/%u window=%u rope_base=%g "
                    "softcap=%g logit_scale=%g eps=%g\n",
                    n_swa, n_layer, sliding, rope_base, softcap, logit_scale,
                    norm_eps);
        std::printf("[muse-loader] bos=%d eos=%d eot=%d eom=%d\n",
                    out.bos_id, out.eos_id, out.eos_chat_id, out.eom_id);
        std::fflush(stdout);
    }

    TargetLoadPlan plan = plan_in;
    if (plan.layer_begin < 0) plan.layer_begin = 0;
    if (plan.layer_end   < 0) plan.layer_end   = (int)n_layer;
    if (plan.layer_begin > plan.layer_end || plan.layer_end > (int)n_layer) {
        char e[160];
        std::snprintf(e, sizeof(e),
                      "muse: invalid layer range [%d,%d) for n_layer=%u",
                      plan.layer_begin, plan.layer_end, n_layer);
        set_last_error(e);
        gguf_free(gctx); return false;
    }

    // ── Map tensors ────────────────────────────────────────────────────
    MuseMmap mmap;
    {
        std::string err;
        if (!mmap.open_ro(path, err)) {
            set_last_error("muse mmap: " + err);
            gguf_free(gctx); return false;
        }
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    std::vector<MuseTensorAlloc> allocs;
    size_t alloc_total = 0;
    const size_t data_offset = gguf_get_data_offset(gctx);
    const int n_tensors = gguf_get_n_tensors(gctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        ggml_tensor * t = ggml_get_tensor(meta_ctx, name);
        if (!t || !should_load_muse_tensor(name, plan)) continue;
        alloc_total = align_up_size(alloc_total, alignment);
        MuseTensorAlloc a;
        a.tensor        = t;
        a.file_offset   = data_offset + gguf_get_tensor_offset(gctx, i);
        a.file_size     = gguf_get_tensor_size(gctx, i);
        a.buffer_offset = alloc_total;
        alloc_total += ggml_backend_buft_get_alloc_size(buft, t);
        allocs.push_back(a);
    }
    if (allocs.empty()) {
        set_last_error("muse: load plan selected no tensors");
        mmap.close_map();
        gguf_free(gctx); return false;
    }

    out.buf = ggml_backend_alloc_buffer(backend, alloc_total);
    if (!out.buf) {
        set_last_error("muse: backend alloc failed");
        mmap.close_map();
        gguf_free(gctx); return false;
    }
    ggml_backend_buffer_set_usage(out.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    char * base = (char *)ggml_backend_buffer_get_base(out.buf);
    for (const MuseTensorAlloc & a : allocs) {
        if (ggml_backend_tensor_alloc(out.buf, a.tensor,
                                      base + a.buffer_offset) != GGML_STATUS_SUCCESS) {
            set_last_error("muse: tensor alloc failed");
            mmap.close_map();
            gguf_free(gctx); return false;
        }
    }

    size_t tok_embd_off = 0, tok_embd_sz = 0;
    ggml_type tok_embd_type = GGML_TYPE_COUNT;
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        ggml_tensor * t = ggml_get_tensor(meta_ctx, name);
        if (!t) continue;
        const size_t offset = data_offset + gguf_get_tensor_offset(gctx, i);
        const size_t sz     = gguf_get_tensor_size(gctx, i);
        if (std::strcmp(name, "token_embd.weight") == 0) {
            tok_embd_off  = offset;
            tok_embd_sz   = sz;
            tok_embd_type = gguf_get_tensor_type(gctx, i);
        }
        if (should_load_muse_tensor(name, plan)) {
            ggml_backend_tensor_set(t, (const char *)mmap.addr + offset, 0, sz);
        }
    }

    out.embedder.mmap_addr = mmap.addr;
    out.embedder.mmap_len  = mmap.len;
#if defined(_WIN32)
    out.embedder.mmap_hfile = mmap.hFile;
    out.embedder.mmap_hmap  = mmap.hMap;
#else
    out.embedder.mmap_fd   = mmap.fd;
#endif
    out.embedder.tok_embd_bytes = (const uint8_t *)mmap.addr + tok_embd_off;
    out.embedder.tok_embd_type  = tok_embd_type;
    out.embedder.n_embd         = n_embd;
    out.embedder.n_vocab        = (int64_t)n_vocab;
    out.embedder.row_bytes      = tok_embd_sz / (size_t)n_vocab;
    // Ownership of the mapping moves to the embedder.
    mmap.addr = nullptr;
#if defined(_WIN32)
    mmap.hFile = INVALID_HANDLE_VALUE;
    mmap.hMap  = nullptr;
#else
    mmap.fd = -1;
#endif

    // ── Bind tensors ───────────────────────────────────────────────────
    out.tok_embd = ggml_get_tensor(meta_ctx, "token_embd.weight");
    out.out_norm = ggml_get_tensor(meta_ctx, "output_norm.weight");
    out.output   = ggml_get_tensor(meta_ctx, "output.weight");
    // Muse ships an UNTIED lm_head. Falling back to tok_embd would still
    // produce fluent-looking output, so refuse rather than silently serve a
    // model whose head is the embedding table.
    if (!out.output) {
        set_last_error("muse: output.weight missing (this family does not tie "
                       "the lm_head to token_embd)");
        gguf_free(gctx); return false;
    }

    out.layers.resize(n_layer);
    char buf[256];
    for (uint32_t il = 0; il < n_layer; ++il) {
        auto & L = out.layers[il];
        auto get = [&](const char * suffix) -> ggml_tensor * {
            std::snprintf(buf, sizeof(buf), "blk.%u.%s", il, suffix);
            return ggml_get_tensor(meta_ctx, buf);
        };

        L.attn_norm      = get("attn_norm.weight");
        L.wq             = get("attn_q.weight");
        L.wk             = get("attn_k.weight");
        L.wv             = get("attn_v.weight");
        L.wo             = get("attn_output.weight");
        L.q_norm         = get("attn_q_norm.weight");
        L.k_norm         = get("attn_k_norm.weight");
        L.attn_gate      = get("attn_gate.weight");
        L.attn_post_norm = get("post_attention_norm.weight");

        L.ffn_norm       = get("ffn_norm.weight");
        L.ffn_gate       = get("ffn_gate.weight");
        L.ffn_up         = get("ffn_up.weight");
        L.ffn_down       = get("ffn_down.weight");
        L.ffn_post_norm  = get("post_ffw_norm.weight");
    }

    // Every layer in range must be complete. A missing projection would
    // otherwise surface as a null-deref deep in graph building, or worse, as
    // a silently skipped branch.
    for (int il = plan.layer_begin; il < plan.layer_end; ++il) {
        const auto & L = out.layers[il];
        const std::pair<const char *, ggml_tensor *> required[] = {
            {"attn_norm",           L.attn_norm},
            {"attn_q",              L.wq},
            {"attn_k",              L.wk},
            {"attn_v",              L.wv},
            {"attn_output",         L.wo},
            {"attn_gate",           L.attn_gate},
            {"post_attention_norm", L.attn_post_norm},
            {"ffn_norm",            L.ffn_norm},
            {"ffn_gate",            L.ffn_gate},
            {"ffn_up",              L.ffn_up},
            {"ffn_down",            L.ffn_down},
            {"post_ffw_norm",       L.ffn_post_norm},
        };
        for (const auto & r : required) {
            if (!r.second) {
                set_last_error("muse: blk." + std::to_string(il) + "." +
                               r.first + ".weight missing");
                gguf_free(gctx); return false;
            }
        }
    }

    std::printf("[muse-loader] loaded %zu/%d tensors, layers=[%d,%d) output=%d\n",
                allocs.size(), n_tensors, plan.layer_begin, plan.layer_end,
                plan.load_output ? 1 : 0);
    std::fflush(stdout);

    gguf_free(gctx);
    return true;
}

void free_muse_weights(MuseWeights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx); w.ctx = nullptr; }
    w.layers.clear();
}

}  // namespace dflash::common
