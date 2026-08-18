#include "dmix2_sidecar.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

extern "C" void ggml_cuda_rocmfp3_mix_register_host(
        const void * base, size_t nb02, int n_experts, int out, int in,
        const void * codebooks_bf16_host, const uint8_t * modes_host,
        const uint8_t * rotations_host);
extern "C" void ggml_cuda_rocmfp2_mix_register_host(
        const void * base, size_t nb02, int n_experts, int out, int in,
        const void * codebooks_bf16_host, const uint8_t * modes_host,
        const uint8_t * rotations_host);
extern "C" void ggml_cuda_rocmfp3_mix_unregister(const void * base);
extern "C" void ggml_cuda_rocmfp2_mix_unregister(const void * base);

namespace dflash {
namespace common {

static const char DMIX2_KV_KEY[] = "geoquant.dmix2.sidecar";
static const uint8_t DMIX2_MAGIC[8] = { 'D', 'M', 'X', '2', 's', '1', 0, 0 };
static const int64_t DMIX2_ROW_ALIGN = 128;

struct dmix2_entry {
    int32_t qtype = 0;
    uint8_t mode  = 0;
    std::vector<uint8_t> codebook;
    bool matched  = false;
};

static bool is_mix_qtype(ggml_type t) {
    return t == GGML_TYPE_Q3_1_ROCMFP3_MIX || t == GGML_TYPE_Q2_1_ROCMFP2_MIX;
}

static int dmix2_levels_for_qtype(int32_t qtype) {
    switch (qtype) {
        case GGML_TYPE_Q3_1_ROCMFP3_MIX: return 8;
        case GGML_TYPE_Q2_1_ROCMFP2_MIX: return 4;
        default:                         return 0;
    }
}

static bool dmix2_parse(const uint8_t * blob, size_t n,
                       std::map<std::string, dmix2_entry> & entries) {
    if (n < 16) {
        std::fprintf(stderr, "[dmix2]: '%s' truncated (%zu B < 16 B header)\n",
                     DMIX2_KV_KEY, n);
        return false;
    }
    if (std::memcmp(blob, DMIX2_MAGIC, 8) != 0) {
        std::fprintf(stderr, "[dmix2]: bad magic (expected DMX2s1)\n");
        return false;
    }
    uint32_t count = 0, reserved = 0;
    std::memcpy(&count, blob + 8, 4);
    std::memcpy(&reserved, blob + 12, 4);
    if (reserved != 0) {
        std::fprintf(stderr, "[dmix2]: reserved field %u != 0\n", reserved);
        return false;
    }
    size_t off = 16;
    for (uint32_t i = 0; i < count; ++i) {
        if (off + 4 > n) {
            std::fprintf(stderr, "[dmix2]: entry %u truncated before name_len\n", i);
            return false;
        }
        uint32_t name_len = 0;
        std::memcpy(&name_len, blob + off, 4);
        off += 4;
        if (name_len == 0 || name_len > 1024 || off + name_len > n) {
            std::fprintf(stderr, "[dmix2]: entry %u bad name_len %u\n", i, name_len);
            return false;
        }
        std::string name(reinterpret_cast<const char *>(blob + off), name_len);
        off += name_len;
        if (entries.count(name)) {
            std::fprintf(stderr, "[dmix2]: duplicate entry for '%s'\n", name.c_str());
            return false;
        }
        if (off + 16 > n) {
            std::fprintf(stderr, "[dmix2]: '%s' truncated metadata\n", name.c_str());
            return false;
        }
        uint32_t qtype = 0, c = 0, k = 0;
        std::memcpy(&qtype, blob + off + 0, 4);
        std::memcpy(&c,     blob + off + 4, 4);
        std::memcpy(&k,     blob + off + 8, 4);
        const uint8_t mode = blob[off + 12];
        off += 16;
        const int want_k = dmix2_levels_for_qtype((int32_t) qtype);
        if (want_k == 0) {
            std::fprintf(stderr, "[dmix2]: '%s' qtype %u is not 105/106\n",
                         name.c_str(), qtype);
            return false;
        }
        if (c != 2 || (int) k != want_k || mode > 1) {
            std::fprintf(stderr, "[dmix2]: '%s' bad C/K/mode C=%u K=%u mode=%u\n",
                         name.c_str(), c, k, mode);
            return false;
        }
        const size_t cb_bytes = (size_t) c * k * 2;
        if (off + cb_bytes > n) {
            std::fprintf(stderr, "[dmix2]: '%s' truncated codebook\n", name.c_str());
            return false;
        }
        dmix2_entry e;
        e.qtype = (int32_t) qtype;
        e.mode  = mode;
        e.codebook.assign(blob + off, blob + off + cb_bytes);
        off += cb_bytes;
        entries.emplace(std::move(name), std::move(e));
    }
    if (off != n) {
        std::fprintf(stderr, "[dmix2]: %zu trailing bytes after %u entries\n",
                     n - off, count);
        return false;
    }
    return true;
}

bool register_dmix2_sidecar(const std::string & gguf_path, ggml_context * ctx) {
    std::vector<ggml_tensor *> mix_tensors;
    if (ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr;
             t = ggml_get_next_tensor(ctx, t)) {
            if (is_mix_qtype(t->type) && t->data) {
                mix_tensors.push_back(t);
            }
        }
    }
    if (mix_tensors.empty()) {
        return true;
    }

    struct gguf_init_params gip = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
    struct gguf_context * g = gguf_init_from_file(gguf_path.c_str(), gip);
    if (!g) {
        std::fprintf(stderr, "[dmix2]: cannot reopen %s to read '%s'\n",
                     gguf_path.c_str(), DMIX2_KV_KEY);
        return false;
    }
    const int64_t kid = gguf_find_key(g, DMIX2_KV_KEY);
    if (kid < 0 || gguf_get_kv_type(g, kid) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(g, kid) != GGUF_TYPE_UINT8) {
        gguf_free(g);
        std::fprintf(stderr, "[dmix2]: %zu qtype-105/106 tensor(s) resident but "
                     "'%s' is missing or not a u8 array\n",
                     mix_tensors.size(), DMIX2_KV_KEY);
        return false;
    }
    std::map<std::string, dmix2_entry> entries;
    const bool parsed = dmix2_parse(
        static_cast<const uint8_t *>(gguf_get_arr_data(g, kid)),
        (size_t) gguf_get_arr_n(g, kid),
        entries);
    gguf_free(g);
    if (!parsed) {
        return false;
    }

    for (const ggml_tensor * t : mix_tensors) {
        const std::string name = t->name;
        const auto it = entries.find(name);
        if (it == entries.end()) {
            std::fprintf(stderr, "[dmix2]: resident tensor '%s' (type %s) has no sidecar entry\n",
                         name.c_str(), ggml_type_name(t->type));
            return false;
        }
        if (it->second.qtype != (int32_t) t->type) {
            std::fprintf(stderr, "[dmix2]: '%s' sidecar qtype %d != tensor %d\n",
                         name.c_str(), it->second.qtype, (int) t->type);
            return false;
        }
        if (t->ne[2] != 1 || t->ne[3] != 1) {
            std::fprintf(stderr, "[dmix2]: '%s' is not 2-D\n", name.c_str());
            return false;
        }
        if (t->ne[0] % DMIX2_ROW_ALIGN != 0) {
            std::fprintf(stderr, "[dmix2]: '%s' row length %ld is off the %ld-weight grid\n",
                         name.c_str(), (long) t->ne[0], (long) DMIX2_ROW_ALIGN);
            return false;
        }
        if (!ggml_is_contiguous(t)) {
            std::fprintf(stderr, "[dmix2]: '%s' is not contiguous\n", name.c_str());
            return false;
        }
        it->second.matched = true;
    }

    size_t unmatched = 0;
    for (const auto & kv : entries) {
        if (!kv.second.matched) {
            ++unmatched;
        }
    }
    if (unmatched) {
        std::fprintf(stderr, "[dmix2]: %zu sidecar entr%s name no resident tensor "
                     "(expected for MTP-block weights outside an MTP context)\n",
                     unmatched, unmatched == 1 ? "y" : "ies");
    }

    for (const ggml_tensor * t : mix_tensors) {
        const dmix2_entry & e = entries.at(t->name);
        const int in  = (int) t->ne[0];
        const int out = (int) t->ne[1];
        const uint8_t mode = e.mode;
        if (t->type == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
            ggml_cuda_rocmfp3_mix_register_host(
                t->data, ggml_nbytes(t), /*n_experts=*/ 1, out, in,
                e.codebook.data(), &mode, /*rotations=*/ nullptr);
        } else {
            ggml_cuda_rocmfp2_mix_register_host(
                t->data, ggml_nbytes(t), /*n_experts=*/ 1, out, in,
                e.codebook.data(), &mode, /*rotations=*/ nullptr);
        }
    }
    std::fprintf(stderr, "[dmix2]: registered %zu tensor(s) from '%s'\n",
                 mix_tensors.size(), DMIX2_KV_KEY);
    return true;
}

void unregister_dmix2_sidecar(ggml_context * ctx) {
    if (!ctx) {
        return;
    }
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr;
         t = ggml_get_next_tensor(ctx, t)) {
        if (!t->data) {
            continue;
        }
        if (t->type == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
            ggml_cuda_rocmfp3_mix_unregister(t->data);
        } else if (t->type == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
            ggml_cuda_rocmfp2_mix_unregister(t->data);
        }
    }
}

}  // namespace common
}  // namespace dflash
