// dmix2 sidecar registration for muse-glimmer (qtype 105/106).
//
// The 3.30 bpw artifact stores 71 tensors at Q3_1_ROCMFP3_MIX (105) and
// Q2_1_ROCMFP2_MIX (106). Those wires are byte-identical to their fixed-level
// twins, so the GGUF carries only codes and the per-tensor codebook + mode
// ride out of band in a GGUF KV — `geoquant.dmix2.sidecar`, name-keyed
// (the DS4 `.dmix` wire is keyed by (layer, class) over five fixed attention
// classes and cannot name a dense muse tensor at all).
//
//   header : magic "DMX2s1\0\0" (8) | entry_count u32 | reserved u32 (=0)
//   entry  : name_len u32 | name utf-8 | qtype u32 (105|106) | C u32 (=2)
//            | K u32 (8 for 105, 4 for 106) | mode u8 (0 fixed, 1 adaptive)
//            | pad[3] | codebook (C*K) bf16 little-endian, row-major
//
// The cover must be EXACT: every resident 105/106 tensor covered once, no
// entries naming anything else. A missing entry decodes against fixed levels
// (silently wrong for a mode-1 tensor); a stray entry means the file and the
// sidecar disagree about what the model is. Both fail the load.

#include "muse_internal.h"
#include "internal.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

// Registry entry points live in the CUDA/HIP backend (ggml-cuda/rocmfp*_mix.cu)
// and are shared with the DeepSeek4 line.
extern "C" void ggml_cuda_rocmfp3_mix_register_host(
    const void * base, size_t nb02, int n_experts, int out, int in,
    const void * codebooks_bf16_host, const uint8_t * modes_host,
    const uint8_t * rotations_host);
extern "C" void ggml_cuda_rocmfp3_mix_unregister(const void * base);
extern "C" void ggml_cuda_rocmfp2_mix_register_host(
    const void * base, size_t nb02, int n_experts, int out, int in,
    const void * codebooks_bf16_host, const uint8_t * modes_host,
    const uint8_t * rotations_host);
extern "C" void ggml_cuda_rocmfp2_mix_unregister(const void * base);

namespace dflash::common {

namespace {

constexpr int      kQtype105 = 105;
constexpr int      kQtype106 = 106;
constexpr uint32_t kBooks    = 2;      // C
constexpr int      kRowAlign = 128;    // fused kernels' wide-load grid
const char         kMagic[8] = {'D','M','X','2','s','1','\0','\0'};

uint32_t rd_u32(const uint8_t * p) {
    uint32_t v; std::memcpy(&v, p, 4); return v;
}

int levels_for(int qtype) { return qtype == kQtype105 ? 8 : 4; }

}  // namespace

bool muse_parse_dmix2_sidecar(const uint8_t * blob, size_t len,
                              std::vector<MuseDmix2Entry> & out) {
    out.clear();
    if (len < 16) { set_last_error("dmix2: blob shorter than its header"); return false; }
    if (std::memcmp(blob, kMagic, 8) != 0) {
        set_last_error("dmix2: bad magic (expected DMX2s1)");
        return false;
    }
    const uint32_t count    = rd_u32(blob + 8);
    const uint32_t reserved = rd_u32(blob + 12);
    if (reserved != 0) {
        set_last_error("dmix2: reserved field is not zero — format drift");
        return false;
    }

    size_t off = 16;
    for (uint32_t i = 0; i < count; ++i) {
        if (off + 4 > len) { set_last_error("dmix2: truncated before name length"); return false; }
        const uint32_t nlen = rd_u32(blob + off); off += 4;
        if (nlen == 0 || off + nlen > len) {
            set_last_error("dmix2: bad name length"); return false;
        }
        MuseDmix2Entry e;
        e.name.assign((const char *)(blob + off), nlen); off += nlen;

        if (off + 16 > len) { set_last_error("dmix2: truncated metadata for " + e.name); return false; }
        e.qtype = (int)rd_u32(blob + off);
        const uint32_t c = rd_u32(blob + off + 4);
        const uint32_t k = rd_u32(blob + off + 8);
        e.mode = blob[off + 12];
        off += 16;   // 3 u32 + u8 + pad[3]

        if (e.qtype != kQtype105 && e.qtype != kQtype106) {
            set_last_error("dmix2: " + e.name + " has qtype " +
                           std::to_string(e.qtype) + " (expected 105 or 106)");
            return false;
        }
        if (c != kBooks || (int)k != levels_for(e.qtype)) {
            set_last_error("dmix2: " + e.name + " codebook shape mismatch");
            return false;
        }
        if (e.mode > 1) {
            set_last_error("dmix2: " + e.name + " mode " + std::to_string(e.mode) +
                           " not in {0,1}");
            return false;
        }
        const size_t cb_bytes = (size_t)c * k * 2;
        if (off + cb_bytes > len) {
            set_last_error("dmix2: truncated codebook for " + e.name); return false;
        }
        e.codebook.resize(c * k);
        std::memcpy(e.codebook.data(), blob + off, cb_bytes);
        off += cb_bytes;
        out.push_back(std::move(e));
    }
    if (off != len) {
        set_last_error("dmix2: " + std::to_string(len - off) +
                       " trailing bytes after " + std::to_string(count) + " entries");
        return false;
    }
    return true;
}

bool muse_register_dmix2(const std::vector<MuseDmix2Entry> & entries,
                         const std::map<std::string, ggml_tensor *> & resident,
                         const std::set<std::string> & all_in_file) {
    // Exact cover, both directions, before touching the registry: a partial
    // registration that later fails would leave live device buffers behind.
    std::map<std::string, const MuseDmix2Entry *> by_name;
    for (const auto & e : entries) {
        if (!by_name.emplace(e.name, &e).second) {
            set_last_error("dmix2: duplicate entry for " + e.name);
            return false;
        }
    }
    for (const auto & kv : resident) {
        auto it = by_name.find(kv.first);
        if (it == by_name.end()) {
            set_last_error("dmix2: resident mix tensor " + kv.first +
                           " has no sidecar entry — it would decode against "
                           "fixed levels");
            return false;
        }
        if (it->second->qtype != (int)kv.second->type) {
            set_last_error("dmix2: " + kv.first + " sidecar says qtype " +
                           std::to_string(it->second->qtype) +
                           " but the tensor is " +
                           std::to_string((int)kv.second->type));
            return false;
        }
    }
    // The other direction is checked against the FILE, not against what this
    // plan happens to load: a layer-split target legitimately loads a subset,
    // and its entries for the other layers are not drift. An entry naming a
    // tensor that is not a mix tensor anywhere in the file IS drift.
    for (const auto & kv : by_name) {
        if (all_in_file.find(kv.first) == all_in_file.end()) {
            set_last_error("dmix2: entry names " + kv.first +
                           ", which is not a qtype-105/106 tensor in this file");
            return false;
        }
    }

    std::vector<std::pair<const void *, int>> done;   // (base, qtype) for unwind
    for (const auto & kv : resident) {
        ggml_tensor * t = kv.second;
        const MuseDmix2Entry & e = *by_name[kv.first];
        const int in  = (int)t->ne[0];
        const int out = (int)t->ne[1];

        // GPU-only decode: the registry maps DEVICE base pointers, and the
        // fused kernels are the only implementation of these qtypes. A
        // host-resident tensor would register a pointer the kernels cannot
        // read — on a discrete part that is a fault at decode time, and on
        // unified memory (Strix Halo) it may quietly read something valid but
        // wrong. Refuse at load, naming the tensor, exactly as the ds4 path
        // does for partial offload.
        if (!t->data || (t->buffer && ggml_backend_buffer_is_host(t->buffer))) {
            set_last_error("dmix2: " + kv.first + " is host-resident — qtype "
                           "105/106 decode is GPU-only, so every mix layer "
                           "must be fully offloaded");
            for (const auto & d : done) {
                if (d.second == kQtype105) ggml_cuda_rocmfp3_mix_unregister(d.first);
                else                       ggml_cuda_rocmfp2_mix_unregister(d.first);
            }
            return false;
        }
        if (in % kRowAlign != 0) {
            set_last_error("dmix2: " + kv.first + " row length " +
                           std::to_string(in) + " is off the " +
                           std::to_string(kRowAlign) + "-weight kernel grid");
            for (const auto & d : done) {
                if (d.second == kQtype105) ggml_cuda_rocmfp3_mix_unregister(d.first);
                else                       ggml_cuda_rocmfp2_mix_unregister(d.first);
            }
            return false;
        }

        // Dense: one "expert". Rotations are not implemented by the kernels;
        // pass zeros, matching the DS4 path.
        const uint8_t mode = e.mode;
        const uint8_t rot  = 0;
        if (e.qtype == kQtype105) {
            ggml_cuda_rocmfp3_mix_register_host(t->data, t->nb[2], 1, out, in,
                                                e.codebook.data(), &mode, &rot);
        } else {
            ggml_cuda_rocmfp2_mix_register_host(t->data, t->nb[2], 1, out, in,
                                                e.codebook.data(), &mode, &rot);
        }
        done.emplace_back(t->data, e.qtype);
    }

    std::printf("[muse-dmix2] registered %zu of %zu qtype-105/106 tensor(s) "
                "(%zu sidecar entries)\n",
                done.size(), all_in_file.size(), entries.size());
    std::fflush(stdout);
    return true;
}

}  // namespace dflash::common
