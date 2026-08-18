#include "gqh_headers.h"

#include "ggml.h"
#include "gguf.h"
#include "gqh-stride.h"   // planar layout: gqh_planar_enabled, gqh_plane_row_bytes

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace dflash {
namespace common {

// ---- GQH (108/109) per-tensor headers, embedded in the GGUF ----
// gqh3/gqh2_h scale every weight by a 5-byte per-tensor header (float32
// tensor_scale + uint8 grid code) that a fixed-size ggml block cannot hold, so it
// rides in the "geoquant.gqh.headers" KV. Wire (schema v1, FROZEN, mirrored in the
// geo-quant exporter):
//   header : magic "GQHh1\0\0\0" (8) | entry_count u32 | reserved u32 (=0)
//   entry  : name_len u32 | name utf-8 | qtype u32 (108|109)
//            | tensor_scale f32 LE | grid_code u8 | pad[3] (=0)
// gqh2_c (110) needs no entry: its scale is fp16 in-block.
//
// Unlike the mix sidecars there is no loose-file fallback -- the payload is five
// bytes per tensor, so it is always embedded.

static const char GQH_KV_KEY[] = "geoquant.gqh.headers";
static const uint8_t GQH_MAGIC[8] = { 'G', 'Q', 'H', 'h', '1', 0, 0, 0 };
static const uint32_t GQH_GRID_CODES_MAX = 12;   // GAMMA_GRID / A_GRID length
static const int64_t  GQH_ROW_ALIGN      = 256;  // superblock, spec C4

struct gqh_kv_entry {
    int32_t qtype        = 0;
    float   tensor_scale = 0.0f;
    uint8_t grid_code    = 0;
    bool    matched      = false;
};

static bool gqh_qtype_has_header(int32_t q) {
    return q == GGML_TYPE_GQH3 || q == GGML_TYPE_GQH2_H;
}

bool register_gqh_headers(const std::string & gguf_path, ggml_context * ctx) {
    // Collect the resident header-bearing GQH tensors. Walk the context rather than
    // the named members: GQH is not tied to a surface the way the mix qtypes are.
    std::vector<ggml_tensor *> gqh_tensors;
    if (ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr;
             t = ggml_get_next_tensor(ctx, t)) {
            if (gqh_qtype_has_header((int32_t) t->type)) {
                gqh_tensors.push_back(t);
            }
        }
    }
    if (gqh_tensors.empty()) {
        return true;
    }

    std::vector<uint8_t> blob;
    {
        struct gguf_init_params gip = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
        struct gguf_context * g = gguf_init_from_file(gguf_path.c_str(), gip);
        if (!g) {
            std::fprintf(stderr, "[gqh]: cannot reopen %s to read '%s'\n",
                         gguf_path.c_str(), GQH_KV_KEY);
            return false;
        }
        const int64_t id = gguf_find_key(g, GQH_KV_KEY);
        if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(g, id) != GGUF_TYPE_UINT8) {
            gguf_free(g);
            std::fprintf(stderr, "[gqh]: %zu qtype-108/109 tensor(s) resident but "
                         "'%s' is missing or not a u8 array -- their per-tensor scale and "
                         "grid code are out-of-band and they cannot be decoded\n",
                         gqh_tensors.size(), GQH_KV_KEY);
            return false;
        }
        const size_t n = gguf_get_arr_n(g, id);
        const uint8_t * d = (const uint8_t *) gguf_get_arr_data(g, id);
        blob.assign(d, d + n);
        gguf_free(g);
    }

    // Parse. Every field is bounds-checked before use; a corrupt KV must fail the
    // load, not produce a tensor that aborts at decode time.
    std::map<std::string, gqh_kv_entry> entries;
    {
        const size_t n = blob.size();
        auto fail = [&](const std::string & msg) {
            std::fprintf(stderr, "[gqh]: invalid '%s': %s\n", GQH_KV_KEY, msg.c_str());
            return false;
        };
        if (n < 16) {
            return fail("blob truncated (" + std::to_string(n) + " B < 16 B header)");
        }
        if (std::memcmp(blob.data(), GQH_MAGIC, 8) != 0) {
            return fail("bad magic (expected GQHh1)");
        }
        uint32_t count = 0, reserved = 0;
        std::memcpy(&count,    blob.data() +  8, 4);
        std::memcpy(&reserved, blob.data() + 12, 4);
        if (reserved != 0) {
            return fail("reserved field " + std::to_string(reserved) + " != 0 (format drift?)");
        }
        size_t off = 16;
        for (uint32_t i = 0; i < count; ++i) {
            const std::string at = "entry " + std::to_string(i);
            if (off + 4 > n) return fail(at + ": truncated before name_len");
            uint32_t name_len = 0;
            std::memcpy(&name_len, blob.data() + off, 4);
            off += 4;
            if (name_len == 0 || name_len > 1024 || off + name_len > n) {
                return fail(at + ": bad name_len " + std::to_string(name_len));
            }
            std::string name((const char *) blob.data() + off, name_len);
            off += name_len;
            if (entries.count(name)) return fail("duplicate entry for '" + name + "'");
            if (off + 12 > n) return fail("'" + name + "': truncated metadata");
            uint32_t qtype = 0;
            float    scale = 0.0f;
            std::memcpy(&qtype, blob.data() + off + 0, 4);
            std::memcpy(&scale, blob.data() + off + 4, 4);
            const uint8_t grid_code = blob[off + 8];
            if (blob[off + 9] || blob[off + 10] || blob[off + 11]) {
                return fail("'" + name + "': padding is not zero (format drift?)");
            }
            off += 12;
            if (!gqh_qtype_has_header((int32_t) qtype)) {
                return fail("'" + name + "': qtype " + std::to_string(qtype) + " is not 108/109");
            }
            if (grid_code >= GQH_GRID_CODES_MAX) {
                return fail("'" + name + "': grid code " + std::to_string(grid_code) +
                            " >= " + std::to_string(GQH_GRID_CODES_MAX));
            }
            // The encoder sets tensor_scale to max|w|, falling back to 1.0 for an
            // all-zero tensor, so a non-finite or non-positive scale means the KV is
            // wrong rather than the tensor being unusual.
            if (!std::isfinite(scale) || scale <= 0.0f) {
                return fail("'" + name + "': tensor_scale is not finite and positive");
            }
            gqh_kv_entry e;
            e.qtype        = (int32_t) qtype;
            e.tensor_scale = scale;
            e.grid_code    = grid_code;
            entries.emplace(std::move(name), e);
        }
        if (off != n) {
            return fail(std::to_string(n - off) + " trailing bytes after " +
                        std::to_string(count) + " entries");
        }
    }

    // Validate the whole cover BEFORE registering anything, so a bad KV leaves no
    // partial state behind.
    for (const ggml_tensor * t : gqh_tensors) {
        const std::string name = t->name;
        auto it = entries.find(name);
        if (it == entries.end()) {
            std::fprintf(stderr, "[gqh]: resident tensor '%s' (%s) has no '%s' "
                         "entry -- refusing to load a tensor that would abort at decode\n",
                         name.c_str(), ggml_type_name(t->type), GQH_KV_KEY);
            return false;
        }
        if (it->second.qtype != (int32_t) t->type) {
            std::fprintf(stderr, "[gqh]: entry for '%s' says qtype %d but the "
                         "tensor is %d\n", name.c_str(), it->second.qtype, (int32_t) t->type);
            return false;
        }
        if (t->ne[2] != 1 || t->ne[3] != 1) {
            // tensor_scale is fitted per 2-D matrix and this KV is keyed by tensor
            // name, so a fused 3-D expert stack has no way to carry one scale per
            // expert. Refuse rather than apply expert 0's scale to all of them.
            std::fprintf(stderr, "[gqh]: '%s' is not 2-D -- the header KV carries "
                         "one scale per tensor, so fused expert stacks are unsupported\n",
                         name.c_str());
            return false;
        }
        if (t->ne[0] % GQH_ROW_ALIGN != 0) {
            std::fprintf(stderr, "[gqh]: '%s' row length %" PRId64 " is not a "
                         "multiple of %" PRId64 " -- the exporter must leave short rows on "
                         "a stock qtype\n", name.c_str(), (int64_t) t->ne[0], GQH_ROW_ALIGN);
            return false;
        }
        if (!ggml_is_contiguous(t)) {
            std::fprintf(stderr, "[gqh]: '%s' is not contiguous\n", name.c_str());
            return false;
        }
        it->second.matched = true;
    }

    // Entries with no resident tensor are EXPECTED, not an error: MTP-block weights
    // are skipped outside an MTP context, so a correct artifact carries entries this
    // load will never match. Refusing them is the trap that cost a build on the mix
    // line. A MIS-NAMED entry still fails, via the loop above.
    size_t unmatched = 0;
    for (const auto & kv : entries) {
        if (!kv.second.matched) ++unmatched;
    }
    if (unmatched) {
        std::fprintf(stderr, "[gqh]: %zu header entr%s name no resident tensor "
                     "(expected for MTP-block weights outside an MTP context)\n",
                     unmatched, unmatched == 1 ? "y" : "ies");
    }

    for (const ggml_tensor * t : gqh_tensors) {
        const gqh_kv_entry & e = entries.at(t->name);
        // Span the whole device allocation, not ggml_nbytes(): under the planar layout
        // the image is padded (gqh-stride.h) and lookup resolves interior row slices by
        // pointer range, so a tight span would miss the tail rows. Recomputed here from
        // the same helper the buffer type's get_alloc_size uses rather than pulled from
        // the backend, to keep this file free of ggml-backend internals.
        // ne0 lets the planar decode rebuild the plane offsets; harmless when tight.
        // planar iff the loader actually planarized this tensor -- the same condition
        // ggml_backend_cuda_gqh_set_planar applies, so the two cannot disagree.
        int planar = 0;
        size_t span = ggml_nbytes(t);
        if (gqh_planar_enabled() && t->ne[0] % GQH_SUPERBLOCK == 0) {
            planar = 1;
            const int     nsb  = (int) (t->ne[0] / GQH_SUPERBLOCK);
            const int     is3  = t->type == GGML_TYPE_GQH3;
            const int64_t rows = ggml_nelements(t) / t->ne[0];
            span = (size_t) rows * (size_t) gqh_plane_row_bytes(nsb, is3);
        }
        ggml_gqh_register_ex(t->data, span, e.tensor_scale, (int) e.grid_code,
                             t->ne[0], planar);
    }
    std::fprintf(stderr, "[gqh]: registered %zu tensor(s) from '%s'\n",
                 gqh_tensors.size(), GQH_KV_KEY);
    return true;
}


void unregister_gqh_headers(ggml_context * ctx) {
    if (!ctx) {
        return;
    }
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr;
         t = ggml_get_next_tensor(ctx, t)) {
        if ((t->type == GGML_TYPE_GQH3 || t->type == GGML_TYPE_GQH2_H) && t->data) {
            ggml_gqh_unregister(t->data);
        }
    }
}

}  // namespace common
}  // namespace dflash
