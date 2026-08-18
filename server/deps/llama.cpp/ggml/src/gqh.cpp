#include "gqh.h"
#include "ggml-impl.h"

#include <cinttypes>
#include <cstring>
#include <mutex>
#include <vector>

namespace {
struct gqh_entry {
    const void * base;
    size_t       nbytes;
    float        tensor_scale;
    int          grid_code;
    // Row length (ne[0]) of the registered tensor, or 0 when unknown. Only the
    // planar device layout needs it: plane offsets are a function of nsb =
    // ne0/GQH_SUPERBLOCK, and the dequant converters see a flat element count.
    int64_t      ne0;
    // True when this tensor's device image is the 4-plane layout. Per tensor, not
    // global: a process can hold planar model weights while a test or a scratch
    // tensor registered through ggml_gqh_register stays tight.
    bool         planar;
};
std::mutex                  g_gqh_mtx;
std::vector<gqh_entry>      g_gqh_registry;
}  // namespace

void ggml_gqh_register_ex(const void * base, size_t nbytes, float tensor_scale, int grid_code,
                          int64_t ne0, int planar) {
    std::lock_guard<std::mutex> lk(g_gqh_mtx);
    const gqh_entry ne{base, nbytes, tensor_scale, grid_code, ne0, planar != 0};
    for (auto & e : g_gqh_registry) {
        if (e.base == base) { e = ne; return; }
    }
    g_gqh_registry.push_back(ne);
}

void ggml_gqh_register(const void * base, size_t nbytes, float tensor_scale, int grid_code) {
    // ne0 unknown: callers on the tight layout (the CPU decoders and the tests) never
    // need it, and the planar path registers through ggml_gqh_register_ex.
    ggml_gqh_register_ex(base, nbytes, tensor_scale, grid_code, 0, /*planar=*/0);
}

void ggml_gqh_unregister(const void * base) {
    std::lock_guard<std::mutex> lk(g_gqh_mtx);
    for (size_t i = 0; i < g_gqh_registry.size(); ++i) {
        if (g_gqh_registry[i].base == base) {
            g_gqh_registry.erase(g_gqh_registry.begin() + i);
            return;
        }
    }
}

bool ggml_gqh_lookup(const void * p, float * tensor_scale, int * grid_code) {
    return ggml_gqh_lookup_ex(p, tensor_scale, grid_code, nullptr, nullptr, nullptr);
}

bool ggml_gqh_lookup_ex(const void * p, float * tensor_scale, int * grid_code,
                        int64_t * ne0, const void ** base, int * planar) {
    std::lock_guard<std::mutex> lk(g_gqh_mtx);
    for (const auto & e : g_gqh_registry) {
        const uint8_t * b = (const uint8_t *) e.base;
        if ((const uint8_t *) p >= b && (const uint8_t *) p < b + e.nbytes) {
            *tensor_scale = e.tensor_scale;
            *grid_code    = e.grid_code;
            if (ne0)    { *ne0    = e.ne0;  }
            if (base)   { *base   = e.base; }
            if (planar) { *planar = e.planar ? 1 : 0; }
            return true;
        }
    }
    return false;
}

// The tables hold raw float32 bit patterns so no compiler can re-round them.
static inline float gqh_f32(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static void gqh_header_or_abort(const void * vx, int64_t k, size_t sb_bytes,
                                float * scale, int * code) {
    if (!ggml_gqh_lookup(vx, scale, code)) {
        GGML_ABORT("gqh: tensor slice %p is not registered -- the per-tensor header KV "
                   "was not read at load time", vx);
    }
    if (k % GQH_SUPERBLOCK != 0) {
        GGML_ABORT("gqh: dequant of %" PRId64 " elements is not a whole number of superblocks", k);
    }
    GGML_UNUSED(sb_bytes);
}

// gqh3 and gqh2_h share a head: d_real = e4m3(d) * tensor_scale, then
// s_b = d_real * (ratio/15) per 16-weight sub-block.
static inline float gqh_subblock_scale(const uint8_t * b, int sub, float tensor_scale) {
    const uint8_t d = b[0];
    const float d_real = gqh_f32(GQH_E4M3_LUT[d >> 3][d & 7]) * tensor_scale;
    const uint8_t rb = b[1 + (sub >> 1)];
    const int ratio = (sub & 1) ? (rb >> 4) : (rb & 0x0f);
    return d_real * gqh_f32(GQH_RATIO_Q[ratio][0]);
}

void dequantize_row_gqh3(const void * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    float scale;
    int   code;
    gqh_header_or_abort(vx, k, GQH3_SB_BYTES, &scale, &code);

    const uint8_t * b = (const uint8_t *) vx;
    for (int64_t sb = 0; sb < k / GQH_SUPERBLOCK; ++sb, b += GQH3_SB_BYTES) {
        for (int sub = 0; sub < GQH_N_SUB; ++sub) {
            const float s_b = gqh_subblock_scale(b, sub, scale);
            for (int t = 0; t < GQH_SUBBLOCK; ++t) {
                const int j  = sub * GQH_SUBBLOCK + t;
                const int lo = (b[ 9 + (j >> 2)] >> (2 * (j & 3))) & 0x03;
                const int hi = (b[73 + (j >> 3)] >> (j & 7)) & 0x01;
                y[sb * GQH_SUPERBLOCK + j] = gqh_f32(GQH3_GRID[code][lo | (hi << 2)]) * s_b;
            }
        }
    }
}

void dequantize_row_gqh2_h(const void * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    float scale;
    int   code;
    gqh_header_or_abort(vx, k, GQH2H_SB_BYTES, &scale, &code);

    const uint8_t * b = (const uint8_t *) vx;
    for (int64_t sb = 0; sb < k / GQH_SUPERBLOCK; ++sb, b += GQH2H_SB_BYTES) {
        for (int sub = 0; sub < GQH_N_SUB; ++sub) {
            const float s_b = gqh_subblock_scale(b, sub, scale);
            for (int t = 0; t < GQH_SUBBLOCK; ++t) {
                const int j = sub * GQH_SUBBLOCK + t;
                const int c = (b[9 + (j >> 2)] >> (2 * (j & 3))) & 0x03;
                y[sb * GQH_SUPERBLOCK + j] = gqh_f32(GQH2H_GRID[code][c]) * s_b;
            }
        }
    }
}

// gqh2_c: 66 B superblock = fp16 d, then 8 blocks of 32. Each block is 4 codebook
// indices plus a uint32 holding four 7-bit sign indices and a uint4 ratio.
// No registry: everything it needs is in-block or a frozen table.
void dequantize_row_gqh2_c(const void * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    if (k % GQH_SUPERBLOCK != 0) {
        GGML_ABORT("gqh2_c: dequant of %" PRId64 " elements is not a whole number of superblocks", k);
    }
    const uint8_t * b = (const uint8_t *) vx;
    for (int64_t sb = 0; sb < k / GQH_SUPERBLOCK; ++sb, b += GQH2C_SB_BYTES) {
        ggml_fp16_t dh;
        memcpy(&dh, b, sizeof(dh));
        const float d = GGML_FP16_TO_FP32(dh);

        for (int blk = 0; blk < GQH2C_BLOCKS_PER_SB; ++blk) {
            const uint8_t * p = b + 2 + blk * 8;
            uint32_t u;
            memcpy(&u, p + 4, sizeof(u));
            const float s_blk = d * gqh_f32(GQH_RATIO_Q[(u >> 28) & 0x0f][0]);

            for (int grp = 0; grp < GQH2C_GROUPS_PER_BLOCK; ++grp) {
                const uint8_t mask = GQH2C_SIGN_MASK[(u >> (7 * grp)) & 0x7f];
                const uint32_t * cb = GQH2C_CODEBOOK[p[grp]];
                float * out = y + sb * GQH_SUPERBLOCK + blk * GQH2C_BLOCK + grp * GQH2C_GROUP;
                for (int e = 0; e < GQH2C_GROUP; ++e) {
                    // reference is sign * mag * s_blk, left to right -- keep the
                    // sign on the magnitude so a zero s_blk yields -0.0 the same way
                    const float mag = ((mask >> e) & 1) ? -gqh_f32(cb[e]) : gqh_f32(cb[e]);
                    out[e] = mag * s_blk;
                }
            }
        }
    }
}
