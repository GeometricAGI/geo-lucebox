#pragma once
// 4-plane GQH device layout. Ported from SuperSonic (kernels/gqh-stride.h) so the
// two implementations stay byte-comparable; keep the offset arithmetic identical.
//
// On disk (and on the CPU decode path) a GQH tensor stays the tight AoS stream of
// 105 / 73 / 66-byte superblocks. The device copy splits each ROW into planes:
//
//   d[sb]          1 B
//   ratio[sb << 3] 8 B
//   lo[sb << 6]    64 B
//   hi[sb << 5]    32 B   (gqh3 only)
//
// Why: in the tight stream a warp's 64 B low-plane read starts at sb*105 + 9. 105 is
// odd, so the start walks all 64 residues mod 64 -- 252 of 256 superblocks straddle
// two cache lines, and half land on odd addresses (which is why the tight kernel has
// to memcpy the uint16 instead of loading it). Planar puts off_lo on a 64 B boundary
// and steps it by 64, so every warp read is exactly one aligned line.
//
// gqh2_c is NOT planarized: its scale is fp16 in-block and its fused kernel splits
// lane -> (block, group) cleanly on the tight layout already.
//
// The row stride is padded to GQH_PLANE_ALIGN, so a planar tensor is slightly larger
// than ggml_nbytes() (+0.64% on the qwen38 27B strong-band artifact). The CUDA buffer
// type's get_alloc_size() reports the padded figure; nothing else may assume that a
// GQH row is ggml_row_size() bytes.

#include "gqh-tables.h"

#define GQH_PLANE_ALIGN 64

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GQH_HD __host__ __device__
#else
#define GQH_HD
#endif

// nsb : superblocks per row (= cols / GQH_SUPERBLOCK)
// is3 : 1 for gqh3 (has the high-bit plane), 0 for gqh2_h
GQH_HD static inline void gqh_plane_offsets(
        int nsb, int is3, int * off_ratio, int * off_lo, int * off_hi, int * stride) {
    int o = (nsb + 7) & ~7;
    *off_ratio = o;
    o += nsb << 3;
    o = (o + (GQH_PLANE_ALIGN - 1)) & ~(GQH_PLANE_ALIGN - 1);
    *off_lo = o;
    o += nsb << 6;
    *off_hi = o;
    if (is3) {
        o += nsb << 5;
    }
    *stride = (o + (GQH_PLANE_ALIGN - 1)) & ~(GQH_PLANE_ALIGN - 1);
}

GQH_HD static inline int gqh_plane_row_bytes(int nsb, int is3) {
    int off_ratio, off_lo, off_hi, stride;
    gqh_plane_offsets(nsb, is3, &off_ratio, &off_lo, &off_hi, &stride);
    return stride;
}

// ---- host-side helpers ----
// Plain (non-__device__) inlines: callable from .cpp and from the host half of a
// .cu/.hip translation unit, which is where both are used.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Host-only opt-in. OFF keeps the tight device layout and every existing code path
// byte-for-byte; ON planarizes at load and switches the kernels to plane addressing.
// Read once per translation unit -- the value must not change between the
// get_alloc_size() call that sizes the buffer and the kernel that reads it.
static inline int gqh_planar_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * e = getenv("DFLASH_GQH_PLANAR");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}

// Host-only: scatter one tensor's tight AoS superblocks into the 4-plane layout.
// `tight` is rows*nsb*sb_bytes; `out` must be rows*gqh_plane_row_bytes(nsb, is3)
// and is fully written (padding zeroed).
static inline void gqh_planarize(
        int is3, int64_t rows, int nsb, const uint8_t * tight, uint8_t * out) {
    const int payload = is3 ? GQH3_SB_BYTES : GQH2H_SB_BYTES;
    int off_ratio, off_lo, off_hi, stride;
    gqh_plane_offsets(nsb, is3, &off_ratio, &off_lo, &off_hi, &stride);
    memset(out, 0, (size_t) rows * (size_t) stride);
    for (int64_t r = 0; r < rows; ++r) {
        const uint8_t * src_row = tight + r * (int64_t) nsb * payload;
        uint8_t       * dst_row = out   + r * (int64_t) stride;
        for (int sb = 0; sb < nsb; ++sb) {
            const uint8_t * src = src_row + (int64_t) sb * payload;
            dst_row[sb] = src[0];
            memcpy(dst_row + off_ratio + (sb << 3), src + 1, 8);
            memcpy(dst_row + off_lo    + (sb << 6), src + 9, 64);
            if (is3) {
                memcpy(dst_row + off_hi + (sb << 5), src + 73, 32);
            }
        }
    }
}
