#include "gqh.cuh"
#include "../gqh.h"
#include "../gqh-stride.h"

#include <cstdio>
#include <cstring>

// Device-side constants, single-sourced with the host tables through the *_INIT
// macros in gqh-tables.h so the two cannot drift.
static __constant__ uint32_t GQH_E4M3_D[32][8]    = GQH_E4M3_LUT_INIT;
static __constant__ uint32_t GQH_RATIO_Q_D[16][1] = GQH_RATIO_Q_INIT;
static __constant__ uint32_t GQH2C_CB_D[256][8]   = GQH2C_CODEBOOK_INIT;
static __constant__ uint8_t  GQH2C_SIGN_D[128]    = GQH2C_SIGN_MASK_INIT;

// The grid is per-tensor and only 4-8 floats, so it travels as a by-value kernel
// argument instead of a device lookup. That is also what keeps an MMVQ vec-dot
// reachable later: the header resolves once at graph time, not per block.
struct gqh_grid8 { float v[8]; };
struct gqh_grid4 { float v[4]; };

static __device__ __forceinline__ float gqh_bits(uint32_t u) {
    return __int_as_float((int) u);
}

static __device__ __forceinline__ void gqh_store(float * p, float v) { *p = v; }
static __device__ __forceinline__ void gqh_store(half  * p, float v) { *p = __float2half(v); }

// gqh3 and gqh2_h share a head: E4M3 superblock scale times the per-tensor scale,
// then the uint4 sub-block ratio. The operation order matches gqh.py exactly --
// d_real = e4m3(d) * tensor_scale, then s_b = d_real * (ratio/15). Do not reassociate.
static __device__ __forceinline__ float gqh_subblock_scale(
        const uint8_t * __restrict__ b, int sub, float tensor_scale) {
    const uint8_t d = b[0];
    const float d_real = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * tensor_scale;
    const uint8_t rb = b[1 + (sub >> 1)];
    const int ratio = (sub & 1) ? (rb >> 4) : (rb & 0x0f);
    return d_real * gqh_bits(GQH_RATIO_Q_D[ratio][0]);
}

// --- gqh3 -------------------------------------------------------------------
// 105 B superblock: [0] E4M3 d, [1:9] 16x uint4 ratios, [9:73] low-2-bit code
// plane (4/byte), [73:105] high-1-bit code plane (8/byte).

// `nsb_row` is the superblocks-per-row needed to split the flat superblock index
// back into (row, sb) for the planar layout; it is unused when PLANAR is false.
template <typename dst_t, bool PLANAR>
static __global__ void gqh3_decode_kernel(
        const uint8_t * __restrict__ wire, float tensor_scale, gqh_grid8 grid,
        dst_t * __restrict__ dst, int nsb_row) {
    const int64_t sb_flat = blockIdx.x;
    const int j = threadIdx.x;                                  // 0..255
    float s_b;
    int lo, hi;
    if (PLANAR) {
        const int64_t row = sb_flat / nsb_row;
        const int     sb  = (int) (sb_flat - row * nsb_row);
        int off_r, off_lo, off_hi, stride;
        gqh_plane_offsets(nsb_row, 1, &off_r, &off_lo, &off_hi, &stride);
        const uint8_t * __restrict__ rb_ = wire + row * (int64_t) stride;
        const int sub = j >> 4;
        const uint8_t d = rb_[sb];
        const float d_real = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * tensor_scale;
        const uint8_t rbyte = rb_[off_r + (sb << 3) + (sub >> 1)];
        const int ratio = (sub & 1) ? (rbyte >> 4) : (rbyte & 0x0f);
        s_b = d_real * gqh_bits(GQH_RATIO_Q_D[ratio][0]);
        lo = (rb_[off_lo + (sb << 6) + (j >> 2)] >> (2 * (j & 3))) & 0x03;
        hi = (rb_[off_hi + (sb << 5) + (j >> 3)] >> (j & 7)) & 0x01;
    } else {
        const uint8_t * __restrict__ b = wire + sb_flat * GQH3_SB_BYTES;
        s_b = gqh_subblock_scale(b, j >> 4, tensor_scale);
        lo = (b[ 9 + (j >> 2)] >> (2 * (j & 3))) & 0x03;
        hi = (b[73 + (j >> 3)] >> (j & 7)) & 0x01;
    }

    gqh_store(&dst[sb_flat * GQH_SUPERBLOCK + j], grid.v[lo | (hi << 2)] * s_b);
}

template <typename dst_t>
static void gqh3_decode_cuda(const void * wire, float tensor_scale, int grid_code,
                             dst_t * dst, int64_t nsb_total, int nsb_row, bool planar,
                             cudaStream_t stream) {
    gqh_grid8 grid;
    memcpy(grid.v, GQH3_GRID[grid_code], sizeof(grid.v));
    if (planar) {
        GGML_ASSERT(nsb_row > 0 && "gqh3 planar decode needs the row length");
        gqh3_decode_kernel<dst_t, true><<<nsb_total, GQH_SUPERBLOCK, 0, stream>>>(
            (const uint8_t *) wire, tensor_scale, grid, dst, nsb_row);
    } else {
        gqh3_decode_kernel<dst_t, false><<<nsb_total, GQH_SUPERBLOCK, 0, stream>>>(
            (const uint8_t *) wire, tensor_scale, grid, dst, nsb_row);
    }
}

void ggml_cuda_gqh3_decode(const void * wire, float tensor_scale, int grid_code,
                           float * dst, int64_t rows, int64_t nsb, cudaStream_t stream) {
    gqh3_decode_cuda(wire, tensor_scale, grid_code, dst, rows * nsb, (int) nsb,
                     /*planar=*/false, stream);
}

// --- gqh2_h -----------------------------------------------------------------
// 73 B superblock: [0] E4M3 d, [1:9] 16x uint4 ratios, [9:73] uint2 codes (4/byte).

template <typename dst_t, bool PLANAR>
static __global__ void gqh2h_decode_kernel(
        const uint8_t * __restrict__ wire, float tensor_scale, gqh_grid4 grid,
        dst_t * __restrict__ dst, int nsb_row) {
    const int64_t sb_flat = blockIdx.x;
    const int j = threadIdx.x;
    float s_b;
    int code;
    if (PLANAR) {
        const int64_t row = sb_flat / nsb_row;
        const int     sb  = (int) (sb_flat - row * nsb_row);
        int off_r, off_lo, off_hi, stride;
        gqh_plane_offsets(nsb_row, 0, &off_r, &off_lo, &off_hi, &stride);
        const uint8_t * __restrict__ rb_ = wire + row * (int64_t) stride;
        const int sub = j >> 4;
        const uint8_t d = rb_[sb];
        const float d_real = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * tensor_scale;
        const uint8_t rbyte = rb_[off_r + (sb << 3) + (sub >> 1)];
        const int ratio = (sub & 1) ? (rbyte >> 4) : (rbyte & 0x0f);
        s_b = d_real * gqh_bits(GQH_RATIO_Q_D[ratio][0]);
        code = (rb_[off_lo + (sb << 6) + (j >> 2)] >> (2 * (j & 3))) & 0x03;
    } else {
        const uint8_t * __restrict__ b = wire + sb_flat * GQH2H_SB_BYTES;
        s_b  = gqh_subblock_scale(b, j >> 4, tensor_scale);
        code = (b[9 + (j >> 2)] >> (2 * (j & 3))) & 0x03;
    }

    gqh_store(&dst[sb_flat * GQH_SUPERBLOCK + j], grid.v[code] * s_b);
}

template <typename dst_t>
static void gqh2h_decode_cuda(const void * wire, float tensor_scale, int grid_code,
                              dst_t * dst, int64_t nsb_total, int nsb_row, bool planar,
                              cudaStream_t stream) {
    gqh_grid4 grid;
    memcpy(grid.v, GQH2H_GRID[grid_code], sizeof(grid.v));
    if (planar) {
        GGML_ASSERT(nsb_row > 0 && "gqh2_h planar decode needs the row length");
        gqh2h_decode_kernel<dst_t, true><<<nsb_total, GQH_SUPERBLOCK, 0, stream>>>(
            (const uint8_t *) wire, tensor_scale, grid, dst, nsb_row);
    } else {
        gqh2h_decode_kernel<dst_t, false><<<nsb_total, GQH_SUPERBLOCK, 0, stream>>>(
            (const uint8_t *) wire, tensor_scale, grid, dst, nsb_row);
    }
}

void ggml_cuda_gqh2h_decode(const void * wire, float tensor_scale, int grid_code,
                            float * dst, int64_t rows, int64_t nsb, cudaStream_t stream) {
    gqh2h_decode_cuda(wire, tensor_scale, grid_code, dst, rows * nsb, (int) nsb,
                      /*planar=*/false, stream);
}

// --- gqh2_c -----------------------------------------------------------------
// 66 B superblock: fp16 d, then 8 blocks of 32. Each block is 4 codebook indices
// plus a uint32 holding four 7-bit sign indices and a uint4 ratio. Needs no
// per-tensor header -- the scale is in-block and the codebook is frozen.

template <typename dst_t>
static __global__ void gqh2c_decode_kernel(
        const uint8_t * __restrict__ wire, dst_t * __restrict__ dst) {
    const int64_t sb = blockIdx.x;
    const int j = threadIdx.x;                                  // 0..255
    const uint8_t * __restrict__ b = wire + sb * GQH2C_SB_BYTES;

    const int blk = j / GQH2C_BLOCK;
    const int grp = (j % GQH2C_BLOCK) / GQH2C_GROUP;
    const int e   = j % GQH2C_GROUP;

    __half dh;
    memcpy(&dh, b, sizeof(dh));
    const uint8_t * p = b + 2 + blk * 8;
    uint32_t u;
    memcpy(&u, p + 4, sizeof(u));

    const float s_blk = __half2float(dh) * gqh_bits(GQH_RATIO_Q_D[(u >> 28) & 0x0f][0]);
    const uint8_t mask = GQH2C_SIGN_D[(u >> (7 * grp)) & 0x7f];
    const float raw = gqh_bits(GQH2C_CB_D[p[grp]][e]);
    // sign rides on the magnitude, so a zero s_blk yields -0.0 like the reference
    const float mag = ((mask >> e) & 1) ? -raw : raw;

    gqh_store(&dst[sb * GQH_SUPERBLOCK + j], mag * s_blk);
}

template <typename dst_t>
static void gqh2c_decode_cuda(const void * wire, dst_t * dst, int64_t nsb_total,
                              cudaStream_t stream) {
    gqh2c_decode_kernel<dst_t><<<nsb_total, GQH_SUPERBLOCK, 0, stream>>>(
        (const uint8_t *) wire, dst);
}

void ggml_cuda_gqh2c_decode(const void * wire, float * dst,
                            int64_t rows, int64_t nsb, cudaStream_t stream) {
    gqh2c_decode_cuda(wire, dst, rows * nsb, stream);
}

// --- fused batch-1 matvec ---------------------------------------------------
// A dedicated kernel rather than an MMVQ vec_dot, following the ROCmFPX mix
// precedent: vec_dot_q_cuda_t carries no per-tensor argument, and GQH needs the
// grid (8 floats) plus tensor_scale, which live in the per-tensor header. Passing
// them as kernel arguments here avoids changing a signature every qtype shares.
// f32 activations also sidestep q8_1's 32-weight block straddling GQH's 16-weight
// sub-blocks, which carry different uint4 ratios.
//
// One warp per output row, 8 weights per lane, so each lane reads 2 adjacent
// bytes of the low-2-bit plane and 1 byte of the high-1-bit plane -- the warp
// covers bytes [9,73) and [73,105) contiguously.

#define GQH_WARP          32
#define GQH_MATVEC_WARPS   4
#define GQH_PER_LANE       (GQH_SUPERBLOCK / GQH_WARP)   // 8
// Columns handled by one pass over the weights. The weight matrix is the whole
// DRAM cost of a decode matvec, so re-reading it per column makes an N-slot server
// N times slower than it should be -- measured as 8 slots buying only 1.35x over
// single-stream. Every column is accumulated against one load instead.
#define GQH_MAX_COLS       8

// Down-shift shuffle confined to a 32-lane logical group. The explicit width is
// what keeps the reduction self-contained on wave64 (GFX8/9), and is a no-op on
// wave32 (gfx1151 Strix Halo, gfx1201 RDNA4) and NVIDIA. HIP keeps the bare
// mask-free __shfl_down and its vendor shim does not cover __shfl_down_sync, so
// branch -- same reason rocmfp3_mix.cu carries mix_warp_shfl_down.
static __device__ __forceinline__ float gqh_warp_shfl_down(float v, int off) {
#if defined(__HIP_PLATFORM_AMD__)
    return __shfl_down(v, off, GQH_WARP);
#else
    return __shfl_down_sync(0xffffffffu, v, off, GQH_WARP);
#endif
}

// Both grids are symmetric about zero, so the level is a SIGN plus one of four
// magnitudes. Selecting from registers beats indexing the table: the SASS showed
// the table version dominated by divergent constant-bank loads (8 distinct grid
// entries per warp serialise into 8 replays), 78 LDC against 23 FP ops.
// gqh3:   grid = [-m3,-m2,-m1,-m0, m0,m1,m2,m3]
// gqh2_h: grid = [-1, -a, +a, +1], so m.x = a and the outer level is 1.
template <bool IS_GQH3>
static __device__ __forceinline__ float gqh_level(int code, const float4 & m) {
    if (IS_GQH3) {
        const int hi  = (code >> 2) & 1;
        const int k   = code & 3;
        const int idx = hi ? k : (3 - k);
        const float e0  = (idx & 1) ? m.y : m.x;
        const float e1  = (idx & 1) ? m.w : m.z;
        const float mag = (idx & 2) ? e1 : e0;
        return hi ? mag : -mag;
    }
    const int hi = (code >> 1) & 1;
    const float mag = ((code & 1) != hi) ? m.x : 1.0f;
    return hi ? mag : -mag;
}

template <bool IS_GQH3, bool PLANAR>
static __global__ void gqh_matvec_kernel(
        const uint8_t * __restrict__ data, const float * __restrict__ x,
        float * __restrict__ y, int in, int out, int ncols, float tensor_scale, float4 mag,
        int64_t x_col_stride, int64_t y_col_stride) {
    const int sb_bytes = IS_GQH3 ? GQH3_SB_BYTES : GQH2H_SB_BYTES;
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int row  = blockIdx.x * warps_per_block + (threadIdx.x / GQH_WARP);
    const int lane = threadIdx.x % GQH_WARP;

    // ratio/15 in LDS, not constant memory: the index is the lane's sub-block, so
    // it is divergent, and a divergent constant-bank load serialises per address.
    // 16 consecutive floats sit in 16 distinct LDS banks, so this is conflict-free.
    // Hoisted above the early return -- __syncthreads needs the whole block, and
    // `row >= out` retires whole warps in the tail block.
    __shared__ float s_ratio[16];
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
    }
    __syncthreads();
    if (row >= out) return;

    const int nsb = in / GQH_SUPERBLOCK;
    // Tight: one AoS stream, superblock sb at row*nsb*sb_bytes + sb*sb_bytes.
    // Planar: four per-row planes, each stepped by its own power-of-two stride, so
    // the low-plane read below is a single aligned 64 B line per warp instead of a
    // straddling one (see gqh-stride.h).
    int off_r = 0, off_lo = 0, off_hi = 0, pstride = 0;
    if (PLANAR) {
        gqh_plane_offsets(nsb, IS_GQH3 ? 1 : 0, &off_r, &off_lo, &off_hi, &pstride);
    }
    const uint8_t * __restrict__ rowbase =
        PLANAR ? data + (int64_t) row * pstride
               : data + (int64_t) row * nsb * sb_bytes;

    const int j0  = lane * GQH_PER_LANE;   // this lane's first weight in the superblock
    const int sub = j0 >> 4;               // two lanes share a 16-weight sub-block

    float acc[GQH_MAX_COLS] = { 0.0f };
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t * __restrict__ b = PLANAR ? rowbase : rowbase + (int64_t) sb * sb_bytes;

        // same order as the reference: d_real = e4m3(d) * tensor_scale, then
        // s_b = d_real * (ratio/15). The d byte is warp-uniform so its table read
        // broadcasts. Identical arithmetic in both layouts -- only the addresses move,
        // so the result stays bit-identical.
        const uint8_t d  = PLANAR ? b[sb] : b[0];
        const float d_real = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * tensor_scale;
        const uint8_t rb = PLANAR ? b[off_r + (sb << 3) + (sub >> 1)] : b[1 + (sub >> 1)];
        const float s_b = d_real * s_ratio[(sub & 1) ? (rb >> 4) : (rb & 0x0f)];

        uint16_t lo2;                                  // 8 codes x 2 bits
        if (PLANAR) {
            // off_lo is 64 B-aligned and (sb<<6) keeps it so, hence a real aligned load.
            lo2 = *(const uint16_t *) (b + off_lo + (sb << 6) + lane * 2);
        } else {
            memcpy(&lo2, b + 9 + lane * 2, sizeof(lo2));
        }
        const uint8_t hi1 = IS_GQH3 ? (PLANAR ? b[off_hi + (sb << 5) + lane] : b[73 + lane]) : 0;

        // Decode this lane's 8 weights ONCE, then reuse them for every column.
        float w[GQH_PER_LANE];
#pragma unroll
        for (int t = 0; t < GQH_PER_LANE; ++t) {
            const int lo = (lo2 >> (2 * t)) & 0x03;
            const int code = IS_GQH3 ? (lo | (((hi1 >> t) & 1) << 2)) : lo;
            w[t] = gqh_level<IS_GQH3>(code, mag) * s_b;
        }

        // Fully unrolled so acc[]/xs[] stay in registers; the per-column term order
        // is unchanged, so each output is bit-identical to the one-column-per-block
        // version this replaces.
#pragma unroll
        for (int c = 0; c < GQH_MAX_COLS; ++c) {
            if (c >= ncols) {
                continue;
            }
            const float * __restrict__ xcc = x + (int64_t) c * x_col_stride
                                               + sb * GQH_SUPERBLOCK + j0;
            const float4 x0 = *(const float4 *) (xcc);
            const float4 x1 = *(const float4 *) (xcc + 4);
            const float xs[GQH_PER_LANE] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w };
#pragma unroll
            for (int t = 0; t < GQH_PER_LANE; ++t) {
                acc[c] += w[t] * xs[t];
            }
        }
    }

#pragma unroll
    for (int c = 0; c < GQH_MAX_COLS; ++c) {
        if (c >= ncols) {
            continue;
        }
#pragma unroll
        for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
            acc[c] += gqh_warp_shfl_down(acc[c], off);
        }
        if (lane == 0) {
            y[(int64_t) c * y_col_stride + row] = acc[c];
        }
    }
}

// gqh2_c fused matvec. 256 weights / 32 lanes = 8, which is exactly one codebook
// group, so lane -> (block, group) is a clean split with no straddling.
static __global__ void gqh2c_matvec_kernel(
        const uint8_t * __restrict__ data, const float * __restrict__ x,
        float * __restrict__ y, int in, int out,
        int64_t x_col_stride, int64_t y_col_stride) {
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int row  = blockIdx.x * warps_per_block + (threadIdx.x / GQH_WARP);
    const int lane = threadIdx.x % GQH_WARP;
    const int col  = blockIdx.y;
    if (row >= out) return;

    const int nsb = in / GQH_SUPERBLOCK;
    const uint8_t * __restrict__ rowbase = data + (int64_t) row * nsb * GQH2C_SB_BYTES;
    const float   * __restrict__ xc      = x + (int64_t) col * x_col_stride;

    const int blk = lane / GQH2C_GROUPS_PER_BLOCK;
    const int grp = lane % GQH2C_GROUPS_PER_BLOCK;
    const int j0  = blk * GQH2C_BLOCK + grp * GQH2C_GROUP;

    float acc = 0.0f;
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t * __restrict__ b = rowbase + (int64_t) sb * GQH2C_SB_BYTES;
        __half dh;
        memcpy(&dh, b, sizeof(dh));
        const uint8_t * __restrict__ p = b + 2 + blk * 8;
        uint32_t u;
        memcpy(&u, p + 4, sizeof(u));

        const float s_blk = __half2float(dh) * gqh_bits(GQH_RATIO_Q_D[(u >> 28) & 0x0f][0]);
        const uint8_t mask = GQH2C_SIGN_D[(u >> (7 * grp)) & 0x7f];
        const uint32_t * cb = GQH2C_CB_D[p[grp]];
        const float * __restrict__ xs = xc + sb * GQH_SUPERBLOCK + j0;

#pragma unroll
        for (int e = 0; e < GQH2C_GROUP; ++e) {
            const float raw = gqh_bits(cb[e]);
            const float m   = ((mask >> e) & 1) ? -raw : raw;
            acc += (m * s_blk) * xs[e];
        }
    }

#pragma unroll
    for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
        acc += gqh_warp_shfl_down(acc, off);
    }
    if (lane == 0) {
        y[(int64_t) col * y_col_stride + row] = acc;
    }
}

bool ggml_cuda_gqh_mul_mat_vec(
        ggml_type type, const void * vx, const float * x, float * y,
        int in, int out, int ncols, int64_t x_col_stride, int64_t y_col_stride,
        cudaStream_t stream) {
    if (type != GGML_TYPE_GQH3 && type != GGML_TYPE_GQH2_H && type != GGML_TYPE_GQH2_C) {
        return false;
    }
    if (in % GQH_SUPERBLOCK != 0 || ncols <= 0 || ncols > GQH_MAX_COLS) {
        return false;   // wider batches keep the dequant->GEMM path
    }
    // The kernel reads activations as float4. in is a multiple of 256 and lanes are
    // 8 floats apart, so every offset is 32-byte aligned -- but only if the base is
    // 16-byte aligned to begin with. A misaligned 128-bit load FAULTS on AMD rather
    // than just running slow, so check instead of assuming.
    if (((uintptr_t) x) % sizeof(float4) != 0 ||
        (x_col_stride * sizeof(float)) % sizeof(float4) != 0) {
        return false;
    }
    const dim3 blocks_c((out + GQH_MATVEC_WARPS - 1) / GQH_MATVEC_WARPS, ncols, 1);
    const dim3 threads_c(GQH_WARP * GQH_MATVEC_WARPS, 1, 1);
    if (type == GGML_TYPE_GQH2_C) {
        // no header to resolve: fp16 scale in-block, frozen codebook
        gqh2c_matvec_kernel<<<blocks_c, threads_c, 0, stream>>>(
            (const uint8_t *) vx, x, y, in, out, x_col_stride, y_col_stride);
        return true;
    }

    float scale;
    int   code;
    int   planar = 0;
    if (!ggml_gqh_lookup_ex(vx, &scale, &code, nullptr, nullptr, &planar)) {
        return false;   // unregistered -> caller keeps the dequant fallback
    }

    // Positive half of the grid: gqh3 keeps all four magnitudes, gqh2_h needs only
    // the inner level a (its outer level is exactly 1).
    float4 mag{};
    if (type == GGML_TYPE_GQH3) {
        memcpy(&mag.x, &GQH3_GRID[code][4], 4 * sizeof(float));
    } else {
        memcpy(&mag.x, &GQH2H_GRID[code][2], sizeof(float));
    }

    const dim3 blocks((out + GQH_MATVEC_WARPS - 1) / GQH_MATVEC_WARPS, 1, 1);
    const dim3 threads(GQH_WARP * GQH_MATVEC_WARPS, 1, 1);
    if (type == GGML_TYPE_GQH3) {
        if (planar) {
            gqh_matvec_kernel<true, true><<<blocks, threads, 0, stream>>>(
                (const uint8_t *) vx, x, y, in, out, ncols, scale, mag, x_col_stride, y_col_stride);
        } else {
            gqh_matvec_kernel<true, false><<<blocks, threads, 0, stream>>>(
                (const uint8_t *) vx, x, y, in, out, ncols, scale, mag, x_col_stride, y_col_stride);
        }
    } else {
        if (planar) {
            gqh_matvec_kernel<false, true><<<blocks, threads, 0, stream>>>(
                (const uint8_t *) vx, x, y, in, out, ncols, scale, mag, x_col_stride, y_col_stride);
        } else {
            gqh_matvec_kernel<false, false><<<blocks, threads, 0, stream>>>(
                (const uint8_t *) vx, x, y, in, out, ncols, scale, mag, x_col_stride, y_col_stride);
        }
    }
    return true;
}

// --- registry-aware converters ----------------------------------------------
// These match the ggml to_fp16 / to_fp32 signatures, which carry no tensor, so
// the header comes from the registry. k is an element count and is always a whole
// number of superblocks (a GQH row is cols/256 superblocks and cols % 256 == 0).

template <typename dst_t>
static void gqh_convert(bool is_gqh3, const void * vx, dst_t * y, int64_t k, cudaStream_t stream) {
    float scale = 0.0f;
    int   code  = 0;
    int64_t ne0 = 0;
    int     planar = 0;
    if (!ggml_gqh_lookup_ex(vx, &scale, &code, &ne0, nullptr, &planar)) {
        GGML_ABORT("gqh: tensor slice %p is not registered -- the per-tensor header KV "
                   "was not read at load time", vx);
    }
    if (k % GQH_SUPERBLOCK != 0) {
        GGML_ABORT("gqh: dequant of %lld elements is not a whole number of superblocks",
                   (long long) k);
    }
    const int64_t nsb = k / GQH_SUPERBLOCK;
    // Planar rows are strided by the padded plane stride, so the decode has to know
    // the row length. It only reaches the registry via ggml_gqh_register_ex, which is
    // the same call that planarizes -- if it is missing here the buffer is planar but
    // the row length was never recorded, and guessing would silently corrupt.
    int nsb_row = 0;
    if (planar) {
        if (ne0 <= 0 || ne0 % GQH_SUPERBLOCK != 0) {
            GGML_ABORT("gqh: planar dequant of %p needs the tensor row length, but the "
                       "registry has ne0=%lld -- register through ggml_gqh_register_ex",
                       vx, (long long) ne0);
        }
        nsb_row = (int) (ne0 / GQH_SUPERBLOCK);
    }
    if (is_gqh3) {
        gqh3_decode_cuda(vx, scale, code, y, nsb, nsb_row, planar != 0, stream);
    } else {
        gqh2h_decode_cuda(vx, scale, code, y, nsb, nsb_row, planar != 0, stream);
    }
}

void dequantize_gqh3_to_fp16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    gqh_convert(true, vx, y, k, stream);
}
void dequantize_gqh2h_to_fp16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    gqh_convert(false, vx, y, k, stream);
}
void dequantize_gqh3_to_fp32_cuda(const void * vx, float * y, int64_t k, cudaStream_t stream) {
    gqh_convert(true, vx, y, k, stream);
}
void dequantize_gqh2h_to_fp32_cuda(const void * vx, float * y, int64_t k, cudaStream_t stream) {
    gqh_convert(false, vx, y, k, stream);
}

// gqh2_c takes no registry lookup: nothing about its decode is out of band.
void dequantize_gqh2c_to_fp16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    gqh2c_decode_cuda(vx, y, k / GQH_SUPERBLOCK, stream);
}
void dequantize_gqh2c_to_fp32_cuda(const void * vx, float * y, int64_t k, cudaStream_t stream) {
    gqh2c_decode_cuda(vx, y, k / GQH_SUPERBLOCK, stream);
}
