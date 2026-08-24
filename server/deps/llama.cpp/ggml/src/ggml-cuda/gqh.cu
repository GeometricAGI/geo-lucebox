#include "gqh.cuh"
#include "unary.cuh"   // ggml_cuda_op_silu_single, for the GLU-fused pre-pass
#include "../gqh.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

// Device-side constants, single-sourced with the host tables through the *_INIT
// macros in gqh-tables.h so the two cannot drift.
static __constant__ uint32_t GQH_E4M3_D[32][8]    = GQH_E4M3_LUT_INIT;
static __constant__ uint32_t GQH_RATIO_Q_D[16][1] = GQH_RATIO_Q_INIT;
static __constant__ uint32_t GQH2C_CB_D[256][8]   = GQH2C_CODEBOOK_INIT;
static __constant__ uint8_t  GQH2C_SIGN_D[128]    = GQH2C_SIGN_MASK_INIT;

// The grid is per-tensor and only 4-8 floats, so it travels as a by-value kernel
// argument instead of a device lookup. That is also what keeps an MMVQ vec-dot
// reachable later: the header resolves once at graph time, not per block.
struct gqh_grid16 { float v[16]; };
struct gqh_grid8 { float v[8]; };
struct gqh_grid4 { float v[4]; };

static __device__ __forceinline__ float gqh_bits(uint32_t u) {
    return __int_as_float((int) u);
}

static __device__ __forceinline__ void gqh_store(float * p, float v) { *p = v; }
static __device__ __forceinline__ void gqh_store(half  * p, float v) { *p = __float2half(v); }

// gqh4, gqh3 and gqh2_h share a head: E4M3 superblock scale times the per-tensor scale,
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

template <typename dst_t>
static __global__ void gqh3_decode_kernel(
        const uint8_t * __restrict__ wire, float tensor_scale, gqh_grid8 grid,
        dst_t * __restrict__ dst) {
    const int64_t sb = blockIdx.x;
    const int j = threadIdx.x;                                  // 0..255
    const uint8_t * __restrict__ b = wire + sb * GQH3_SB_BYTES;

    const float s_b = gqh_subblock_scale(b, j >> 4, tensor_scale);

    const int lo = (b[ 9 + (j >> 2)] >> (2 * (j & 3))) & 0x03;
    const int hi = (b[73 + (j >> 3)] >> (j & 7)) & 0x01;

    gqh_store(&dst[sb * GQH_SUPERBLOCK + j], grid.v[lo | (hi << 2)] * s_b);
}

template <typename dst_t>
static void gqh3_decode_cuda(const void * wire, float tensor_scale, int grid_code,
                             dst_t * dst, int64_t nsb_total, cudaStream_t stream) {
    gqh_grid8 grid;
    memcpy(grid.v, GQH3_GRID[grid_code], sizeof(grid.v));
    gqh3_decode_kernel<dst_t><<<nsb_total, GQH_SUPERBLOCK, 0, stream>>>(
        (const uint8_t *) wire, tensor_scale, grid, dst);
}

void ggml_cuda_gqh3_decode(const void * wire, float tensor_scale, int grid_code,
                           float * dst, int64_t rows, int64_t nsb, cudaStream_t stream) {
    gqh3_decode_cuda(wire, tensor_scale, grid_code, dst, rows * nsb, stream);
}

// --- gqh4 -------------------------------------------------------------------
// 137 B superblock: [0] E4M3 d, [1:9] 16x uint4 ratios, [9:137] uint4 codes packed
// two per byte -- even weight in the low nibble, odd in the high nibble.
//
// The 16-level grid is staged into LDS rather than indexed out of the by-value
// argument: `code` is divergent, and an array kernel arg indexed divergently either
// spills to local memory or expands into a 7-deep select tree. 16 consecutive floats
// occupy 16 distinct LDS banks and each bank sees a single address, so the divergent
// read broadcasts conflict-free -- the same reasoning as s_ratio in the matvec.

template <typename dst_t>
static __global__ void gqh4_decode_kernel(
        const uint8_t * __restrict__ wire, float tensor_scale, gqh_grid16 grid,
        dst_t * __restrict__ dst) {
    const int64_t sb = blockIdx.x;
    const int j = threadIdx.x;                                  // 0..255
    const uint8_t * __restrict__ b = wire + sb * GQH4_SB_BYTES;

    __shared__ float s_grid[16];
    if (j < 16) {
        s_grid[j] = grid.v[j];
    }
    __syncthreads();

    const float s_b = gqh_subblock_scale(b, j >> 4, tensor_scale);

    const uint8_t cb = b[9 + (j >> 1)];
    const int code = (j & 1) ? (cb >> 4) : (cb & 0x0f);

    gqh_store(&dst[sb * GQH_SUPERBLOCK + j], s_grid[code] * s_b);
}

template <typename dst_t>
static void gqh4_decode_cuda(const void * wire, float tensor_scale, int grid_code,
                             dst_t * dst, int64_t nsb_total, cudaStream_t stream) {
    gqh_grid16 grid;
    memcpy(grid.v, GQH4_GRID[grid_code], sizeof(grid.v));
    gqh4_decode_kernel<dst_t><<<nsb_total, GQH_SUPERBLOCK, 0, stream>>>(
        (const uint8_t *) wire, tensor_scale, grid, dst);
}

void ggml_cuda_gqh4_decode(const void * wire, float tensor_scale, int grid_code,
                           float * dst, int64_t rows, int64_t nsb, cudaStream_t stream) {
    gqh4_decode_cuda(wire, tensor_scale, grid_code, dst, rows * nsb, stream);
}

// --- gqh2_h -----------------------------------------------------------------
// 73 B superblock: [0] E4M3 d, [1:9] 16x uint4 ratios, [9:73] uint2 codes (4/byte).

template <typename dst_t>
static __global__ void gqh2h_decode_kernel(
        const uint8_t * __restrict__ wire, float tensor_scale, gqh_grid4 grid,
        dst_t * __restrict__ dst) {
    const int64_t sb = blockIdx.x;
    const int j = threadIdx.x;
    const uint8_t * __restrict__ b = wire + sb * GQH2H_SB_BYTES;

    const float s_b = gqh_subblock_scale(b, j >> 4, tensor_scale);
    const int code = (b[9 + (j >> 2)] >> (2 * (j & 3))) & 0x03;

    gqh_store(&dst[sb * GQH_SUPERBLOCK + j], grid.v[code] * s_b);
}

template <typename dst_t>
static void gqh2h_decode_cuda(const void * wire, float tensor_scale, int grid_code,
                              dst_t * dst, int64_t nsb_total, cudaStream_t stream) {
    gqh_grid4 grid;
    memcpy(grid.v, GQH2H_GRID[grid_code], sizeof(grid.v));
    gqh2h_decode_kernel<dst_t><<<nsb_total, GQH_SUPERBLOCK, 0, stream>>>(
        (const uint8_t *) wire, tensor_scale, grid, dst);
}

void ggml_cuda_gqh2h_decode(const void * wire, float tensor_scale, int grid_code,
                            float * dst, int64_t rows, int64_t nsb, cudaStream_t stream) {
    gqh2h_decode_cuda(wire, tensor_scale, grid_code, dst, rows * nsb, stream);
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
// ROW_XMASK DPP for the warp reduction (see gqh_dpp_xor). gfx10+ and wave32 rows of 16;
// everything else keeps the ds_bpermute tree. `-DGQH_DPP_REDUCE=0` on the compile line
// forces the old network back for an A/B -- it cannot be an env knob, the choice is
// device code and both networks in one binary would double every instantiation.
#if !defined(GQH_DPP_REDUCE)
#  if defined(__HIP_PLATFORM_AMD__) && \
      (defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__))
#    define GQH_DPP_REDUCE 1
#  else
#    define GQH_DPP_REDUCE 0
#  endif
#endif
// REDUCE-SCATTER instead of ALL-REDUCE for the arms whose accumulator count is exactly
// the wave width (ROWS * NCOLS_MAX == GQH_WARP, i.e. the dispatched <*,8,4> pair). See
// gqh_rs_level for the network and the reduction block itself for the epilogue it
// unlocks. gqh_lane_id is load-bearing, not cosmetic -- read its comment before touching
// the lane arithmetic here.
// `-DGQH_RS_REDUCE=0` forces the all-reduce network + lane-0 store back for an A/B.
#if !defined(GQH_RS_REDUCE)
#  define GQH_RS_REDUCE 1
#endif
// The reduce-scatter's level-0 (off == 16) exchange on the VALU instead of the LDS
// crossbar. Independent of GQH_PERMLANE16, which is the same instruction in the
// all-reduce network and stays OFF there (refuted: it breaks that network's VOPD
// packing). See gqh_rs_xor for why the reduce-scatter wants the opposite default.
#if !defined(GQH_RS_PERMLANE16)
#  define GQH_RS_PERMLANE16 1
#endif
// The fifth level (off == 16, the only one that crosses a DPP row) as ONE VALU op --
// BUILT, MEASURED AND REFUTED; see gqh_xor16 for the mechanism. Default OFF.
// `-DGQH_PERMLANE16=1` selects it; it is kept because the refutation is about issue
// slots, not about correctness, and a future arm with VGPR headroom could revisit it.
#if !defined(GQH_PERMLANE16)
#  define GQH_PERMLANE16 0
#endif
// The superblock E4M3 scale byte decoded by the HARDWARE fp8 unit instead of the
// GQH_E4M3_D table (see gqh_e4m3_f32) -- BUILT, MEASURED AND REFUTED, default OFF.
// It deletes 29 SALU and 11 waits per trip and STILL LOSES 1.17% of HumanEval, because
// the four instructions it adds are VALU and the ones it deletes are not. The mechanism,
// and the instrument correction that falls out of it, are on gqh_e4m3_f32.
//
// Kept in the tree for the same reason GQH_PERMLANE16 is: the refutation is about which
// PIPE the work lands on, not about correctness -- the decode itself is bit-exact (all
// eight reference FNVs reproduce) -- so an arm that ever needs `d_real` in a VGPR anyway
// gets it here for free. `-DGQH_FP8_CVT=1` turns it back on.
//
// The two branches are NOT interchangeable spellings of one default. `v_cvt_f32_fp8` is
// gated on the `fp8-conversion-insts` target feature, gfx1100 does not have it, and this
// tree builds FAT (`GPU_TARGETS=gfx1100;gfx1201`) -- so setting the else-branch to 1 is a
// hard compile error on the gfx1100 half, not a silently slower path. An A/B driver that
// seds `define GQH_FP8_CVT [01]` hits both lines and breaks the build (it did); flip the
// marked line and only the marked line.
#if !defined(GQH_FP8_CVT)
#  if defined(__HIP_PLATFORM_AMD__) && defined(__GFX12__)
#    define GQH_FP8_CVT 0  /* <-- A/B: flip THIS line to 1 for the hardware-cvt arm */
#  else
#    define GQH_FP8_CVT 0  /* no fp8-conversion-insts on this target */
#  endif
#endif
// GQH_FP8DOT: the N=8 int8 arm's FMA block re-based on `v_dot4_f32_fp8_fp8` -- BUILT,
// MEASURED, FASTER, AND REFUSED ON NUMERICS. Default OFF. `-DGQH_FP8DOT=1` turns it on.
//
// gfx1201 has a 4-wide e4m3 dot that accumulates in **f32**, and it issues at the SAME
// rate as `v_dot4_i32_iu8`: 32.625 ms against 32.536 ms over 20 saturating launches of a
// 16-way-ILP loop (`v_fma_f32` 31.471, `v_xor_b32` 29.166 in the same harness), i.e.
// +0.3%, the same slot. f32 accumulation is the point -- `acc += s * (float) dot` becomes
// `acc += s * dot`, so all 32 `v_cvt_f32_i32` in the trip go away and NOTHING else moves:
// same pair LUT, same four `ds_load_u16` per row, same `v_lshl_or_b32` pair, same 8
// activation BYTES per lane per superblock. Trip VALU 217 -> 185 (-14.7%) on
// `<GQH4,8,4,unpaired,0,i8>` and 258 -> 226 (-12.4%) on the GQH3 pair; `mem` UNCHANGED at
// 37 / 41; VGPRs 96 -> 93 (i.e. AWAY from the 16-waves/SIMD cliff, not onto it), zero
// scratch, zero spills, LDS unchanged; the f32 exact arm byte-identical. It is the
// cleanest VALU-only lever this kernel has ever had, and it is **+1.17% of HumanEval**
// (139.185 against 137.578, order-balanced ABBA, 4/4 candidate runs above 4/4 controls).
//
// AND IT IS THE WRONG TRADE, because e4m3 is the wrong format for a value that already
// carries a group scale:
//
//   RMS deviation from the f32-exact arm, same vectors, same build pair:
//     GQH3 pair 5120->17408   int8 0.4063%   fp8 1.7259%   **4.25x**
//     GQH4 17408->5120        int8 0.3857%   fp8 2.4474%   **6.34x**
//
// The decomposition says the codebook is not the problem and cannot be made into the
// fix. Per-8-group e4m3 activations measure 1.89-1.98% RMS relative against int8's
// 0.33-0.40% -- **4.8x to 6.0x on the ACTIVATION SIDE ALONE**, independent of the grid,
// which is the whole of the 4.25x / 6.34x above. The mechanism: int8 under a per-group
// scale spends all 8 bits on the mantissa inside a range the scale already pinned; e4m3
// spends 4 of its 8 on an exponent that scale made redundant, leaving 3 mantissa bits
// against int8's effective ~7. **A shared scale and a per-element exponent are redundant,
// and e4m3 pays for the exponent out of the mantissa.** Not tunable -- it is the format.
//
// So the one free knob does not rescue it either. The level table carries a
// multiplicative constant that `s_b` absorbs, and searching it over an octave does bring
// the WEIGHT side back to int8 parity (GQH4 grid 8: e4m3 0.01096 -> 0.00261 at K=0.691
// against int8's 0.00228; GQH3 grid 11: 0.00781 -> 0.00055 at K=1.626). Grid 0 is exactly
// representable and needs no K at all. But with the weight half driven to zero the
// activation half still leaves ~4.8x, so K cannot reach the threshold. Not built.
//
// Kept in the tree for the same reason GQH_PERMLANE16 and GQH_FP8_CVT are: the refutation
// is about the FORMAT, not correctness. The arm is fully built and verified -- the f32
// output is byte-identical across both builds, and `avg_commit` is bit-identical at
// 7.4400 per prompt -- so a shape with no group scale, or a longer-mantissa 4-wide dot,
// takes it for free. What would actually unlock it is an 8-bit dot with more mantissa
// (there is none on gfx1201) or dropping the group scale (which buys nothing: e4m3's
// relative precision is scale-invariant).
//
// THE CALIBRATION IS THE MORE VALUABLE HALF, and it re-prices the whole next-list.
// -14.7% of trip VALU, with mem, registers, occupancy, LDS and AL all pinned, bought
// **+1.17% of HE: a conversion rate of 0.08x, not the 0.3x iteration 6 estimated from a
// single point.** Iteration 6's change moved `d_real` out of an SGPR and lengthened the
// per-row scale chain, so its -1.17% was a dependency effect being read as a VALU-count
// effect. Screening candidates on VALU count is now refuted from a THIRD direction, and
// the three agree: iteration 2 deleted 86 of 773 instructions for 0.15% of wall clock,
// iteration 3 deleted 29 of 336 and got 8.3% SLOWER, and this deletes 32 of 217 for
// +1.17%. **This loop is not issue-bound.** With DRAM bandwidth already refuted
// (iteration 3: a b128 stream does 638 of 640 GB/s), what is left is LATENCY -- critical
// paths and bytes-in-flight -- and that, not op counts, is where a further iteration
// should look.
//
// Two branches, not two spellings of one default: `v_dot4_f32_fp8_fp8` and
// `v_cvt_pk_fp8_f32` are gated on gfx12 target features gfx1100 does not have, and this
// tree builds FAT -- so setting the else-branch to 1 is a hard compile error on the
// gfx1100 half. Flip the marked line and only the marked line (an A/B driver that seds
// `define GQH_FP8DOT [01]` hits both and breaks the build; that happened once already
// with GQH_FP8_CVT).
#if !defined(GQH_FP8DOT)
#  if defined(__HIP_PLATFORM_AMD__) && defined(__GFX12__)
#    define GQH_FP8DOT 0  /* <-- A/B: flip THIS line to 1 for the fp8 f32-accumulate arm */
#  else
#    define GQH_FP8DOT 0  /* no fp8 dot / conversion insts on this target */
#  endif
#endif
// GQH_WIRE_COVER: how much of the trip the exact-width arm's wire prefetch is actually
// outstanding for. This is a LOAD SCHEDULE knob and nothing else -- the same loads, in the
// same issue order, folded into acc[] in the same term order, so every output is
// bit-identical (microbench FNV unchanged, and ctest covers the f32 arm).
//
// The trip issues its loads in two groups and the ORDER of the two is load-bearing:
// loadcnt retires in issue order, so the activations (consumed in the trip that issues
// them) must stay AHEAD of the wire prefetch (consumed in the NEXT trip) or the partial
// wait that drains the activations drains the prefetch with it. That constraint is
// already respected and this knob does not touch it.
//
// What sat BETWEEN the two groups is the E4M3 scale decode -- four `s_load_u8` for the
// per-row `d` byte, four `s_load_b32` into GQH_E4M3_D, and the `s_wait_kmcnt 0x0` behind
// them. Read off the gfx1201 dump of `<GQH4,8,4,unpaired,68,i8>`: the activation clause
// ends at trip instruction 45, the kmcnt drain lands at 163, and the 8-load wire clause is
// not issued until 196 of a 332-instruction trip -- so the prefetch is outstanding for the
// last 137 instructions (41%) of the trip that issues it, and the wave pays a scalar-cache
// round trip before the DRAM request for the next superblock's weights is even on the bus.
// Moving the decode below the prefetch costs nothing (it feeds the epilogue scale, not the
// dot) and hands the prefetch the whole trip.
//
// This is the family iteration 7's s5 left open: the trip does not respond to arithmetic
// (-14.7% VALU bought 0.00% of isolated kernel time) and it is not DRAM-bound (a b128
// stream does 638 of 640 GB/s), so what is left is the load schedule. The in-file
// precedent with a number is the batch-1 DEPTH sweep -- 11.93 -> 10.75 ms (-10%) for
// 1 -> 3 superblocks in flight. That is an imperfect precedent (DEPTH buys BYTES in
// flight, this buys the DUTY CYCLE of the bytes already in flight), so it supports "the
// load schedule on this arm converts", not a predicted size.
//
// One #define, no target-feature branch: this is pure scheduling, it emits no gfx12-only
// instruction, and the two-branch `#if defined(__GFX12__)` shape is what broke the
// gfx1100 half of the fat build twice (GQH_FP8_CVT, GQH_FP8DOT).
#if !defined(GQH_WIRE_COVER)
#  define GQH_WIRE_COVER 0  /* <-- A/B: flip THIS line to 1 for the early-prefetch arm */
#endif
// GQH_WIRE_DEPTH2: how many superblocks of WIRE the exact-width arm holds in flight, 1 or
// 2. The activations stay depth-1 either way -- they are consumed in the trip that issues
// them, so pipelining them is what the batch-1 arm's xpipe does and it costs
// GQH_PER_LANE VGPRs per stage twice over.
//
// This is the axis GQH_WIRE_COVER does NOT cover, and the distinction is the whole reason
// both flags exist. COVER moves WHERE the single prefetch is issued (its duty cycle);
// this moves HOW MANY superblocks are outstanding (its size). COVER measured +0.03% of
// HumanEval, so the duty cycle of a depth-1 window is refuted -- but `sbn` stayed `sb + 1`
// throughout, so bytes-in-flight per wave never changed and nothing was learned about it.
// The in-file precedent for the size axis is MEASURED, not estimated: the batch-1 arm's
// own DEPTH sweep is 11.93 -> 10.75 ms (-10%) for 1 -> 3 (see GQH_MATVEC_DEPTH), and
// iteration 3's arithmetic-deleted ablation -- which reaches 525-625 GB/s against the
// shipped 443 -- has effectively unbounded superblocks in flight, so its advantage fits
// this axis and not COVER's.
//
// The price is registers and it is why this is a flag rather than a default: a second
// stage is ROWS x (one codes dword + the d/rb/hi1 bytes) live across the whole trip, on
// an arm sitting at 89 VGPRs where 96 is the last allocation that still gives 16
// waves/SIMD (1536 / 96). 97 rounds up to a 104-register allocation and 14 waves, i.e.
// -12.5% occupancy for +100% wire depth, which is not a trade worth making. SCREEN THE
// RESOURCE TABLE BEFORE BELIEVING ANY TIMING FROM THIS FLAG.
//
// Rotation is by register copy, not by a parity index: `wire_cur[r] = wire_nxt[r]` is
// ROWS x 2 v_mov the scheduler folds into its consumers, against duplicating a 200-line
// body to make the stage index compile-time. VALU converts at 0.08x here (iteration 7),
// so the copies are free at this arm's margins. Same bytes, same superblock, same term
// order into acc[] -- bit-identical, and the microbench FNV is the gate.
#if !defined(GQH_WIRE_DEPTH2)
#  define GQH_WIRE_DEPTH2 0  /* <-- A/B: flip THIS line to 1 for the 2-deep wire arm */
#endif
#define GQH_MATVEC_WARPS   4
#define GQH_PER_LANE       (GQH_SUPERBLOCK / GQH_WARP)   // 8
// Columns handled by one pass over the weights. The weight matrix is the whole
// DRAM cost of a decode matvec, so re-reading it per column makes an N-slot server
// N times slower than it should be -- measured as 8 slots buying only 1.35x over
// single-stream. Every column is accumulated against one load instead.
// Columns one fused dispatch can fold. 16 covers DFlash2 `--draft-block-size 12`
// verify (and the default block-8) without the dequant->GEMM fallback; wider
// batches still take that path. Exact-width arms stop at GQH_MULTICOL_SPEC_MAX;
// 9..16 use the generic runtime-guarded instantiation.
#define GQH_MAX_COLS       16
// Output rows one warp owns on the batch-1 path. The kernel is memory-level-parallelism
// bound, not issue bound (measured: a 14% instruction cut bought 2.5%, an 8-instruction
// increase that kept two more requests in flight bought 4.4%), so the axis that pays is
// outstanding DRAM bytes per wave. Four rows give the wave four independent wire streams
// instead of one, and quarter the activation traffic per row -- a warp pulls 1024 B of x
// per 137 B of weights, and the prefetched xnext is now shared by all four rows.
//
// Swept on the R9700 against a same-session ROWS == 1 control (rocprofv3, ±0.3%):
// GQH4 total per N=1 forward 26.408 / 25.151 / 23.707 ms at ROWS 1 / 2 / 4. Costs VGPRs
// 33 -> 46 -> 70, all still 16 waves/SIMD with no spills, so 4 is the last step before
// the occupancy cliff. NOT uniformly better per shape: the out == 17408 (gate/up)
// dispatches, the only ones already near DRAM peak at ROWS == 1, lose 9%. Which shapes
// take this value and which fall back to one row per warp is decided by
// gqh_rows1_selected(), which carries the per-shape sweep table.
#define GQH_MATVEC_ROWS    4

// Occupancy rounds' worth of waves that a ROWS == 1 launch has to supply before
// ROWS == GQH_MATVEC_ROWS stops paying. This is a MEASURED TABLE, not a law -- read
// gqh_rows1_selected() below before touching it.
//
// 5, not 8. The threshold is `rounds * 2048` output rows, so 5 admits out == 10240 and
// out == 12288 to the deep ROWS == 1 arm and keeps out == 6144 (and the nsb == 68
// down-proj at out == 5120) on ROWS == 4. Measured per bucket with rocprofv3 on ONE
// binary, A/B'd through GGML_GQH_ROWS1_ROUNDS so both arms come from the same build,
// us per dispatch, two runs each (bucket figures repeat to ~0.5%):
//
//   out     nsb   waves@R4 rounds   ROWS==4    ROWS==1/DEPTH==3   arm
//    5120    68     1280    0.63     78.0 78.2        --          ROWS==4
//    6144    20     1536    0.75     41.9 42.1     49.1           ROWS==4
//   10240    20     2560    1.25     61.9 61.9     53.2 53.5      ROWS==1  <- moved
//   12288    20     3072    1.50     69.1 69.3     60.0 60.0      ROWS==1  <- moved
//   17408    20     4352    2.13    (102)         85.1 85.2       ROWS==1
//
// The mechanism is occupancy-round quantisation, and it only reads correctly when you
// compare `ceil(waves/2048) / (waves/2048)` between the two arms rather than trusting
// either arm's raw rate: out == 10240 and out == 12288 sit at 1.25 and 1.50 rounds on
// ROWS == 4, i.e. they pay for 2 rounds and use 1.25/1.50, while ROWS == 1 puts them at
// exactly 5 and 6 whole rounds with zero waste. out == 6144 is the case that keeps this
// a table and not a formula: ROWS == 1 gives it 3 whole rounds and it still LOSES by
// 13%, because ROWS == 4 amortises the 1024 B activation read over four rows and that
// is worth more than its 33% round waste.
//
// This supersedes iteration 8's sweep, which read R = 6/5/3/2 as "all worse" off
// `hot_ms`. R == 5 is worth 0.52 ms of a 37 ms forward -- 1.4%, under that signal's
// 1.6% noise floor. Only the per-bucket kernel trace can see it; do not re-sweep this
// constant with hot_ms.
#define GQH_ROWS1_ROUNDS   5

// Superblocks the batch-1 ROWS == 1 arm keeps in flight per wave. ROWS and DEPTH buy
// the SAME thing -- outstanding DRAM bytes per wave -- on two different axes, and the
// ROWS == 1 arm exists precisely for the shapes where the ROWS axis is closed: ROWS
// divides the wave count, and out == 17408 at ROWS == 4 falls from 8.5 occupancy
// rounds to 2.125, which quantises up to 3 and throws away 29% of the machine
// (measured: 11.93 -> 12.99 ms). DEPTH costs registers instead and leaves the wave
// count alone.
//
// Little's law says that is exactly what this arm needs, and the same model reads the
// whole sweep table below correctly, which is why it is worth trusting. This arm
// sustains 508 GB/s with 128 SIMDs x 16 waves x 137 B = 280 KB in flight, i.e. an
// effective ~551 ns of DRAM latency; the same 551 ns needs ~358 KB to reach the
// R9700's ~640 GB/s peak, and two superblocks per wave give 560 KB. The rest of the
// table: the down-proj bucket at ROWS == 4 holds 701 KB in flight and measures 613
// GB/s (96% of peak), and gate/up at ROWS == 4 holds 1.1 MB and still loses -- to
// round quantisation, not to bandwidth (74.5 us saturated x 3/2.125 rounds = 105 us,
// measured 102).
//
// Swept on the R9700 against the DEPTH == 1 control, rocprofv3, ms of <111,1,1> per
// N=1 forward (this arm's only bucket: out == 17408, in == 5120, 127 dispatches).
// Rows 1-2 are iteration 7's sweep; 3 and 4 were re-measured same-session in
// iteration 8 against a 10.897 control and reproduced iteration 7 to 0.03%:
//
//   DEPTH      ms   vs ctl   GB/s   VGPRs   occupancy
//     1    11.866      --     510      32      16      <- the one-superblock pipeline
//     2    10.896   1.089x    556      50      16
//     3    10.731   1.106x    565      76      16      <- landed
//     4       --       --      --      97      12      <- occupancy cliff, by ONE VGPR
//
// 3, and it is the last step this axis has. 4 is refuted outright and not by a little:
// 97 VGPRs is one over the 96 that 16 waves/SIMD allows, and crossing that cliff does
// not merely cost occupancy, it halves the blocks-per-MP that gqh_rows1_selected()
// multiplies its threshold by, which silently drags two more GQH4 buckets onto this arm
// with no signal in hot_ms or the sha gate (read that function's warning). At 76 VGPRs
// this leaves 20 of headroom, so ANY future change to this arm has to re-read
// .amdhsa_next_free_vgpr -- there is no longer room to be casual about it.
//
// Do not try to buy DEPTH == 4 by moving the activation prefetch after the fold to free
// the xcur copy. Measured in iteration 8: that schedule DOES fit (81 VGPRs, 16
// waves/SIMD at DEPTH == 4) but the reordering costs 3.2% on its own, and DEPTH 4 on
// top of it only got back to 10.751 -- worse than plain DEPTH == 3. Nor by decoupling
// the wire prefetch distance from the trip length (a WSTAGES knob, wire queue
// 2*DEPTH deep at +4 VGPRs): 11.356 / 11.301 / 11.694 at 2 / 3 / 4 stages against
// 10.889 at 1, i.e. worse everywhere.
//
// Diminishing returns, and they are nearly exhausted: at 565 GB/s this arm is at 88% of
// the R9700's ~640 peak, and 613 GB/s is the best any GQH4 bucket has ever reached on
// this device, so there is at most ~8% left in it from ANY amount of extra depth.
#define GQH_MATVEC_DEPTH   3

// Output rows one warp owns on the SPECIALIZED multi-column arm, and the whole reason
// that arm exists.
//
// A multi-column matvec reads the weight stream ONCE for all N columns, but the generic
// instantiation re-reads the ACTIVATIONS per (row, superblock, column): 1024 B per warp
// per column against 137 B of wire. Over a forward that is ~100 GB of x reads PER COLUMN
// (105 M superblock-rows x 1 KB), and the measured llama-bench pp ladder --
// 42.4 / 56.8 / 64.0 / 71.9 / 80.1 ms per forward at N = 1 / 2 / 3 / 4 / 5 -- puts each
// extra column at ~8 ms, i.e. ~12 TB/s effective. That is AT L0-hit bandwidth for 64 CUs,
// so those loads are not latency-starved; there are simply too many of them, and the only
// fix is to issue fewer. Every row a warp owns folds the SAME activations, so ROWS rows
// per warp cut the activation traffic by ROWS -- the same lever GQH_MATVEC_ROWS pulls on
// the batch-1 path (worth 1.3-1.6x there), which is why the hoist below and this constant
// are one change and not two: at ROWS == 1 the hoist has nothing to share.
//
// Swept on the R9700 against a SAME-BINARY control (GGML_GQH_MULTICOL=0 sends every
// ncols > 1 dispatch back to the generic instantiation), llama-bench pp3 per-forward ms,
// -r 24, one build per row. The control reproduces to 0.3% across all four builds, which
// is what makes the cross-build comparison of the candidates fair:
//
//   ROWS   pp3 ms   control   speedup   VGPRs (GQH4 / GQH3)   occupancy
//     1     50.41    59.31     1.177x        47 /  56            16
//     2     46.35    59.41     1.282x        68 /  83            16
//     3     42.76    59.43     1.390x        87 /  97         16 / 12   <- landed
//     4     43.15    59.49     1.379x       102 / 122         12 / 10
//
// **That sweep predates the wave-uniform addressing (UNIFORM_ADDR in the kernel), which
// deleted ~16 VGPRs of 64-bit address chain from every instantiation on this arm and so
// moved the cliff this table is fitted against. RE-MEASURED after it, same rig:**
//
//   ROWS   pp3 ms   control   speedup   VGPRs (GQH4 / GQH3)   occupancy
//     3     39.44    59.20     1.501x        71 /  93            16      <- landed
//     4     39.31    59.03     1.502x        85 /  93            16
//     5       --       --        --          98 / 106         12 / 12
//
// So ROWS == 4 is no longer refuted by occupancy -- it is simply a NON-EVENT (0.33%,
// inside the ~0.5% cross-build spread of this rig; ROWS == 3 itself read 39.26 and 39.44
// on two builds). The traffic it saves is 1/12 of the activation stream, and after the
// addressing fix that stream is no longer what this arm is waiting on. 3 stays because it
// is the measured point with the most headroom (25 VGPRs to the cliff, against 11) and
// because the NCOLS axis spends ~10-11 VGPRs per column out of that headroom -- which is
// what raising GQH_MULTICOL_SPEC_MAX will need. ROWS == 5 is refuted outright at 98.
//
// GQH3 was the loser of the old fit at 97 VGPRs (12 waves/SIMD, one over the cliff); it is
// now 93 and gets its 16 waves back for free. Read `.amdhsa_next_free_vgpr` (or clang
// -Rpass-analysis=kernel-resource-usage) before moving this constant either way.
#define GQH_MULTICOL_ROWS  3

// Output rows one warp owns on the ncols == 8 exact-width arm -- the DFlash2 native
// verify width, and the only width this model's spec-decode issues. Separate from
// GQH_MULTICOL_ROWS because that arm is bound on a DIFFERENT RESOURCE, and that is the
// whole point of this constant. Scoped to ncols == 8 on purpose: ncols 5..7 keep
// GQH_MULTICOL_ROWS (they are what pp5 measures, and GQH3 ncols == 5 at ROWS == 6 with
// kNsb spills 4 VGPRs to scratch -- an unmeasured regression for a width nothing here
// dispatches).
//
// **The ncols == 8 arm is INSTRUCTION-ISSUE bound, not DRAM bound and not occupancy
// bound.** Measured on the R9700, rocprofv3 kernel trace of a real HE decode: the
// paired GQH3 gate/up dispatch (in 5120, out 17408, both tensors) is 32.6% of all
// decode kernel time at 320.6 us, against 73.1 MB of wire bytes -- i.e. 228 GB/s,
// only 36% of this device's ~640 GB/s peak. Its loop body is 445 instructions per
// superblock for ROWS*NCOLS*8 == 192 useful FMAs (2.32 instructions per FMA), and
// 232160 wave-superblocks x 445 at 128 instructions/clk (2 SIMD32 per CU x 64 CUs)
// is ~330 us at ~2.5 GHz. That accounts for the whole dispatch: there is no latency
// to hide, only instructions to delete.
//
// Two independent measurements say occupancy is NOT the axis:
//   - Same-binary A/B (GGML_GQH_KNSB_WIDE): 8 effective waves/SIMD (146 VGPRs, kNsb)
//     beat 12 (98 VGPRs, runtime nsb) by 2.0% on 10-prompt HE, 2 reps each --
//     i.e. +50% resident waves made it SLOWER. Occupancy is already sufficient.
//   - The prior handoff's ROWS == 2 experiment lost 14% (98 -> 84 tok/s) while
//     RAISING occupancy, for the same reason in reverse.
//
// ROWS is the lever that cuts instructions per FMA, because the per-superblock cost
// splits into ~148 instructions that a warp pays ONCE (16 b128 activation loads for
// the 8 columns, the waits behind them, the loop's SALU) and ~108 per row (decode 8
// GQH3 weights: ~37 VALU + 9 LDS + 8 scale muls, then 64 FMAs). More rows per warp
// amortises the fixed half AND cuts the activation stream by 1/ROWS. Loop-body
// instructions per useful FMA, `clang -S --offload-arch=gfx1201`, GQH3 ncols == 8,
// no spills or scratch at any cell:
//
//   ROWS   kNsb   runtime nsb   VGPRs (kNsb / runtime)   waves/SIMD
//     3    2.318     2.458            146 /  98            8 / 12   <- was shipped
//     4    2.242     2.410            175 / 101            8 / 12
//     6    2.305     2.117            190 / 162            8 /  9
//     8    2.473     2.105            190 / 179            8 /  8
//
// 6, not 8, and the tie-breaker is occupancy-round quantisation rather than the
// instruction count (2.117 vs 2.105 is 0.6%): out == 17408 at 4 warps x 6 rows is
// 1452 blocks = 5808 waves against ~1024 resident, i.e. 5.67 rounds -> 6 (5.8% of
// the machine idle in the last round), the same waste the shipped ROWS == 3 pays.
// ROWS == 8 is 4.25 rounds -> 5, which throws away 17.6% and more than eats its
// 0.6%. GGML_GQH_N8_ROWS=3/6/8 selects between them in one binary.
#ifndef GQH_MULTICOL_ROWS_WIDE
#define GQH_MULTICOL_ROWS_WIDE 6
#endif

// Widest ncols that gets an exact-width instantiation. These kernels write EXACTLY
// NCOLS_MAX columns -- their `c >= ncols` guard is compile-time dead, which is what
// collapses the superblock body to one basic block and lets the activation read leave the
// row loop -- so the launcher MUST dispatch them at ncols == NCOLS_MAX, and anything
// wider stays on the generic runtime-guarded instantiation. Bounded by register pressure,
// not by taste: acc[ROWS][NCOLS] + xshared[NCOLS][8] both scale with NCOLS, and the arm
// has to stay at or under 96 VGPRs to keep 16 waves/SIMD (see the sweep in
// gqh_multicol_launch).
//
// Measured VGPRs / occupancy at ROWS == 3, `clang -Rpass-analysis=kernel-resource-usage`,
// gfx1201, no spills and no scratch anywhere in the table:
//
//   NCOLS      2        3        4        5        6         7
//   GQH4    61/16    72/16    83/16    94/16    95/16     99/12
//   GQH3    85/16    92/16    92/16    96/16    95/16    100/12
//   GQH2_H  60/16    71/16    82/16    93/16    97/12     99/12
//
// So the wall is at 6-7, not at 5: the NCOLS axis costs ~11 VGPRs per column up to 4 and
// then flattens as the allocator starts folding the column addressing, which is why 5
// lands at 93-96 (GQH3 sits exactly ON the 96 cliff and keeps its 16 waves) instead of
// the ~103 a linear extrapolation from NCOLS 4 predicts. Cap is 8 because that is the
// DFlash2 default verify width (GGUF `dflash.block_size`); 6 stays 16 waves on GQH4/GQH3,
// 7-8 drop to 12 waves on the measured VGPR grid (still far cheaper than dequant->GEMM).
// Staging N=8 x into 8 KB LDS dropped GQH3 to 94 VGPRs (under the cliff) but
// gfx1201 has 64 KB LDS/WGP so residency is 7 blocks: 4-warp HE 94 tok/s,
// 8-warp 101 tok/s, both below the register-xshared 104. Do not revive without
// a smaller tile. 9..12 (DFlash2 --draft-block-size 12) use ROWS==1 exact-width
// so they do not sit on that cliff; 13..GQH_MAX_COLS stay on the generic
// instantiation.
#define GQH_MULTICOL_SPEC_MAX 12

// Activation base POINTERS the exact-width arm carries; every further column is addressed
// as base 0 plus a uniform 32-bit byte offset. This is a statement about how many SGPR
// base pairs LLVM hands that load group, not a tuning knob -- the hoist in
// gqh_matvec_kernel carries the per-width dump it is read off. Re-read that dump (and the
// v_dual_fmac counts next to it) before moving this either way.
#define GQH_MULTICOL_XBASES 2

// Instruction classes gqh_sched_fence() still lets the scheduler move across:
// VALU | SALU | DS read | DS write | transcendental. Only VMEM is pinned, because
// VMEM is the prefetch -- the LDS gathers and the activation loads stay free to
// float up into the shadow of the E4M3 table read.
#define GQH_SCHED_MASK     0x0486

// The classes allowed to cross GQH_WIRE_COVER's barrier: VALU | DS | transcendental.
// GQH_SCHED_MASK minus SALU, and the SALU bit is the whole difference -- gfx12 SMEM is
// SALU-classified for this purpose, so with SALU allowed the E4M3 `s_load`s float back
// above the prefetch and the reorder measures backwards (cover 41% -> 36%). VALU must
// stay free: pinning it too (a bare `sched_barrier(0)`) traps the prefetch's OWN
// consumers -- the `v_lshl_or_b32` that packs the (d, rb) pair out of the loaded bytes --
// directly behind the clause, which then drains 16 of the 17 loads by instruction 65 of
// the trip and gives back more than the earlier issue won.
#define GQH_COVER_MASK     0x0482

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

// XOR (butterfly) partner of the same reduction tree. Interchangeable with the shift
// form for the value LANE 0 ends up with, and that is the only lane that stores: at
// level `off` the shift form gives lane l `v[l] + v[l+off]` and the xor form
// `v[l] + v[l^off]`, which agree exactly whenever bit `off` of l is 0 -- lane 0 at
// every level -- and are the same two summands commuted otherwise. IEEE addition is
// commutative, so no lane's value changes by more than the operand order, and lane 0's
// does not change at all.
static __device__ __forceinline__ float gqh_warp_shfl_xor(float v, int lane_mask) {
#if defined(__HIP_PLATFORM_AMD__)
    return __shfl_xor(v, lane_mask, GQH_WARP);
#else
    return __shfl_xor_sync(0xffffffffu, v, lane_mask, GQH_WARP);
#endif
}

// DPP for four of the reduction's five levels. Bit-identical, and it deletes a
// 160-deep serialised LDS dependency chain -- which measured as a WASH. Read both halves.
//
// HIP's __shfl_down / __shfl_xor lower to ds_bpermute_b32 -- an LDS crossbar op, tracked
// by dscnt -- and the compiler emits ONE s_wait_dscnt per shuffle, because level k+1 of a
// value's tree reads level k's result. On the ncols == 8 arm the reduction is
// ROWS * NCOLS == 32 accumulators x 5 levels, and it lands as **160 ds_bpermute_b32
// behind 164 s_wait_dscnt** inside a 1074-instruction post-loop block
// (<GQH4,8,4,unpaired,68,PAIRLUT,i8>; prologue 156, trip 336). Per wave that post-loop
// block plus the prologue is 1230 of the 7950 instructions an nsb == 20 dispatch issues,
// i.e. **15.5%** -- 13.9% on the GQH3 pair (1298 of 9318) -- and about half of it is the
// reduction.
//
// Four of the five levels never need to leave the row. ROW_XMASK (gfx10+, DPP ctrl
// 0x160|i) exchanges lane n with lane n^i inside a DPP row of 16 lanes, so off 8/4/2/1
// are pure VALU -- and GCNDPPCombine folds the v_mov_b32_dpp into the consumer, so each
// level costs exactly ONE v_add_f32_dpp: no LDS op, no dscnt, no wait. Only off == 16
// crosses the two rows of a wave32 and keeps its ds_bpermute. Measured on the i8 arm:
// post-loop block **1074 -> 711**, ds_bpermute **160 -> 32**, s_wait_dscnt **164 -> 29**,
// v_add_f32 132 -> 38 + 96 v_add_f32_dpp; the f32 exact arm goes 1282 -> 792. The trip is
// untouched (336 -> 331 on gqh4, 401 -> 402 on the gqh3 pair -- allocation churn only),
// no live instantiation moves a VGPR class, no scratch anywhere.
//
// WORTH +1.74% OF HumanEval -- AND THE MICROBENCH SAYS -0.3%. Read both, because the gap
// between them is the most useful thing in this comment.
//
// Same-session interleaved A/B, `-DGQH_DPP_REDUCE=0` control. Microbench first, median us
// at copies=8, reps=20, 2 reps per arm (control reps agree to 0.4%):
//
//   dispatch                    share   bpermute        DPP            delta
//   GQH3  5120->17408 paired    41.3%   167.43/167.24   168.34/168.03  +0.5%
//   GQH4 17408->5120           (30.3%)  106.07/106.03   105.87/105.85  -0.2%
//   GQH2_H 5120->5120          (30.3%)   35.06/ 34.80    35.22/ 35.05  +0.4%
//   GQH4  5120->10240           11.8%    76.14/ 75.89    74.75/ 75.00  -1.5%
//   GQH4  5120->6144             8.8%    57.02/ 57.22    54.28/ 55.09  -2.6%
//   GQH4  5120->12288            4.0%    87.31/ 87.32    84.65/ 84.55  -3.1%
//   f32 exact, GQH3 pair          --    285.97/285.03   283.60/284.56  -0.6%
//
// Share-weighted: **-0.3% of gqh_matvec**, i.e. +0.16% of HE predicted. Now the same
// change through the full scorer, candidate / control / candidate / control in ONE
// session, 10-prompt HumanEval tok/s: **138.11 / 135.73 / 137.95 / 135.61.** Each arm
// replicates to <= 0.12% and the arms are interleaved, so this is **+1.74%**, ten times
// the microbench's prediction and outside anything drift can produce.
//
// The mechanism, and it generalises: **the microbench runs 8 copies x N windows of ONE
// dispatch back to back, so the next copy's waves hide the previous copy's tail. The real
// decode issues 326 dependency-SEPARATED gqh_matvec dispatches per verify cycle and
// exposes every one of those tails on the critical path.** So the microbench is
// structurally blind to per-dispatch prologue/epilogue cost and will report ~zero for it.
// A/B that class of change on HE; keep the microbench for changes inside the trip, where
// it remains the right ranker.
//
// Corollary worth having before the next attempt: the A/B above cannot separate the
// reduction from the rest of the per-wave fixed block (prologue 156-192 instructions plus
// two or three __syncthreads that every wave hits at dispatch start, and 1074-1106
// after-loop instructions). All of the 1.74% is attributed to "the fixed block got 363
// instructions and 128 LDS round-trips smaller"; how much of it was the prologue is
// UNMEASURED. Do not conclude the prologue is exhausted.
//
// Correct here because EXEC is full: every early return above this point is
// wave-uniform (`row >= out` and `ncols != NCOLS_MAX` both retire whole warps), so no
// lane reads a disabled partner and the bound_ctrl:1 clang emits never fires.
template <int OFF>
static __device__ __forceinline__ float gqh_dpp_xor(float v) {
    static_assert(OFF > 0 && OFF < 16, "ROW_XMASK cannot cross a DPP row of 16");
    return __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x160 | OFF, 0xf, 0xf, false));
}

// The reduction's FIFTH level, off == 16 -- the one DPP cannot reach, because ROW_XMASK
// exchanges inside a row of 16 and this level is the exchange BETWEEN a wave32's two
// rows. It was all that still used the LDS crossbar after the DPP change above: 32
// ds_bpermute_b32 behind 29 s_wait_dscnt per wave, plus the lane*4 byte-address VGPR
// they are indexed by.
//
// v_permlanex16_b32 is that exact exchange as one VALU op (gfx10+). Each half-row picks
// its source out of the OTHER half with a 4-bit selector per lane, packed 8-per-SGPR:
// the IDENTITY selectors 0x76543210 / 0xfedcba98 give lane n <- lane n^16, which is
// __shfl_xor(v, 16, 32). Verified on gfx1201 against __shfl_xor over all 32 lanes, not
// assumed from the ISA doc -- the selector packing is easy to get backwards and a wrong
// one is a silent wrong answer, not a compile error.
//
// BIT-IDENTICAL, and for a stronger reason than the DPP change needed: this is the same
// VALUE, not merely the same value in lane 0. permlanex16 and ds_bpermute both return
// lane n^16's copy of v to lane n, so every lane's summand is unchanged at every level
// and the whole 32-accumulator network folds the same floats in the same order. All
// seven reference FNVs reproduced with it on (gqh3 pair e3367b7f56060183, gqh4 17408
// bb98ee6f19230b63, gqh2_h b309e4af74cefa43, gqh4 10240 c481449351c6c943, gqh4 6144
// f677207a391586c3, XSCALE=0 gqh3 54d716eaf6b29383, f32 arm bad6ce4ad97f7483).
//
// EXEC is full wherever this runs (see gqh_dpp_xor's last paragraph), so the disabled-
// lane behaviour of the `old` operand never comes up: bound_ctrl and fetch_inactive are
// both false and `old` is dead.
//
// REFUTED, iteration 5, and the mechanism is VOPD dual issue -- not registers, and not
// the LDS traffic it successfully deletes.
//
// It does delete the traffic: on the four instantiations the HE decode actually
// dispatches, the after-loop block goes ds_bpermute 32 -> 0 and s_wait_dscnt 16 -> 0, so
// the reduction network ends up with no LDS op and no dscnt wait anywhere. The block
// still gets BIGGER, 382 -> 402 instructions on <GQH4,8,4,unpaired,nsb0,PAIRLUT,i8>
// (5336 of 7567 traced dispatches), and the opcode diff says exactly why:
//
//   ds_bpermute_b32   32 ->  0     v_dual_add_f32   28 -> 13     s_delay_alu   3 -> 22
//   v_permlanex16_b32  0 -> 32     v_add_f32_e32     6 -> 38     v_mov_b32     15 -> 24
//   s_wait_dscnt      16 ->  0
//
// With ds_bpermute the reduction's adds sit next to each other and the compiler packs
// them as **v_dual_add_f32**, two adds per issue slot. v_permlanex16_b32 is not a
// VOPD-eligible opcode, so interleaving one per accumulator breaks that packing: 15
// dual-issue pairs unpack into 32 singles and 19 s_delay_alu appear to cover the
// permlane -> add hazard. Net per wave +0.49% (GQH3 pair, nsb 20) to +0.86% (GQH4, nsb
// 20) MORE instructions than the ds_bpermute form.
//
// Scored, order-balanced, 8 runs, same session, arms alternating which one goes first
// (the rig drifts down within a session, so a fixed order confounds the sign):
//
//   rep  slot1              slot2              delta(cand - ctl)
//    1   ctl  137.97        cand 137.44        -0.38%
//    2   ctl  137.39        cand 137.04        -0.25%
//    3   cand 137.53        ctl  137.11        +0.31%
//    4   cand 136.86        ctl  137.10        -0.18%
//
// Mean -0.13% +/- 0.15% -- indistinguishable from zero, and the SIGN TRACKS SLOT ORDER,
// which is what an order-balanced design is for. AL was 7.4400 on all eight runs.
//
// The obvious repair does not fit: batching the 32 permlanes ahead of the 32 adds would
// let the adds re-pack, but it needs 32 live floats and the hot <GQH4,8,4,...,i8> arm is
// at exactly 96 VGPRs, which IS the 16-waves/SIMD cliff. There is no headroom to buy the
// packing back. Same lesson the magic-constant refutation above records: on this arm the
// ISSUE SLOT is the resource, and a change that deletes LDS traffic can still lose if it
// perturbs what the compiler can dual-issue.
//
// Left in the tree, off, because none of the above is a correctness result: the selector
// packing below is hardware-verified and is the expensive part to rediscover.
static __device__ __forceinline__ float gqh_permlanex16(float v) {
#if defined(__HIP_PLATFORM_AMD__)
    return __int_as_float(__builtin_amdgcn_permlanex16(
        0, __float_as_int(v), 0x76543210u, 0xfedcba98u, false, true));
#else
    return gqh_warp_shfl_xor(v, GQH_WARP / 2);
#endif
}

static __device__ __forceinline__ float gqh_xor16(float v) {
#if GQH_PERMLANE16
    return gqh_permlanex16(v);
#else
    return gqh_warp_shfl_xor(v, GQH_WARP / 2);
#endif
}

// The reduction's xor exchange at one offset, picking the cheapest primitive for it:
// ROW_XMASK DPP inside a DPP row of 16, gqh_xor16 for the level that crosses the row.
template <int OFF>
static __device__ __forceinline__ float gqh_rs_xor(float v) {
    if constexpr (OFF == GQH_WARP / 2) {
        // permlanex16 rather than the ds_bpermute gqh_xor16 defaults to. What it buys,
        // MEASURED, is 16 LDS crossbar round trips and their s_wait_dscnt out of the
        // exposed dispatch tail -- with it the after-loop holds ZERO ds_bpermute and ZERO
        // s_wait_dscnt -- plus the lane * 4 byte-address VGPR the crossbar is indexed by.
        // Same exchange, bit-exact (the selector packing is hardware-verified; see
        // gqh_xor16's history).
        //
        // It is NOT the fix for this arm's register pressure, and the first draft of this
        // comment claimed it was. Both spellings measured 98 VGPRs on the 5336-dispatch
        // arm; what took it to 95 was gqh_lane_id. Read that one.
        //
        // The opposite default from GQH_PERMLANE16, deliberately: that flag is the same
        // instruction in the ALL-reduce network, where it is refuted because interleaving
        // one permlane per accumulator breaks the network's VOPD packing. The
        // reduce-scatter's level 0 is a single pass of 16 with no adds to pack against.
        return GQH_RS_PERMLANE16 ? gqh_permlanex16(v) : gqh_xor16(v);
    } else {
        return gqh_dpp_xor<OFF>(v);
    }
}

// The lane index, RECOMPUTED rather than carried.
//
// The reduce-scatter needs the lane's bit OFF at every level, plus the (row, column) its
// survivor stores to. Spelled as `lane` (i.e. `threadIdx.x % GQH_WARP`) LLVM derives all
// six from `threadIdx.x` ITSELF -- `lane & OFF` is `tid & OFF` for OFF < GQH_WARP -- and
// then keeps **v0 live across the whole superblock loop**. On the 5336-dispatch arm that
// is +1 VGPR on a trip sitting at exactly 96, which is the 16-waves/SIMD cliff.
//
// mbcnt over an all-ones mask counts the active lanes below this one, which IS the lane
// index while EXEC is full -- and EXEC is full wherever the reduction runs (see
// gqh_dpp_xor's last paragraph). It reads EXEC, not a register, so there is nothing to
// keep alive: the compiler materialises it after the loop. This is not a new trick here,
// it is the one LLVM already applies to the all-reduce network on its own -- that
// network's ds_bpermute lane address comes out of a post-loop `v_mbcnt_lo_u32_b32 v0,
// -1, 0` rather than out of a carried v0. Saying it explicitly is what stops it
// regressing when the consumers change: with it the hot arm goes 96 -> **95** VGPRs, one
// BELOW the pristine tree, and all four dispatched arms keep 16 / 16 / 12 / 16
// waves/SIMD.
static __device__ __forceinline__ int gqh_lane_id() {
#if defined(__HIP_PLATFORM_AMD__)
    static_assert(GQH_WARP == 32, "wave32: EXEC is one 32-bit mask, so mbcnt_lo suffices");
    return (int) __builtin_amdgcn_mbcnt_lo(~0u, 0u);
#else
    return (int) (threadIdx.x % GQH_WARP);
#endif
}

// ONE LEVEL of a reduce-SCATTER, and the point is what it does NOT do.
//
// The five-level butterfly the exact-width arm has always used is an ALL-REDUCE: at every
// level all NCOLS_MAX*ROWS accumulators are exchanged and added, so after five levels
// every lane holds every total -- and then exactly one lane (lane 0) stores them and the
// other 31 copies are thrown away. At ROWS == 4, NCOLS_MAX == 8 that is 5 x 32 = 160
// exchanges and 160 adds to produce 32 floats.
//
// A reduce-scatter does the same tree and keeps ONE copy: at each level a lane keeps half
// the accumulators and hands the other half to its xor partner, so the live count halves
// every level -- 16 + 8 + 4 + 2 + 1 = 31 exchanges and 31 adds total, and the survivor
// lands in a different lane for every accumulator. It only closes when the accumulator
// count equals the wave width, which is exactly the dispatched <*, ncols 8, ROWS 4> arm.
//
// BIT-EXACT, not rounding-equivalent, and the argument is worth stating because the whole
// change rides on it. The offsets are applied in the SAME order (16, 8, 4, 2, 1), so an
// accumulator's summation TREE is the one the butterfly built: at each level the survivor
// computes `mine + partner's`, where `partner's` is that partner's partial sum of the SAME
// accumulator over the SAME lane subset. The only difference from the butterfly's
// expression at that lane is that the two summands may arrive in the opposite order, and
// float addition is exactly commutative. Every microbench FNV must reproduce -- that is
// the gate, on both the i8 and the f32-exact arm at ncols == 8.
//
// The `hi` select is the price: 2 v_cndmask per surviving value (62 in total), because
// which half a lane keeps is by construction divergent. VALU converts at 0.08x on this
// arm, so that is the right side of the trade against 129 exchanges and adds -- and the
// exchange it deletes at OFF == 16 is an LDS crossbar round trip behind an s_wait_dscnt.
template <int OFF>
static __device__ __forceinline__ void gqh_rs_level(float (&v)[GQH_WARP], int lane) {
    const bool hi = (lane & OFF) != 0;
#pragma unroll
    for (int k = 0; k < OFF; ++k) {
        // The lane that keeps accumulator k holds it in `mine`; the one that gives it
        // away holds it in `other`, which is what the xor exchange fetches. Both lanes
        // of a pair therefore read the same accumulator out of the exchange.
        const float mine  = hi ? v[k + OFF] : v[k];
        const float other = hi ? v[k]       : v[k + OFF];
        v[k] = mine + gqh_rs_xor<OFF>(other);
    }
}

// One superblock's wire bytes, as the matvec pipeline carries them from the
// iteration that loads them into the one that decodes them.
struct gqh_wire {
    uint32_t codes;   // this lane's 8 packed codes
    uint8_t  d;       // superblock E4M3 scale byte (warp-uniform)
    uint8_t  rb;      // the byte holding this lane's sub-block uint4 ratio
    uint8_t  hi1;     // gqh3 high-1-bit code plane byte; 0 for the other rungs
};

// This lane's byte offsets within a superblock. Loop-invariant, and uint32_t rather
// than int on purpose: every wire read is addressed as `row base + 32-bit offset`,
// never as a 64-bit pointer add. A global load only takes its SADDR form (SGPR base
// pair + one 32-bit VGPR offset, no VALU at all) when the zero-extension of the
// offset is selected in the same basic block as the load. Fold the lane term into a
// 64-bit pointer instead and LICM hoists it into a 64-bit VGPR pair in the preheader,
// after which every load in the loop pays a v_add_co_u32/v_add_co_ci_u32 pair plus
// the s_wait_alu depctr_va_vcc hazard behind it. Pure addressing: which bytes are
// read, and in what order, is unchanged.
struct gqh_wire_offsets {
    uint32_t rb;      // 1 + (sub >> 1)              -- this lane's sub-block ratio byte
    uint32_t codes;   // 9 + lane * (4 or 2)         -- this lane's packed codes
    uint32_t hi1;     // 73 + lane                   -- gqh3 high-1-bit plane byte
};

// Every global read of one superblock, in one place, so the matvec can issue a
// whole superblock's worth of wire loads a full iteration ahead of the decode
// that consumes them. `rowbase` is the wave-uniform start of this row and `sb_off`
// the 32-bit byte offset of superblock `sb` inside it; keeping the two apart is what
// leaves the SADDR pattern intact.
template <ggml_type RUNG>
static __device__ __forceinline__ gqh_wire gqh_wire_load(
        const uint8_t * __restrict__ rowbase, uint32_t sb_off,
        const gqh_wire_offsets & off) {
    constexpr bool IS_GQH3 = RUNG == GGML_TYPE_GQH3;
    constexpr bool IS_GQH4 = RUNG == GGML_TYPE_GQH4;

    gqh_wire wire;
    wire.d  = rowbase[sb_off];
    wire.rb = rowbase[sb_off + off.rb];
    // memcpy, not a cast: 9 + lane*k is odd and superblocks are an odd stride
    // apart, so these are unaligned. memcpy lets the compiler pick byte loads
    // instead of emitting an access that faults on AMD.
    if (IS_GQH4) {
        memcpy(&wire.codes, rowbase + (sb_off + off.codes), sizeof(uint32_t));
    } else {
        uint16_t lo2;
        memcpy(&lo2, rowbase + (sb_off + off.codes), sizeof(lo2));
        wire.codes = lo2;
    }
    wire.hi1 = IS_GQH3 ? rowbase[sb_off + off.hi1] : 0;
    return wire;
}

// The eight activations this lane folds against one superblock, as two 128-bit loads.
// Addressed as `column base + 32-bit BYTE offset` for the SADDR reason in
// gqh_wire_offsets -- indexing the float* instead makes the address zext(off)*4, the
// selector will not hoist the scale out of the zero-extension, and the pair of 64-bit
// VALU adds comes back. in <= 17408, so sb*1024 + j0*4 cannot overflow 32 bits, and
// alignment is unchanged: j0 is a multiple of 8 floats, so every offset is 32-byte
// aligned and the 128-bit loads stay legal.
// `col_off` is a WAVE-UNIFORM byte offset added to that 32-bit lane offset rather than to
// the base pointer, and it exists as a parameter for exactly that reason -- a caller that
// baked it into `xc` would get a fourth, fifth, ... base pointer, which is the thing the
// xcol hoist in gqh_matvec_kernel could not make LLVM keep in SGPRs. It does not change
// the overflow bound in any way that matters: the widest x on this arm is
// GQH_MULTICOL_SPEC_MAX columns of `in` floats (8 x 17408 x 4 B = 557 KB), so
// `sb*1024 + j0*4 + col_off` still cannot reach 32 bits, and col_off is a multiple of
// `in`*4 (>= 20 KB), so the 32-byte alignment the 128-bit loads need is unchanged.
static __device__ __forceinline__ void gqh_load_x(
        const float * __restrict__ xc, int sb, int j0, float (&xs)[GQH_PER_LANE],
        uint32_t col_off = 0) {
    const uint8_t * __restrict__ xb = (const uint8_t *) xc;
    const uint32_t xo = (uint32_t) (sb * GQH_SUPERBLOCK + j0) * sizeof(float) + col_off;
    const float4 x0 = *(const float4 *) (xb + xo);
    const float4 x1 = *(const float4 *) (xb + xo + sizeof(float4));
    xs[0] = x0.x; xs[1] = x0.y; xs[2] = x0.z; xs[3] = x0.w;
    xs[4] = x1.x; xs[5] = x1.y; xs[6] = x1.z; xs[7] = x1.w;
}

// ---------------------------------------------------------------------------
// GGML_GQH_I8DOT: the activation side. Not bit-exact; see gqh_n8_dot_mode().
//
// The requant-in-the-loop form this replaces quantized ALREADY-DECODED f32 weights
// and activations inside the superblock body, and that is why it lost 39% of HE: the
// hot loop went 1163 -> 2621 instructions (128 v_dot4 buried under 128 v_rndne + 128
// v_cvt_i32_f32 + 128 v_med3_i32 + 64 v_div_scale_f32 -- `127.0f/amax` emitted the
// full IEEE division sequence -- and 280 v_mul). The int8 dot is only worth its
// 4-MACs-per-instruction if NOTHING is quantized in the loop, so:
//
//   * the WEIGHT side never materialises an f32 weight at all. Every GQH grid has amax
//     exactly 1.0 (GQH3_GRID / GQH4_GRID / GQH2H_GRID all run -1 .. +1), so the
//     codebook -> int8 scale is the compile-time constant 1/127 and the level table is
//     baked to int8 ONCE per block, in the prologue. See gqh_q8_lut_fill.
//   * the ACTIVATION side is quantized ONCE per dispatch by this pre-pass kernel,
//     not once per superblock per wave. Every block re-reads all of `x`, so in-loop
//     quantization is paid out/(warps*ROWS) times over.
//
// One scale per 8-weight group (one lane's worth), so the reconstruction granularity
// matches the wire's own 16-weight sub-block ratio. GGML_GQH_I8_XSCALE=1 (default)
// shares that scale across the ncols columns -- the group max moves very little across
// columns because per-channel activation outliers sit at the same channel index for
// every token position -- and =0 gives each column its own, at one extra v_mul per
// (row, column) in the loop. Same buffer layout either way: the shared arm writes the
// reduced max into all ncols slots and the kernel reads only column 0's.
//
// The stored scale folds BOTH /127s (the codebook's and the activation's), so the
// kernel's per-row scale chain stays d_real * ratio * qxs -- one extra v_mul per row
// per superblock, none per column.
#define GQH_Q8_XSCALE (1.0f / (127.0f * 127.0f))
#define GQH_Q4_XSCALE (1.0f / (7.0f * 7.0f))

// The fp8 arm's activation range target, and its scale. e4m3 carries no /127: the LUT byte
// IS the level, so only the activation normaliser folds into qxs. GQH_FP8_XSHIFT is a power
// of two (so amax -> 256 is exact and every |x| <= amax lands on a representable magnitude)
// chosen well inside e4m3's 448 ceiling and well above its 2^-6 smallest normal -- a value
// at 2^-14 of the group max is still normal, which is 8 binades of headroom below anything
// that contributes to a dot.
#define GQH_FP8_XSHIFT 256.0f
#define GQH_FP8_XSCALE (1.0f / GQH_FP8_XSHIFT)

// REFUTED, iteration 3 -- magic-constant int -> float on the int8 arm.
//
// Seeding the first dp4a's integer accumulator with the bit pattern of 1.5 * 2^23
// (0x4B400000) makes the result's bit pattern the float 12582912 + dot EXACTLY (floats in
// [2^23, 2^24) are spaced 1.0 apart and |dot| <= 8*127*127 = 129032 stays inside that
// binade), so __int_as_float replaces v_cvt_f32_i32 at zero instructions. Iteration 1
// deferred it on a numeric argument; iteration 3 BUILT it, with the s * 12582912 bias
// accumulated per (row, lane) in the trip and subtracted in the epilogue.
//
// It did exactly what it was designed to do and LOST. Trip instructions fell 336 -> 307
// on GQH4 and 388 -> 348 on the GQH3 pair (all 32 v_cvt gone, +4 v_add), VGPRs 93 -> 96,
// occupancy 16 waves/SIMD either way, no spills -- and the microbench (copies=8, 2 reps,
// same binary, GGML_GQH_I8_MAGIC selecting):
//
//   dispatch                     cvt (shipped)   magic   delta
//   GQH4 17408->5120                   107.6     116.6   +8.3%
//   GQH4  5120->10240                   82.3      85.3   +3.7%
//   GQH4  5120->6144                    57.0      58.6   +2.8%
//   GQH4  5120->12288                   97.5      99.4   +2.0%
//   GQH3  5120->17408 paired           171.2     170.1   -0.6%
//
// So -8.6% of the trip's instructions bought +8.3% of its TIME. The leading hypothesis is
// the accumulator operand: `dp4a(w, x, 0)` takes the inline constant 0 as src2, and
// seeding it makes all 32 first-of-pair dp4a read a third VGPR, which changes what can
// dual-issue. Whatever the mechanism, the lesson generalises past this change: on this arm
// the issue slot, not the instruction, is the resource, and an instruction cut that
// perturbs operand shape can cost more than it saves. Do not re-derive this from the
// instruction count.


// One (8-activation group, column) quantized, factored out of gqh_quant_x_kernel so the
// GLU-fused producer below emits BYTE-IDENTICAL codes and scales. That identity is the
// whole correctness argument for the fusion: the down-projection matvec reads the same
// bytes either way, so the decode stays bit-exact and the HE token stream cannot move.
// Do not reorder the amax / shfl / round chain -- it is the wire format.
template <bool XSSHARED, bool I4DOT = false>
static __device__ __forceinline__ void gqh_quant_group(
        const float (&v)[GQH_PER_LANE], int c, int g, int ngroups,
        int8_t * __restrict__ qx, float * __restrict__ qxs, int64_t qx_col_stride) {
    float amax = fabsf(v[0]);
#pragma unroll
    for (int t = 1; t < GQH_PER_LANE; ++t) {
        amax = fmaxf(amax, fabsf(v[t]));
    }
    if constexpr (XSSHARED) {
#pragma unroll
        for (int m = 1; m < 8; m <<= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, m, GQH_WARP));
        }
    }

#if GQH_FP8DOT
    // The fp8 arm: the group is normalised to GQH_FP8_XSHIFT purely so no element can
    // reach e4m3's 448 ceiling or its NaN encodings, and the hardware packer writes two
    // bytes at a time -- 4 `v_cvt_pk_fp8_f32` for the lane's 8 activations, against the
    // int8 arm's rnd / cvt / clamp / shift / or chain. Same buffer, same 8 bytes per
    // (column, group), same scale slot; only the byte format moves.
    // Spelled out rather than looped: the packer's word-select is a compile-time operand,
    // and a `#pragma unroll` index is not constant enough for the front end.
    static_assert(GQH_PER_LANE == 8, "fp8 pre-pass packs exactly 8 activations per lane");
    const float inv = amax > 0.0f ? GQH_FP8_XSHIFT / amax : 0.0f;
    int packed[2];
    packed[0] = __builtin_amdgcn_cvt_pk_fp8_f32(v[0] * inv, v[1] * inv, 0,         false);
    packed[0] = __builtin_amdgcn_cvt_pk_fp8_f32(v[2] * inv, v[3] * inv, packed[0], true);
    packed[1] = __builtin_amdgcn_cvt_pk_fp8_f32(v[4] * inv, v[5] * inv, 0,         false);
    packed[1] = __builtin_amdgcn_cvt_pk_fp8_f32(v[6] * inv, v[7] * inv, packed[1], true);
    *(int2 *) (qx + c * qx_col_stride + (int64_t) g * GQH_PER_LANE) =
        make_int2(packed[0], packed[1]);
    qxs[c * (int64_t) ngroups + g] = amax * GQH_FP8_XSCALE;
#else
    if constexpr (I4DOT) {
        const float inv = amax > 0.0f ? 7.0f / amax : 0.0f;
        uint32_t packed = 0;
#pragma unroll
        for (int t = 0; t < GQH_PER_LANE; ++t) {
            int q = __float2int_rn(v[t] * inv);
            q = q > 7 ? 7 : (q < -7 ? -7 : q);
            packed |= (uint32_t) (q & 0x0f) << (4 * t);
        }
        *(uint32_t *) (qx + c * qx_col_stride + (int64_t) g * GQH_PER_LANE) = packed;
        qxs[c * (int64_t) ngroups + g] = amax * GQH_Q4_XSCALE;
        return;
    }
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    int packed[2] = { 0, 0 };
#pragma unroll
    for (int t = 0; t < GQH_PER_LANE; ++t) {
        int q = __float2int_rn(v[t] * inv);
        q = q > 127 ? 127 : (q < -127 ? -127 : q);
        packed[t >> 2] |= (q & 0xff) << (8 * (t & 3));
    }
    *(int2 *) (qx + c * qx_col_stride + (int64_t) g * GQH_PER_LANE) =
        make_int2(packed[0], packed[1]);
    qxs[c * (int64_t) ngroups + g] = amax * GQH_Q8_XSCALE;
#endif
}

template <bool XSSHARED, bool I4DOT = false>
static __global__ void gqh_quant_x_kernel(
        const float * __restrict__ x, int ngroups, int64_t x_col_stride,
        int8_t * __restrict__ qx, float * __restrict__ qxs, int64_t qx_col_stride) {
    // One thread per (8-weight group, column); the ncols == 8 columns of a group are
    // lanes 0..7 of a wave, so the shared-scale reduction is three intra-wave xors.
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int g   = tid >> 3;
    const int c   = tid & 7;
    if (g >= ngroups) {
        return;
    }
    // Same addressing (and the same 16-byte alignment) as gqh_load_x.
    const float4 * __restrict__ xv =
        (const float4 *) (x + c * x_col_stride + (int64_t) g * GQH_PER_LANE);
    const float4 x0 = xv[0];
    const float4 x1 = xv[1];
    const float v[GQH_PER_LANE] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w };
    gqh_quant_group<XSSHARED, I4DOT>(v, c, g, ngroups, qx, qxs, qx_col_stride);
}

// GGML_GQH_FUSE_GLU: the FFN's SwiGLU and the down-projection's activation pre-pass are
// two separate dispatches over the SAME 17408 x 8 tensor -- 1210 + 1472 of the ~37 800
// dispatches a 1.2 s decode window issues (iteration 10's trace), for 0.40% + 0.45% of
// decode KERNEL time. Neither is compute: the SwiGLU reads 1.1 MB and writes 0.56 MB, the
// pre-pass reads that 0.56 MB straight back, and each is a 68-block launch. Fusing them
// keeps the glu tensor materialised (so nothing downstream changes) and emits the int8
// codes from the SAME registers, so the second read and the second launch both disappear.
//
// Bit-exact by construction, and that is the gate: `v[t]` is
// `ggml_cuda_op_silu_single(gate) * up` -- unary_gated_op_kernel<op_silu>'s expression, in
// its order -- and the codes come out of `gqh_quant_group`, the same device function
// gqh_quant_x_kernel uses. The down-projection therefore reads byte-identical activations,
// so the whole decode is unchanged and the HE token stream and `avg_commit` must not move.
template <bool XSSHARED, bool QUANT>
static __global__ void gqh_glu_quant_kernel(
        const float * __restrict__ gate, const float * __restrict__ up,
        float * __restrict__ glu, int ngroups,
        int64_t gate_col_stride, int64_t up_col_stride, int64_t glu_col_stride,
        int8_t * __restrict__ qx, float * __restrict__ qxs, int64_t qx_col_stride) {
    // Same (group, column) split as the pre-pass it replaces -- lanes 0..7 of a wave own
    // the 8 columns of one group, which is what makes the shared-scale reduction three
    // intra-wave xors and what makes the codes identical rather than merely equivalent.
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int g   = tid >> 3;
    const int c   = tid & 7;
    if (g >= ngroups) {
        return;
    }
    const int64_t off = (int64_t) g * GQH_PER_LANE;
    const float4 * __restrict__ gv = (const float4 *) (gate + c * gate_col_stride + off);
    const float4 * __restrict__ uv = (const float4 *) (up   + c * up_col_stride   + off);
    const float4 g0 = gv[0];
    const float4 g1 = gv[1];
    const float4 u0 = uv[0];
    const float4 u1 = uv[1];
    // op_silu(gate) * up, one term at a time, in unary_gated_op_kernel's order.
    const float v[GQH_PER_LANE] = {
        ggml_cuda_op_silu_single(g0.x) * u0.x,
        ggml_cuda_op_silu_single(g0.y) * u0.y,
        ggml_cuda_op_silu_single(g0.z) * u0.z,
        ggml_cuda_op_silu_single(g0.w) * u0.w,
        ggml_cuda_op_silu_single(g1.x) * u1.x,
        ggml_cuda_op_silu_single(g1.y) * u1.y,
        ggml_cuda_op_silu_single(g1.z) * u1.z,
        ggml_cuda_op_silu_single(g1.w) * u1.w,
    };
    // The glu tensor stays materialised: it is a graph node other passes may read, and
    // leaving it unwritten would make the fusion a graph-shape assumption rather than a
    // dispatch merge.
    float4 * __restrict__ ov = (float4 *) (glu + c * glu_col_stride + off);
    ov[0] = make_float4(v[0], v[1], v[2], v[3]);
    ov[1] = make_float4(v[4], v[5], v[6], v[7]);
    if constexpr (QUANT) {
        gqh_quant_group<XSSHARED>(v, c, g, ngroups, qx, qxs, qx_col_stride);
    }
}

// Scheduling fence for the software pipeline. With the ncols == 1 body collapsed to
// a single basic block, nothing structural stops LLVM sinking the prefetch across the
// back-edge onto its own use -- and it does, which un-does the pipeline entirely.
// Measured: the specialization alone handed back all of iteration 1's 1.10x (scored
// 1.0125). GQH_SCHED_MASK names the classes still allowed to cross it, so the
// prefetch stays issued a full superblock ahead of the decode that consumes it. Costs no
// instructions, and is a no-op on NVIDIA and on the multi-column instantiation, whose
// eight predicated blocks already pin the prefetch.
static __device__ __forceinline__ void gqh_sched_fence() {
#if defined(__HIP_PLATFORM_AMD__)
    __builtin_amdgcn_sched_barrier(GQH_SCHED_MASK);
#endif
}

// Asserts to the compiler that `v` is the same in every lane of the wave. `row` is
// wave-uniform by construction -- one warp owns one output row -- but LLVM's
// divergence analysis cannot see through threadIdx.x / GQH_WARP and marks the row,
// and therefore every pointer derived from it, divergent. That costs a 64-bit VALU
// address chain per global load (v_add_co_u32 + v_add_co_ci_u32, each with an
// s_wait_alu depctr_va_vcc hazard behind it) where a scalar base plus a 32-bit lane
// offset would do. readfirstlane is exact here, not an approximation: the value
// already is uniform, so lane 0's copy is every lane's copy.
static __device__ __forceinline__ int gqh_uniform(int v) {
#if defined(__HIP_PLATFORM_AMD__)
    return __builtin_amdgcn_readfirstlane(v);
#else
    return v;
#endif
}

// The superblock's E4M3 scale, times the per-tensor scale.
//
// `d` is the superblock's E4M3 byte -- one address for the whole wave -- but it is
// adjacent to the per-lane ratio byte `rb` in gqh_wire, LLVM merges the two into one
// 16-bit value, and rb's divergence infects d. Re-asserting uniformity is what turns
// the table read into an `s_load_b32`: the lookup leaves the vector memory pipe for the
// scalar one, so it is tracked by kmcnt instead of loadcnt and can no longer force a
// full `s_wait_loadcnt 0x0` that drains the prefetch behind it. It also deletes the
// 64-bit VALU address chain the divergent form needed.
template <int NCOLS_MAX>
static __device__ __forceinline__ float gqh_d_real(uint8_t d_raw, float tensor_scale) {
    const int d = NCOLS_MAX == 1 ? gqh_uniform(d_raw) : d_raw;
    return gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * tensor_scale;
}

// The same decode as ONE VALU instruction, for the arms that take it (GQH_FP8_CVT).
//
// GQH_E4M3_D is torch.float8_e4m3fn -> f32, and gfx12's `v_cvt_f32_fp8` decodes exactly
// that format (OCP e4m3, bias 7, max 448) -- NOT the e4m3fnuz that gfx94x's same-named
// instruction reads. So the table is not an approximation of the hardware unit here, it
// IS the hardware unit, and the lookup can go away entirely:
//
//   * VERIFIED EXHAUSTIVELY on gfx1201, all 256 bytes: 254 of 256 reproduce the table's
//     float32 bit pattern EXACTLY. The two that differ are 0x7f / 0xff, which the table
//     itself marks NaN (0x7ff00000 / 0xfff00000) and the hardware returns as a
//     differently-payloaded NaN. A NaN superblock scale cannot come out of the
//     quantizer, and if one ever did both forms poison the row identically.
//   * The `_e32` form reads src[7:0] and IGNORES bits 8..31 -- also verified on device,
//     over all 256 bytes with 0xABCD00 / 0xFFFFFF00 / 0x00FF00 in the high bytes. That
//     matters because the value handed in is the PIPELINE register, which carries
//     `(rb << 8) | d`: LLVM merges the wire's two adjacent bytes to save a VGPR across
//     the back edge. The byte select consumes that merge for free -- no mask, no shift.
//
// What it deletes, per row per superblock, is the whole scalar address chain the table
// needed: `s_getpc_b64` + `s_sext_i32_i16` + two `s_add_co`, then per row two
// `s_lshl_b32` + two `s_and_b32` + `s_add_nc_u64` + `s_load_b32` to reach
// `GQH_E4M3_D[d >> 3][d & 7]`, plus the `s_wait_alu depctr_sa_sdst` hazard behind nearly
// every one of those and the `s_wait_kmcnt` on the load.
//
// `d` stays wave-uniform through gqh_uniform on the caller's side: VOP1 takes an SGPR
// src0, so `v_cvt_f32_fp8_e32 v, s22` keeps the wire's `d` byte on the SCALAR load path
// (kmcnt, not loadcnt) -- the property the gqh_d_real comment above is about.
//
// AND IT LOSES ANYWAY. Order-balanced 10-prompt HE, rebuilt between arms, control first
// in reps 1-2 and candidate first in reps 3-4 (so slot position cannot carry the sign):
//
//   rep 1  ctl 138.00  cand 136.40   -1.16%
//   rep 2  ctl 137.42  cand 135.40   -1.47%
//   rep 3  cand 135.77  ctl 137.21   -1.05%
//   rep 4  cand 135.68  ctl 137.05   -1.00%     mean -1.17%, 4/4 negative
//
// The ISA says why, and it is the useful part. On `<GQH4,8,4,unpaired,0,i8>` the trip
// goes 328 -> 296 instructions (-9.8%) but its VALU ops go 217 -> 225 (+3.7%); on the
// GQH3 pair, 387 -> 362 (-6.5%) with VALU 258 -> 270 (+4.7%). Everything deleted is
// SALU (-29, -28) and waits (-11, -5); everything added is VALU -- the `v_cvt` itself,
// and the `t_scale` multiply, which the table arm got for FREE as an `s_mul_f32` on an
// SGPR-resident `d_real` that then rode into `v_mul_f32 v, s26, v` as a scalar operand.
//
// So the table lookup was never costing this kernel anything: it executed entirely on
// the scalar pipe, which co-issues with VALU and is idle here. Share-weighted the change
// is about +4.0% VALU for -8% instructions, and HE moved -1.17% -- i.e. on this arm
// **VALU converts at roughly 0.3x and SALU at roughly 0x**. Iteration 5's "1% fewer
// per-wave instructions is worth 0.7-1.0% of HE" is therefore a VALU-and-memory rule
// wearing a total-instruction-count costume; it was calibrated on a change (DPP) that
// removed LDS round-trips and waits without adding VALU. Screen candidates on VALU ops
// and memory ops, and treat SALU as free.
//
// Corollary worth keeping: moving wave-uniform work ONTO the scalar pipe is a live
// optimization direction on this arm, and moving anything off it is not.
static __device__ __forceinline__ float gqh_e4m3_f32(int d) {
#if GQH_FP8_CVT
    return __builtin_amdgcn_cvt_f32_fp8(d, 0);
#else
    return gqh_bits(GQH_E4M3_D[(d >> 3) & 31][d & 7]);
#endif
}

// One superblock's contribution to one output row, for every column.
//
// This is a SECOND copy of the fold that gqh_matvec_kernel's generic loop spells out
// inline, and the duplication is deliberate. The deep-pipeline arm needs the fold in
// two places (its main trip and its leftover-superblock tail), and calling this from
// the generic loop as well was tried and rejected: it changed the codegen of all six
// DEPTH == 1 instantiations (e.g. <108,1,4> 657 -> 646 instructions, register
// allocation shuffled throughout), and four of those six are on paths NOTHING in the
// scoring rig executes -- so the drift would have been unmeasurable in either
// direction. Keeping the generic loop textually untouched is what makes its ISA
// diff-to-identical, which is the only instrument those paths have. If you change the
// term order here, change it in gqh_matvec_kernel too.
//
// same order as the reference: d_real = e4m3(d) * tensor_scale, then
// s_b = d_real * (ratio/15), then the levels. Do not reassociate.
template <ggml_type RUNG, int NCOLS_MAX>
static __device__ __forceinline__ void gqh_fold_superblock(
        float d_real, uint32_t codes, uint8_t rb, uint8_t hi1, int sub,
        const float * __restrict__ s_grid, const float * __restrict__ s_ratio,
        const float (&xcur)[GQH_PER_LANE], const float * __restrict__ x, int sb, int j0,
        int ncols, int64_t x_col_stride, float (&acc)[NCOLS_MAX]) {
    constexpr bool IS_GQH3 = RUNG == GGML_TYPE_GQH3;
    constexpr bool IS_GQH4 = RUNG == GGML_TYPE_GQH4;

    const float s_b = d_real * s_ratio[(sub & 1) ? (rb >> 4) : (rb & 0x0f)];

    // Decode this lane's 8 weights ONCE, then reuse them for every column.
    float w[GQH_PER_LANE];
#pragma unroll
    for (int t = 0; t < GQH_PER_LANE; ++t) {
        // gqh4: little-endian, byte t>>1, low nibble for even t -> bit 4*t
        // either way. gqh3: the low-2-bit plane carries bits [1:0] of the code
        // and the high plane bit 2. gqh2_h: the 2-bit code is the whole index.
        const int code = IS_GQH4
            ? ((codes >> (4 * t)) & 0x0f)
            : (((codes >> (2 * t)) & 0x03) |
               (IS_GQH3 ? (((hi1 >> t) & 1) << 2) : 0));
        w[t] = s_grid[code] * s_b;
    }

    // Fully unrolled so acc[]/xs[] stay in registers; the per-column term order is
    // unchanged, so each output is bit-identical to the one-superblock-per-trip loop.
#pragma unroll
    for (int c = 0; c < NCOLS_MAX; ++c) {
        if (NCOLS_MAX > 1 && c >= ncols) {
            continue;
        }
        float xs[GQH_PER_LANE];
        if (NCOLS_MAX == 1) {
#pragma unroll
            for (int t = 0; t < GQH_PER_LANE; ++t) {
                xs[t] = xcur[t];
            }
        } else {
            gqh_load_x(x + (int64_t) c * x_col_stride, sb, j0, xs);
        }
#pragma unroll
        for (int t = 0; t < GQH_PER_LANE; ++t) {
            acc[c] += w[t] * xs[t];
        }
    }
}

// PAIR-DECODED LEVEL TABLE. Two weights per table entry, one LDS read, one index.
//
// This is the weight-decode half of the exact-width loop, and on the ncols == 8 arm it
// is BIGGER than the FMA half: `clang --cuda-device-only -S`, <GQH3,8,6,paired,runtime>,
// 813 instructions per superblock for 384 useful FMAs, of which the FMAs occupy ~313
// (195 v_fmac + 118 v_dual_*) and the decode occupies 342 -- 234 integer VALU, 54
// ds_load_b32, 54 v_mul_f32. Per weight that is 4.9 VALU to turn (lo2, hi1) into a
// 3-bit code and an LDS byte address, for ONE float.
//
// Two weights' codes are ADJACENT BIT FIELDS in both planes, so one index addresses
// both: weights 2p and 2p+1 own `codes` bits [4p, 4p+3] (a nibble) and `hi1` bits
// [2p, 2p+1] (a bit pair). Concatenating them gives a 6-bit index (gqh3), 8-bit (gqh4,
// no hi plane and 4-bit codes) or 4-bit (gqh2_h), and a table of float2 turns the pair
// into ONE ds_load_b64. Per pair: 2 shifts + 2 masks + one 64-bit LDS read + 2 muls,
// against 2 x (2 shifts + 2 masks + a byte address + a ds_load_b32 + a mul).
//
// BIT-EXACT, not rounding-equivalent: entry[idx] holds the SAME float32s the per-weight
// gather would have loaded (see gqh_pair_lut_fill -- code0/code1 invert the packing), and
// `w[t] = level * s_b` keeps its form and its order. Nothing is reassociated.
//
// This is NOT the closed 8 KB LDS-x family (staging x[8][8] through LDS, which cost 6%).
// That added an LDS WRITE, an LDS READ and a BARRIER to the superblock loop. This adds
// nothing to the loop -- the table is read-only, filled once in the prologue behind the
// __syncthreads that was already there, and REPLACES 48 in-loop ds_load_b32 with 24
// ds_load_b64. Loop LDS instructions go down, not up.
//
// The one thing the instruction count cannot see -- and it turned out to be the whole
// story on gqh3: an 8-byte stride gives up the conflict-free broadcast the
// RUNG_LEVELS-float table had. RESOLVED, not open: the table is now stored as two
// stride-1 planes rather than one plane of float2, which restores conflict-free banking
// at the same LDS instruction count. See gqh_pair_lut_lds and the microbench table in
// gqh_pairlut_mode().
template <ggml_type RUNG>
struct gqh_pair_lut {
    static constexpr bool IS_GQH3 = RUNG == GGML_TYPE_GQH3;
    static constexpr bool IS_GQH4 = RUNG == GGML_TYPE_GQH4;
    // gqh4: two 4-bit codes. gqh3: two 2-bit codes + two high bits. gqh2_h: two 2-bit
    // codes and no high plane.
    static constexpr int BITS = IS_GQH4 ? 8 : (IS_GQH3 ? 6 : 4);
    static constexpr int SIZE = 1 << BITS;

    // The pair index this lane's weight pair `p` lands on. The shifts are the SAME
    // fields the per-weight decode extracts, read two at a time.
    static __device__ __forceinline__ int index(uint32_t codes, uint8_t hi1, int p) {
        return IS_GQH4
            ? (int) ((codes >> (8 * p)) & 0xff)
            : (int) (((codes >> (4 * p)) & 0x0f) |
                     (IS_GQH3 ? (uint32_t) (((hi1 >> (2 * p)) & 3) << 4) : 0u));
    }
    // Inverses of index(): the code of the even / odd weight of the pair.
    static constexpr int code0(int idx) {
        return IS_GQH4 ? (idx & 0x0f)
                       : ((idx & 3) | (IS_GQH3 ? ((idx >> 4) & 1) << 2 : 0));
    }
    static constexpr int code1(int idx) {
        return IS_GQH4 ? ((idx >> 4) & 0x0f)
                       : (((idx >> 2) & 3) | (IS_GQH3 ? ((idx >> 5) & 1) << 2 : 0));
    }
};

// The table's LDS storage, in a function so the allocation only exists for the
// instantiations that ask for it. A kernel that never calls this has no reference to the
// global, so its LDS Size (and every other line of its resource dump) is unchanged --
// the ~50 instantiations nothing in the scoring rig executes are this file's only
// instrument, and they stay byte-identical.
//
// TWO PLANES OF float, NOT ONE PLANE OF float2, and that is the whole change. A float2
// table puts the pair's two levels in ADJACENT dwords, so `s_pair[idx]` is a
// `ds_load_b64` at a 2-dword stride: the LDS crossbar serves it as two 32-bank phases,
// and in each phase every lane's address is EVEN, so only 16 of the 32 banks are
// reachable and entries `i` and `i+16` collide. Splitting the pair into plane 0
// (`lut[idx]`, the even weight) and plane 1 (`lut[SIZE + idx]`, the odd one) makes both
// reads stride-1 over the full 32 banks, and LLVM's load/store optimizer merges the two
// into one `ds_read2_b32` whose two immediate offsets differ by SIZE dwords -- so the
// LDS instruction count is unchanged and only the bank mapping moves.
template <ggml_type RUNG>
static __device__ __forceinline__ float * gqh_pair_lut_lds() {
    __shared__ float lut[2 * gqh_pair_lut<RUNG>::SIZE];
    return lut;
}

// Fill it from the tensor's own level table, reading the LDS copy rather than the kernarg
// `grid`: a DYNAMIC index into a by-value kernarg struct puts that struct on the stack,
// and the PAIRED arm carries two of them -- measured 132 bytes/lane of scratch, which is
// why this takes the (prologue-only) second barrier instead. 64 (gqh3) / 256 (gqh4)
// entries against a block that then folds nsb x ROWS superblocks: under 0.5% of it.
//
// Plane 0 holds the even weight of every pair and plane 1 the odd one, so the two floats
// a pair index selects are SIZE dwords apart. Same floats as the float2 layout and the
// same floats the per-weight gather loaded -- only their LDS addresses change.
template <ggml_type RUNG>
static __device__ __forceinline__ void gqh_pair_lut_fill(
        float * lut, const float * __restrict__ g_levels) {
    constexpr int SIZE = gqh_pair_lut<RUNG>::SIZE;
    for (int i = threadIdx.x; i < SIZE; i += blockDim.x) {
        lut[i]        = g_levels[gqh_pair_lut<RUNG>::code0(i)];
        lut[SIZE + i] = g_levels[gqh_pair_lut<RUNG>::code1(i)];
    }
}

// The codebook -> int8 scale is 1/127 exactly, not a measured amax: every rung's grid
// spans -1 .. +1, so 127 lands on the extreme level with no clamping and the levels in
// between round to within half a step (worst case 0.4% of amax, and the residual is a
// fixed perturbation of the codebook -- the same 8 or 16 numbers for every weight in
// the tensor -- not per-weight noise).
//
// Both fills below take the LDS level copy (`s_grid`), NOT the kernarg `grid`, and that is
// a measured constraint rather than a style choice: a DYNAMIC index into a by-value kernarg
// struct puts the whole struct on the stack, and the PAIRED arm carries two of them --
// 132 bytes/lane of scratch when it was tried. That is why the fills pay a prologue-only
// second __syncthreads instead of reading `grid` directly.
static __device__ __forceinline__ int gqh_q8_level(float level) {
    const int q = __float2int_rn(level * 127.0f);
    return q > 127 ? 127 : (q < -127 ? -127 : q);
}

static __device__ __forceinline__ int gqh_q4_level(float level) {
    const int q = __float2int_rn(level * 7.0f);
    return q > 7 ? 7 : (q < -7 ? -7 : q);
}

// The fp8 arm's codebook byte. Every rung's grid spans -1 .. +1 so there is no clamping to
// do and no scale to carry: e4m3 has 8 binades below 1.0, the level lands on its own
// exponent, and the byte the hardware dot reads IS the level to 3 mantissa bits. Rounding
// is the unit's own round-to-nearest-even, which is what makes the LUT and the dot agree by
// construction rather than by a table that has to be kept in step (the trap gqh_e4m3_f32
// documents on the scale side).
static __device__ __forceinline__ int gqh_fp8_level(float level) {
#if GQH_FP8DOT
    return __builtin_amdgcn_cvt_pk_fp8_f32(level, 0.0f, 0, false) & 0xff;
#else
    (void) level;
    return 0;
#endif
}

// One LUT byte, whichever arm is compiled. Both fills go through this so the weight side
// cannot drift out of step with the dot opcode.
static __device__ __forceinline__ int gqh_w8_level(float level) {
#if GQH_FP8DOT
    return gqh_fp8_level(level);
#else
    return gqh_q8_level(level);
#endif
}

// The int8 twin of the pair table, for GGML_GQH_I8DOT. ONE plane of uint16, not two of
// float: a pair index selects TWO int8 weights and dp4a wants them side by side in one
// dword, so `wq = lut[i0] | (lut[i1] << 16)` is one ds_read_u16 per pair plus one
// v_lshl_or_b32 per dword -- 4 LDS + 2 VALU for the lane's 8 weights, against the f32
// table's 4 LDS + 8 v_mul (the `level * s_b` muls are gone entirely; s_b rides the
// accumulator instead). Byte 0 is the EVEN weight, so the dword's byte order matches the
// four consecutive activation bytes a b64 load brings in, little-endian either side.
//
// The layout used to carry a second claim -- that 64 uint16 entries are 32 dwords and so
// the gqh3 table is "conflict-free by construction". That reasoning was wrong (the index
// is a weight CODE, i.e. data, so lanes land on banks at random however the table is laid
// out) and it does not matter either way, per the refutation below.
//
// REFUTED, and left in place because of it: a register-resident codebook gathered with
// v_perm_b32 instead, which deletes ALL FOUR of these LDS reads per row. gqh3 (one perm
// per dword, 8 levels in 2 VGPRs) measured 194.31 us against 193.88 -- dead flat -- and
// gqh4 (16 levels need THREE perms: two octet gathers plus a per-byte merge under code
// bit 3) measured 122.32 against 107.46, i.e. -14%. So the divergent LDS gather is NOT
// what this loop is bound on, and the f32 arm's GGML_GQH_PAIRLUT 8-vs-4 A/B (1.13x on
// gqh3, 1.16x on gqh4) does not price LDS -- it prices the four extra index computations
// and eight extra v_mul that ride along with it. The perm decode is bit-identical
// (verified exhaustively on gfx1201 over all 65536 gqh3 `codes` x 7 `hi1` planes and all
// 65536 gqh4 low halves x 16 high halves) and the microbench FNV was unchanged; it simply
// buys nothing. Do not rebuild it. Two side facts worth keeping: v_perm_b32 is full rate
// on gfx1201 (1.99 cycles/instr against v_xor_b32's 1.84 on a 16-way-ILP loop) and its
// selector values 0..7 are byte selects while 8..12 give 0x00 and 13..15 give 0xff.
template <ggml_type RUNG>
static __device__ __forceinline__ uint16_t * gqh_q8_lut_lds() {
    __shared__ uint16_t lut[gqh_pair_lut<RUNG>::SIZE];
    return lut;
}

template <ggml_type RUNG>
static __device__ __forceinline__ void gqh_q8_lut_fill(
        uint16_t * lut, const float * __restrict__ g_levels, bool q4 = false) {
    constexpr int SIZE = gqh_pair_lut<RUNG>::SIZE;
    for (int i = threadIdx.x; i < SIZE; i += blockDim.x) {
        const int q0 = q4 ? gqh_q4_level(g_levels[gqh_pair_lut<RUNG>::code0(i)])
                          : gqh_w8_level(g_levels[gqh_pair_lut<RUNG>::code0(i)]);
        const int q1 = q4 ? gqh_q4_level(g_levels[gqh_pair_lut<RUNG>::code1(i)])
                          : gqh_w8_level(g_levels[gqh_pair_lut<RUNG>::code1(i)]);
        lut[i] = q4 ? (uint16_t) ((q0 & 0x0f) | ((q1 & 0x0f) << 4))
                    : (uint16_t) ((q0 & 0xff) | ((q1 & 0xff) << 8));
    }
}

// The E4M3 scale decode for one row of the exact-width trip, factored out of the loop
// body only so GQH_WIRE_COVER can move it BELOW the wire prefetch without duplicating the
// expression. Same table, same `gqh_uniform`, same two multiplies in the same order, so
// `d_real` is the identical float either side of the move -- see GQH_WIRE_COVER.
template <bool UNIFORM_ADDR, bool I8DOT>
static __device__ __forceinline__ float gqh_d_real_uni(uint8_t d_raw, float t_scale) {
    const int d = UNIFORM_ADDR ? gqh_uniform(d_raw) : d_raw;
    // Bit-identical to the table lookup on every scale byte a quantizer can emit (see
    // gqh_e4m3_f32), and the same two multiplies in the same order after it -- only the
    // way the E4M3 byte becomes a float moves. Confined to the int8 arm on purpose: the
    // f32 and batch-1 instantiations are this file's only instrument on the ~270 paths
    // the scoring rig never executes, so they stay byte-for-byte what they were.
    return (I8DOT && GQH_FP8_CVT)
        ? gqh_e4m3_f32(d) * t_scale
        : gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * t_scale;
}

// RUNG selects the wire layout and the level decode at compile time. gqh2_c has its
// own kernel below (different block geometry), so RUNG is one of GQH3/GQH2_H/GQH4.
// `grid` carries the rung's whole signed level table by value; the kernel reads its
// first 16 / 8 / 4 entries and ignores the rest.
//
// NCOLS_MAX is the compile-time column bound. With the runtime `ncols` alone, the
// eight `c >= ncols` guards are opaque to the compiler and it emits eight predicated
// basic blocks -- so a batch-1 decode still walks ~20 dead scalar compare/branch
// instructions per superblock, and, worse, the machine scheduler cannot move loads
// across those block boundaries. Instantiating NCOLS_MAX == 1 makes the guard
// compile-time false and collapses the superblock body to a single basic block; the
// generic NCOLS_MAX == GQH_MAX_COLS instantiation keeps the runtime guard and is
// byte-for-byte the kernel this replaces. Only the codegen changes: the values and
// the order they fold into acc[] are untouched, so every output stays bit-identical.
//
// ROWS is how many output rows one warp owns. Each row keeps its own rowbase, its own
// wire stream and its own acc, folded in the same term order as the one-row kernel, so
// every output is still bit-identical; only `x` is shared between them. ROWS > 1 is the
// batch-1 path -- the generic multi-column instantiation stays at ROWS == 1 so its
// codegen does not move.
template <ggml_type RUNG, int NCOLS_MAX, int ROWS, bool PAIRED, int NSB = 0,
          bool PAIRLUT = false, bool F16DOT = false, bool I8DOT = false,
          bool XSSHARED = true, bool I4DOT = false>
static __global__ void gqh_matvec_kernel(
        const uint8_t * __restrict__ data, const float * __restrict__ x,
        float * __restrict__ y, int in, int out, int ncols, float tensor_scale,
        gqh_grid16 grid, int64_t x_col_stride, int64_t y_col_stride,
        const uint8_t * __restrict__ data_b, float * __restrict__ y_b,
        float tensor_scale_b, gqh_grid16 grid_b,
        const int8_t * __restrict__ qx = nullptr,
        const float * __restrict__ qxs = nullptr) {
    // PAIRED folds two same-shaped weight tensors that share `x` into ONE dispatch,
    // selected by blockIdx.y. See ggml_cuda_gqh_mul_mat_vec_pair(). The four extra
    // kernargs are appended, so every pre-existing kernarg keeps its offset, and the
    // if constexpr below vanishes for PAIRED == false -- the seven unpaired
    // instantiations keep a byte-identical opcode stream (verified by ISA diff).
    //
    // A block still folds exactly the terms it folded before, in the same order, for
    // the same output row of whichever tensor it belongs to. Bit-identical by
    // construction: the only thing that changes is which dispatch carries the block.
    const uint8_t * __restrict__ w_base = data;
    float * __restrict__ y_base = y;
    float t_scale = tensor_scale;
    // Each half carries its OWN level table: 26% of this model's gate/up pairs are
    // quantized against different GQH4 grids (measured: 53 of 200), and requiring a
    // shared table would silently leave those pairs unfused.
    const float * g_levels = grid.v;
    if constexpr (PAIRED) {
        if (blockIdx.y != 0) {
            w_base   = data_b;
            y_base   = y_b;
            t_scale  = tensor_scale_b;
            g_levels = grid_b.v;
        }
    }
    constexpr bool IS_GQH3 = RUNG == GGML_TYPE_GQH3;
    constexpr bool IS_GQH4 = RUNG == GGML_TYPE_GQH4;
    // Exact-width multi-column instantiation: ncols == NCOLS_MAX by the launcher's
    // construction, so the column guards below are compile-time dead and the activation
    // read can be hoisted out of the row loop and shared by all ROWS rows. The generic
    // NCOLS_MAX == GQH_MAX_COLS instantiation keeps its runtime guard and its per-column
    // load, so its codegen does not move (ISA-diffed).
    constexpr bool XSHARED = NCOLS_MAX > 1 && NCOLS_MAX < GQH_MAX_COLS;
    // Arms that hand LLVM the uniformity it cannot prove. `row` is wave-uniform by
    // construction (one warp owns ROWS consecutive rows) and so is a superblock's E4M3
    // byte, but divergence analysis cannot see through threadIdx.x / GQH_WARP -- and a
    // pointer it thinks is divergent costs a 64-bit VALU address chain
    // (v_add_co_u32 + v_add_co_ci_u32, each with an s_wait_alu depctr hazard behind it)
    // on EVERY global load in the loop, where an SGPR base plus a 32-bit lane offset
    // would do. The generic NCOLS_MAX == GQH_MAX_COLS instantiation is deliberately
    // excluded: it is the same-binary A/B control (GGML_GQH_MULTICOL=0), so its codegen
    // has to stay byte-for-byte what it was.
    constexpr bool UNIFORM_ADDR = NCOLS_MAX == 1 || XSHARED;
    const int sb_bytes = IS_GQH3 ? GQH3_SB_BYTES : (IS_GQH4 ? GQH4_SB_BYTES : GQH2H_SB_BYTES);
    const int warps_per_block = blockDim.x / GQH_WARP;
    const int row_raw = (blockIdx.x * warps_per_block + (threadIdx.x / GQH_WARP)) * ROWS;
    const int row  = UNIFORM_ADDR ? gqh_uniform(row_raw) : row_raw;
    const int lane = threadIdx.x % GQH_WARP;

    // ratio/15 in LDS, not constant memory: the index is the lane's sub-block, so
    // it is divergent, and a divergent constant-bank load serialises per address.
    // 16 consecutive floats sit in 16 distinct LDS banks, so this is conflict-free.
    // Hoisted above the early return -- __syncthreads needs the whole block, and
    // `row >= out` retires whole warps in the tail block.
    __shared__ float s_ratio[16];
    // The signed level grid rides in LDS for EVERY rung, for the same reason: `code` is
    // divergent, and RUNG_LEVELS consecutive floats land in that many distinct banks with
    // one address each, so the gather broadcasts conflict-free.
    //
    // gqh3/gqh2_h used to decode from a register select tree instead, on the strength of
    // an NVIDIA SASS reading where the table lived in CONSTANT memory and 8 distinct
    // addresses per warp serialised into 8 constant-bank replays. LDS does not replay --
    // that is the whole reason gqh4's 16 levels are staged here -- and the tree was far
    // from free: ~5 v_cndmask per weight put gqh3 at 126 VALU per row-superblock against
    // gqh4's 40, which made the 248320-row output head VALU-ISSUE BOUND (626 M VALU
    // instructions in 1.65 ms = 128/clk x 2.9 GHz, i.e. saturated) at only 309 GB/s.
    // Do not reinstate the tree without re-measuring that dispatch.
    //
    // Bit-exact, not rounding-equivalent: GQH3_GRID / GQH2H_GRID are antisymmetric to the
    // bit (G[i] == -G[N-1-i], sign bit only), so table[code] is the same float32 the tree
    // selected out of the positive half.
    constexpr int RUNG_LEVELS = IS_GQH4 ? 16 : (IS_GQH3 ? 8 : 4);
    __shared__ float s_grid[RUNG_LEVELS];
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
        if (threadIdx.x < RUNG_LEVELS) {
            s_grid[threadIdx.x] = g_levels[threadIdx.x];
        }
    }
    __syncthreads();
    // Pair table for the exact-width arm; see gqh_pair_lut. It reads s_grid, so it needs
    // a barrier on each side -- both in the prologue, and both compiled away for
    // PAIRLUT == false, so no other instantiation moves. Ahead of the row clamp for the
    // same reason the first barrier is: __syncthreads needs the whole block.
    float * s_pair = nullptr;
    if constexpr (PAIRLUT && !I8DOT && !I4DOT) {
        s_pair = gqh_pair_lut_lds<RUNG>();
        gqh_pair_lut_fill<RUNG>(s_pair, s_grid);
        __syncthreads();
    }
    // The int8 arm decodes out of its OWN table and never touches s_pair or s_grid in the
    // loop, so it takes the pair table unconditionally (GGML_GQH_PAIRLUT does not gate
    // it) and the f32 one is not even allocated -- 128 B of LDS on gqh3 against 608 B.
    uint16_t * s_q8 = nullptr;
    if constexpr (I8DOT || I4DOT) {
        s_q8 = gqh_q8_lut_lds<RUNG>();
        gqh_q8_lut_fill<RUNG>(s_q8, s_grid, I4DOT);
        __syncthreads();
    }
    if (row >= out) return;
    // The exact-width instantiations write EVERY column they carry -- their `c >= ncols`
    // guard is compile-time dead -- so a caller that reached one of them with a narrower
    // ncols would store PAST THE END of `y`, which is the one way this arm can do worse
    // than run slowly. gqh_matvec_launch's switch guarantees the match; this makes the
    // contract self-enforcing instead of conventional. Compile-time dead on the generic
    // and batch-1 arms (so their codegen does not move), one wave-uniform scalar compare
    // outside the superblock loop on the others.
    if (XSHARED && ncols != NCOLS_MAX) return;

    // SuperSonic kNsb: compile-time superblock count unrolls the sb loop.
    // NSB == 0 is the pre-existing runtime `in/256` (ISA-identical).
    const int nsb = NSB > 0 ? NSB : (in / GQH_SUPERBLOCK);
    // Clamped, not branched: a tail warp whose second row runs past `out` re-reads the
    // last real row (an L2 hit) and simply does not store it. Branching would split the
    // superblock body across basic blocks, and the machine scheduler only clusters loads
    // within one block -- which is the whole basis of the software pipeline below.
    const uint8_t * __restrict__ rowbase[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int rr = r == 0 ? row : (row + r < out ? row + r : out - 1);
        rowbase[r] = w_base + (int64_t) rr * nsb * sb_bytes;
    }

    const int j0  = lane * GQH_PER_LANE;   // this lane's first weight in the superblock
    const int sub = j0 >> 4;               // two lanes share a 16-weight sub-block
    // Which nibble of `rb` this lane's sub-block ratio sits in, as a SHIFT rather than a
    // select. `(rb >> ((sub & 1) << 2)) & 0x0f` is the same index as
    // `(sub & 1) ? (rb >> 4) : (rb & 0x0f)` for every uint8_t rb -- the high branch's
    // mask is a no-op and the low branch's shift is zero -- but `sub` is loop-invariant,
    // so the shift amount leaves the superblock loop while the select cannot: LLVM keeps
    // both nibbles live and picks between them per row per superblock (v_lshrrev_b16 +
    // two v_and_b16 + v_cndmask_b16, in 16-bit halves so it can pack two rows per
    // register). Same nibble, same float, ~2 VALU per row per superblock cheaper.
    const int rb_shift = (sub & 1) << 2;
    const gqh_wire_offsets woff = {
        (uint32_t) (1 + (sub >> 1)),
        (uint32_t) (9 + lane * (IS_GQH4 ? 4 : 2)),
        (uint32_t) (73 + lane),
    };

    float acc[ROWS][NCOLS_MAX] = {};

    // Two arms, and the split is the whole reason this kernel has a ROWS parameter at
    // all. Both buy the same thing -- outstanding DRAM bytes per wave -- but ROWS pays
    // for it by dividing the wave count, so on a shape with no wave supply to spare it
    // is a losing trade (see gqh_rows1_selected). The batch-1 one-row-per-warp arm is
    // exactly that case, so it goes deeper in the superblock dimension instead, where
    // the price is registers. Everything else keeps the one-superblock-deep pipeline.
    if constexpr (NCOLS_MAX == 1 && ROWS == 1) {
        // Software pipeline, GQH_MATVEC_DEPTH superblocks deep. One superblock deep, a
        // wave holds one wire request in flight and the loop is not unrolled, so there
        // is nothing else of this wave's to overlap the DRAM read with; the ISA shows
        // the request issued and then consumed ~70 instructions later in the SAME trip,
        // i.e. 0.7 superblocks of cover, with one activation load left outstanding
        // across the back-edge. Issuing DEPTH superblocks ahead makes it ~1.7. Purely a
        // load schedule: the values and the order they fold into acc are untouched, so
        // every output stays bit-identical.
        constexpr int DEPTH = GQH_MATVEC_DEPTH;
        gqh_wire wire[DEPTH];
        // The activations ride the same pipeline stage as the wire. loadcnt retires in
        // issue order, so as long as ANY load in the body is consumed in the trip that
        // issues it, the partial wait that consumes it also drains the prefetches
        // sitting behind it -- measured on ISA, loading x for the current superblock
        // left the wire prefetch 6 instructions of cover. With x prefetched too, every
        // load in the body belongs to a later superblock. Costs GQH_PER_LANE VGPRs per
        // stage, twice over: a stage's activations stay live while the next trip's load
        // into the same slot is already in flight.
        float xpipe[DEPTH][GQH_PER_LANE];
#pragma unroll
        for (int p = 0; p < DEPTH; ++p) {
            // Clamped so a tensor with fewer superblocks than DEPTH still fills the
            // pipeline (it re-reads superblock nsb-1); p == 0 is spelled out separately
            // so the first stage keeps a literal zero offset.
            const int sbp = p == 0 ? 0 : (p < nsb ? p : nsb - 1);
            wire[p] = gqh_wire_load<RUNG>(
                rowbase[0], (uint32_t) sbp * (uint32_t) sb_bytes, woff);
            gqh_load_x(x, sbp, j0, xpipe[p]);
        }
        // The pipeline advances DEPTH stages per trip, so the main loop covers a
        // multiple of DEPTH and at most DEPTH-1 superblocks are left for the tail.
        const int nsb_main = nsb - nsb % DEPTH;
        for (int sb = 0; sb < nsb_main; sb += DEPTH) {
            // Read every stage out of the loop-carried registers before the prefetch
            // below overwrites them.
            float    d_real[DEPTH];
            uint32_t codes[DEPTH];
            uint8_t  rb[DEPTH];
            uint8_t  hi1[DEPTH];
            float    xcur[DEPTH][GQH_PER_LANE];
#pragma unroll
            for (int p = 0; p < DEPTH; ++p) {
                d_real[p] = gqh_d_real<NCOLS_MAX>(wire[p].d, t_scale);
                codes[p]  = wire[p].codes;
                rb[p]     = wire[p].rb;
                hi1[p]    = wire[p].hi1;
#pragma unroll
                for (int t = 0; t < GQH_PER_LANE; ++t) {
                    xcur[p][t] = xpipe[p][t];
                }
            }

            // Every stage's loads are issued here, in one group, so the wave holds
            // DEPTH independent DRAM requests in flight instead of one.
            //
            // ONE clamped base superblock, then a compile-time per-stage stride off it.
            // Clamping each stage separately instead reads to LLVM as DEPTH unrelated
            // addresses, and it strength-reduces the later ones into 64-bit pointer
            // induction variables -- which loses the SADDR form iteration 3's whole
            // addressing scheme exists to keep (measured on ISA: a v_add_co_u32 /
            // v_add_co_ci_u32 pair came back, and one stage's two b128 activation loads
            // split into b64 + flat b128 + b64). With a common base every offset is
            // `base + constant`, so it folds into the load's immediate.
            //
            // Clamped, not branched: the last trip re-reads superblocks it already
            // holds (a cache hit) instead of splitting the body in two, which would put
            // the prefetch and the decode in different basic blocks -- the machine
            // scheduler only clusters loads within one block, which is the whole basis
            // of this. nsb >= DEPTH whenever this loop runs, so the clamp cannot go
            // negative.
            const int sbn0 = sb + DEPTH <= nsb - DEPTH ? sb + DEPTH : nsb - DEPTH;
#pragma unroll
            for (int p = 0; p < DEPTH; ++p) {
                wire[p] = gqh_wire_load<RUNG>(
                    rowbase[0], (uint32_t) (sbn0 + p) * (uint32_t) sb_bytes, woff);
                gqh_load_x(x, sbn0 + p, j0, xpipe[p]);
            }
            gqh_sched_fence();

            // Stage p before stage p+1, so acc still sums its superblocks in ascending
            // sb -- bit-identical to the one-superblock-per-trip loop.
#pragma unroll
            for (int p = 0; p < DEPTH; ++p) {
                gqh_fold_superblock<RUNG, NCOLS_MAX>(
                    d_real[p], codes[p], rb[p], hi1[p], sub, s_grid, s_ratio,
                    xcur[p], x, sb + p, j0, ncols, x_col_stride, acc[0]);
            }
        }

        // Leftover superblocks, when nsb is not a multiple of DEPTH. These load their
        // own wire and activations rather than reading them out of the pipeline
        // registers: the clamped prefetch base above means stage p does NOT reliably
        // hold superblock nsb_main + p on the final trip, and an unpipelined load here
        // costs nothing -- this runs at most DEPTH-1 times per dispatch, outside the hot
        // loop's basic block.
        //
        // THIS IS ON THE HOT PATH, and it stopped being dead the moment DEPTH became 3.
        // Every weight tensor in the model has `in` 5120 or 17408, i.e. nsb 20 or 68,
        // and 20 % 3 == 68 % 3 == 2 -- so at DEPTH == 2 this ran NEVER and at DEPTH == 3
        // it runs TWICE PER DISPATCH, 2 of every 20 superblocks, on the arm that is 47%
        // of all matvec time. Its cost is already inside the 10.731 ms in the sweep
        // table above.
        //
        // Grouping the two tail loads ahead of both folds (so one DRAM latency is
        // exposed per dispatch instead of two) was tried in iteration 8 and LOST:
        // 10.804 vs 10.731, because keeping wt[]/xt[] live perturbs the main loop's
        // allocation. Leave the load/fold chain alone.
        //
        // Correctness here is covered by forcing gqh_rows1_selected() to true in a
        // throwaway build and running all 12 test-gqh-backend cases at --nvec=1, which
        // compares BITWISE against the geo-quant f32 reference; nsb 1 / 2 / 4 in those
        // vectors give a skipped main loop, a two-stage tail, and a main trip plus a
        // tail respectively.
#pragma unroll
        for (int p = 0; p < DEPTH - 1; ++p) {
            const int sbt = nsb_main + p;
            if (sbt < nsb) {
                const gqh_wire wt = gqh_wire_load<RUNG>(
                    rowbase[0], (uint32_t) sbt * (uint32_t) sb_bytes, woff);
                float xt[GQH_PER_LANE];
                gqh_load_x(x, sbt, j0, xt);
                gqh_fold_superblock<RUNG, NCOLS_MAX>(
                    gqh_d_real<NCOLS_MAX>(wt.d, t_scale), wt.codes, wt.rb, wt.hi1,
                    sub, s_grid, s_ratio, xt, x, sbt, j0, ncols, x_col_stride, acc[0]);
            }
        }
    } else {
        // Software pipeline, one superblock deep. Without it a wave holds exactly one
        // superblock in flight and every iteration pays the whole serial chain
        // b[0] -> E4M3 table -> s_ratio -> s_grid -> FMA; the loop is not unrolled, so
        // there is nothing else of this wave's to overlap the DRAM read with. Issuing
        // sb+1's wire bytes here, ahead of the decode they feed, hides that read
        // behind this superblock's table lookup, LDS gathers and FMAs. Purely a load
        // schedule: the values and the order they fold into acc[] are untouched, so
        // every output stays bit-identical.
        gqh_wire wire[ROWS];
    #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            wire[r] = gqh_wire_load<RUNG>(rowbase[r], 0, woff);
        }
        // The second wire stage, see GQH_WIRE_DEPTH2. Sized away to one dead element off
        // the arm so every instantiation that does not take it keeps byte-identical
        // codegen. Clamped like the batch-1 pipeline's fill: a tensor with one superblock
        // re-reads superblock 0 into the second stage and never uses it.
        constexpr bool DEPTH2 = GQH_WIRE_DEPTH2 && NCOLS_MAX > 1 && NCOLS_MAX < GQH_MAX_COLS;
        gqh_wire wire2[DEPTH2 ? ROWS : 1];
        if constexpr (DEPTH2) {
            const uint32_t sb1_off = (uint32_t) (nsb > 1 ? 1 : 0) * (uint32_t) sb_bytes;
    #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                wire2[r] = gqh_wire_load<RUNG>(rowbase[r], sb1_off, woff);
            }
        }
        // Batch-1 carries the activations through the same pipeline stage as the wire.
        // loadcnt retires in issue order, so as long as ANY load in the body is consumed
        // in the iteration that issues it, the partial wait that consumes it also drains
        // the prefetch sitting behind it -- measured on ISA, that left the wire prefetch
        // 6 instructions of cover. With the activations prefetched too, every load in the
        // body belongs to sb+1 and the only wait is at the top of the next trip, so both
        // requests stay outstanding across the whole FMA block and the back-edge. Costs 8
        // VGPRs of the ~70 this kernel has spare at 16 waves/SIMD.
        float xnext[GQH_PER_LANE];
        if (NCOLS_MAX == 1) {
            gqh_load_x(x, 0, j0, xnext);
        }
        // The exact-width arm's per-column activation addressing, hoisted out of the
        // superblock loop. The SPELLING is the whole point: written at the load site as
        // `x + c * x_col_stride`, LLVM commons the divergent `x + lane_offset` part
        // across the columns FIRST and then adds the (uniform) column strides to that
        // VGPR pair -- so every column but the zeroth pays a v_add_co_u32 /
        // v_add_co_ci_u32 pair per load, with an s_wait_alu depctr hazard behind each.
        //
        // Hoisting to one loop-invariant BASE POINTER per column was the first half of
        // that fix and it only ever got TWO columns: LLVM hands this load group exactly
        // GQH_MULTICOL_XBASES SGPR base pairs and addresses every column past the second
        // in the VADDR form off a 64-bit chain rebuilt IN the loop, once per trip. Read
        // straight off the gfx1201 dump, `<111,NCOLS,3>` trips:
        //
        //   NCOLS      2      3      4      5
        //   x SADDR    4      4      4      4
        //   x VADDR    0      2      4      6
        //   carry      0      4      8     10     <- v_add_co_u32 / v_add_co_ci_u32
        //
        // So the second half is to stop handing out base pointers: columns at or past
        // XBASES take base 0 plus a uniform 32-bit BYTE offset, which folds into the
        // lane offset gqh_load_x already builds -- one v_add_nc_u32, no carry, no
        // s_wait_alu depctr -- and every column's load keeps SADDR at every width.
        //
        // Only the columns that are broken move. Putting the first two on offsets as well
        // (i.e. XBASES == 0, one base for everything) also zeroes the carry column, but it
        // re-allocates registers the FMA block is packed against: v_dual_fmac 26 -> 21 at
        // NCOLS 3 and 48 -> 40 at NCOLS 5, and the trip grows 248 -> 257 at NCOLS 2, which
        // has nothing to fix. Measured on the bare rig it is a wash against this form on
        // pp3/pp4 and ~1.1% worse on pp5, and it perturbs the NCOLS 2 arm for nothing --
        // hence 2 and not 0. Both forms were built and measured; see the handoff.
        //
        // Do NOT "simplify" this into a reconstructed pointer. Round-tripping the base
        // through readfirstlane + inttoptr to force it into an SGPR pair made LLVM drop
        // all six activation loads from the trip while keeping all 72 FMAs -- silently
        // wrong, caught on an ISA dump, never built.
        //
        // Zero and unused on the other arms, so it costs them nothing (NCOLS 1, 2 and the
        // generic 8 are byte-identical across all three rungs, ISA-diffed). Pure
        // addressing: the same bytes, read in the same order, so every output is
        // bit-identical.
        const float * __restrict__ xcol[NCOLS_MAX];
        uint32_t xoff[NCOLS_MAX] = {};
        if constexpr (XSHARED && !I8DOT && !I4DOT) {
    #pragma unroll
            for (int c = 0; c < NCOLS_MAX; ++c) {
                const int b = c < GQH_MULTICOL_XBASES ? c : 0;
                xcol[c] = x + (int64_t) b * x_col_stride;
                xoff[c] = (uint32_t) ((int64_t) (c - b) * x_col_stride
                                      * (int64_t) sizeof(float));
            }
        }
        // The int8 arm's activation addressing, same two-bases-plus-byte-offsets shape
        // and for the same reason -- the column stride is now in BYTES per column
        // (1 byte per activation, not 4), and the group index a lane reads its scale at
        // is `sb * 32 + lane`, one coalesced 128 B request per wave per superblock.
        const int8_t * __restrict__ qxcol[NCOLS_MAX];
        uint32_t qxoff[NCOLS_MAX] = {};
        if constexpr (XSHARED && (I8DOT || I4DOT)) {
    #pragma unroll
            for (int c = 0; c < NCOLS_MAX; ++c) {
                const int b = c < GQH_MULTICOL_XBASES ? c : 0;
                qxcol[c] = qx + (int64_t) b * in;
                qxoff[c] = (uint32_t) ((int64_t) (c - b) * in);
            }
        }
        for (int sb = 0; sb < nsb; ++sb) {
            // The specialized multi-column arm's activation read: ONE grouped load per
            // superblock, shared by every row this warp owns, issued BEFORE the wire
            // prefetch below. Two things ride on that placement. (a) Sharing -- the
            // generic instantiation loads x inside the row loop, so ROWS rows would
            // re-read the same 1024 B per column; hoisting it is what turns ROWS into a
            // 1/ROWS cut in activation traffic, which is the entire point of this arm.
            // (b) Order -- loadcnt retires in issue order, so a load issued AFTER the
            // wire prefetch cannot be waited on without draining the prefetch behind it
            // (the trap the xnext note below describes); the fence stops the scheduler
            // sinking these past it. Purely a load schedule and a load count: the values
            // and the order they fold into acc[] are untouched, so every output stays
            // bit-identical.
            float xshared[NCOLS_MAX][GQH_PER_LANE];
            half2 xh[NCOLS_MAX][GQH_PER_LANE / 2];
            int2  xq[NCOLS_MAX];
            float xq_scale[NCOLS_MAX];
            if constexpr (XSHARED && !I8DOT && !I4DOT) {
    #pragma unroll
                for (int c = 0; c < NCOLS_MAX; ++c) {
                    gqh_load_x(xcol[c], sb, j0, xshared[c], xoff[c]);
                }
                gqh_sched_fence();
                if constexpr (F16DOT) {
    #pragma unroll
                    for (int c = 0; c < NCOLS_MAX; ++c) {
    #pragma unroll
                        for (int p = 0; p < GQH_PER_LANE / 2; ++p) {
                            xh[c][p] = __floats2half2_rn(
                                xshared[c][2 * p], xshared[c][2 * p + 1]);
                        }
                    }
                }
            }
            // The int8 arm reads the pre-pass output IN PLACE OF the f32 loads, not on
            // top of them: 8 bytes per column per lane against 32, one b64 against two
            // b128, and 16 VGPRs of live activation against 64. The scale rides the same
            // load group so the wait that consumes it also covers the codes -- and both
            // stay ahead of the wire prefetch below, for the loadcnt reason above.
            if constexpr (XSHARED && (I8DOT || I4DOT)) {
                const uint32_t qxo = (uint32_t) (sb * GQH_SUPERBLOCK + j0);
    #pragma unroll
                for (int c = 0; c < NCOLS_MAX; ++c) {
                    // ONE 32-bit offset added to the base, and the parentheses are the
                    // whole point: `(base + qxo) + qxoff[c]` adds the DIVERGENT lane
                    // offset first, which materialises a divergent 64-bit pointer and
                    // costs a v_add_co_u32 / v_add_co_ci_u32 pair per column (measured:
                    // 7 of the 8 loads lost SADDR). Summing the two 32-bit offsets first
                    // leaves `SGPR base + 32-bit voffset`, the form gqh_load_x keeps.
                    if constexpr (I4DOT) {
                        xq[c].x = *(const int *) (qxcol[c] + (qxo + qxoff[c]));
                    } else {
                        xq[c] = *(const int2 *) (qxcol[c] + (qxo + qxoff[c]));
                    }
                }
                const int grp = sb * (GQH_SUPERBLOCK / GQH_PER_LANE) + lane;
                if constexpr (XSSHARED) {
                    xq_scale[0] = qxs[grp];
                } else {
    #pragma unroll
                    for (int c = 0; c < NCOLS_MAX; ++c) {
                        xq_scale[c] = qxs[c * (int64_t) (in / GQH_PER_LANE) + grp];
                    }
                }
                gqh_sched_fence();
            }

            // same order as the reference: d_real = e4m3(d) * tensor_scale, then
            // s_b = d_real * (ratio/15). Do not reassociate.
            //
            // `d` is the superblock's E4M3 byte -- one address for the whole wave -- but it
            // is adjacent to the per-lane ratio byte `rb` in gqh_wire, LLVM merges the two
            // into one 16-bit value, and rb's divergence infects d. Re-asserting uniformity
            // is what turns the table read into an `s_load_b32`: the lookup leaves the
            // vector memory pipe for the scalar one, so it is tracked by kmcnt instead of
            // loadcnt and can no longer force a full `s_wait_loadcnt 0x0` that drains the
            // prefetch behind it. It also deletes the 64-bit VALU address chain the
            // divergent form needed.
            //
            // COVER moves ONLY the scale decode, and only on the exact-width arm: the
            // wire registers are drained into codes/rb/hi1/d_raw here either way (they
            // have to be, the prefetch below overwrites them), and the four `s_load`s
            // plus the `s_wait_kmcnt 0x0` behind them go after the prefetch instead of
            // in front of it. `d_real` feeds the per-row scale in the epilogue, not the
            // dot, so nothing in the FMA block waits on it any earlier than it did.
            constexpr bool COVER = GQH_WIRE_COVER && XSHARED;
            float    d_real[ROWS];
            uint32_t codes[ROWS];
            uint8_t  rb[ROWS];
            uint8_t  hi1[ROWS];
            // Sized away to a dead byte off the COVER arm so the ~600 instantiations the
            // scoring rig never dispatches keep byte-identical codegen -- they are this
            // file's only instrument on those paths (iteration 5). With `d_raw` a plain
            // local in the else arm below, the !COVER body is the loop this replaces,
            // statement for statement, after `if constexpr` elimination.
            uint8_t  d_cover[COVER ? ROWS : 1];
    #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if constexpr (COVER) {
                    d_cover[r] = wire[r].d;
                } else {
                    const uint8_t d_raw = wire[r].d;
                    d_real[r] = gqh_d_real_uni<UNIFORM_ADDR, I8DOT || I4DOT>(d_raw, t_scale);
                }
                codes[r] = wire[r].codes;
                rb[r]    = wire[r].rb;
                hi1[r]   = wire[r].hi1;
            }

            float xcur[GQH_PER_LANE];
            if (NCOLS_MAX == 1) {
    #pragma unroll
                for (int t = 0; t < GQH_PER_LANE; ++t) {
                    xcur[t] = xnext[t];
                }
            }

            // Clamped, not branched: the last iteration re-reads its own superblock (a
            // cache hit) instead of splitting the body in two, which would put the
            // prefetch and the decode in different basic blocks -- the machine
            // scheduler only clusters loads within one block.
            const int sbn = sb + 1 < nsb ? sb + 1 : sb;
            // Hand the superblock byte offset to the wire loads through an SGPR that LSR
            // cannot see through, on the int8 arm only.
            //
            // `rowbase[r] + (sbn_off + woff.field)` is the SADDR pattern -- uniform
            // 64-bit base, 32-bit lane offset -- but SADDR is only selected when the
            // zero-extension of that offset lands in the SAME basic block as the load.
            // With the int8 arm's shorter body there is register budget to spare, so LSR
            // rewrites the whole address as a strength-reduced pointer induction variable:
            // 16 s_add_nc_u64 per trip advancing the ROWS row bases, and the lane offset
            // left LOOP-INVARIANT in a VGPR, i.e. zero-extended in the preheader. Every
            // one of the 24 wire loads then falls back to the 64-bit VADDR form and pays a
            // v_add_co_u32 / v_add_co_ci_u32 pair plus the s_wait_alu depctr hazard behind
            // it -- measured 27+27 carry adds and 39 s_wait_alu in a 773-instruction loop.
            //
            // readfirstlane is exact, not an approximation: `sbn` is derived from the
            // superblock loop counter and is wave-uniform by construction, so lane 0's
            // copy is every lane's copy (same argument as gqh_uniform on `row`). What it
            // buys is that the offset is now an opaque convergent intrinsic LSR will not
            // fold into a pointer, so `sbn_off + woff.field` is computed IN the trip --
            // three v_add_nc_u32 shared by all ROWS rows, and every wire load keeps
            // SGPR base + 32-bit voffset. Pure addressing: the same bytes, read in the
            // same order, so every output is bit-identical (microbench FNV unchanged).
            //
            // Gated on I8DOT so the f32 and batch-1 instantiations keep byte-identical
            // codegen -- they are this file's only instrument on the paths the scoring rig
            // does not execute, and LSR made a different (already-good) choice there.
            const uint32_t sbn_off_raw = (uint32_t) sbn * (uint32_t) sb_bytes;
            const uint32_t sbn_off = (I8DOT || I4DOT)
                ? (uint32_t) gqh_uniform((int) sbn_off_raw) : sbn_off_raw;
            // The 2-deep arm prefetches sb+2, clamped the same way -- the last two trips
            // re-read the final superblock (an L2 hit) rather than splitting the body.
            const int sbn2 = sb + 2 < nsb ? sb + 2 : nsb - 1;
            const uint32_t sbn2_off_raw = (uint32_t) sbn2 * (uint32_t) sb_bytes;
            const uint32_t sbn2_off = (I8DOT || I4DOT)
                ? (uint32_t) gqh_uniform((int) sbn2_off_raw) : sbn2_off_raw;
            // Every row's wire load is issued here, in one group, so the wave holds ROWS
            // independent DRAM requests in flight instead of one. That is the whole point
            // of ROWS > 1: bytes-in-flight per wave is the axis this kernel is bound on.
    #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if constexpr (DEPTH2) {
                    // Rotate, then refill the slot just vacated with sb+2. `wire[r]` was
                    // read into codes/rb/hi1/d above, so the copy is of registers already
                    // dead as a wire stage.
                    wire[r]  = wire2[r];
                    wire2[r] = gqh_wire_load<RUNG>(rowbase[r], sbn2_off, woff);
                } else {
                    wire[r] = gqh_wire_load<RUNG>(rowbase[r], sbn_off, woff);
                }
            }
            if (NCOLS_MAX == 1) {
                gqh_load_x(x, sbn, j0, xnext);
            }
            // Pin the wire prefetch, on the exact-width arm as well as batch-1.
            //
            // Without a fence HERE, nothing structural stops LLVM sinking these loads
            // across the back-edge onto their uses at the top of the next trip -- and on
            // this arm it does exactly that for `codes`, which is 128 of the 137 wire
            // bytes a warp reads per superblock. The result reads as a pipeline but the
            // biggest load in it is issued in the trip that consumes it, i.e. one DRAM
            // latency exposed per superblock instead of none. batch-1 already got this
            // fence (it sits after its own xnext load); the exact-width arm inherited the
            // generic instantiation's fence-free tail when it was split off. The generic
            // instantiation is deliberately still excluded -- it is the same-binary A/B
            // control, so its codegen has to stay put.
            if constexpr (COVER) {
                // A FULL barrier, not gqh_sched_fence(), and the difference is the whole
                // change. gqh_sched_fence()'s mask lets SALU/SMEM cross, and on this arm
                // they do: with the decode merely moved below the fence in source order
                // its four `s_load_b32` and the `s_wait_kmcnt 0x0` behind them floated
                // straight back above the prefetch, and the scheduler pushed the 8-load
                // wire clause 195 -> 215 of the trip on top of that -- cover 41% -> 36%,
                // i.e. the reorder measured BACKWARDS on the static instrument. Nothing
                // crosses this one, so the clause is issued where the source puts it.
                __builtin_amdgcn_sched_barrier(GQH_COVER_MASK);
            } else if (NCOLS_MAX == 1 || XSHARED) {
                gqh_sched_fence();
            }
            // The scale decode, now BEHIND the prefetch it used to delay: four `s_load`s
            // and a `s_wait_kmcnt 0x0` that the wave used to pay for BEFORE the next
            // superblock's weight request was on the bus. It feeds the epilogue scale,
            // not the dot, so nothing in the FMA block waits on it any earlier.
            if constexpr (COVER) {
    #pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    d_real[r] = gqh_d_real_uni<UNIFORM_ADDR, I8DOT || I4DOT>(d_cover[r], t_scale);
                }
            }

    #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                // The shift spelling is confined to UNIFORM_ADDR. It is a win on the
                // exact-width and batch-1 arms (pp3 -1.6%, pp1 and greedy tg -2.5%) and a
                // LOSS on the generic one, which is the only arm that reaches ncols 5..8:
                // measured pp5 74.82 -> 78.61 and the same-binary control pp3 59.38 ->
                // 63.16 when it was applied everywhere, on a hot loop the ISA says is
                // unchanged (326 instrs / 144 VALU either way) -- so it is the generic
                // arm's eight predicated blocks reacting to it, not its inner loop. Same
                // nibble, same float, either spelling.
                const int ratio_idx = XSHARED
                    ? ((rb[r] >> rb_shift) & 0x0f)
                    : ((sub & 1) ? (rb[r] >> 4) : (rb[r] & 0x0f));
                const float s_b = d_real[r] * s_ratio[ratio_idx];

                // The int8 arm's decode: FOUR ds_read_u16 and TWO v_lshl_or_b32 for the
                // lane's 8 weights, and not one f32 weight in sight. `s_b` does not
                // multiply anything here -- it rides the accumulator, folded with the
                // activation group's scale (which already carries both /127s), so the
                // eight `level * s_b` muls the f32 arm pays per row per superblock are
                // gone and the per-row scale chain grows by exactly one v_mul.
                if constexpr ((I8DOT || I4DOT) && XSHARED) {
                    if constexpr (I4DOT) {
                        uint32_t wq = 0;
#pragma unroll
                        for (int d = 0; d < 4; ++d) {
                            const int idx = gqh_pair_lut<RUNG>::index(codes[r], hi1[r], d);
                            wq |= (uint32_t) (s_q8[idx] & 0xff) << (8 * d);
                        }
                        const float s_row = s_b * (XSSHARED ? xq_scale[0] : 1.0f);
#pragma unroll
                        for (int c = 0; c < NCOLS_MAX; ++c) {
                            const int dot = __builtin_amdgcn_sudot8(true, (int) wq, true, xq[c].x, 0, false);
                            const float s = XSSHARED ? s_row : (s_b * xq_scale[c]);
                            acc[r][c] += s * (float) dot;
                        }
                        continue;
                    }
                    int wq[2];
    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        const int i0 = gqh_pair_lut<RUNG>::index(codes[r], hi1[r], 2 * d);
                        const int i1 = gqh_pair_lut<RUNG>::index(codes[r], hi1[r], 2 * d + 1);
                        wq[d] = (int) s_q8[i0] | ((int) s_q8[i1] << 16);
                    }
                    const float s_row = s_b * (XSSHARED ? xq_scale[0] : 1.0f);
    #if GQH_FP8DOT
                    // The whole of the fp8 arm, and the reason for it: the dot ALREADY
                    // accumulates in f32, so the `v_cvt_f32_i32` the int8 form needs per
                    // (row, column) is not replaced by anything. 3 VALU per (row, column)
                    // against 4, on a dot that issues at the same rate. The 0.0f seed is
                    // an inline VOP3P src2, not a materialised zero -- checked in the ISA,
                    // because a `v_mov` per (row, column) would eat a third of the win.
    #pragma unroll
                    for (int c = 0; c < NCOLS_MAX; ++c) {
                        float dot = __builtin_amdgcn_dot4_f32_fp8_fp8(wq[0], xq[c].x, 0.0f);
                        dot = __builtin_amdgcn_dot4_f32_fp8_fp8(wq[1], xq[c].y, dot);
                        const float s = XSSHARED ? s_row : (s_b * xq_scale[c]);
                        acc[r][c] += s * dot;
                    }
    #else
    #pragma unroll
                    for (int c = 0; c < NCOLS_MAX; ++c) {
                        int dot = ggml_cuda_dp4a(wq[0], xq[c].x, 0);
                        dot = ggml_cuda_dp4a(wq[1], xq[c].y, dot);
                        const float s = XSSHARED ? s_row : (s_b * xq_scale[c]);
                        acc[r][c] += s * (float) dot;
                    }
    #endif
                    continue;
                }

                // Decode this lane's 8 weights ONCE, then reuse them for every column.
                float w[GQH_PER_LANE];
                if constexpr (PAIRLUT) {
                    // Two weights per index, one stride-1 pair of LDS dwords SIZE apart
                    // (see gqh_pair_lut_lds); same floats, same muls, same order.
                    constexpr int LUT_SIZE = gqh_pair_lut<RUNG>::SIZE;
    #pragma unroll
                    for (int p = 0; p < GQH_PER_LANE / 2; ++p) {
                        const int idx = gqh_pair_lut<RUNG>::index(codes[r], hi1[r], p);
                        w[2 * p]     = s_pair[idx]            * s_b;
                        w[2 * p + 1] = s_pair[LUT_SIZE + idx] * s_b;
                    }
                } else {
    #pragma unroll
                    for (int t = 0; t < GQH_PER_LANE; ++t) {
                        // gqh4: little-endian, byte t>>1, low nibble for even t -> bit 4*t
                        // either way. gqh3: the low-2-bit plane carries bits [1:0] of the
                        // code and the high plane bit 2. gqh2_h: the 2-bit code is the
                        // whole index.
                        const int code = IS_GQH4
                            ? ((codes[r] >> (4 * t)) & 0x0f)
                            : (((codes[r] >> (2 * t)) & 0x03) |
                               (IS_GQH3 ? (((hi1[r] >> t) & 1) << 2) : 0));
                        w[t] = s_grid[code] * s_b;
                    }
                }

                // Fully unrolled so acc[]/xs[] stay in registers. F16DOT / I8DOT are
                // not bit-exact; they are separate instantiations, default off.
                if constexpr (F16DOT && XSHARED) {
                    half2 wh[GQH_PER_LANE / 2];
    #pragma unroll
                    for (int p = 0; p < GQH_PER_LANE / 2; ++p) {
                        wh[p] = __floats2half2_rn(w[2 * p], w[2 * p + 1]);
                    }
    #pragma unroll
                    for (int c = 0; c < NCOLS_MAX; ++c) {
    #pragma unroll
                        for (int p = 0; p < GQH_PER_LANE / 2; ++p) {
                            ggml_cuda_mad(acc[r][c], wh[p], xh[c][p]);
                        }
                    }
                } else {
    #pragma unroll
                    for (int c = 0; c < NCOLS_MAX; ++c) {
                        if (!XSHARED && c >= ncols) {
                            continue;
                        }
                        float xs[GQH_PER_LANE];
                        if (NCOLS_MAX == 1) {
    #pragma unroll
                            for (int t = 0; t < GQH_PER_LANE; ++t) {
                                xs[t] = xcur[t];
                            }
                        } else if (XSHARED) {
    #pragma unroll
                            for (int t = 0; t < GQH_PER_LANE; ++t) {
                                xs[t] = xshared[c][t];
                            }
                        } else {
                            gqh_load_x(x + (int64_t) c * x_col_stride, sb, j0, xs);
                        }
    #pragma unroll
                        for (int t = 0; t < GQH_PER_LANE; ++t) {
                            acc[r][c] += w[t] * xs[t];
                        }
                    }
                }
            }
        }
    }

    static_assert(GQH_WARP == 32, "the reduction network below assumes wave32 segments");

    // ONE accumulator per lane: reduce-scatter, then ONE wave-wide store.
    //
    // This arm's epilogue was the last untouched block in the per-dispatch fixed region
    // -- the one region on this kernel with a MEASURED conversion rate (iteration 4:
    // 363 instructions off the fixed block bought +1.74% of HumanEval, while the
    // microbench read -0.3%). Two things were paying for the all-reduce's redundancy:
    //
    //   * the network itself, 160 exchanges + 160 adds to produce 32 floats, of which
    //     31 of every 32 copies are discarded (see gqh_rs_level), and
    //   * the STORE, which is `if (lane == 0)` around 32 scalar dwords: 32
    //     `global_store_b32` behind 32 `s_add_nc_u64`, 16 `s_wait_alu depctr_sa_sdst`
    //     and 16 `s_clause`, all under an EXEC of ONE lane, to write 128 bytes.
    //
    // A reduce-scatter fixes both at once, because its survivor for accumulator k lands
    // in lane k. Flattening k as `c * ROWS + r` puts consecutive lanes on consecutive
    // output ROWS, which are consecutive floats in y -- so the 32 lane-0 dwords become
    // ONE wave-wide `global_store_b32` covering eight 16-byte runs.
    //
    // Gated on the accumulator count being exactly the wave width, which is the whole
    // reason it closes. The generic (`ncols < NCOLS_MAX`) and batch-1 arms keep the old
    // network untouched -- the generic one is the same-binary A/B control and its codegen
    // must not move -- and so does anything whose ROWS is not a power of two.
#if GQH_RS_REDUCE
    if constexpr (XSHARED && ROWS * NCOLS_MAX == GQH_WARP && (ROWS & (ROWS - 1)) == 0) {
        const int rlane = gqh_lane_id();
        // Pure register renaming: every index is a compile-time constant, so the flatten
        // itself emits nothing.
        float v[GQH_WARP];
#pragma unroll
        for (int c = 0; c < NCOLS_MAX; ++c) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                v[c * ROWS + r] = acc[r][c];
            }
        }
        // Same five levels in the same order as the butterfly, so v[0] is bit-for-bit
        // the float lane 0 used to store for accumulator `lane`.
        gqh_rs_level<GQH_WARP / 2>(v, rlane);
        gqh_rs_level<8>(v, rlane);
        gqh_rs_level<4>(v, rlane);
        gqh_rs_level<2>(v, rlane);
        gqh_rs_level<1>(v, rlane);

        static_assert((ROWS & (ROWS - 1)) == 0, "the lane -> (row, col) map needs ROWS 2^k");
        const int r_out = rlane & (ROWS - 1);
        const int c_out = rlane / ROWS;
        // Wave-uniform SGPR base + a 32-bit lane BYTE offset, for the SADDR reason in
        // gqh_wire_offsets: the lane-0 form's address was uniform and therefore SGPRs,
        // and this is the spelling that keeps the per-lane form to one 32-bit VGPR
        // instead of a 64-bit VALU pair plus its s_wait_alu hazards.
        //
        // The 32-bit offset cannot wrap on this arm: it is only reached when
        // ncols == NCOLS_MAX <= 8, and the exact-width launcher's widest live shape is
        // out == 17408, i.e. 8 * 17408 * 4 = 557 056 bytes.
        const uint32_t yoff =
            ((uint32_t) c_out * (uint32_t) y_col_stride + (uint32_t) r_out) * 4u;
        // The row clamp, unchanged in meaning: a tail warp recomputed row out-1 in slot
        // r > 0 and must not write it back. One wave-wide compare instead of ROWS
        // scalar branches.
        if (r_out == 0 || row + r_out < out) {
            *(float *) ((char *) (y_base + row) + yoff) = v[0];
        }
        return;
    }
#endif

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        // The only place the row clamp is paid for: a tail warp recomputed row out-1
        // in slot r > 0 and must not write it back.
        const bool store = r == 0 || row + r < out;
#if GQH_DPP_REDUCE
#pragma unroll
        for (int c = 0; c < NCOLS_MAX; ++c) {
            if (!XSHARED && c >= ncols) {
                continue;
            }
            // Same five levels in the same order (16, 8, 4, 2, 1) and the same two
            // summands at each level, so lane 0's float is bit-for-bit the one the shift
            // tree produced -- verified, every microbench FNV is unchanged. off 8/4/2/1
            // fold into one v_add_f32_dpp each; off == 16 crosses the wave's two DPP
            // rows and keeps its ds_bpermute. gqh_xor16 records why the one-VALU-op
            // replacement for that level was built, measured and turned off again.
            //
            // NOT batched across columns. Hoisting the NCOLS_MAX off == 16 shuffles into
            // their own pass first does put them behind ONE s_wait_dscnt instead of one
            // each (164 -> 28), and it measured the same on all six live shapes to within
            // 0.5% -- while costing NCOLS_MAX live floats, which stepped four
            // <GQH3,8,4,i8,XSSHARED=0> and four <GQH3,8,8,i8,XSSHARED=0> instantiations
            // from 16 to 12 and 9 to 8 waves/SIMD (96 -> 102, 167 -> 171 VGPRs). This
            // form leaves every LIVE instantiation's occupancy exactly where it was; 22
            // never-dispatched ones (NCOLS 4/6 at ROWS 3, the XSSHARED == 0 x-scale
            // escape hatch, ROWS == 6) still step down, against 20 that step up.
            float v = acc[r][c];
            v += gqh_xor16(v);
            v += gqh_dpp_xor<8>(v);
            v += gqh_dpp_xor<4>(v);
            v += gqh_dpp_xor<2>(v);
            v += gqh_dpp_xor<1>(v);
            acc[r][c] = v;
        }
#else
#pragma unroll
        for (int c = 0; c < NCOLS_MAX; ++c) {
            if (!XSHARED && c >= ncols) {
                continue;
            }
#pragma unroll
            for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
                acc[r][c] += gqh_warp_shfl_down(acc[r][c], off);
            }
        }
#endif
#pragma unroll
        for (int c = 0; c < NCOLS_MAX; ++c) {
            // Only the generic instantiation can be dispatched with ncols < NCOLS_MAX;
            // the exact-width ones write every column they carry.
            if (!XSHARED && c >= ncols) {
                continue;
            }
            if (lane == 0 && store) {
                y_base[(int64_t) c * y_col_stride + row + r] = acc[r][c];
            }
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

// Rows-per-warp selector for the batch-1 path, and the one place the ROWS trade-off
// is decided. ROWS == GQH_MATVEC_ROWS gives a wave ROWS independent wire streams,
// which is what a LATENCY-STARVED wave wants; it is not free, because it also divides
// the wave count by ROWS. Whether that trade pays is a property of the SHAPE, and the
// swept GQH4 buckets on the R9700 (rocprofv3, GQH4 ms per N=1 forward, +-0.3%) say so
// plainly -- ROWS == 4 lands the whole in == 5120 family on a ~470-490 GB/s plateau
// regardless of shape, while ROWS == 1 tracks wave supply from 252 up to 514 GB/s:
//
//   out    in     n   ROWS=1  ROWS=4  r1->r4   GB/s r1->r4   rounds at ROWS=1
//   17408  5120  127  11.790  12.988  0.908x   514 -> 467    8.5   <- ROWS=1 wins
//   12288  5120   12   1.081   0.826  1.308x   374 -> 489    6.0
//   10240  5120   48   3.813   2.864  1.331x   354 -> 471    5.0
//    6144  5120   45   3.010   1.878  1.602x   252 -> 403    3.0
//    5120 17408   66   6.690   5.134  1.303x   471 -> 613    2.5
//
// The selector as it stands, A/B'd from ONE binary against its own rounds == 0 control
// (the only fair comparison at this effect size), GQH4 ms per N=1 forward:
//
//   out      control (all ROWS=4)   selector    delta
//   17408             12.960          11.934    1.086x   <- the only bucket switched
//    5120              5.132           5.138    ~1.00x
//   10240              2.868           2.881    ~1.00x
//    6144              1.879           1.881    ~1.00x
//   12288              0.827           0.831    ~1.00x
//   total             23.666          22.665    1.044x
//
// So exactly one bucket loses, and it is the only one that was ALREADY near DRAM peak
// at ROWS == 1 -- there was no starvation left to fix, only wave count to give up.
// `out` (i.e. the wave supply, since ROWS == 1 is one warp per row) is the only
// variable that separates it: out == 12288 has the same `in`, the same nsb == 20, and
// gains 31%.
//
// **This is a fit on five shapes of one model, and it is deliberately biased.** The
// threshold below now sits between 6144 and 10240 rows (it was between 12288 and 17408
// while the ROWS == 1 arm was one superblock deep); nothing measured here pins it
// inside that interval, and neither hot_ms nor the greedy-sha gate can falsify it.
// It is therefore written as the NARROW EXCEPTION: a shape has to clear
// GQH_ROWS1_ROUNDS full occupancy rounds before it takes the ROWS == 1 arm, so any
// shape this table does not cover lands on the ROWS == 4 plateau instead of on the arm
// that can regress 13%. Scoped to GQH4 for the same reason -- the sweep was
// GQH4-only, and GQH3/GQH2_H measured flat across ROWS (2.47 -> 2.44 ms) so they have
// nothing to gain from being included and a shape-dependent regression to lose.
//
// **The table above is iteration 5's, taken when the ROWS == 1 arm was DEPTH == 1 and
// ran at 510 GB/s. Iterations 7-8 took that arm to 565 GB/s, which moved the crossover
// two buckets down -- see GQH_ROWS1_ROUNDS, re-measured per bucket in iteration 9.**
// The rows above are still the right shape of evidence; their verdicts for out == 10240
// and out == 12288 are stale.
//
// GGML_GQH_ROWS1_ROUNDS overrides the threshold; 0 disables the ROWS == 1 arm outright,
// which is how a single binary can be A/B'd against its own selector-off control (the
// kernel trace's cross-session spread is ~1.4%, so same-build is the only fair
// comparison for a 5% effect).
static int gqh_rows1_rounds() {
    static const int rounds = []() {
        const char * e = getenv("GGML_GQH_ROWS1_ROUNDS");
        return e ? atoi(e) : GQH_ROWS1_ROUNDS;
    }();
    return rounds;
}

// Waves the device holds resident for the ROWS == 1 batch-1 kernel, asked of the
// runtime rather than assumed. Both halves of this matter:
//   - `nsm` is whatever unit the backend counts multiprocessors in. On RDNA, HIP
//     reports WGPs (2 CUs), so this box says nsm == 32 for a 64-CU R9700 -- a
//     hand-rolled "waves per CU" constant is silently off by 2x, which is exactly
//     how the first cut of this selector put the threshold at 8192 rows instead of
//     16384 and cost 1.2 ms. (That regression was iteration 5's, when the ROWS == 1
//     arm was DEPTH == 1; at DEPTH == 3 out == 10240 and out == 12288 are FASTER on
//     that arm and the threshold is deliberately 10240 now. The lesson survives the
//     inversion: a hand-rolled `nsm` still moves the threshold by 2x, and the shapes
//     it would drag over -- out == 5120 and out == 6144 -- lose 13% there.)
//   - blocks-per-MP comes from the occupancy API, so it tracks the kernel's real
//     register pressure instead of a hard-coded 16 waves/SIMD.
// Product on the R9700: 32 MPs x 16 blocks x 4 warps = 2048 waves, the physically
// correct figure. Cached per device -- this is on the per-dispatch path, and the
// forward's real launch-gap budget is only ~4.4 ms (1891 dispatches; the ~13 ms an
// earlier note claimed was rocprofv3's own per-dispatch cost, see the handoff).
static int gqh_rows1_resident_waves() {
    const int id = ggml_cuda_get_device();
    static int cached[GGML_CUDA_MAX_DEVICES] = {};
    if (cached[id] == 0) {
        const int threads = GQH_WARP * GQH_MATVEC_WARPS;
        int blocks_per_mp = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_mp, gqh_matvec_kernel<GGML_TYPE_GQH4, 1, 1, false>,
            threads, 0);
        const int nsm = ggml_cuda_info().devices[id].nsm;
        cached[id] = blocks_per_mp * nsm * GQH_MATVEC_WARPS;
    }
    return cached[id];
}

static bool gqh_rows1_selected(int out) {
    const int rounds = gqh_rows1_rounds();
    if (rounds <= 0) {
        return false;
    }
    const int resident = gqh_rows1_resident_waves();
    if (resident <= 0) {
        return false;   // occupancy query failed: take the safe arm
    }
    // One warp per row at ROWS == 1, so the wave count IS `out`.
    return (int64_t) out >= (int64_t) rounds * resident;
}

// SuperSonic iter 14: 2-wave CTAs on two GQH3 decode singles. Occupancy is
// unchanged (16 waves/SIMD at 2 or 4 warps/block); only block retire/refill
// granularity changes, which lets tiny co-resident kernels fill gaps. Bit-exact:
// each output row is still one wave on the same superblock loop.
//     tight GQH3 nsb=68 o=5120  (ffn down)
//     tight GQH3 nsb=20 o=10240 (attn qkv)
// GGML_GQH_WARPS2=0 is the same-binary control.
static int gqh_decode_warps(ggml_type rung, int in, int out, int ncols) {
    static const bool warps2_on = []() {
        const char * e = getenv("GGML_GQH_WARPS2");
        return e ? atoi(e) != 0 : true;
    }();
    if (warps2_on && ncols == 1 && rung == GGML_TYPE_GQH3 &&
        ((in == 17408 && out == 5120) || (in == 5120 && out == 10240))) {
        return 2;
    }
    return GQH_MATVEC_WARPS;
}

// Env gate for the specialized multi-column arm. GGML_GQH_MULTICOL=0 sends every
// ncols > 1 dispatch back to the generic runtime-guarded instantiation, so ONE binary
// can be A/B'd against the exact pristine multi-column path -- the discipline the rest
// of this file's constants are measured under.
static bool gqh_multicol_on() {
    static const bool on = []() {
        const char * e = getenv("GGML_GQH_MULTICOL");
        return e ? atoi(e) != 0 : true;
    }();
    return on;
}

// One exact-width multi-column dispatch: NCOLS is the column count at compile time AND
// the value passed as `ncols`, which is the contract the kernel's dead column guard
// relies on. A block covers GQH_MATVEC_WARPS * GQH_MULTICOL_ROWS output rows -- getting
// that divisor wrong launches ROWS times too many blocks, which still computes the right
// answer (the `row >= out` return retires them) and reads exactly like "ROWS did not
// help".
//
// ROWS is GQH_MULTICOL_ROWS for EVERY specialized width. It used to drop to 2 at NCOLS == 4
// because both axes spend the same registers -- acc[ROWS][NCOLS] and xshared[NCOLS][8] each
// scale with NCOLS -- and NCOLS == 4 at ROWS == 3 measured 98 VGPRs, two over the
// 16-waves/SIMD cliff. The wave-uniform addressing (UNIFORM_ADDR in the kernel) took ~16
// VGPRs of 64-bit address chain out of every cell, and the whole grid now fits. Measured
// VGPRs, clang -Rpass-analysis=kernel-resource-usage on gfx1201, GQH4 / GQH3; 96 is the
// cliff:
//
//   NCOLS   ROWS=1    ROWS=2    ROWS=3     ROWS=4
//     2     32 / 40   46 / 62   60 /  82   73 /  93
//     3     41 / 50   57 / 72   71 /  93   85 /  93
//     4     51 / 60   67 / 82   82 /  93   97 / 108
//
// (The pre-UNIFORM_ADDR grid, for the record: 76/97 and 94/100 at NCOLS 2, 87/97 and
// 102/122 at NCOLS 3, 98/111 and 118/124 at NCOLS 4 for ROWS 3 and 4.) The generic
// instantiation is 37 VGPRs, so this arm is still spending headroom the multi-column path
// never used -- ~10-11 VGPRs per extra column at ROWS == 3, which is the budget for
// raising GQH_MULTICOL_SPEC_MAX.
//
// Dropping the NCOLS == 4 special case was MEASURED, not inferred from the table:
// llama-bench pp4 per-forward ms, -r 24, one build each, against the same-binary generic
// control at 66.59 ms -- ROWS == 2 47.69 (1.396x), ROWS == 3 42.99 (1.549x), i.e. 1.109x
// for the switch. pp2 is unmoved (36.76 vs 36.81) as it must be, since NCOLS == 2 was
// already on GQH_MULTICOL_ROWS. ncols == 2 and 4 are not what hot_ms scores (that is
// pp3 = the MTP n-max=2 verify width); they are measured here directly because
// llama-bench can dispatch them, not fitted from the VGPR rule.

// Tiled N=8 verify (DFlash2 native block). K-outer over superblocks, M-tile =
// 4 warps x 3 rows, 4 KB LDS holding a 4-column x tile. Bit-identical to
// gqh_matvec_kernel<RUNG,8,3> (sb ascending, t=0..7). Default OFF on gfx1201:
// 161 VGPRs, one-prompt HE 87 tok/s vs generic N=8 at ~104. Opt in with
// GGML_GQH_N8_TILE=1.
#define GQH_N8_TILE_ROWS  3
#define GQH_N8_TILE_WARPS 4
#define GQH_N8_XCHUNK     4

static bool gqh_n8_tile_on() {
    static const bool on = []() {
        const char * e = getenv("GGML_GQH_N8_TILE");
        // Default OFF: gfx1201 GQH3 nsb=20 is 161 VGPRs / 4 KB LDS, HE 87 tok/s
        // vs the generic exact-width N=8 arm at ~100 VGPRs / 104 tok/s.
        return e ? atoi(e) != 0 : false;
    }();
    return on;
}

template <ggml_type RUNG, bool PAIRED, int NSB>
static __global__ void gqh_matvec_n8_tile_kernel(
        const uint8_t * __restrict__ data, const float * __restrict__ x,
        float * __restrict__ y, int in, int out, float tensor_scale,
        gqh_grid16 grid, int64_t x_col_stride, int64_t y_col_stride,
        const uint8_t * __restrict__ data_b, float * __restrict__ y_b,
        float tensor_scale_b, gqh_grid16 grid_b) {
    constexpr int NCOLS = 8;
    constexpr int ROWS  = GQH_N8_TILE_ROWS;
    constexpr bool IS_GQH3 = RUNG == GGML_TYPE_GQH3;
    constexpr bool IS_GQH4 = RUNG == GGML_TYPE_GQH4;

    const uint8_t * __restrict__ w_base = data;
    float * __restrict__ y_base = y;
    float t_scale = tensor_scale;
    const float * g_levels = grid.v;
    if constexpr (PAIRED) {
        if (blockIdx.y != 0) {
            w_base   = data_b;
            y_base   = y_b;
            t_scale  = tensor_scale_b;
            g_levels = grid_b.v;
        }
    }

    const int sb_bytes = IS_GQH3 ? GQH3_SB_BYTES : (IS_GQH4 ? GQH4_SB_BYTES : GQH2H_SB_BYTES);
    const int warp = gqh_uniform((int) (threadIdx.x / GQH_WARP));
    const int lane = threadIdx.x % GQH_WARP;
    const int row  = gqh_uniform((int) ((blockIdx.x * GQH_N8_TILE_WARPS + warp) * ROWS));

    constexpr int RUNG_LEVELS = IS_GQH4 ? 16 : (IS_GQH3 ? 8 : 4);
    __shared__ float s_ratio[16];
    __shared__ float s_grid[RUNG_LEVELS];
    // 4 columns x 256 activations = 4 KB. 16 blocks x 4 KB fills the 64 KB
    // WGP LDS file; 4 warps x 16 blocks = 64 waves. Do NOT raise this to 8 KB.
    __shared__ float s_x[GQH_N8_XCHUNK][GQH_PER_LANE][GQH_WARP];
    if (threadIdx.x < 16) {
        s_ratio[threadIdx.x] = gqh_bits(GQH_RATIO_Q_D[threadIdx.x][0]);
        if (threadIdx.x < RUNG_LEVELS) {
            s_grid[threadIdx.x] = g_levels[threadIdx.x];
        }
    }
    __syncthreads();

    const int nsb = NSB > 0 ? NSB : (in / GQH_SUPERBLOCK);
    const uint8_t * __restrict__ rowbase[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int rr = (row + r < out) ? row + r : (out > 0 ? out - 1 : 0);
        rowbase[r] = w_base + (int64_t) rr * nsb * sb_bytes;
    }

    const int j0 = lane * GQH_PER_LANE;
    const int sub = j0 >> 4;
    const int rb_shift = (sub & 1) << 2;
    const gqh_wire_offsets woff = {
        (uint32_t) (1 + (sub >> 1)),
        (uint32_t) (9 + lane * (IS_GQH4 ? 4 : 2)),
        (uint32_t) (73 + lane),
    };

    const float * __restrict__ xcol[NCOLS];
    uint32_t xoff[NCOLS];
#pragma unroll
    for (int c = 0; c < NCOLS; ++c) {
        const int b = c < GQH_MULTICOL_XBASES ? c : 0;
        xcol[c] = x + (int64_t) b * x_col_stride;
        xoff[c] = (uint32_t) ((int64_t) (c - b) * x_col_stride * (int64_t) sizeof(float));
    }

    float acc[ROWS][NCOLS] = {};
    gqh_wire wire[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        wire[r] = gqh_wire_load<RUNG>(rowbase[r], 0, woff);
    }

    const int tid = (int) threadIdx.x;
    const int fill_col = tid / GQH_WARP;     // 0..3
    const int fill_ln  = tid % GQH_WARP;

    for (int sb = 0; sb < nsb; ++sb) {
        float    d_real[ROWS];
        uint32_t codes[ROWS];
        uint8_t  rb[ROWS];
        uint8_t  hi1[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int d = gqh_uniform((int) wire[r].d);
            d_real[r] = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * t_scale;
            codes[r]  = wire[r].codes;
            rb[r]     = wire[r].rb;
            hi1[r]    = wire[r].hi1;
        }

        // Cooperative 4-col x fill, then wire prefetch. 128 threads, 128
        // groups of 8 floats: one gqh_load_x per thread.
        {
            float xt[GQH_PER_LANE];
            gqh_load_x(xcol[fill_col], sb, fill_ln * GQH_PER_LANE, xt, xoff[fill_col]);
#pragma unroll
            for (int t = 0; t < GQH_PER_LANE; ++t) {
                s_x[fill_col][t][fill_ln] = xt[t];
            }
        }
        __syncthreads();
        gqh_sched_fence();

        const int sbn = sb + 1 < nsb ? sb + 1 : sb;
        const uint32_t sbn_off = (uint32_t) sbn * (uint32_t) sb_bytes;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            wire[r] = gqh_wire_load<RUNG>(rowbase[r], sbn_off, woff);
        }
        gqh_sched_fence();

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const float s_b = d_real[r] * s_ratio[(rb[r] >> rb_shift) & 0x0f];
            float w[GQH_PER_LANE];
#pragma unroll
            for (int t = 0; t < GQH_PER_LANE; ++t) {
                const int code = IS_GQH4
                    ? ((codes[r] >> (4 * t)) & 0x0f)
                    : (((codes[r] >> (2 * t)) & 0x03) |
                       (IS_GQH3 ? (((hi1[r] >> t) & 1) << 2) : 0));
                w[t] = s_grid[code] * s_b;
            }
#pragma unroll
            for (int k = 0; k < GQH_N8_XCHUNK; ++k) {
                float xs[GQH_PER_LANE];
#pragma unroll
                for (int t = 0; t < GQH_PER_LANE; ++t) {
                    xs[t] = s_x[k][t][lane];
                }
#pragma unroll
                for (int t = 0; t < GQH_PER_LANE; ++t) {
                    acc[r][k] += w[t] * xs[t];
                }
            }
        }
        __syncthreads();

        {
            float xt[GQH_PER_LANE];
            gqh_load_x(xcol[GQH_N8_XCHUNK + fill_col], sb,
                       fill_ln * GQH_PER_LANE, xt, xoff[GQH_N8_XCHUNK + fill_col]);
#pragma unroll
            for (int t = 0; t < GQH_PER_LANE; ++t) {
                s_x[fill_col][t][fill_ln] = xt[t];
            }
        }
        __syncthreads();

#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const float s_b = d_real[r] * s_ratio[(rb[r] >> rb_shift) & 0x0f];
            float w[GQH_PER_LANE];
#pragma unroll
            for (int t = 0; t < GQH_PER_LANE; ++t) {
                const int code = IS_GQH4
                    ? ((codes[r] >> (4 * t)) & 0x0f)
                    : (((codes[r] >> (2 * t)) & 0x03) |
                       (IS_GQH3 ? (((hi1[r] >> t) & 1) << 2) : 0));
                w[t] = s_grid[code] * s_b;
            }
#pragma unroll
            for (int k = 0; k < GQH_N8_XCHUNK; ++k) {
                float xs[GQH_PER_LANE];
#pragma unroll
                for (int t = 0; t < GQH_PER_LANE; ++t) {
                    xs[t] = s_x[k][t][lane];
                }
#pragma unroll
                for (int t = 0; t < GQH_PER_LANE; ++t) {
                    acc[r][GQH_N8_XCHUNK + k] += w[t] * xs[t];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const bool store = row + r < out;
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
#pragma unroll
            for (int off = GQH_WARP / 2; off > 0; off >>= 1) {
                acc[r][c] += gqh_warp_shfl_down(acc[r][c], off);
            }
            if (lane == 0 && store) {
                y_base[(int64_t) c * y_col_stride + row + r] = acc[r][c];
            }
        }
    }
}

template <ggml_type RUNG, bool PAIRED>
static void gqh_n8_tile_launch(
        cudaStream_t stream,
        const uint8_t * a, float * ya, const uint8_t * b, float * yb,
        const float * x, int in, int out,
        float scale_a, gqh_grid16 grid_a, float scale_b, gqh_grid16 grid_b,
        int64_t x_col_stride, int64_t y_col_stride) {
    const int rows_per_block = GQH_N8_TILE_WARPS * GQH_N8_TILE_ROWS;
    const dim3 threads(GQH_WARP * GQH_N8_TILE_WARPS, 1, 1);
    const dim3 blocks((out + rows_per_block - 1) / rows_per_block, PAIRED ? 2 : 1, 1);
    const int nsb_spec = (in == 5120) ? 20 : (in == 17408) ? 68 : 0;
    if (nsb_spec == 20) {
        gqh_matvec_n8_tile_kernel<RUNG, PAIRED, 20><<<blocks, threads, 0, stream>>>(
            a, x, ya, in, out, scale_a, grid_a, x_col_stride, y_col_stride,
            b, yb, scale_b, grid_b);
    } else if (nsb_spec == 68) {
        gqh_matvec_n8_tile_kernel<RUNG, PAIRED, 68><<<blocks, threads, 0, stream>>>(
            a, x, ya, in, out, scale_a, grid_a, x_col_stride, y_col_stride,
            b, yb, scale_b, grid_b);
    } else {
        gqh_matvec_n8_tile_kernel<RUNG, PAIRED, 0><<<blocks, threads, 0, stream>>>(
            a, x, ya, in, out, scale_a, grid_a, x_col_stride, y_col_stride,
            b, yb, scale_b, grid_b);
    }
}

// Warps per block, resolved PER DISPATCH on the int8 ncols == 8 arm.
// GGML_GQH_N8_WARPS forces one value for every dispatch and is the same-binary A/B
// control (=4 reproduces the single-constant behaviour this replaces).
//
// Every warp in a block reads the SAME activations, so `warps` looks like a sharing
// factor -- and it is NOT why this rule exists. Cross-warp activation sharing was
// measured directly (GGML_GQH_N8_XABL, iteration 3: all 8 columns read column 0, i.e.
// activation traffic AND per-wave footprint /8) and is worth 2.0% on the GQH3 pair and
// 0.8% on GQH4 at ROWS == 4. What `warps` buys is block RETIRE/REFILL granularity, and
// like the ROWS axis it is worth measuring per dispatch rather than deriving: on the
// same binary, microbench median us at copies=8, 2 reps (FNV identical on every arm --
// this is launch geometry, every output is bit-identical):
//
//   dispatch                     rows_total   w=1     w=2    w=4     w=8   best
//   GQH4  5120->12288                12288    95.1    94.1   97.5    96.8   2 (-3.2%)
//   GQH3  5120->12288                12288      --    77.6   76.6      --   4 (+1.3%)
//   GQH3  5120->17408 paired         34816   169.8   170.4  171.1   177.9   2 == 4
//   GQH4  5120->10240                10240    92.0    89.9   82.4    83.0   4
//   GQH4  5120->6144                  6144    57.5    57.1   57.0    57.0   4 (flat)
//   GQH4 17408->5120                  5120   108.4   108.1  107.6   107.6   4
//   GQH2_H 5120->5120                 5120    37.8    37.9   37.8    37.8   4 (flat)
//
// So the band, not a threshold, and the band is the smallest claim the data supports:
// out == 12288 is the ONLY dispatch where 2 warps measured better (-3.2%, replicated
// over 4 runs in 2 builds), the 34816 pair is NEUTRAL (171.0 / 171.2 against 171.6 /
// 171.0 -- 4 runs, inside noise), and both dispatches below 12288 that were tried at 2
// LOSE (8.7% on 10240, 0.5% on 5120). Extending the band up to the pair would change the
// geometry of 41% of gqh_matvec for zero measured return, so it does not.
//
// The band is keyed on rows_total alone, so the GQH3 out == 12288 dispatch falls in it
// too, and 2 warps are 1.3% WORSE there (76.6 -> 77.6). It is left in: that dispatch is
// 0.5% of gqh_matvec against the GQH4 shape's 4.0%, so the row costs 0.007% of the 0.13%
// the band buys, and a rung condition would be more surface than the 20:1 it protects.
// Weighted by the rocprofv3 decode shares below, the band is -0.12% of gqh_matvec, i.e.
// +0.07% of HE -- under the rig's resolution, kept because it is free and
// bit-identical, not because it is visible.
//
// The w=4 column is from the FINAL build and the w=1 / w=2 / w=8 columns from a probe
// build that also carried one extra kernarg. That kernarg cost 3.2% on the 34816 pair at
// w=4 alone (176.9 there against 171.1 here) and nothing on any other shape, which is why
// this table looked like a 3.7% win for w=2 on the pair before the probe was reverted.
// Cross-build columns on this arm are not comparable; see the handoff's instrument note.
//
// Gated on `i8`: the f32 exact arm (ctest) and every batch-1 path keep GQH_MATVEC_WARPS,
// so their launch geometry does not move. 2-warp blocks on the ncols == 8 arm at
// ROWS == 6 on the f32 arm were measured and refuted (104.26 vs 104.82 tok/s); this is
// the int8 arm at ROWS == 4, which is a different regime -- see the handoff.
static int gqh_verify_warps(int ncols, int rows_total, bool i8) {
    static const int forced = []() {
        const char * e = getenv("GGML_GQH_N8_WARPS");
        return e ? atoi(e) : 0;
    }();
    if (forced && ncols == 8) {
        return forced;
    }
    if (i8 && ncols == 8 && rows_total >= 12288 && rows_total < 16384) {
        return 2;
    }
    return GQH_MATVEC_WARPS;
}

// Rows per warp for one exact-width width.
template <int NCOLS>
static constexpr int gqh_multicol_rows() {
    return NCOLS >= 9 ? 1 : NCOLS == 8 ? GQH_MULTICOL_ROWS_WIDE : GQH_MULTICOL_ROWS;
}

// Whether the ncols == 8 verify arm still WANTS the SuperSonic kNsb unroll.
//
// kNsb hands the kernel `nsb` as a template parameter so the superblock loop's trip
// count is a literal (20 for in == 5120, 68 for in == 17408). It is a win at the
// narrow widths and at ROWS == 3, and a LOSS on the wide arm once ROWS grows -- the
// unrolled body's register demand starts costing more issue slots than the loop
// overhead it deletes. Loop-body instructions per useful FMA, gfx1201 GQH3 ncols == 8
// (this arm is instruction-ISSUE bound, see GQH_MULTICOL_ROWS_WIDE):
//
//   ROWS      kNsb      runtime nsb
//     3      2.318  <- shipped      2.458   (measured: runtime nsb is 2.0% WORSE on HE)
//     4      2.242                  2.410
//     6      2.305                  2.117
//     8      2.473                  2.105
//
// So the kNsb decision is not independent of ROWS: at ROWS == 3 it pays, at ROWS >= 6
// it costs, so the default below is keyed off GQH_MULTICOL_ROWS_WIDE rather than being
// a constant. Every width except ncols == 8 keeps kNsb exactly as it was.
// GGML_GQH_KNSB_WIDE=0/1 forces the kNsb decision on the wide arm, so ONE binary can
// be A/B'd on this axis -- the discipline the rest of this file is measured under.
static bool gqh_knsb_wide_on() {
    static const int forced = []() {
        const char * e = getenv("GGML_GQH_KNSB_WIDE");
        return e ? atoi(e) : -1;
    }();
    return forced < 0 ? (GQH_MULTICOL_ROWS_WIDE <= 4) : forced != 0;
}

// The compile-time superblock count this dispatch should carry; 0 means "runtime".
template <int NCOLS>
static int gqh_knsb_nsb(int in) {
    if (NCOLS == 8 && !gqh_knsb_wide_on()) {
        return 0;
    }
    return (in == 5120) ? 20 : (in == 17408) ? 68 : 0;
}

// Pair-decoded level table on the ncols == 8 verify arm (see gqh_pair_lut). The decode
// half of that loop is bigger than the FMA half -- 342 of 813 instructions per superblock
// at <GQH3,8,6,paired,runtime> -- and this halves its LDS reads and its index arithmetic.
// GGML_GQH_PAIRLUT=0 is the same-binary control: both instantiation sets stay compiled,
// so the A/B comes out of ONE build, which is the only way to resolve a few percent on a
// rig whose cross-build spread is ~1.5%.
//
// **ON FOR EVERY RUNG, and the per-rung gate this replaces was an artefact of the
// instrument, not of the hardware.** The gate existed because one rocprofv3
// --kernel-trace reading put the gqh3 paired dispatch at 291.6 us with the table off and
// 297.1 us with it on, and a 3-rep same-binary HE A/B could not resolve 0.2-0.6 pp. Both
// instruments were too coarse. The table below is a per-dispatch microbench of the exact
// shapes the HE decode issues (hipEvent, 8 dispatches per window, 20-25 windows, median;
// weights rotated over 8 copies so each dispatch reads them cold past the 64 MB MALL,
// which is the regime the real decode runs in). Its SAME-ARM replicate spread is 0.13%
// and its cross-build control reproduces to 0.05%, so it resolves what the trace could
// not -- and it reproduces the trace's ranking on every shape the trace could rank.
// Median us per dispatch, gfx1201, all three arms from ONE build:
//
//   dispatch                          off   float2   split   split vs shipped
//   GQH3 5120->17408 paired         315.1    308.6   300.5   -4.62%  (was off)
//   GQH4 17408->5120                179.3    174.1   162.2   -6.83%  (was float2)
//   GQH4 5120->10240                113.3    113.3   108.3   -4.41%
//   GQH4 5120->6144                  71.7     71.3    67.3   -5.61%
//   GQH2_H 5120->5120                60.9     59.2    56.0   -5.41%
//
// Two findings. (1) The `float2` layout, not the table, was what held gqh3 back: at a
// 2-dword stride the LDS crossbar can only reach 16 of 32 banks per phase (see
// gqh_pair_lut_lds), and that cost more than the 10.5% instruction cut bought. The
// split-plane layout is faster than BOTH prior arms on every shape. (2) With the banks
// fixed the elasticity story changes too -- the same 10.5% instruction cut that read as
// -1.9% now reads as -4.6%, so the loop was never insensitive to instruction count, it
// was paying for LDS serialisation with the change that removed them.
//
// GGML_GQH_PAIRLUT=0 is the same-binary control: both instantiation sets stay compiled,
// so the A/B comes out of ONE build, which is the only way to resolve a few percent on a
// rig whose cross-build spread is ~1.5%. 0 = off everywhere. 1 = the OLD default
// (gqh4/gqh2_h only), kept reachable so the gate can be re-measured without a rebuild.
// 2 = every rung, the measured default.
static int gqh_pairlut_mode() {
    static const int mode = []() {
        const char * e = getenv("GGML_GQH_PAIRLUT");
        return e ? atoi(e) : 2;
    }();
    return mode;
}

// GGML_GQH_I8DOT: N=8 FMA uses codebook->int8 + v_sudot4 (gfx1201). Default ON.
// GGML_GQH_I8DOT=0 restores the f32 bit-exact arm (ctest). GGML_GQH_F16DOT=1 is
// the closed v_dot2 experiment and only applies when I8DOT is explicitly 0.
static int gqh_n8_dot_mode() {
    static const int mode = []() {
        const char * i4 = getenv("GGML_GQH_I4DOT");
        if (i4 && atoi(i4) != 0) {
            return 3;
        }
        const char * i8 = getenv("GGML_GQH_I8DOT");
        if (i8) {
            return atoi(i8) != 0 ? 2 : 0;
        }
        const char * f16 = getenv("GGML_GQH_F16DOT");
        if (f16 && atoi(f16) != 0) {
            return 1;
        }
        return 2;
    }();
    return mode;
}

// GGML_GQH_I8_XSCALE=1 (default): one activation scale per 8-weight group, SHARED by the
// ncols columns, so it folds into the per-row scale and the loop pays no per-column mul.
// =0: one scale per (column, group), at one extra v_mul per (row, column) per superblock.
// Both are instantiated, so this is a same-binary A/B like GGML_GQH_PAIRLUT -- and it is
// the escape hatch if sharing ever costs `avg_commit`, which is a cliff and not a slope.
static bool gqh_i8_xscale_shared() {
    static const bool shared = []() {
        const char * e = getenv("GGML_GQH_I8_XSCALE");
        return e ? atoi(e) != 0 : true;
    }();
    return shared;
}

template <ggml_type RUNG>
static bool gqh_pairlut_on() {
    const int mode = gqh_pairlut_mode();
    return mode >= 2 || (mode == 1 && RUNG != GGML_TYPE_GQH3);
}

// The int8 arm's activation scratch: codes then scales, one buffer per device, grown on
// demand and never freed -- the decode loop asks for the same two shapes (in 5120 and
// 17408) every step, so this allocates twice per process and then never again.
//
// A pointer-keyed cache would be wrong, not just fragile: ggml's graph allocator reuses
// the same activation buffer across layers, so the same `x` address carries different
// numbers within one forward. The pre-pass runs per dispatch, on the dispatch's own
// stream, ahead of the matvec that reads it.
struct gqh_qx_view {
    const int8_t * q = nullptr;
    const float  * s = nullptr;
};

static gqh_qx_view gqh_quant_x(
        cudaStream_t stream, const float * x, int in, int ncols,
        int64_t x_col_stride, bool shared, bool i4 = false) {
    // ncols == 8 is baked into the kernel's lane -> (group, column) split (`tid & 7`),
    // which is what makes the shared-scale reduction three intra-wave xors. The only
    // caller is the NCOLS == 8 branch of gqh_exact_dispatch; this keeps that contract
    // self-enforcing rather than conventional, like the XSHARED width check.
    // in % GQH_SUPERBLOCK, not % GQH_PER_LANE: the SUPERBLOCK multiple is what makes
    // `ngroups * 8 == in` land on an exact multiple of the 256-thread block, so the
    // kernel's `g >= ngroups` return is never taken and the shared-scale __shfl_xor
    // never reads an inactive lane. The matvec already requires it (nsb = in / 256).
    GGML_ASSERT(ncols == 8 && in % GQH_SUPERBLOCK == 0);
    struct scratch { void * p = nullptr; size_t bytes = 0; };
    static scratch bufs[GGML_CUDA_MAX_DEVICES];

    const int    ngroups = in / GQH_PER_LANE;
    const size_t codes   = (size_t) ncols * in;
    const size_t s_off   = (codes + 255) & ~(size_t) 255;
    const size_t bytes   = s_off + (size_t) ncols * ngroups * sizeof(float);

    scratch & buf = bufs[ggml_cuda_get_device()];
    if (buf.bytes < bytes) {
        if (buf.p) {
            CUDA_CHECK(cudaFree(buf.p));
        }
        CUDA_CHECK(cudaMalloc(&buf.p, bytes));
        buf.bytes = bytes;
    }
    int8_t * q = (int8_t *) buf.p;
    float  * s = (float *) ((char *) buf.p + s_off);

    constexpr int threads = 256;
    const int blocks = (ngroups * ncols + threads - 1) / threads;
    if (shared) {
        if (i4) {
            gqh_quant_x_kernel<true, true><<<blocks, threads, 0, stream>>>(
                x, ngroups, x_col_stride, q, s, in);
        } else {
            gqh_quant_x_kernel<true><<<blocks, threads, 0, stream>>>(
                x, ngroups, x_col_stride, q, s, in);
        }
    } else {
        if (i4) {
            gqh_quant_x_kernel<false, true><<<blocks, threads, 0, stream>>>(
                x, ngroups, x_col_stride, q, s, in);
        } else {
            gqh_quant_x_kernel<false><<<blocks, threads, 0, stream>>>(
                x, ngroups, x_col_stride, q, s, in);
        }
    }
    return { q, s };
}

// GGML_GQH_FUSE_GLU=0 forces the unfused {swiglu dispatch, pre-pass dispatch} pair, so the
// fusion is a same-binary A/B against the constant it replaces (the rig's build-to-build
// step is ~0.63%, an order of magnitude above its run-to-run scatter, so a compile-time
// knob could not resolve a ~1% effect at all).
static bool gqh_fuse_glu_on() {
    static const bool on = []() {
        const char * e = getenv("GGML_GQH_FUSE_GLU");
        return e ? atoi(e) != 0 : true;
    }();
    return on;
}

// The one-shot handoff from the fused GLU to the down-projection's matvec.
//
// SINGLE SLOT, READ-AND-CLEAR, and both properties are load-bearing. Iteration 5 refused a
// pointer-keyed pre-pass cache for the right reason: ggml's graph allocator reuses activation
// buffers across layers, so an address alone does not identify a tensor's CONTENTS. What makes
// this safe is the lifetime, not the key: the slot is cleared by the very next ncols == 8 GQH
// dispatch whether it hits or not, and the glu tensor is an INPUT to that dispatch, so ggml's
// allocator cannot have handed its buffer to anything else in between -- a live tensor's
// storage is not reusable while it is live. A stale hit would need the immediately-following
// GQH dispatch to read the same address with the same in / stride / ncols / scale mode and
// not be the consumer that kept it alive, which the graph order (gate/up -> glu -> down)
// does not admit. The generation counter is also what makes the paths that return BEFORE
// the take (gqh_n8_tile_on(), the ncols != 8 widths, the type/shape rejections at the entry
// points) safe rather than holes: they bump the generation, so the slot becomes untakeable
// instead of surviving to a later dispatch.
struct gqh_qx_memo {
    const float  * x            = nullptr;
    const int8_t * q            = nullptr;
    const float  * s            = nullptr;
    int            in           = 0;
    int            ncols        = 0;
    int64_t        x_col_stride = 0;
    uint64_t       gen          = 0;
    bool           shared       = false;
    bool           valid        = false;
};

static gqh_qx_memo g_gqh_qx_memo[GGML_CUDA_MAX_DEVICES];

// Bumped by every GQH matvec entry point, of every width, so "the next GQH dispatch" is a
// fact and not a graph-order assumption: a memo written at generation N is only takeable at
// generation N + 1. Without it the slot could survive an intervening ncols == 1 dispatch.
static uint64_t g_gqh_dispatch_gen = 0;

// Take the memo if it describes exactly this dispatch's activations; clear it either way.
static gqh_qx_view gqh_qx_memo_take(
        const float * x, int in, int ncols, int64_t x_col_stride, bool shared) {
    gqh_qx_memo & m = g_gqh_qx_memo[ggml_cuda_get_device()];
    const bool hit = m.valid && m.x == x && m.in == in && m.ncols == ncols
        && m.x_col_stride == x_col_stride && m.shared == shared
        && m.gen + 1 == g_gqh_dispatch_gen;
    const gqh_qx_view v = hit ? gqh_qx_view{ m.q, m.s } : gqh_qx_view{};
    m = gqh_qx_memo{};
    return v;
}

// Fused SwiGLU + down-projection pre-pass. Returns false (caller keeps ggml's swiglu) unless
// every shape precondition the fused kernel's addressing assumes actually holds.
bool ggml_cuda_gqh_glu_quant(
        const float * gate, const float * up, float * glu, int nc, int ncols,
        int64_t gate_col_stride, int64_t up_col_stride, int64_t glu_col_stride,
        cudaStream_t stream) {
    // ncols == 8 is the pre-pass's lane -> (group, column) split; nc % GQH_SUPERBLOCK is
    // what makes ngroups * 8 an exact multiple of the 256-thread block, so the kernel's
    // `g >= ngroups` return is never taken and the shared-scale shfl never reads an
    // inactive lane. The float4 addressing needs 16-byte alignment on all three strides.
    if (!gqh_fuse_glu_on() || gqh_n8_dot_mode() != 2 || ncols != 8
            || nc % GQH_SUPERBLOCK != 0
            || gate_col_stride % 4 != 0 || up_col_stride % 4 != 0
            || glu_col_stride % 4 != 0) {
        return false;
    }
    struct scratch { void * p = nullptr; size_t bytes = 0; };
    static scratch bufs[GGML_CUDA_MAX_DEVICES];

    const int    ngroups = nc / GQH_PER_LANE;
    const size_t codes   = (size_t) ncols * nc;
    const size_t s_off   = (codes + 255) & ~(size_t) 255;
    const size_t bytes   = s_off + (size_t) ncols * ngroups * sizeof(float);

    const int dev = ggml_cuda_get_device();
    scratch & buf = bufs[dev];
    if (buf.bytes < bytes) {
        if (buf.p) {
            CUDA_CHECK(cudaFree(buf.p));
        }
        CUDA_CHECK(cudaMalloc(&buf.p, bytes));
        buf.bytes = bytes;
    }
    // Its OWN buffer, not gqh_quant_x's: the two are then never live at the same time on
    // the same bytes, so a missed memo costs a redundant pre-pass and never a wrong read.
    int8_t * q = (int8_t *) buf.p;
    float  * s = (float *) ((char *) buf.p + s_off);

    constexpr int threads = 256;
    const int blocks = (ngroups * ncols + threads - 1) / threads;
    const bool shared = gqh_i8_xscale_shared();
    if (shared) {
        gqh_glu_quant_kernel<true, true><<<blocks, threads, 0, stream>>>(
            gate, up, glu, ngroups, gate_col_stride, up_col_stride, glu_col_stride,
            q, s, nc);
    } else {
        gqh_glu_quant_kernel<false, true><<<blocks, threads, 0, stream>>>(
            gate, up, glu, ngroups, gate_col_stride, up_col_stride, glu_col_stride,
            q, s, nc);
    }
    g_gqh_qx_memo[dev] =
        { glu, q, s, nc, ncols, glu_col_stride, g_gqh_dispatch_gen, shared, true };
    return true;
}

// The int8 arm costs one extra dispatch (the pre-pass) and buys ~25% of the matvec's
// instructions, so it needs a shape floor: below it the launch is a bigger number than
// the saving. Gate on the row total the launch covers, the same term gqh_n8_rows uses.
// Measured floor is in the handoff; 5120 -> 1024 (2.0% of matvec time, ~29 us) is the
// one live shape below it.
//
// 16 Mi is now CONSERVATIVE and deliberately left so. Once the int8 arm takes ROWS == 4
// it beats the f32 fallback on 5120 -> 1024 as well -- 25.36 us against 29.37, 1.16x,
// pre-pass launch included -- so the floor could drop to 4 Mi. It does not, because that
// dispatch is 2.0% of gqh_matvec and gqh_matvec is ~30% of the decode wall, so the whole
// prize is ~0.08% of HE, an order of magnitude under the run-to-run noise, and the price
// is turning the one remaining bit-exact N=8 dispatch into an approximate one. Revisit
// only if `avg_commit` headroom is being spent on something that pays.
static bool gqh_i8_shape_ok(int in, int rows_total) {
    return (int64_t) in * rows_total >= 16 * 1024 * 1024;
}

// Rows per warp on the ncols == 8 verify arm, resolved PER DISPATCH out of the
// instantiated set {3, 4, GQH_MULTICOL_ROWS_WIDE, 8}. GGML_GQH_N8_ROWS forces one
// value for every dispatch and is the same-binary A/B control (=6 reproduces the
// single-constant behaviour this replaces).
//
// One constant cannot fit this arm, because ROWS trades two things against each other
// and the exchange rate is set by `out`:
//   - instructions per useful FMA falls with ROWS (the ~164 instructions a warp pays
//     ONCE per superblock -- 16 b128 activation loads for the 8 columns, their waits,
//     the loop SALU -- amortise over more rows), and
//   - the wave count is out / (4 * ROWS), so ROWS divides the parallelism.
// On `out == 17408` (the GQH3 gate/up pair) there are waves to spare and the
// instruction count wins. On `out == 5120` (the GQH4 down projection) ROWS == 6 leaves
// only 856 waves for ~1280 slots, so a third of the machine sits idle for the whole
// dispatch and the instruction count is not what the dispatch is waiting on.
//
// Measured on the R9700, microbench median us at copies=8, 2 reps each, ONE binary
// (GGML_GQH_N8_ROWS selects the arm), so this table carries no cross-build offset. Every
// cell is a REAL decode dispatch; the shares are the rocprofv3 kernel trace's:
//
//   dispatch                        share    R=3     R=4     R=5     R=6     R=8   best
//   GQH3  5120->17408 paired        41.6%   308.9   299.2   297.5   300.2  *296.4*  8
//   GQH4 17408->5120                        213.5  *154.9*  178.4   162.2   170.8   4
//   GQH2_H 5120->5120                       54.4   *51.5*   56.6    56.2    57.8   4
//     (those two share out == 5120, so the trace merges them at 34.0%)
//   GQH4  5120->10240               11.0%   113.5  *105.5*  120.3   107.9   120.5   4
//   GQH4  5120->6144                 7.4%    79.0   *64.1*   75.3    67.2    71.6   4
//   GQH4  5120->12288                4.1%   122.3   121.7    --    *120.0*  127.3   6
//   * 5120->1024                     2.0%  *22.3*   29.1     --     35.7    43.8    3
//
// Same-arm replicate spread was <= 0.4% everywhere except the two slowest GQH2_H arms
// (1.5% / 1.25%), against an 8.3% effect there. Weighted by the shares, picking the
// per-shape best is -3.2% of gqh_matvec time against the single ROWS == 6 constant; the
// rule below, re-measured as a whole against the GGML_GQH_N8_ROWS=6 control in ONE
// binary, lands -3.4% there and **+1.47% on 10-prompt HumanEval** (paired deltas
// +0.97 / +1.67 / +1.78 over 3 interleaved reps, mean avg_commit 7.440 on both arms).
//
// The instantiated set is {3, 4, GQH_MULTICOL_ROWS_WIDE, 8} and BOTH ends of the axis
// are refuted, so do not widen it: ROWS == 12 spills (192 VGPRs + 12 B/lane of scratch
// on GQH3, the same failure mode as the closed nsb == 68 unroll), ROWS == 10 is
// scratch-free but measured +39% on the paired 17408 shape (its 872 blocks are 3.03
// rounds of wave supply, which quantises up to 4) against -1.9% on out == 10240 -- one
// small win on an 11% dispatch is not worth a third regime -- and ROWS == 5 loses to
// both 4 and 6 on every shape below AND is the only value whose kNsb instantiations
// spill (16-52 B/lane; the dispatched NSB == 0 arm does not, so the spill is not the
// explanation for its measured loss -- two separate facts).
//
// ROWS is NOT monotone -- 5 is worse than both 4 and 6 on every GQH4 shape -- so this
// is a measured table, not a fitted curve. The reason is that ROWS moves TWO quantised
// things at once: VGPRs (GQH3 100 / 103 / 128 / 137 / 166 at R = 3 / 4 / 5 / 6 / 8, i.e.
// occupancy 12 / 12 / 10 / 10 / 9) and the wave count. R == 4 wins the small-`out`
// shapes because it is the largest ROWS that still keeps 12 waves/SIMD AND, at
// out == 5120, the smallest ROWS whose 1280 waves still fit ONE round: R == 6 leaves
// 856 waves in ~1280 slots (a third of the machine idle for the whole dispatch) and
// R == 3 spills over into a second round at 1708.
//
// Every ROWS value produces a bit-identical output (each row is its own sum, folded in
// the same term order; the microbench output FNV is unchanged across the whole sweep,
// on all five shapes), so this is purely a launch-geometry choice.
static int gqh_n8_rows_forced() {
    static const int rows = []() {
        const char * e = getenv("GGML_GQH_N8_ROWS");
        const int v = e ? atoi(e) : 0;
        return (v == 3 || v == 4 || v == 8 || v == GQH_MULTICOL_ROWS_WIDE)
            ? v : 0;
    }();
    return rows;
}

// The per-dispatch choice. The shape term that matters is the TOTAL number of output
// rows the launch covers -- `out`, doubled when the pair rides one dispatch as grid.y ==
// 2 -- because that is what sets the wave count. `in` cancels: it multiplies every
// candidate's per-wave work equally.
//
// TWO regimes, deliberately, and the table above shows what each end gives up. At
// R == 8 a launch covers 32 rows per block and issues 4 waves, so `rows_total / 8`
// waves against ~1152 resident: 16384 is where R == 8 first has ~1.8 rounds of wave
// supply to hide behind its lower instruction count.
//
// The two shapes this rule knowingly leaves money on the table for are both small and
// both measured, not assumed:
//   - out == 12288 (4.1%) wants 6, and gets 4: +1.0% on that dispatch, i.e. +0.04% of
//     gqh_matvec time. A third regime is not worth 0.04%.
//   - out == 1024 (2.0%) wants the SMALLEST ROWS available: its time is nearly
//     proportional to ROWS (22.3 / 29.1 / 35.7 / 43.8 us at R = 3/4/6/8, i.e. ~7 us per
//     row) because 1024 rows is 128..344 waves however you slice it -- one round, way
//     under the ~1536 slots, so per-wave work IS the dispatch. The rule's R == 4 is
//     already -21% against the old constant 6 there; R == 3 would be -39%, and a ROWS
//     of 1 or 2 (not instantiated at ncols == 8) is the real answer. Worth ~-1% of
//     gqh_matvec time -- the largest single item left on this axis. See the handoff.
//
// THE INT8 ARM DOES NOT SHARE THIS TABLE, and the reason is the VGPR cliff, not the
// instruction count. Everything above was measured on the f32 arm, where the ROWS axis is
// nearly flat (296.4 .. 308.9 us across R = 3..8 on the paired 17408 shape, a 4% spread)
// because that arm's ~1163-instruction trip is long enough to hide a memory latency at 8
// waves/SIMD. The int8 trip is 687 instructions and hides much less, so on that arm ROWS
// stops being an instructions-per-FMA knob and becomes an OCCUPANCY knob:
//
//   ROWS  VGPRs  waves/SIMD      because acc[ROWS][8] alone is 8 * ROWS registers
//     3      84          16
//     4      93          16      <- the largest ROWS that still clears the 16-wave ceiling
//     6     128          10
//     8     163           9
//
// Measured on the int8 arm, microbench median us at copies=8, ONE binary
// (GGML_GQH_N8_ROWS selects), shares from the rocprofv3 kernel trace:
//
//   dispatch                     share      R3      R4      R6      R8
//   GQH3  5120->17408 paired     41.6%   179.1  *171.7*  186.7   193.5
//   GQH4 17408->5120             25.2%   111.9  *107.5*  124.9   120.2
//   GQH2_H 5120->5120             8.8%    38.2   *37.9*   41.2    42.2
//   GQH4  5120->10240            11.0%    86.4    82.7    97.1   *79.9*
//   GQH4  5120->6144              7.4%    62.5   *57.1*   64.5    62.3
//   GQH4  5120->12288             4.1%   *87.3*   96.9   101.5    99.1
//
// R == 4 wins four of six outright and every dispatch that matters. The f32 rule sends
// the biggest one (rows_total 34816) to R == 8 and pays 11.3% for it: replicated over 3
// interleaved reps at 20 windows each, R8 193.47 / 193.43 / 193.97 against R4 171.59 /
// 171.92 / 171.65, output FNV identical on every arm.
//
// So: one constant, not a table. The two shapes that want something else are worth
// 0.11 * 3.4% + 0.041 * 9.9% = 0.8% of gqh_matvec between them, which does not pay for a
// third and fourth regime on an axis this noisy -- and R == 6 and R == 8 are refuted
// everywhere else, so a wrong guess there is expensive. Every ROWS value produces a
// bit-identical output (each row is its own sum in its own term order), so this is purely
// a launch-geometry choice; GGML_GQH_N8_ROWS still forces one value for the A/B.
static int gqh_n8_rows(int out, bool paired, bool i8) {
    const int forced = gqh_n8_rows_forced();
    if (forced) {
        return forced;
    }
    if (i8) {
        return 4;
    }
    const int rows_total = paired ? 2 * out : out;
    return rows_total >= 16384 ? 8 : 4;
}

// kNsb / ROWS / grid for ONE exact-width instantiation. Unpaired callers pass the
// same tensor twice (as they always did) and get a grid.y == 1 launch; PAIRED gets
// grid.y == 2 and blockIdx.y selects gate vs up inside the kernel.
template <ggml_type RUNG, int NCOLS, int ROWS, bool PAIRED, bool PAIRLUT = false,
          bool F16DOT = false, bool I8DOT = false, bool XSSHARED = true,
          bool I4DOT = false>
static void gqh_exact_launch(
        cudaStream_t stream,
        const uint8_t * a, float * ya, const uint8_t * b, float * yb,
        const float * x, int in, int out,
        float scale_a, gqh_grid16 grid_a, float scale_b, gqh_grid16 grid_b,
        int64_t x_col_stride, int64_t y_col_stride, gqh_qx_view qxv = {}) {
    const int warps = gqh_verify_warps(NCOLS, PAIRED ? 2 * out : out, I8DOT || I4DOT);
    const int rows_per_block = warps * ROWS;
    const dim3 threads(GQH_WARP * warps, 1, 1);
    const dim3 blocks((out + rows_per_block - 1) / rows_per_block, PAIRED ? 2 : 1, 1);
    const int nsb_spec = gqh_knsb_nsb<NCOLS>(in);
    if (nsb_spec == 20) {
        gqh_matvec_kernel<RUNG, NCOLS, ROWS, PAIRED, 20, PAIRLUT, F16DOT, I8DOT, XSSHARED, I4DOT><<<blocks, threads, 0, stream>>>(
            a, x, ya, in, out, NCOLS, scale_a, grid_a,
            x_col_stride, y_col_stride, b, yb, scale_b, grid_b, qxv.q, qxv.s);
    } else if (nsb_spec == 68) {
        gqh_matvec_kernel<RUNG, NCOLS, ROWS, PAIRED, 68, PAIRLUT, F16DOT, I8DOT, XSSHARED, I4DOT><<<blocks, threads, 0, stream>>>(
            a, x, ya, in, out, NCOLS, scale_a, grid_a,
            x_col_stride, y_col_stride, b, yb, scale_b, grid_b, qxv.q, qxv.s);
    } else {
        gqh_matvec_kernel<RUNG, NCOLS, ROWS, PAIRED, 0, PAIRLUT, F16DOT, I8DOT, XSSHARED, I4DOT><<<blocks, threads, 0, stream>>>(
            a, x, ya, in, out, NCOLS, scale_a, grid_a,
            x_col_stride, y_col_stride, b, yb, scale_b, grid_b, qxv.q, qxv.s);
    }
}

// Binds the runtime pair-table choice to the kernel's PAIRLUT template parameter, so
// the ROWS switch below stays one line per case. Both instantiation sets are still
// compiled for every ROWS, which is what keeps GGML_GQH_PAIRLUT a same-binary control.
template <ggml_type RUNG, int NCOLS, int ROWS, bool PAIRED>
static void gqh_n8_launch(
        bool lut, int dot, cudaStream_t stream,
        const uint8_t * a, float * ya, const uint8_t * b, float * yb,
        const float * x, int in, int out,
        float scale_a, gqh_grid16 grid_a, float scale_b, gqh_grid16 grid_b,
        int64_t x_col_stride, int64_t y_col_stride, gqh_qx_view qxv = {}) {
    // dot: 0=f32, 1=f16 v_dot2, 2=i8 dp4a, 3=i4 dot8.
    const bool f16 = dot == 1;
    const bool i8  = dot == 2;
    const bool i4  = dot == 3;
    // The i8 arm carries its OWN level table, so PAIRLUT is dead for it and only the
    // x-scale layout forks -- two instantiations, not four.
    if (i8) {
        if (gqh_i8_xscale_shared()) {
            gqh_exact_launch<RUNG, NCOLS, ROWS, PAIRED, true, false, true, true>(
                stream, a, ya, b, yb, x, in, out, scale_a, grid_a, scale_b, grid_b,
                x_col_stride, y_col_stride, qxv);
        } else {
            gqh_exact_launch<RUNG, NCOLS, ROWS, PAIRED, true, false, true, false>(
                stream, a, ya, b, yb, x, in, out, scale_a, grid_a, scale_b, grid_b,
                x_col_stride, y_col_stride, qxv);
        }
        return;
    }
    if (i4) {
        gqh_exact_launch<RUNG, NCOLS, ROWS, PAIRED, true, false, false, true, true>(
            stream, a, ya, b, yb, x, in, out, scale_a, grid_a, scale_b, grid_b,
            x_col_stride, y_col_stride, qxv);
        return;
    }
    if (lut) {
        if (f16) {
            gqh_exact_launch<RUNG, NCOLS, ROWS, PAIRED, true, true, false>(
                stream, a, ya, b, yb, x, in, out, scale_a, grid_a, scale_b, grid_b,
                x_col_stride, y_col_stride);
        } else {
            gqh_exact_launch<RUNG, NCOLS, ROWS, PAIRED, true, false, false>(
                stream, a, ya, b, yb, x, in, out, scale_a, grid_a, scale_b, grid_b,
                x_col_stride, y_col_stride);
        }
    } else {
        if (f16) {
            gqh_exact_launch<RUNG, NCOLS, ROWS, PAIRED, false, true, false>(
                stream, a, ya, b, yb, x, in, out, scale_a, grid_a, scale_b, grid_b,
                x_col_stride, y_col_stride);
        } else {
            gqh_exact_launch<RUNG, NCOLS, ROWS, PAIRED, false, false, false>(
                stream, a, ya, b, yb, x, in, out, scale_a, grid_a, scale_b, grid_b,
                x_col_stride, y_col_stride);
        }
    }
}

// One exact-width dispatch, paired or not. The ncols == 8 arm (the DFlash2 native
// verify width, and the only width this model's spec-decode actually issues) carries
// its rows-per-warp as a RUNTIME choice so one binary can A/B the whole ROWS axis --
// see GQH_MULTICOL_ROWS_WIDE. Every other width is compile-time constant, unchanged.
template <ggml_type RUNG, int NCOLS, bool PAIRED>
static void gqh_exact_dispatch(
        cudaStream_t stream,
        const uint8_t * a, float * ya, const uint8_t * b, float * yb,
        const float * x, int in, int out,
        float scale_a, gqh_grid16 grid_a, float scale_b, gqh_grid16 grid_b,
        int64_t x_col_stride, int64_t y_col_stride) {
    if constexpr (NCOLS == 8) {
        if (gqh_n8_tile_on()) {
            gqh_n8_tile_launch<RUNG, PAIRED>(
                stream, a, ya, b, yb, x, in, out,
                scale_a, grid_a, scale_b, grid_b, x_col_stride, y_col_stride);
            return;
        }
        const bool lut = gqh_pairlut_on<RUNG>();
        int  dot = gqh_n8_dot_mode();
        if (dot == 3 && RUNG != GGML_TYPE_GQH3) {
            dot = 2;
        }
        // The int8 arm's one pre-pass, ahead of the matvec on the same stream, shared by
        // both halves of a PAIRED dispatch (gate and up read the same x). Below the shape
        // floor the extra launch costs more than the arm saves, so the dispatch falls
        // back to the f32 instantiation -- which is compiled either way.
        gqh_qx_view qxv;
        if (dot == 2) {
            if (gqh_i8_shape_ok(in, PAIRED ? 2 * out : out) || getenv("GGML_GQH_I8_FORCE")) {
                // Read-and-clear FIRST, unconditionally: the memo's safety comes from
                // living exactly one ncols == 8 GQH dispatch (see gqh_qx_memo), so it
                // has to be cleared even on a miss and even on the fallback below.
                const bool shared = gqh_i8_xscale_shared();
                qxv = gqh_qx_memo_take(x, in, NCOLS, x_col_stride, shared);
                if (!qxv.q) {
                    qxv = gqh_quant_x(stream, x, in, NCOLS, x_col_stride, shared);
                }
            } else {
                gqh_qx_memo_take(x, in, NCOLS, x_col_stride, gqh_i8_xscale_shared());
                dot = 0;
            }
        } else if (dot == 3) {
            // Explicit probe mode bypasses the production i8 profitability floor so
            // the small reference vectors exercise the real dot8 kernel.
            qxv = gqh_quant_x(stream, x, in, NCOLS, x_col_stride,
                              gqh_i8_xscale_shared(), true);
            gqh_qx_memo_take(x, in, NCOLS, x_col_stride, gqh_i8_xscale_shared());
        } else {
            gqh_qx_memo_take(x, in, NCOLS, x_col_stride, gqh_i8_xscale_shared());
        }
        // `dot == 2` after the shape-floor check above, so a dispatch that fell back to
        // the f32 instantiation also falls back to the f32 ROWS rule -- the two choices
        // have to agree, since it is the int8 arm's register footprint that wants R == 4.
        switch (gqh_n8_rows(out, PAIRED, dot == 2 || dot == 3)) {
            case 3:
                gqh_n8_launch<RUNG, NCOLS, 3, PAIRED>(
                    lut, dot, stream, a, ya, b, yb, x, in, out, scale_a, grid_a, scale_b,
                    grid_b, x_col_stride, y_col_stride, qxv);
                return;
            case 4:
                gqh_n8_launch<RUNG, NCOLS, 4, PAIRED>(
                    lut, dot, stream, a, ya, b, yb, x, in, out, scale_a, grid_a, scale_b,
                    grid_b, x_col_stride, y_col_stride, qxv);
                return;
            case 8:
                gqh_n8_launch<RUNG, NCOLS, 8, PAIRED>(
                    lut, dot, stream, a, ya, b, yb, x, in, out, scale_a, grid_a, scale_b,
                    grid_b, x_col_stride, y_col_stride, qxv);
                return;
            default:
                gqh_n8_launch<RUNG, NCOLS, GQH_MULTICOL_ROWS_WIDE, PAIRED>(
                    lut, dot, stream, a, ya, b, yb, x, in, out, scale_a, grid_a, scale_b,
                    grid_b, x_col_stride, y_col_stride, qxv);
                return;
        }
    } else {
        gqh_exact_launch<RUNG, NCOLS, gqh_multicol_rows<NCOLS>(), PAIRED>(
            stream, a, ya, b, yb, x, in, out, scale_a, grid_a, scale_b, grid_b,
            x_col_stride, y_col_stride);
    }
}

// Binds the runtime `ncols` to the kernel's compile-time NCOLS_MAX. Decode is
// ncols == 1 and is where all the time goes, so it gets its own instantiation with
// the column guards folded away and gqh_rows1_selected()'s choice of rows per warp;
// everything wider keeps the generic one-row-per-warp instantiation unchanged. The
// grid follows: a block covers warps * ROWS output rows.
template <ggml_type RUNG>
static void gqh_matvec_launch(
        const dim3 & threads, cudaStream_t stream,
        const uint8_t * data, const float * x, float * y, int in, int out, int ncols,
        float tensor_scale, gqh_grid16 grid,
        int64_t x_col_stride, int64_t y_col_stride) {
    if (ncols == 1) {
        const int warps = gqh_decode_warps(RUNG, in, out, ncols);
        const dim3 t(GQH_WARP * warps, 1, 1);
        // if constexpr, not a plain if: the selector is GQH4-only, and this keeps
        // <GQH3, 1, 1> / <GQH2_H, 1, 1> from being instantiated at all rather than
        // emitted and never dispatched.
        if constexpr (RUNG == GGML_TYPE_GQH4) {
            if (gqh_rows1_selected(out)) {
                const dim3 blocks((out + warps - 1) / warps, 1, 1);
                gqh_matvec_kernel<RUNG, 1, 1, false><<<blocks, t, 0, stream>>>(
                    data, x, y, in, out, ncols, tensor_scale, grid,
                    x_col_stride, y_col_stride, data, y, tensor_scale, grid);
                return;
            }
        }
        const int rows_per_block = warps * GQH_MATVEC_ROWS;
        const dim3 blocks((out + rows_per_block - 1) / rows_per_block, 1, 1);
        gqh_matvec_kernel<RUNG, 1, GQH_MATVEC_ROWS, false>
            <<<blocks, t, 0, stream>>>(
                data, x, y, in, out, ncols, tensor_scale, grid,
                x_col_stride, y_col_stride, data, y, tensor_scale, grid);
    } else {
        // Exact-width arm for spec-decode verify (DFlash2 block 8 => ncols 8).
        // The switch is exhaustive over 2..GQH_MULTICOL_SPEC_MAX; anything wider
        // keeps the generic instantiation below.
        static_assert(GQH_MULTICOL_SPEC_MAX == 12, "add or remove a case below to match");
        if (gqh_multicol_on() && ncols <= GQH_MULTICOL_SPEC_MAX) {
            switch (ncols) {
                case 2:
                    gqh_exact_dispatch<RUNG, 2, false>(
                        stream, data, y, data, y, x, in, out,
                        tensor_scale, grid, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                case 3:
                    gqh_exact_dispatch<RUNG, 3, false>(
                        stream, data, y, data, y, x, in, out,
                        tensor_scale, grid, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                case 4:
                    gqh_exact_dispatch<RUNG, 4, false>(
                        stream, data, y, data, y, x, in, out,
                        tensor_scale, grid, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                case 5:
                    gqh_exact_dispatch<RUNG, 5, false>(
                        stream, data, y, data, y, x, in, out,
                        tensor_scale, grid, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                case 6:
                    gqh_exact_dispatch<RUNG, 6, false>(
                        stream, data, y, data, y, x, in, out,
                        tensor_scale, grid, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                case 7:
                    gqh_exact_dispatch<RUNG, 7, false>(
                        stream, data, y, data, y, x, in, out,
                        tensor_scale, grid, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                case 8:
                    gqh_exact_dispatch<RUNG, 8, false>(
                        stream, data, y, data, y, x, in, out,
                        tensor_scale, grid, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                case 9:
                    gqh_exact_dispatch<RUNG, 9, false>(
                        stream, data, y, data, y, x, in, out,
                        tensor_scale, grid, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                case 10:
                    gqh_exact_dispatch<RUNG, 10, false>(
                        stream, data, y, data, y, x, in, out,
                        tensor_scale, grid, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                case 11:
                    gqh_exact_dispatch<RUNG, 11, false>(
                        stream, data, y, data, y, x, in, out,
                        tensor_scale, grid, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                case 12:
                    gqh_exact_dispatch<RUNG, 12, false>(
                        stream, data, y, data, y, x, in, out,
                        tensor_scale, grid, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                default:
                    break;      // ncols == 1 is handled above; nothing else can land here
            }
        }
        const dim3 blocks((out + GQH_MATVEC_WARPS - 1) / GQH_MATVEC_WARPS, 1, 1);
        gqh_matvec_kernel<RUNG, GQH_MAX_COLS, 1, false>
            <<<blocks, threads, 0, stream>>>(
                data, x, y, in, out, ncols, tensor_scale, grid,
                x_col_stride, y_col_stride, data, y, tensor_scale, grid);
    }
}

bool ggml_cuda_gqh_mul_mat_vec(
        ggml_type type, const void * vx, const float * x, float * y,
        int in, int out, int ncols, int64_t x_col_stride, int64_t y_col_stride,
        cudaStream_t stream) {
    ++g_gqh_dispatch_gen;   // see gqh_qx_memo: one GQH dispatch is the memo's whole life
    if (type != GGML_TYPE_GQH3 && type != GGML_TYPE_GQH2_H && type != GGML_TYPE_GQH2_C &&
        type != GGML_TYPE_GQH4) {
        return false;
    }
    if (in % GQH_SUPERBLOCK != 0 || ncols <= 0 || ncols > GQH_MAX_COLS) {
        return false;   // wider batches keep the dequant->GEMM path
    }
    // The kernel addresses both the wire and the activations as a uniform base plus a
    // 32-bit byte offset (that is what lets the loads take their SADDR form). The
    // largest such offset is in*sizeof(float) for the activations and (in/256)*137 for
    // the wire, so a row of 2^24 elements leaves better than 2 orders of magnitude of
    // headroom over any real matvec. Bail rather than wrap.
    if (in > (1 << 24)) {
        return false;
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
    if (!ggml_gqh_lookup(vx, &scale, &code)) {
        return false;   // unregistered -> caller keeps the dequant fallback
    }

    // The whole signed level grid, out of the same per-rung tables the dequant kernels
    // read, so the two decoders cannot drift. The kernel stages the first 16 / 8 / 4
    // entries into LDS and gathers by code; one gqh_grid16 covers every rung.
    gqh_grid16 grid{};
    if (type == GGML_TYPE_GQH3) {
        memcpy(grid.v, GQH3_GRID[code], 8 * sizeof(float));
    } else if (type == GGML_TYPE_GQH4) {
        memcpy(grid.v, GQH4_GRID[code], 16 * sizeof(float));
    } else {
        memcpy(grid.v, GQH2H_GRID[code], 4 * sizeof(float));
    }

    const dim3 threads(GQH_WARP * GQH_MATVEC_WARPS, 1, 1);
    switch (type) {
        case GGML_TYPE_GQH3:
            gqh_matvec_launch<GGML_TYPE_GQH3>(
                threads, stream, (const uint8_t *) vx, x, y, in, out, ncols,
                scale, grid, x_col_stride, y_col_stride);
            break;
        case GGML_TYPE_GQH4:
            gqh_matvec_launch<GGML_TYPE_GQH4>(
                threads, stream, (const uint8_t *) vx, x, y, in, out, ncols,
                scale, grid, x_col_stride, y_col_stride);
            break;
        default:
            gqh_matvec_launch<GGML_TYPE_GQH2_H>(
                threads, stream, (const uint8_t *) vx, x, y, in, out, ncols,
                scale, grid, x_col_stride, y_col_stride);
            break;
    }
    return true;
}

// Two same-shaped GQH4 weight tensors that share `x`, in ONE dispatch.
//
// WHY THIS EXISTS. Iteration 10 fitted every GQH4 bucket to
// `T = C + q * bytes / 630 GB/s`, where q is occupancy-round quantisation
// (ceil(waves/resident) / (waves/resident)) and C is a per-DISPATCH fixed cost of
// 4-6 us. 630 GB/s is 98% of the R9700's 640 GB/s datasheet peak, so the steady-state
// loop is finished -- every remaining microsecond is C or q, and both are properties of
// the DISPATCH, not of the loop body. This is the one change that attacks both at once,
// on the biggest bucket in the model:
//
//   ffn gate and ffn up are 126 of the 298 GQH4 dispatches per N=1 forward (10.64 ms,
//   47% of all matvec time). Both are out == 17408, in == 5120, and they read the SAME
//   activation. 17408 rows at one row per warp is 8.5 occupancy rounds against 2048
//   resident waves -- a fractional round, the only fractional one on the ROWS == 1 arm
//   (out == 10240 and 12288 are exactly 5.0 and 6.0). Folding the pair into one
//   dispatch of 2 x 4352 blocks makes it 34816 waves = 17.0 rounds EXACTLY, and pays C
//   once for the layer instead of twice.
//
// Do NOT try to buy the same thing by cutting occupancy so the wave count divides
// evenly (16 KB of LDS caps residency at 1024 waves and makes 17408 a whole 17 rounds):
// measured, the whole-round control buckets out == 10240 / 12288 lose 10% of their
// bandwidth to the halved residency, which swamps the 5.9% the quantisation is worth.
//
// Returns false unless BOTH tensors are registered and the same rung
// (gqh3/gqh4/gqh2_h). Mixed-rung pairs keep two singles -- the PAIRED kernel
// decodes both halves with one RUNG. The two halves may carry DIFFERENT level
// tables; 26% of GQH4 pairs do.
//
// Launch geometry follows the unpaired path, not a single ROWS value:
// GQH4 out==17408 takes ROWS==1 (the only fractional occupancy-round bucket);
// GQH3/GQH2_H singles never enter that arm, so pairing them at ROWS==1 was a
// measured ~1% AR regression on Q3KXL (all 65 gate/up layers are GQH3).
bool ggml_cuda_gqh_mul_mat_vec_pair(
        ggml_type type,
        const void * vx_a, float * y_a, const void * vx_b, float * y_b,
        const float * x, int in, int out, int ncols,
        int64_t x_col_stride, int64_t y_col_stride, cudaStream_t stream) {
    ++g_gqh_dispatch_gen;   // see gqh_qx_memo
    if (ncols < 1 || ncols > GQH_MULTICOL_SPEC_MAX ||
            in % GQH_SUPERBLOCK != 0 || in > (1 << 24)) {
        return false;
    }
    if (type != GGML_TYPE_GQH3 && type != GGML_TYPE_GQH4 && type != GGML_TYPE_GQH2_H) {
        return false;
    }
    if (((uintptr_t) x) % sizeof(float4) != 0 ||
            (x_col_stride * (int64_t) sizeof(float)) % sizeof(float4) != 0) {
        return false;
    }
    const bool use_rows1 = ncols == 1 && type == GGML_TYPE_GQH4 && gqh_rows1_selected(out);
    if (ncols == 1 && type == GGML_TYPE_GQH4 && !use_rows1) {
        return false;
    }
    float scale_a, scale_b;
    int   code_a,  code_b;
    if (!ggml_gqh_lookup(vx_a, &scale_a, &code_a) ||
        !ggml_gqh_lookup(vx_b, &scale_b, &code_b)) {
        return false;
    }
    gqh_grid16 grid_a{}, grid_b{};
    if (type == GGML_TYPE_GQH3) {
        memcpy(grid_a.v, GQH3_GRID[code_a], 8 * sizeof(float));
        memcpy(grid_b.v, GQH3_GRID[code_b], 8 * sizeof(float));
    } else if (type == GGML_TYPE_GQH4) {
        memcpy(grid_a.v, GQH4_GRID[code_a], 16 * sizeof(float));
        memcpy(grid_b.v, GQH4_GRID[code_b], 16 * sizeof(float));
    } else {
        memcpy(grid_a.v, GQH2H_GRID[code_a], 4 * sizeof(float));
        memcpy(grid_b.v, GQH2H_GRID[code_b], 4 * sizeof(float));
    }

    const uint8_t * a = (const uint8_t *) vx_a;
    const uint8_t * b = (const uint8_t *) vx_b;
    if (ncols > 1) {
        auto dispatch = [&](auto rung) {
            using R = decltype(rung);
            switch (ncols) {
                case 2:  gqh_exact_dispatch<R::value, 2, true>(stream, a, y_a, b, y_b, x, in, out, scale_a, grid_a, scale_b, grid_b, x_col_stride, y_col_stride); return true;
                case 3:  gqh_exact_dispatch<R::value, 3, true>(stream, a, y_a, b, y_b, x, in, out, scale_a, grid_a, scale_b, grid_b, x_col_stride, y_col_stride); return true;
                case 4:  gqh_exact_dispatch<R::value, 4, true>(stream, a, y_a, b, y_b, x, in, out, scale_a, grid_a, scale_b, grid_b, x_col_stride, y_col_stride); return true;
                case 5:  gqh_exact_dispatch<R::value, 5, true>(stream, a, y_a, b, y_b, x, in, out, scale_a, grid_a, scale_b, grid_b, x_col_stride, y_col_stride); return true;
                case 6:  gqh_exact_dispatch<R::value, 6, true>(stream, a, y_a, b, y_b, x, in, out, scale_a, grid_a, scale_b, grid_b, x_col_stride, y_col_stride); return true;
                case 7:  gqh_exact_dispatch<R::value, 7, true>(stream, a, y_a, b, y_b, x, in, out, scale_a, grid_a, scale_b, grid_b, x_col_stride, y_col_stride); return true;
                case 8:  gqh_exact_dispatch<R::value, 8, true>(stream, a, y_a, b, y_b, x, in, out, scale_a, grid_a, scale_b, grid_b, x_col_stride, y_col_stride); return true;
                case 9:  gqh_exact_dispatch<R::value, 9, true>(stream, a, y_a, b, y_b, x, in, out, scale_a, grid_a, scale_b, grid_b, x_col_stride, y_col_stride); return true;
                case 10: gqh_exact_dispatch<R::value, 10, true>(stream, a, y_a, b, y_b, x, in, out, scale_a, grid_a, scale_b, grid_b, x_col_stride, y_col_stride); return true;
                case 11: gqh_exact_dispatch<R::value, 11, true>(stream, a, y_a, b, y_b, x, in, out, scale_a, grid_a, scale_b, grid_b, x_col_stride, y_col_stride); return true;
                case 12: gqh_exact_dispatch<R::value, 12, true>(stream, a, y_a, b, y_b, x, in, out, scale_a, grid_a, scale_b, grid_b, x_col_stride, y_col_stride); return true;
                default: return false;
            }
        };
        if (type == GGML_TYPE_GQH3) {
            return dispatch(std::integral_constant<ggml_type, GGML_TYPE_GQH3>{});
        }
        if (type == GGML_TYPE_GQH4) {
            return dispatch(std::integral_constant<ggml_type, GGML_TYPE_GQH4>{});
        }
        return dispatch(std::integral_constant<ggml_type, GGML_TYPE_GQH2_H>{});
    }

    // SuperSonic: 2-wave blocks on the GQH3 nsb=20 out=17408 pair path (ncols==1).
    static const bool warps2_on = []() {
        const char * e = getenv("GGML_GQH_WARPS2");
        return e ? atoi(e) != 0 : true;
    }();
    const int pair_warps =
        (warps2_on && type == GGML_TYPE_GQH3 && in == 5120 && out == 17408)
            ? 2 : GQH_MATVEC_WARPS;
    const int rows = use_rows1 ? 1 : GQH_MATVEC_ROWS;
    const int rows_per_block = pair_warps * rows;
    const dim3 threads(GQH_WARP * pair_warps, 1, 1);
    const dim3 blocks((out + rows_per_block - 1) / rows_per_block, 2, 1);
    if (use_rows1) {
        switch (type) {
            case GGML_TYPE_GQH4:
                gqh_matvec_kernel<GGML_TYPE_GQH4, 1, 1, true><<<blocks, threads, 0, stream>>>(
                    a, x, y_a, in, out, ncols, scale_a, grid_a,
                    x_col_stride, y_col_stride, b, y_b, scale_b, grid_b);
                break;
            default:
                return false;
        }
    } else {
        switch (type) {
            case GGML_TYPE_GQH3:
                gqh_matvec_kernel<GGML_TYPE_GQH3, 1, GQH_MATVEC_ROWS, true><<<blocks, threads, 0, stream>>>(
                    a, x, y_a, in, out, ncols, scale_a, grid_a,
                    x_col_stride, y_col_stride, b, y_b, scale_b, grid_b);
                break;
            case GGML_TYPE_GQH2_H:
                gqh_matvec_kernel<GGML_TYPE_GQH2_H, 1, GQH_MATVEC_ROWS, true><<<blocks, threads, 0, stream>>>(
                    a, x, y_a, in, out, ncols, scale_a, grid_a,
                    x_col_stride, y_col_stride, b, y_b, scale_b, grid_b);
                break;
            default:
                return false;
        }
    }
    return true;
}

// --- registry-aware converters ----------------------------------------------
// These match the ggml to_fp16 / to_fp32 signatures, which carry no tensor, so
// the header comes from the registry. k is an element count and is always a whole
// number of superblocks (a GQH row is cols/256 superblocks and cols % 256 == 0).

template <typename dst_t>
static void gqh_convert(ggml_type type, const void * vx, dst_t * y, int64_t k, cudaStream_t stream) {
    float scale = 0.0f;
    int   code  = 0;
    if (!ggml_gqh_lookup(vx, &scale, &code)) {
        GGML_ABORT("gqh: tensor slice %p is not registered -- the per-tensor header KV "
                   "was not read at load time", vx);
    }
    if (k % GQH_SUPERBLOCK != 0) {
        GGML_ABORT("gqh: dequant of %lld elements is not a whole number of superblocks",
                   (long long) k);
    }
    const int64_t nsb = k / GQH_SUPERBLOCK;
    switch (type) {
        case GGML_TYPE_GQH3: gqh3_decode_cuda(vx, scale, code, y, nsb, stream); break;
        case GGML_TYPE_GQH4: gqh4_decode_cuda(vx, scale, code, y, nsb, stream); break;
        default:             gqh2h_decode_cuda(vx, scale, code, y, nsb, stream); break;
    }
}

void dequantize_gqh3_to_fp16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    gqh_convert(GGML_TYPE_GQH3, vx, y, k, stream);
}
void dequantize_gqh2h_to_fp16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    gqh_convert(GGML_TYPE_GQH2_H, vx, y, k, stream);
}
void dequantize_gqh3_to_fp32_cuda(const void * vx, float * y, int64_t k, cudaStream_t stream) {
    gqh_convert(GGML_TYPE_GQH3, vx, y, k, stream);
}
void dequantize_gqh2h_to_fp32_cuda(const void * vx, float * y, int64_t k, cudaStream_t stream) {
    gqh_convert(GGML_TYPE_GQH2_H, vx, y, k, stream);
}

void dequantize_gqh4_to_fp16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    gqh_convert(GGML_TYPE_GQH4, vx, y, k, stream);
}
void dequantize_gqh4_to_fp32_cuda(const void * vx, float * y, int64_t k, cudaStream_t stream) {
    gqh_convert(GGML_TYPE_GQH4, vx, y, k, stream);
}

// gqh2_c takes no registry lookup: nothing about its decode is out of band.
void dequantize_gqh2c_to_fp16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    gqh2c_decode_cuda(vx, y, k / GQH_SUPERBLOCK, stream);
}
void dequantize_gqh2c_to_fp32_cuda(const void * vx, float * y, int64_t k, cudaStream_t stream) {
    gqh2c_decode_cuda(vx, y, k / GQH_SUPERBLOCK, stream);
}
