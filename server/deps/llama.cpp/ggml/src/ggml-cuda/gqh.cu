#include "gqh.cuh"
#include "../gqh.h"

#include <cstdio>
#include <cstdlib>
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
// 9..GQH_MAX_COLS stay on the generic ROWS==1 instantiation.
#define GQH_MULTICOL_SPEC_MAX 8

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
template <ggml_type RUNG, int NCOLS_MAX, int ROWS, bool PAIRED>
static __global__ void gqh_matvec_kernel(
        const uint8_t * __restrict__ data, const float * __restrict__ x,
        float * __restrict__ y, int in, int out, int ncols, float tensor_scale,
        gqh_grid16 grid, int64_t x_col_stride, int64_t y_col_stride,
        const uint8_t * __restrict__ data_b, float * __restrict__ y_b,
        float tensor_scale_b, gqh_grid16 grid_b) {
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
    if (row >= out) return;
    // The exact-width instantiations write EVERY column they carry -- their `c >= ncols`
    // guard is compile-time dead -- so a caller that reached one of them with a narrower
    // ncols would store PAST THE END of `y`, which is the one way this arm can do worse
    // than run slowly. gqh_matvec_launch's switch guarantees the match; this makes the
    // contract self-enforcing instead of conventional. Compile-time dead on the generic
    // and batch-1 arms (so their codegen does not move), one wave-uniform scalar compare
    // outside the superblock loop on the others.
    if (XSHARED && ncols != NCOLS_MAX) return;

    const int nsb = in / GQH_SUPERBLOCK;
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
        if constexpr (XSHARED) {
    #pragma unroll
            for (int c = 0; c < NCOLS_MAX; ++c) {
                const int b = c < GQH_MULTICOL_XBASES ? c : 0;
                xcol[c] = x + (int64_t) b * x_col_stride;
                xoff[c] = (uint32_t) ((int64_t) (c - b) * x_col_stride
                                      * (int64_t) sizeof(float));
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
            if constexpr (XSHARED) {
    #pragma unroll
                for (int c = 0; c < NCOLS_MAX; ++c) {
                    gqh_load_x(xcol[c], sb, j0, xshared[c], xoff[c]);
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
            float    d_real[ROWS];
            uint32_t codes[ROWS];
            uint8_t  rb[ROWS];
            uint8_t  hi1[ROWS];
    #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const uint8_t d_raw = wire[r].d;
                const int d = UNIFORM_ADDR ? gqh_uniform(d_raw) : d_raw;
                d_real[r] = gqh_bits(GQH_E4M3_D[d >> 3][d & 7]) * t_scale;

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
            const uint32_t sbn_off = (uint32_t) sbn * (uint32_t) sb_bytes;
            // Every row's wire load is issued here, in one group, so the wave holds ROWS
            // independent DRAM requests in flight instead of one. That is the whole point
            // of ROWS > 1: bytes-in-flight per wave is the axis this kernel is bound on.
    #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                wire[r] = gqh_wire_load<RUNG>(rowbase[r], sbn_off, woff);
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
            if (NCOLS_MAX == 1 || XSHARED) {
                gqh_sched_fence();
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

                // Decode this lane's 8 weights ONCE, then reuse them for every column.
                float w[GQH_PER_LANE];
    #pragma unroll
                for (int t = 0; t < GQH_PER_LANE; ++t) {
                    // gqh4: little-endian, byte t>>1, low nibble for even t -> bit 4*t
                    // either way. gqh3: the low-2-bit plane carries bits [1:0] of the code
                    // and the high plane bit 2. gqh2_h: the 2-bit code is the whole index.
                    const int code = IS_GQH4
                        ? ((codes[r] >> (4 * t)) & 0x0f)
                        : (((codes[r] >> (2 * t)) & 0x03) |
                           (IS_GQH3 ? (((hi1[r] >> t) & 1) << 2) : 0));
                    w[t] = s_grid[code] * s_b;
                }

                // Fully unrolled so acc[]/xs[] stay in registers; the per-column term order
                // is unchanged, so each output is bit-identical to the one-column-per-block
                // version this replaces. Rows are independent sums, so folding row r+1 after
                // row r changes nothing about either row's term order.
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

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        // The only place the row clamp is paid for: a tail warp recomputed row out-1
        // in slot r > 0 and must not write it back.
        const bool store = r == 0 || row + r < out;
#pragma unroll
        for (int c = 0; c < NCOLS_MAX; ++c) {
            // Only the generic instantiation can be dispatched with ncols < NCOLS_MAX;
            // the exact-width ones write every column they carry.
            if (!XSHARED && c >= ncols) {
                continue;
            }
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
template <ggml_type RUNG, int NCOLS>
static void gqh_multicol_dispatch(
        const dim3 & threads, cudaStream_t stream,
        const uint8_t * data, const float * x, float * y, int in, int out,
        float tensor_scale, gqh_grid16 grid,
        int64_t x_col_stride, int64_t y_col_stride) {
    constexpr int ROWS = GQH_MULTICOL_ROWS;
    const int rows_per_block = GQH_MATVEC_WARPS * ROWS;
    const dim3 blocks((out + rows_per_block - 1) / rows_per_block, 1, 1);
    gqh_matvec_kernel<RUNG, NCOLS, ROWS, false><<<blocks, threads, 0, stream>>>(
        data, x, y, in, out, NCOLS, tensor_scale, grid,
        x_col_stride, y_col_stride, data, y, tensor_scale, grid);
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
        // if constexpr, not a plain if: the selector is GQH4-only, and this keeps
        // <GQH3, 1, 1> / <GQH2_H, 1, 1> from being instantiated at all rather than
        // emitted and never dispatched.
        if constexpr (RUNG == GGML_TYPE_GQH4) {
            if (gqh_rows1_selected(out)) {
                const dim3 blocks((out + GQH_MATVEC_WARPS - 1) / GQH_MATVEC_WARPS, 1, 1);
                gqh_matvec_kernel<RUNG, 1, 1, false><<<blocks, threads, 0, stream>>>(
                    data, x, y, in, out, ncols, tensor_scale, grid,
                    x_col_stride, y_col_stride, data, y, tensor_scale, grid);
                return;
            }
        }
        const int rows_per_block = GQH_MATVEC_WARPS * GQH_MATVEC_ROWS;
        const dim3 blocks((out + rows_per_block - 1) / rows_per_block, 1, 1);
        gqh_matvec_kernel<RUNG, 1, GQH_MATVEC_ROWS, false>
            <<<blocks, threads, 0, stream>>>(
                data, x, y, in, out, ncols, tensor_scale, grid,
                x_col_stride, y_col_stride, data, y, tensor_scale, grid);
    } else {
        // Exact-width arm for spec-decode verify (DFlash2 block 8 => ncols 8).
        // The switch is exhaustive over 2..GQH_MULTICOL_SPEC_MAX; anything wider
        // keeps the generic instantiation below.
        static_assert(GQH_MULTICOL_SPEC_MAX == 8, "add or remove a case below to match");
        if (gqh_multicol_on() && ncols <= GQH_MULTICOL_SPEC_MAX) {
            switch (ncols) {
                case 2:
                    gqh_multicol_dispatch<RUNG, 2>(
                        threads, stream, data, x, y, in, out, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                case 3:
                    gqh_multicol_dispatch<RUNG, 3>(
                        threads, stream, data, x, y, in, out, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                case 4:
                    gqh_multicol_dispatch<RUNG, 4>(
                        threads, stream, data, x, y, in, out, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                case 5:
                    gqh_multicol_dispatch<RUNG, 5>(
                        threads, stream, data, x, y, in, out, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                case 6:
                    gqh_multicol_dispatch<RUNG, 6>(
                        threads, stream, data, x, y, in, out, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                case 7:
                    gqh_multicol_dispatch<RUNG, 7>(
                        threads, stream, data, x, y, in, out, tensor_scale, grid,
                        x_col_stride, y_col_stride);
                    return;
                case 8:
                    gqh_multicol_dispatch<RUNG, 8>(
                        threads, stream, data, x, y, in, out, tensor_scale, grid,
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
// Returns false unless BOTH tensors are registered GQH4 of identical shape and the
// shape takes the ROWS == 1 arm. Narrow on purpose -- the caller keeps its unfused
// path, and no other shape family in this model forms a pair. The two halves may carry
// DIFFERENT level tables; 26% of this model's pairs do.
bool ggml_cuda_gqh_mul_mat_vec_pair(
        const void * vx_a, float * y_a, const void * vx_b, float * y_b,
        const float * x, int in, int out, int ncols,
        int64_t x_col_stride, int64_t y_col_stride, cudaStream_t stream) {
    if (ncols != 1 || in % GQH_SUPERBLOCK != 0 || in > (1 << 24)) {
        return false;
    }
    if (((uintptr_t) x) % sizeof(float4) != 0) {
        return false;
    }
    // The paired kernel is the ROWS == 1 arm only; that is where out == 17408 goes and
    // the arm's own selector has to agree, or the pair would silently take a different
    // code path from the two dispatches it replaces.
    if (!gqh_rows1_selected(out)) {
        return false;
    }
    float scale_a, scale_b;
    int   code_a,  code_b;
    if (!ggml_gqh_lookup(vx_a, &scale_a, &code_a) ||
        !ggml_gqh_lookup(vx_b, &scale_b, &code_b)) {
        return false;
    }
    gqh_grid16 grid_a{}, grid_b{};
    memcpy(grid_a.v, GQH4_GRID[code_a], 16 * sizeof(float));
    memcpy(grid_b.v, GQH4_GRID[code_b], 16 * sizeof(float));

    const dim3 threads(GQH_WARP * GQH_MATVEC_WARPS, 1, 1);
    const dim3 blocks((out + GQH_MATVEC_WARPS - 1) / GQH_MATVEC_WARPS, 2, 1);
    gqh_matvec_kernel<GGML_TYPE_GQH4, 1, 1, true><<<blocks, threads, 0, stream>>>(
        (const uint8_t *) vx_a, x, y_a, in, out, ncols, scale_a, grid_a,
        x_col_stride, y_col_stride, (const uint8_t *) vx_b, y_b, scale_b, grid_b);
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
