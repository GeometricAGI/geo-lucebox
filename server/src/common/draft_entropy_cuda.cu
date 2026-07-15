// GPU temperature-agnostic per-position draft entropy. See .h for the
// contract/rationale. Structurally mirrors draft_topk_cuda.cu's split-grid
// reduction (bandwidth-bound over a huge vocab, tiny row count) minus the
// top-K bookkeeping: two accumulators (sumexp, weighted sum) instead of a
// register-resident sorted top-K, reduced the same way logsumexp already is.

#include "draft_entropy_cuda.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdint>

namespace dflash::common {

namespace {

constexpr int kBlock    = 256;
constexpr int kMaxSplit = 128;

#define DFLASH_ENTROPY_FOLD(L)                                                 \
    do {                                                                       \
        const float _l = (L);                                                 \
        if (_l > lmax) { lsum  = lsum  * __expf(lmax - _l) + 1.0f;            \
                          lwsum = lwsum * __expf(lmax - _l) + _l; lmax = _l; } \
        else { const float _e = __expf(_l - lmax); lsum += _e; lwsum += _e * _l; } \
    } while (0)

// Pass 1: per-(position, split) partial (max, sumexp, weighted) over a raw
// (unscaled -- temperature-agnostic) logit chunk. VEC selects 16-byte float4
// loads, gated the same way as draft_topk_partial's VEC path (same tensor, so
// the same alignment precondition applies): only used when every row's base
// pointer is 16-byte aligned (vocab % 4 == 0 and an aligned tensor).
template <bool VEC>
__global__ void entropy_partial(const float * __restrict__ logits,
                                int vocab, int split,
                                float * __restrict__ part_max,
                                float * __restrict__ part_sum,
                                float * __restrict__ part_wsum) {
    const int row = blockIdx.x;
    const int s   = blockIdx.y;
    const int tid = threadIdx.x;
    const float * __restrict__ li = logits + (size_t)row * vocab;

    float lmax = -FLT_MAX, lsum = 0.0f, lwsum = 0.0f;

    if (VEC) {
        // Partition the float4s of the row into `split` contiguous chunks,
        // mirroring draft_topk_partial exactly (see that file for why).
        const int vocab4 = vocab >> 2;
        const int chunk4 = (vocab4 + split - 1) / split;
        const int b4 = s * chunk4;
        const int e4 = min(b4 + chunk4, vocab4);
        const float4 * __restrict__ li4 = reinterpret_cast<const float4 *>(li);
        for (int j4 = b4 + tid; j4 < e4; j4 += kBlock) {
            const float4 f = li4[j4];
            DFLASH_ENTROPY_FOLD(f.x);
            DFLASH_ENTROPY_FOLD(f.y);
            DFLASH_ENTROPY_FOLD(f.z);
            DFLASH_ENTROPY_FOLD(f.w);
        }
        // Tail elements past the last full float4 (only when vocab % 4 != 0);
        // the last split owns them so no element is scanned twice.
        if (s == split - 1) {
            for (int j = (vocab4 << 2) + tid; j < vocab; j += kBlock)
                DFLASH_ENTROPY_FOLD(li[j]);
        }
    } else {
        const int chunk = (vocab + split - 1) / split;
        const int begin = s * chunk;
        const int end   = min(begin + chunk, vocab);
        for (int j = begin + tid; j < end; j += kBlock)
            DFLASH_ENTROPY_FOLD(li[j]);
    }

    __shared__ float s_max[kBlock];
    __shared__ float s_sum[kBlock];
    __shared__ float s_wsum[kBlock];
    s_max[tid] = lmax; s_sum[tid] = lsum; s_wsum[tid] = lwsum;
    __syncthreads();

    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float am = s_max[tid],          as = s_sum[tid],          aw = s_wsum[tid];
            const float bm = s_max[tid + stride], bs = s_sum[tid + stride], bw = s_wsum[tid + stride];
            const float m = fmaxf(am, bm);
            s_sum[tid]  = as * __expf(am - m) + bs * __expf(bm - m);
            s_wsum[tid] = aw * __expf(am - m) + bw * __expf(bm - m);
            s_max[tid]  = m;
        }
        __syncthreads();
    }

    if (tid == 0) {
        const int idx = row * split + s;
        part_max[idx]  = s_max[0];
        part_sum[idx]  = s_sum[0];
        part_wsum[idx] = s_wsum[0];
    }
}

#undef DFLASH_ENTROPY_FOLD

// Combine one online-logsumexp triple into another: (m,s,w) := (m,s,w) ⊕ (om,os,ow).
__device__ __forceinline__ void combine3(float & m, float & s, float & w,
                                          float om, float os, float ow) {
    const float nm = fmaxf(m, om);
    s = s * __expf(m - nm) + os * __expf(om - nm);
    w = w * __expf(m - nm) + ow * __expf(om - nm);
    m = nm;
}

// Pass 2: merge `split` partials per row into entropy = log_z - weighted/sumexp.
// blockDim == pow2_ceil(split) <= kMaxSplit, i.e. at most 4 warps. Reduces
// within each warp via shuffle (no shared memory, no __syncthreads -- and for
// blockDim < 32, mask must be __activemask(), not a hardcoded full-warp mask:
// the lanes beyond blockDim.x aren't real executing threads, and shfl_sync's
// mask must match exactly the converged set or the result is undefined). Only
// when more than one warp is active (small n_positions can push split, and so
// blockDim, up to kMaxSplit=128) do the ≤4 per-warp partials need a shared-mem
// hop for the final cross-warp combine.
__global__ void entropy_combine(const float * __restrict__ part_max,
                                 const float * __restrict__ part_sum,
                                 const float * __restrict__ part_wsum,
                                 int split,
                                 float * __restrict__ out_entropy) {
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;  // blockDim == pow2_ceil(split)
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int n_warps = (blockDim.x + 31) >> 5;

    float m, s, w;
    if (tid < split) {
        const int idx = row * split + tid;
        m = part_max[idx]; s = part_sum[idx]; w = part_wsum[idx];
    } else {
        m = -FLT_MAX; s = 0.0f; w = 0.0f;
    }

    // __shfl_down_sync only exchanges data within a 32-lane warp, never across
    // warp boundaries. When n_warps==1, blockDim.x itself is <=32 (possibly a
    // partial warp) so the reduction width is blockDim.x; when n_warps>1,
    // blockDim.x is a multiple of 32 (pow2_ceil(split) with split>32), so
    // every warp is fully populated and the per-warp width is a full 32.
    const int warp_width = (n_warps == 1) ? (int)blockDim.x : 32;
    const unsigned mask = __activemask();
    for (int off = warp_width >> 1; off > 0; off >>= 1) {
        combine3(m, s, w, __shfl_down_sync(mask, m, off),
                          __shfl_down_sync(mask, s, off),
                          __shfl_down_sync(mask, w, off));
    }
    // lane 0 of each warp now holds that warp's fully-combined partial.

    if (n_warps == 1) {
        if (tid == 0) out_entropy[row] = m + logf(s) - w / s;
        return;
    }

    __shared__ float sh_m[4], sh_s[4], sh_w[4];  // kMaxSplit=128 -> at most 4 warps
    if (lane == 0) { sh_m[wid] = m; sh_s[wid] = s; sh_w[wid] = w; }
    __syncthreads();

    if (tid == 0) {
        float fm = sh_m[0], fs = sh_s[0], fw = sh_w[0];
        for (int i = 1; i < n_warps; i++) combine3(fm, fs, fw, sh_m[i], sh_s[i], sh_w[i]);
        out_entropy[row] = fm + logf(fs) - fw / fs;
    }
}

struct Scratch {
    int      device  = -1;
    size_t   cap      = 0;   // n_positions*split partial elements
    float *  d_pmax   = nullptr;
    float *  d_psum   = nullptr;
    float *  d_pwsum  = nullptr;
    float *  d_out    = nullptr;  // n_positions
    size_t   out_cap  = 0;
};
Scratch g_scratch;

void free_scratch() {
    if (g_scratch.d_pmax)  cudaFree(g_scratch.d_pmax);
    if (g_scratch.d_psum)  cudaFree(g_scratch.d_psum);
    if (g_scratch.d_pwsum) cudaFree(g_scratch.d_pwsum);
    if (g_scratch.d_out)   cudaFree(g_scratch.d_out);
    g_scratch = Scratch{};
}

bool ensure_scratch(int device, size_t n_parts, size_t n_positions) {
    if (g_scratch.device == device && g_scratch.cap >= n_parts && g_scratch.out_cap >= n_positions)
        return true;
    free_scratch();
    if (cudaMalloc(&g_scratch.d_pmax,  n_parts * sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc(&g_scratch.d_psum,  n_parts * sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc(&g_scratch.d_pwsum, n_parts * sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc(&g_scratch.d_out,   n_positions * sizeof(float)) != cudaSuccess) goto fail;
    g_scratch.device  = device;
    g_scratch.cap     = n_parts;
    g_scratch.out_cap = n_positions;
    return true;
fail:
    free_scratch();
    return false;
}

inline int pow2_ceil(int x) { int p = 1; while (p < x) p <<= 1; return p; }

// SM count of `device`, queried once and cached (topology is static for the
// process lifetime). Mirrors draft_topk_cuda.cu's sm_count/pick_split.
int sm_count(int device) {
    static int cached_device = -1;
    static int cached_sm     = 0;
    if (device != cached_device) {
        cudaDeviceProp prop{};
        cached_sm = (cudaGetDeviceProperties(&prop, device) == cudaSuccess)
                    ? prop.multiProcessorCount
                    : 80;  // fallback: the ~80-SM device this heuristic was tuned on
        cached_device = device;
    }
    return cached_sm;
}

// Aim for ~3 waves of blocks across the device's actual SM count (was a fixed
// 240 = 3*80, tuned on one ~80-SM GPU; see draft_topk_cuda.cu's pick_split for
// the same reasoning), but cap the chunk floor at ~2k elements so a block
// never scans too little to be worth launching.
int pick_split(int vocab, int n_positions, int device) {
    const int target_blocks = 3 * sm_count(device);
    int by_blocks = (target_blocks + n_positions - 1) / n_positions;
    int by_chunk  = vocab / 2048;
    int split = by_blocks < by_chunk ? by_blocks : by_chunk;
    if (split < 1)         split = 1;
    if (split > kMaxSplit) split = kMaxSplit;
    return split;
}

}  // namespace

bool extract_draft_entropy_cuda(const float * d_logits, int vocab, int n_positions,
                                float * out_entropy, cudaStream_t stream) {
    if (!d_logits || vocab <= 0 || n_positions <= 0) return false;

    int dev = 0;
    cudaGetDevice(&dev);
    const int split = pick_split(vocab, n_positions, dev);
    const size_t n_parts = (size_t)n_positions * split;
    if (!ensure_scratch(dev, n_parts, (size_t)n_positions)) return false;

    // float4 loads are safe only when every row base is 16-byte aligned: the
    // tensor base aligned and a vocab stride that is a multiple of 4 (same
    // precondition as draft_topk_cuda.cu's use_vec, same tensor).
    const bool use_vec = (vocab % 4 == 0) &&
                         (reinterpret_cast<uintptr_t>(d_logits) % 16 == 0);

    const dim3 grid1((unsigned)n_positions, (unsigned)split);
    if (use_vec) {
        entropy_partial<true><<<grid1, kBlock, 0, stream>>>(
            d_logits, vocab, split, g_scratch.d_pmax, g_scratch.d_psum, g_scratch.d_pwsum);
    } else {
        entropy_partial<false><<<grid1, kBlock, 0, stream>>>(
            d_logits, vocab, split, g_scratch.d_pmax, g_scratch.d_psum, g_scratch.d_pwsum);
    }
    entropy_combine<<<n_positions, pow2_ceil(split), 0, stream>>>(
        g_scratch.d_pmax, g_scratch.d_psum, g_scratch.d_pwsum, split, g_scratch.d_out);

    const cudaError_t e = cudaMemcpyAsync(out_entropy, g_scratch.d_out,
                                          (size_t)n_positions * sizeof(float),
                                          cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return false;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace dflash::common
