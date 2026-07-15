// Fast GPU top-M candidate extractor for the candidate-restricted LM head.
//
// Given the draft logits [vocab × n_tokens] (column p = position p's vocab
// logits, contiguous), produce cand_ids [M × n_tokens]: M candidate vocab ids
// per position that contain (with high probability) the target's argmax. Used
// to feed restricted_lm_head_q6k. The draft top-M is far cheaper to find
// approximately than exactly: we radix-threshold on the order-preserving uint
// mapping of the float logits via one histogram pass, pick the threshold bin
// whose cumulative top-down count first reaches M, then gather. Bins strictly
// above the threshold are always kept; the threshold bin fills the remainder
// (its intra-bin order is irrelevant — the bin spans <0.05% of the value
// range, so coverage of the true top-M is preserved).
//
// Reads the logits twice, not three times: pass 1 (histogram) must finish
// before the threshold bin is known, so it can't be avoided or fused with the
// gather. But the two gather passes (elements strictly above the threshold
// bin, and elements equal to it) previously each did their own full-vocab scan
// — a third full read of the same data recomputing the same order_key per
// element. They're fused into one pass below: each element is classified once
// against the threshold and routed to one of two disjoint output ranges via
// two independent atomic counters (the "above" range is exactly above_cnt[p]
// elements by construction; the "eq" range fills the rest, offset by
// above_cnt[p]) — same result, one fewer full-vocab DRAM pass.

#include "topm_extract_cuda.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace dflash::common {

namespace {

constexpr int kBits = 16;                 // histogram key bits (16 ⇒ ~exact top-M)
constexpr int kBins = 1 << kBits;
constexpr int kBlock = 256;
constexpr int kThrThreads = 256;          // threads for the parallel threshold scan
static_assert(kBins % kThrThreads == 0, "kBins must divide threshold threads");

// Order-preserving map float→uint32: larger float ⇒ larger uint (radix-sort key).
__device__ __forceinline__ uint32_t order_key(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Per-row alignment peel: how many leading scalar elements (0-3) `col` needs
// before it reaches a 16-byte (float4) boundary. Computed from the row's
// actual runtime address, so it's correct for any vocab/base-pointer
// alignment -- no global "vocab % 4 == 0 && base % 16 == 0" precondition
// needed; each row peels its own head (and, symmetrically, its own <=3
// element tail past the aligned body) so the vectorized body is always
// provably aligned regardless of vocab's divisibility by 4.
__device__ __forceinline__ int align_head(const float * col, int vocab) {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(col);
    const int head = (int)(((16 - (addr & 15u)) & 15u) >> 2);
    return head < vocab ? head : vocab;  // degenerate: row shorter than the peel
}

// SM count of `device`, queried once and cached (topology is static for the
// process lifetime). Mirrors draft_topk_cuda.cu's sm_count.
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

// Blocks per position for the vocab scan: aim for ~3 waves of blocks across
// the device's actual SM count (was a fixed 64, tuned on one ~80-SM GPU —
// same reasoning as draft_topk_cuda.cu's pick_split, just without that file's
// vocab/2048 chunk floor: this kernel's grid-stride loop spans all `nsplit`
// blocks jointly rather than assigning each split a contiguous chunk, so
// there's no "chunk too small" failure mode to guard against, only atomic
// contention on the histogram at extreme nsplit — hence the cap below).
int pick_nsplit(int n_tokens, int device) {
    constexpr int kMaxNsplit = 256;
    const int target_blocks = 3 * sm_count(device);
    int nsplit = (target_blocks + n_tokens - 1) / n_tokens;
    if (nsplit < 1)           nsplit = 1;
    if (nsplit > kMaxNsplit)  nsplit = kMaxNsplit;
    return nsplit;
}

// Pass 1: per-position histogram of the top kBits of order_key over the vocab.
// Always vectorized (float4): each row peels its own <=3-element head to an
// aligned float4 boundary and <=3-element tail past the aligned body (see
// align_head), handled by one designated thread per position so they aren't
// double-counted by every block.
__global__ void histogram_kernel(const float * __restrict__ logits,
                                 int vocab, int n_tokens,
                                 unsigned int * __restrict__ hist) {  // [n_tokens][kBins]
    const int p = blockIdx.y;
    unsigned int * h = hist + (size_t)p * kBins;
    const float * __restrict__ col = logits + (size_t)p * vocab;

    const int head = align_head(col, vocab);
    const int nvec = (vocab - head) >> 2;
    const float4 * __restrict__ col4 = reinterpret_cast<const float4 *>(col + head);

    for (int v4 = blockIdx.x * blockDim.x + threadIdx.x; v4 < nvec;
         v4 += gridDim.x * blockDim.x) {
        const float4 f = col4[v4];
        atomicAdd(&h[order_key(f.x) >> (32 - kBits)], 1u);
        atomicAdd(&h[order_key(f.y) >> (32 - kBits)], 1u);
        atomicAdd(&h[order_key(f.z) >> (32 - kBits)], 1u);
        atomicAdd(&h[order_key(f.w) >> (32 - kBits)], 1u);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int v = 0; v < head; v++)
            atomicAdd(&h[order_key(col[v]) >> (32 - kBits)], 1u);
        for (int v = head + (nvec << 2); v < vocab; v++)
            atomicAdd(&h[order_key(col[v]) >> (32 - kBits)], 1u);
    }
}

// Pass 2: find threshold_bin[p] = the bin where the top-down cumulative count
// first reaches M, and above_cnt[p] = count strictly above it (< M). One block
// per position: each thread owns a contiguous chunk of bins, computes its chunk
// total, thread 0 builds the per-chunk "count above" prefix, the single thread
// whose chunk straddles the M-crossing rescans just its chunk. O(kBins/T + T).
// chunk (= kBins/kThrThreads = 256, fixed by the two constants above) is always
// a multiple of 4, and h+lo is always 16-byte aligned (hist is cudaMalloc'd,
// kBins and chunk are both multiples of 4 uints) — uint4 loads are safe
// unconditionally here, no runtime alignment check needed.
__global__ void threshold_kernel(const unsigned int * __restrict__ hist,
                                 int n_tokens, int M,
                                 int * __restrict__ threshold_bin,
                                 unsigned int * __restrict__ above_cnt) {
    const int p = blockIdx.x;
    if (p >= n_tokens) return;
    const unsigned int * h = hist + (size_t)p * kBins;
    const int t = threadIdx.x;
    const int chunk = kBins / kThrThreads;
    const int lo = t * chunk, hi = lo + chunk;       // bins [lo, hi)

    __shared__ unsigned int part[kThrThreads];       // chunk totals
    __shared__ unsigned int above[kThrThreads];      // count strictly above chunk t
    unsigned int s = 0;
    const uint4 * __restrict__ h4 = reinterpret_cast<const uint4 *>(h + lo);
#pragma unroll
    for (int c4 = 0; c4 < chunk / 4; c4++) {
        const uint4 v4 = h4[c4];
        s += v4.x + v4.y + v4.z + v4.w;
    }
    part[t] = s;
    __syncthreads();

    if (t == 0) {                                    // suffix sums over chunks
        unsigned int acc = 0;
        for (int c = kThrThreads - 1; c >= 0; c--) { above[c] = acc; acc += part[c]; }
    }
    __syncthreads();

    // The crossing chunk: count above it < M, but adding its total reaches M.
    if (above[t] < (unsigned)M && (unsigned)M <= above[t] + part[t]) {
        unsigned int cum = above[t];
        int bb = lo;
        for (int b = hi - 1; b >= lo; b--) {
            if (cum + h[b] >= (unsigned)M) { bb = b; break; }
            cum += h[b];
        }
        threshold_bin[p] = bb;
        above_cnt[p]     = cum;                      // count strictly above bb (< M)
    }
}

// Fused gather pass: classify each vocab id against the threshold bin and
// route it to one of two disjoint output ranges in a single full-vocab scan
// (replaces the previous gather_above + gather_fill pair, which each did
// their own full scan). "above" elements (key > tb) land in slots
// [0, above_cnt[p]) via their own atomic counter; "eq" elements (key == tb)
// fill the remainder [above_cnt[p], M) via a second counter offset by
// above_cnt[p]. The two ranges never collide because membership is disjoint
// and each counter only ever allocates within its own range — intra-range
// order doesn't matter (see file header). Always vectorized, same per-row
// head/tail peel as histogram_kernel (see align_head); bounded writes (never
// an early `return`) keep every thread scanning its whole strided range so no
// in-range element is skipped.
__global__ void gather_kernel(const float * __restrict__ logits,
                              int vocab, int n_tokens, int M,
                              const int * __restrict__ threshold_bin,
                              const unsigned int * __restrict__ above_cnt,
                              int32_t * __restrict__ cand_ids,
                              unsigned int * __restrict__ fill_above,
                              unsigned int * __restrict__ fill_eq) {
    const int p  = blockIdx.y;
    const int tb = threshold_bin[p];
    const unsigned int ac = above_cnt[p];
    const float * __restrict__ col = logits + (size_t)p * vocab;
    int32_t * out = cand_ids + (size_t)p * M;

    auto emit = [&](int v, unsigned int key) {
        if ((int)key > tb) {
            const unsigned int slot = atomicAdd(&fill_above[p], 1u);
            if (slot < (unsigned)M) out[slot] = v;
        } else if ((int)key == tb) {
            const unsigned int slot = atomicAdd(&fill_eq[p], 1u);
            if (ac + slot < (unsigned)M) out[ac + slot] = v;
        }
    };

    const int head = align_head(col, vocab);
    const int nvec = (vocab - head) >> 2;
    const float4 * __restrict__ col4 = reinterpret_cast<const float4 *>(col + head);

    for (int v4 = blockIdx.x * blockDim.x + threadIdx.x; v4 < nvec;
         v4 += gridDim.x * blockDim.x) {
        const float4 f = col4[v4];
        const int base = head + (v4 << 2);
        emit(base + 0, order_key(f.x) >> (32 - kBits));
        emit(base + 1, order_key(f.y) >> (32 - kBits));
        emit(base + 2, order_key(f.z) >> (32 - kBits));
        emit(base + 3, order_key(f.w) >> (32 - kBits));
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int v = 0; v < head; v++)
            emit(v, order_key(col[v]) >> (32 - kBits));
        for (int v = head + (nvec << 2); v < vocab; v++)
            emit(v, order_key(col[v]) >> (32 - kBits));
    }
}

}  // namespace

bool extract_topm_cuda(const float * d_logits, int vocab, int n_tokens, int M,
                       int32_t * d_cand_ids, void * d_scratch, cudaStream_t stream) {
    if (vocab <= 0 || n_tokens <= 0 || M <= 0 || M > vocab) return false;

    // scratch layout: hist[n_tokens*kBins] u32 | threshold_bin[n_tokens] i32 |
    //                 above_cnt[n_tokens] u32 | fill_above[n_tokens] u32 |
    //                 fill_eq[n_tokens] u32
    auto * base = static_cast<unsigned char *>(d_scratch);
    auto * hist          = reinterpret_cast<unsigned int *>(base);
    auto * threshold_bin = reinterpret_cast<int *>(hist + (size_t)n_tokens * kBins);
    auto * above_cnt     = reinterpret_cast<unsigned int *>(threshold_bin + n_tokens);
    auto * fill_above     = above_cnt + n_tokens;
    auto * fill_eq        = fill_above + n_tokens;

    if (cudaMemsetAsync(hist, 0, (size_t)n_tokens * kBins * sizeof(unsigned int), stream) != cudaSuccess)
        return false;
    cudaMemsetAsync(fill_above, 0, (size_t)n_tokens * sizeof(unsigned int), stream);
    cudaMemsetAsync(fill_eq,    0, (size_t)n_tokens * sizeof(unsigned int), stream);

    int dev = 0;
    cudaGetDevice(&dev);
    const int nsplit = pick_nsplit(n_tokens, dev);
    dim3 grid(nsplit, n_tokens);
    histogram_kernel<<<grid, kBlock, 0, stream>>>(d_logits, vocab, n_tokens, hist);

    threshold_kernel<<<n_tokens, kThrThreads, 0, stream>>>(
        hist, n_tokens, M, threshold_bin, above_cnt);

    gather_kernel<<<grid, kBlock, 0, stream>>>(
        d_logits, vocab, n_tokens, M, threshold_bin, above_cnt, d_cand_ids, fill_above, fill_eq);

    return cudaGetLastError() == cudaSuccess;
}

size_t extract_topm_scratch_bytes(int n_tokens) {
    return (size_t)n_tokens * kBins * sizeof(unsigned int)   // hist
         + (size_t)n_tokens * sizeof(int)                     // threshold_bin
         + (size_t)n_tokens * sizeof(unsigned int)            // above_cnt
         + (size_t)n_tokens * sizeof(unsigned int)            // fill_above
         + (size_t)n_tokens * sizeof(unsigned int);           // fill_eq
}

}  // namespace dflash::common
