// Candidate-restricted LM head for the greedy verify path (Q6_K weight).
//
// Instead of the full-vocab head matmul (output.weight [n_embd × n_vocab] Q6_K @
// hidden), compute logits only over a per-position candidate shortlist S (the
// draft's top-M tokens), then argmax within S. See draft top-k calibration:
// the target greedy token lies in the draft top-M with ~94% (M=128) … ~97.5%
// (M=1024) coverage, so a restricted head is near-lossless for greedy and reads
// only M rows of the head instead of all 248k.
//
// This is a SINGLE fused kernel: each (candidate, position) block gathers the
// candidate's Q6_K row, dequantizes it in registers, dots it with that
// position's hidden vector, and folds the result into a per-position argmax via
// a packed 64-bit atomicMax — no intermediate gathered-weights tensor and no
// separate logits buffer. Mirrors the contract in restricted_lm_head_cuda.h.

#include "restricted_lm_head_cuda.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace dflash::common {

namespace {

// Q6_K block layout (must match ggml-common.h block_q6_K exactly): 256 weights
// per super-block, 210 bytes.
constexpr int kQK = 256;
struct __align__(2) BlockQ6K {
    uint8_t ql[kQK / 2];      // 128 B: low 4 bits
    uint8_t qh[kQK / 4];      //  64 B: high 2 bits
    int8_t  scales[kQK / 16]; //  16 B: per-16 sub-block scales
    uint16_t d;               //   2 B: super-block scale (f16 bits)
};
static_assert(sizeof(BlockQ6K) == 210, "block_q6_K layout mismatch");

constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = 4;            // 128-thread blocks, 1 warp per candidate
constexpr int kBlock = kWarp * kWarpsPerBlock;
constexpr int kCandPerWarp = 1;              // candidates each warp pipelines (>1 ⇒ double-buffered)
constexpr int kNBuf = (kCandPerWarp > 1) ? 2 : 1;  // shared-mem row buffers per warp

// Dequantize one Q6_K weight at local index j∈[0,256) of super-block `b`.
// Mirrors dequantize_row_q6_K (ggml-quants.c): the 256 weights split into two
// 128-wide halves; within a half, weight jj uses group=jj/32, l=jj%32.
__device__ __forceinline__ float dequant_q6k(const BlockQ6K & b, int j, float d) {
    const int half    = j >> 7;          // 0/1
    const int jj      = j & 127;
    const int group   = jj >> 5;         // 0..3
    const int l       = jj & 31;
    const int is      = l >> 4;          // 0/1
    const int ql_base = half * 64;
    const int qh_base = half * 32;
    const int sc_base = half * 8;
    const uint8_t qhb = b.qh[qh_base + l];
    int q, sc_idx;
    switch (group) {
        case 0: q = (b.ql[ql_base + l]      & 0xF) | (((qhb >> 0) & 3) << 4); sc_idx = sc_base + is + 0; break;
        case 1: q = (b.ql[ql_base + l + 32] & 0xF) | (((qhb >> 2) & 3) << 4); sc_idx = sc_base + is + 2; break;
        case 2: q = (b.ql[ql_base + l]      >> 4)  | (((qhb >> 4) & 3) << 4); sc_idx = sc_base + is + 4; break;
        default:q = (b.ql[ql_base + l + 32] >> 4)  | (((qhb >> 6) & 3) << 4); sc_idx = sc_base + is + 6; break;
    }
    return d * (float)b.scales[sc_idx] * (float)(q - 32);
}

// Pack (logit, index) into one 64-bit key whose unsigned ordering matches the
// desired argmax (larger logit wins; ties → lower index). The float bits go in
// the high 32 bits (monotonic-mapped so the int compare matches float order),
// the *complemented* index in the low 32 (so lower index ranks higher on ties).
__device__ __forceinline__ uint64_t pack_key(float logit, int idx) {
    uint32_t u = __float_as_uint(logit);
    // Map IEEE float bits to a monotonically-increasing unsigned ordering.
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((uint64_t)u << 32) | (uint32_t)(0xFFFFFFFFu - (uint32_t)idx);
}
__device__ __forceinline__ int unpack_idx(uint64_t key) {
    return (int)(0xFFFFFFFFu - (uint32_t)(key & 0xFFFFFFFFu));
}

// Decode a smem-resident Q6_K row (rw 32-bit words) and dot with hidden h.
__device__ __forceinline__ float row_dot(const uint32_t * sbuf, int blocks_per_row,
                                         const float * __restrict__ h, int lane) {
    const BlockQ6K * row = reinterpret_cast<const BlockQ6K *>(sbuf);
    float acc = 0.0f;
    for (int sb = 0; sb < blocks_per_row; sb++) {
        const BlockQ6K & b = row[sb];
        const float d = __half2float(*reinterpret_cast<const __half *>(&b.d));
        const int base = sb << 8;
#pragma unroll
        for (int j = lane; j < kQK; j += kWarp) acc += dequant_q6k(b, j, d) * h[base + j];
    }
#pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc;
}

// One warp per group of kCandPerWarp candidates for position p. Each warp
// software-pipelines its candidate rows: it issues a cp.async bulk copy of the
// next Q6_K row into shared memory while it decodes/dots the current one (the
// rows are the only sizeable global traffic, so overlapping their loads with
// the tiny dequant+dot compute hides the load latency). logit folds into
// best_key[p] via a packed atomicMax. Each row's 4200 bytes are 4-aligned, so
// the whole row is copied in one coalesced async pass (per-super-block copies
// would hit 2-mod-4 offsets that cp.async can't express).
__global__ void restricted_head_kernel(const BlockQ6K * __restrict__ head,
                                       int n_embd, int blocks_per_row,
                                       const float * __restrict__ hidden,
                                       const int32_t * __restrict__ cand_ids,
                                       int M,
                                       unsigned long long * __restrict__ best_key) {
    extern __shared__ uint32_t smem[];        // [kWarpsPerBlock][2][rw]
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int p    = blockIdx.y;
    const int c0   = (blockIdx.x * kWarpsPerBlock + warp) * kCandPerWarp;
    if (c0 >= M) return;

    const int rw = (blocks_per_row * (int)sizeof(BlockQ6K) + 3) / 4;  // row words
    uint32_t * buf[2] = { smem + (size_t)(warp * kNBuf + 0) * rw,
                          smem + (size_t)(warp * kNBuf + (kNBuf>1?1:0)) * rw };
    const float * __restrict__ h = hidden + (size_t)p * n_embd;
    const int kc = min(kCandPerWarp, M - c0);  // valid candidates for this warp

    auto issue = [&](int slot, int c) {
        const int32_t id = cand_ids[(size_t)p * M + c];
        const uint32_t * src = reinterpret_cast<const uint32_t *>(head + (size_t)id * blocks_per_row);
        for (int w = lane; w < rw; w += kWarp)
            __pipeline_memcpy_async(&buf[slot][w], &src[w], sizeof(uint32_t));
        __pipeline_commit();
    };

    issue(0, c0);                              // prefetch first row
    for (int k = 0; k < kc; k++) {
        const bool has_next = (k + 1 < kc);
        if (has_next) issue((k + 1) & 1, c0 + k + 1);
        __pipeline_wait_prior(has_next ? 1 : 0);   // current row ready (next still in flight)
        __syncwarp();
        const float acc = row_dot(buf[k & 1], blocks_per_row, h, lane);
        if (lane == 0)
            atomicMax(&best_key[p], (unsigned long long)pack_key(acc, c0 + k));
        __syncwarp();
    }
}

// Resolve the packed argmax keys to vocab ids: out_tokens[p] = cand_ids[p*M + c*]
// where c* is the winning candidate index for position p.
__global__ void resolve_kernel(const unsigned long long * __restrict__ best_key,
                               const int32_t * __restrict__ cand_ids, int M,
                               int n_tokens, int32_t * __restrict__ out_tokens) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_tokens) return;
    const int c = unpack_idx(best_key[p]);
    out_tokens[p] = cand_ids[(size_t)p * M + c];
}

}  // namespace

bool restricted_lm_head_q6k(const void * d_head_q6k,
                            int n_embd, int n_vocab,
                            const float * d_hidden, int n_tokens,
                            const int32_t * d_cand_ids, int M,
                            int32_t * d_out_tokens,
                            void * d_scratch_keys,
                            cudaStream_t stream) {
    if (n_embd <= 0 || n_embd % kQK != 0 || n_tokens <= 0 || M <= 0) return false;
    (void)n_vocab;
    const int blocks_per_row = n_embd / kQK;
    auto * best_key = static_cast<unsigned long long *>(d_scratch_keys);

    // Init per-position best keys to the minimum (all bits 0 → most-negative).
    if (cudaMemsetAsync(best_key, 0, (size_t)n_tokens * sizeof(unsigned long long),
                        stream) != cudaSuccess) return false;

    const int cands_per_block = kWarpsPerBlock * kCandPerWarp;
    const int rw = (blocks_per_row * (int)sizeof(BlockQ6K) + 3) / 4;  // row words
    const size_t smem = (size_t)kWarpsPerBlock * kNBuf * rw * sizeof(uint32_t);
    dim3 grid((M + cands_per_block - 1) / cands_per_block, n_tokens);
    restricted_head_kernel<<<grid, kBlock, smem, stream>>>(
        static_cast<const BlockQ6K *>(d_head_q6k), n_embd, blocks_per_row,
        d_hidden, d_cand_ids, M, best_key);

    const int rb = 128;
    resolve_kernel<<<(n_tokens + rb - 1) / rb, rb, 0, stream>>>(
        best_key, d_cand_ids, M, n_tokens, d_out_tokens);

    return cudaGetLastError() == cudaSuccess;
}

}  // namespace dflash::common
