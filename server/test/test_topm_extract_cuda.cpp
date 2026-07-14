// Correctness test for dflash::common::extract_topm_cuda (GPU) vs a CPU
// reference of the same coarse-bin-threshold algorithm.
//
// extract_topm_cuda's contract is NOT "exact top-M by value" -- it partitions
// by the top 16 bits of an order-preserving key, so it's only exact up to
// that coarsening (see the .cu file header). The invariant this test checks
// is the algorithm's actual contract: every element whose coarse bin is
// strictly above the threshold bin must appear in the output (no winner
// dropped), no element whose bin is strictly below the threshold must appear
// (no non-candidate let in), the output has exactly M distinct valid ids, and
// the above/eq split sizes match a CPU-computed reference threshold exactly.
//
// Deliberately includes several vocab sizes NOT divisible by 4 -- the whole
// point of the per-row alignment-peel rewrite (align_head in the .cu) was to
// drop the old "vocab % 4 == 0" precondition, so this is the coverage that
// didn't exist before that change at all.
//
// Build: registered in server/CMakeLists.txt under DFLASH27B_TESTS, CUDA
// backend only (topm_extract_cuda.cu is CUDA-only, gated by
// DFLASH27B_HAVE_TOPK_HEAD). Run: ./test_topm_extract_cuda (0 = pass).

#include "../src/common/topm_extract_cuda.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <unordered_set>
#include <vector>

using dflash::common::extract_topm_cuda;
using dflash::common::extract_topm_scratch_bytes;

namespace {

constexpr int kBits = 16;  // must match topm_extract_cuda.cu's kBits

uint32_t order_key(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

int coarse_bin(float f) { return (int)(order_key(f) >> (32 - kBits)); }

// CPU reference: same coarse-bin histogram + top-down threshold search the
// kernel uses. Returns (threshold_bin, above_cnt) for one row.
std::pair<int, int> cpu_threshold(const float * row, int vocab, int M) {
    std::vector<int64_t> hist(1 << kBits, 0);
    for (int v = 0; v < vocab; v++) hist[coarse_bin(row[v])]++;
    int64_t cum = 0;
    for (int b = (1 << kBits) - 1; b >= 0; b--) {
        if (cum + hist[b] >= (int64_t)M) return {b, (int)cum};
        cum += hist[b];
    }
    return {0, (int)cum};  // unreachable when M <= vocab
}

struct Case {
    int n;
    int vocab;
    int M;
};

bool run_case(const Case & c, unsigned seed) {
    const size_t n_logits = (size_t)c.n * c.vocab;
    std::vector<float> h_logits(n_logits);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 4.f);
    for (auto & x : h_logits) x = dist(rng);

    float * d_logits = nullptr;
    int32_t * d_cand = nullptr;
    void * d_scratch = nullptr;
    cudaMalloc(&d_logits, n_logits * sizeof(float));
    cudaMalloc(&d_cand, (size_t)c.n * c.M * sizeof(int32_t));
    cudaMalloc(&d_scratch, extract_topm_scratch_bytes(c.n));
    cudaMemcpy(d_logits, h_logits.data(), n_logits * sizeof(float), cudaMemcpyHostToDevice);

    const bool ok = extract_topm_cuda(d_logits, c.vocab, c.n, c.M, d_cand, d_scratch, nullptr);
    cudaDeviceSynchronize();

    std::vector<int32_t> h_cand((size_t)c.n * c.M);
    if (ok) cudaMemcpy(h_cand.data(), d_cand, h_cand.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_logits);
    cudaFree(d_cand);
    cudaFree(d_scratch);

    if (!ok) {
        printf("  FAIL: extract_topm_cuda returned false\n");
        return false;
    }

    int bad_rows = 0;
    for (int r = 0; r < c.n; r++) {
        const float * row = h_logits.data() + (size_t)r * c.vocab;
        const auto [thr, above_cnt] = cpu_threshold(row, c.vocab, c.M);
        const int32_t * out = h_cand.data() + (size_t)r * c.M;

        std::unordered_set<int32_t> seen;
        int n_above = 0, n_eq = 0, dup = 0, out_of_range = 0, below_thr = 0;
        for (int i = 0; i < c.M; i++) {
            const int32_t id = out[i];
            if (id < 0 || id >= c.vocab) { out_of_range++; continue; }
            if (!seen.insert(id).second) { dup++; continue; }
            const int bin = coarse_bin(row[id]);
            if (bin > thr) n_above++;
            else if (bin == thr) n_eq++;
            else below_thr++;
        }
        // Completeness: every element with bin > thr must be in the output.
        int total_above_in_data = 0;
        for (int v = 0; v < c.vocab; v++)
            if (coarse_bin(row[v]) > thr) total_above_in_data++;

        const bool row_ok = (dup == 0) && (out_of_range == 0) && (below_thr == 0) &&
                            (n_above == above_cnt) && (n_above == total_above_in_data) &&
                            (n_above + n_eq == c.M);
        if (!row_ok) {
            bad_rows++;
            if (bad_rows <= 4) {
                printf("    row=%d BAD: thr=%d above_cnt(cpu)=%d n_above(gpu)=%d "
                       "n_eq(gpu)=%d dup=%d oor=%d below_thr=%d total_above_in_data=%d\n",
                       r, thr, above_cnt, n_above, n_eq, dup, out_of_range, below_thr,
                       total_above_in_data);
            }
        }
    }

    const bool pass = (bad_rows == 0);
    printf("  [%s] n=%-4d vocab=%-7d M=%-5d bad_rows=%d/%d\n",
           pass ? "PASS" : "FAIL", c.n, c.vocab, c.M, bad_rows, c.n);
    return pass;
}

}  // namespace

int main() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        printf("SKIP: no CUDA device available\n");
        return 0;
    }

    const Case cases[] = {
        // Realistic decode shape, vocab % 4 == 0 (the old use_vec-eligible case).
        {15,  151936, 1024},
        {1,   151936, 1024},
        // vocab % 4 != 0 -- previously always the scalar fallback; now exercises
        // the same vectorized path via per-row alignment peeling.
        {15,  151937, 1024},
        {15,  151933, 512},
        {7,   1023,   128},
        {3,   257,    64},
        // Small/edge shapes: M close to vocab, vocab smaller than the peel width.
        {4,   130,    128},
        {2,   5,      4},
        {1,   4,      1},
        {1,   3,      1},   // vocab < float4 width entirely
        {32,  4096,   2048},
    };

    int failures = 0, idx = 0;
    for (const Case & c : cases) {
        if (!run_case(c, /*seed=*/9001u + idx)) failures++;
        idx++;
    }

    if (failures) {
        printf("\nFAILED: %d/%d cases\n", failures, idx);
        return 1;
    }
    printf("\nALL PASS: %d/%d cases\n", idx, idx);
    return 0;
}
