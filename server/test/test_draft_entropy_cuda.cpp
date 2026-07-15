// Correctness test for dflash::common::extract_draft_entropy_cuda (GPU) vs a
// double-precision CPU reference.
//
// Deliberately stresses both branches of entropy_combine's warp-level
// reduction (src/common/draft_entropy_cuda.cu): the single-warp fast path
// (n_warps==1, including a genuinely partial warp with blockDim.x < 32) and
// the multi-warp cross-warp-combine path (n_warps>1, needs split>32, which
// pick_split only reaches for a small n_positions against a large vocab —
// see the case table below for exactly how each shape is chosen to land on
// one side or the other regardless of the test GPU's actual SM count).
//
// Build: registered in server/CMakeLists.txt under DFLASH27B_TESTS (CUDA
// backend only -- draft_entropy_cuda.cu is CUDA-only, unlike its topk sibling
// which also compiles for HIP). Run: ./test_draft_entropy_cuda (0 = pass).

#include "../src/common/draft_entropy_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using dflash::common::extract_draft_entropy_cuda;

namespace {

// float4/16-byte-aligned VEC path requires vocab % 4 == 0 (see the .cu); all
// cases below use such vocabs so both use_vec branches get exercised across
// the different (n, vocab) shapes.
constexpr float kTol = 8e-3f;

struct Case {
    int n;      // n_positions
    int vocab;
};

// Double-precision CPU reference: temperature-agnostic (raw-logit) entropy in
// nats, H = log_z - sum(p_i * logit_i) = log_z - weighted/sumexp.
double cpu_entropy(const float * row, int vocab) {
    double m = -1e300;
    for (int j = 0; j < vocab; j++) m = std::max(m, (double)row[j]);
    double sumexp = 0.0, weighted = 0.0;
    for (int j = 0; j < vocab; j++) {
        const double e = std::exp((double)row[j] - m);
        sumexp   += e;
        weighted += e * (double)row[j];
    }
    const double log_z = m + std::log(sumexp);
    return log_z - weighted / sumexp;
}

bool run_case(const Case & c, unsigned seed) {
    const size_t n_logits = (size_t)c.n * c.vocab;
    std::vector<float> h_logits(n_logits);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 4.f);
    for (auto & x : h_logits) x = dist(rng);

    std::vector<double> cpu_h(c.n);
    for (int r = 0; r < c.n; r++)
        cpu_h[r] = cpu_entropy(h_logits.data() + (size_t)r * c.vocab, c.vocab);

    float * d_logits = nullptr;
    cudaError_t err = cudaMalloc(&d_logits, n_logits * sizeof(float));
    if (err != cudaSuccess) {
        printf("  cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    cudaMemcpy(d_logits, h_logits.data(), n_logits * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> gpu_h(c.n, 0.f);
    bool ok = extract_draft_entropy_cuda(d_logits, c.vocab, c.n, gpu_h.data());
    cudaFree(d_logits);

    if (!ok) {
        printf("  FAIL: extract_draft_entropy_cuda returned false\n");
        return false;
    }

    double max_err = 0.0;
    int worst = -1;
    for (int r = 0; r < c.n; r++) {
        const double err = std::fabs((double)gpu_h[r] - cpu_h[r]);
        if (err > max_err) { max_err = err; worst = r; }
    }

    const bool pass = max_err <= kTol;
    printf("  [%s] n=%-4d vocab=%-7d max_err=%.3e (row=%d gpu=%.5f cpu=%.5f)\n",
           pass ? "PASS" : "FAIL", c.n, c.vocab, max_err, worst,
           worst >= 0 ? gpu_h[worst] : 0.f, worst >= 0 ? cpu_h[worst] : 0.0);
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
        // n=1, large vocab: by_chunk=vocab/2048=74 and by_blocks=3*SMs/1 both
        // exceed 32 on any real discrete GPU -> split>32 -> pow2_ceil>=64 ->
        // entropy_combine's multi-warp (n_warps>1) cross-warp-combine path.
        {1,   151936},
        // n~100, same vocab: by_blocks=3*SMs/100 is small (<=32 on anything
        // short of an absurdly large GPU) -> split likely single-digit ->
        // pow2_ceil<32 -> entropy_combine's single-warp path with a
        // genuinely partial warp (blockDim.x < 32, exercises __activemask()
        // over fewer than 32 real lanes).
        {100, 151936},
        // n~15 (realistic DDTree draft-batch size): split lands in the
        // teens-to-32 range depending on SM count -> exercises blockDim.x
        // near/at exactly 32 (a full, non-partial single warp).
        {15,  151936},
        // Small/edge shapes: vocab barely above the VEC path's float4 tile,
        // and a vocab not a multiple of 4 (exercises the scalar non-VEC path
        // and, for the VEC-eligible cases, the tail epilogue).
        {7,   1024},
        {32,  4096},
        {3,   260},     // vocab % 4 == 0, small
        {5,   257},     // vocab % 4 != 0 -> scalar path
        {1,   4},       // minimal vocab
    };

    int failures = 0, idx = 0;
    for (const Case & c : cases) {
        if (!run_case(c, /*seed=*/4242u + idx)) failures++;
        idx++;
    }

    if (failures) {
        printf("\nFAILED: %d/%d cases\n", failures, idx);
        return 1;
    }
    printf("\nALL PASS: %d/%d cases\n", idx, idx);
    return 0;
}
