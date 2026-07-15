// Standalone microbenchmark + correctness check for extract_draft_topk_cuda.
// Build:
//   nvcc -O3 -arch=sm_86 -o /tmp/bench_topk \
//       server/src/common/bench_topk.cu server/src/common/draft_topk_cuda.cu
// Run:
//   /tmp/bench_topk [n_positions] [vocab] [K] [iters]
#include "draft_topk_cuda.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

using dflash::common::extract_draft_topk_cuda;

// CPU reference matching the documented semantics.
static void cpu_ref(const float* logits, int n, int vocab, int K, float temp,
                    std::vector<float>& lp, std::vector<int32_t>& ids) {
    const float inv_t = 1.0f / std::fmax(1e-3f, temp);
    lp.assign((size_t)n * K, 0.f);
    ids.assign((size_t)n * K, -1);
    for (int r = 0; r < n; r++) {
        const float* li = logits + (size_t)r * vocab;
        // online logsumexp
        float lmax = -FLT_MAX, lsum = 0.f;
        for (int j = 0; j < vocab; j++) {
            float l = li[j] * inv_t;
            if (l > lmax) { lsum = lsum * std::exp(lmax - l) + 1.f; lmax = l; }
            else lsum += std::exp(l - lmax);
        }
        float log_z = lmax + std::log(lsum);
        // top-K with lower-id-wins on ties
        std::vector<int> idx(vocab);
        for (int j = 0; j < vocab; j++) idx[j] = j;
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
            [&](int a, int b){
                float la = li[a]*inv_t, lb = li[b]*inv_t;
                if (la != lb) return la > lb;
                return a < b;
            });
        for (int k = 0; k < K; k++) {
            int j = idx[k];
            lp[(size_t)r*K+k]  = li[j]*inv_t - log_z;
            ids[(size_t)r*K+k] = j;
        }
    }
}

int main(int argc, char** argv) {
    int n     = argc > 1 ? atoi(argv[1]) : 15;
    int vocab = argc > 2 ? atoi(argv[2]) : 151936;
    int K     = argc > 3 ? atoi(argv[3]) : 8;
    int iters = argc > 4 ? atoi(argv[4]) : 200;
    float temp = 1.0f;

    printf("n=%d vocab=%d K=%d iters=%d\n", n, vocab, K, iters);

    std::vector<float> h_logits((size_t)n * vocab);
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.f, 4.f);
    for (auto& x : h_logits) x = dist(rng);

    float* d_logits = nullptr;
    cudaMalloc(&d_logits, h_logits.size() * sizeof(float));
    cudaMemcpy(d_logits, h_logits.data(), h_logits.size()*sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float>   lp((size_t)n*K);
    std::vector<int32_t> ids((size_t)n*K);

    // correctness
    bool ok = extract_draft_topk_cuda(d_logits, n, vocab, K, lp.data(), ids.data(), temp);
    if (!ok) { printf("FAIL: kernel returned false\n"); return 1; }

    std::vector<float> rlp; std::vector<int32_t> rids;
    cpu_ref(h_logits.data(), n, vocab, K, temp, rlp, rids);
    int mism = 0; float maxerr = 0.f;
    for (size_t i = 0; i < lp.size(); i++) {
        if (ids[i] != rids[i]) { if (mism < 10) printf("  id mismatch @%zu gpu=%d cpu=%d\n", i, ids[i], rids[i]); mism++; }
        maxerr = std::fmax(maxerr, std::fabs(lp[i]-rlp[i]));
    }
    printf("correctness: id_mismatches=%d  max_lp_err=%.3e\n", mism, maxerr);

    // warmup
    for (int i = 0; i < 10; i++) extract_draft_topk_cuda(d_logits, n, vocab, K, lp.data(), ids.data(), temp);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++)
        extract_draft_topk_cuda(d_logits, n, vocab, K, lp.data(), ids.data(), temp);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    printf("avg full call (kernel+sync+copy): %.4f ms\n", ms / iters);

    cudaFree(d_logits);
    return 0;
}
