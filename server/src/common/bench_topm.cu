// Correctness + timing for extract_topm_cuda.
//   nvcc -O3 -arch=sm_86 -o /tmp/bench_topm \
//       server/src/common/bench_topm.cu server/src/common/topm_extract_cuda.cu
#include "topm_extract_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_set>

using dflash::common::extract_topm_cuda;
using dflash::common::extract_topm_scratch_bytes;

int main(int argc, char** argv) {
    const int vocab = argc > 1 ? atoi(argv[1]) : 248320;
    const int n     = argc > 2 ? atoi(argv[2]) : 16;
    const int M     = argc > 3 ? atoi(argv[3]) : 1024;

    std::vector<float> logits((size_t)vocab * n);
    std::mt19937 rng(123);
    std::normal_distribution<float> d(0, 4);
    for (auto& x : logits) x = d(rng);

    float* d_logits; int32_t* d_cand; void* d_scr;
    cudaMalloc(&d_logits, logits.size() * sizeof(float));
    cudaMalloc(&d_cand, (size_t)M * n * sizeof(int32_t));
    cudaMalloc(&d_scr, extract_topm_scratch_bytes(n));
    cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice);

    bool ok = extract_topm_cuda(d_logits, vocab, n, M, d_cand, d_scr, 0);
    cudaDeviceSynchronize();
    std::vector<int32_t> cand((size_t)M * n);
    cudaMemcpy(cand.data(), d_cand, cand.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // verify per position
    int tot_argmax_in = 0, tot_overlap = 0, tot_distinct_ok = 0;
    for (int p = 0; p < n; p++) {
        const float* col = logits.data() + (size_t)p * vocab;
        // true top-M
        std::vector<int> idx(vocab);
        for (int i = 0; i < vocab; i++) idx[i] = i;
        std::nth_element(idx.begin(), idx.begin() + M, idx.end(),
                         [&](int a, int b){ return col[a] > col[b]; });
        std::unordered_set<int> trueM(idx.begin(), idx.begin() + M);
        int argmax = (int)(std::max_element(col, col + vocab) - col);

        std::unordered_set<int> got;
        for (int c = 0; c < M; c++) got.insert(cand[(size_t)p * M + c]);
        if (got.count(argmax)) tot_argmax_in++;
        if ((int)got.size() == M) tot_distinct_ok++;
        int ov = 0; for (int id : got) if (trueM.count(id)) ov++;
        tot_overlap += ov;
    }
    printf("extract_topm vocab=%d n=%d M=%d ok=%d:\n", vocab, n, M, (int)ok);
    printf("  argmax-in-candidates : %d/%d positions\n", tot_argmax_in, n);
    printf("  exactly-M-distinct   : %d/%d positions\n", tot_distinct_ok, n);
    printf("  overlap with true top-M: %.2f%% (avg)\n", 100.0 * tot_overlap / ((double)M * n));

    // timing
    for (int i = 0; i < 10; i++) extract_topm_cuda(d_logits, vocab, n, M, d_cand, d_scr, 0);
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    const int it = 200;
    for (int i = 0; i < it; i++) extract_topm_cuda(d_logits, vocab, n, M, d_cand, d_scr, 0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    printf("  time: %.3f ms/call\n", ms / it);
    return 0;
}
