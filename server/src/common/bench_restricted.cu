// Standalone correctness + timing for restricted_lm_head_q6k.
//   nvcc -O3 -arch=sm_86 -o /tmp/bench_restricted \
//       server/src/common/bench_restricted.cu server/src/common/restricted_lm_head_cuda.cu
//   /tmp/bench_restricted
#include "restricted_lm_head_cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using dflash::common::restricted_lm_head_q6k;

static constexpr int QK = 256;
struct BlockQ6K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
static_assert(sizeof(BlockQ6K) == 210, "layout");

// CPU dequant of weight j∈[0,256) in block b — mirrors dequantize_row_q6_K.
static float cpu_dequant(const BlockQ6K& b, int j) {
    float d = __half2float(*reinterpret_cast<const __half*>(&b.d));
    int half = j >> 7, jj = j & 127, group = jj >> 5, l = jj & 31, is = l >> 4;
    int qlb = half * 64, qhb = half * 32, scb = half * 8;
    uint8_t qh = b.qh[qhb + l];
    int q, si;
    switch (group) {
        case 0: q = (b.ql[qlb + l]      & 0xF) | (((qh >> 0) & 3) << 4); si = scb + is + 0; break;
        case 1: q = (b.ql[qlb + l + 32] & 0xF) | (((qh >> 2) & 3) << 4); si = scb + is + 2; break;
        case 2: q = (b.ql[qlb + l]      >> 4)  | (((qh >> 4) & 3) << 4); si = scb + is + 4; break;
        default:q = (b.ql[qlb + l + 32] >> 4)  | (((qh >> 6) & 3) << 4); si = scb + is + 6; break;
    }
    return d * (float)b.scales[si] * (float)(q - 32);
}

int main(int argc, char** argv) {
    // ---- correctness on small dims ----
    {
        const int n_embd = 5120, n_vocab = 3000, n_tokens = 5, M = 64;
        const int bpr = n_embd / QK;
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> byte(0, 255);
        std::vector<BlockQ6K> head((size_t)n_vocab * bpr);
        for (auto& b : head) {
            for (auto& x : b.ql) x = byte(rng);
            for (auto& x : b.qh) x = byte(rng);
            for (auto& x : b.scales) x = (int8_t)byte(rng);
            b.d = __half_as_ushort(__float2half(0.005f + 0.001f * (byte(rng) / 255.0f)));
        }
        std::normal_distribution<float> nf(0, 1);
        std::vector<float> hidden((size_t)n_embd * n_tokens);
        for (auto& x : hidden) x = nf(rng);
        std::vector<int32_t> cand((size_t)M * n_tokens);
        for (int p = 0; p < n_tokens; p++) {
            // distinct random ids per position
            std::vector<int> ids(n_vocab); for (int i = 0; i < n_vocab; i++) ids[i] = i;
            std::shuffle(ids.begin(), ids.end(), rng);
            for (int c = 0; c < M; c++) cand[(size_t)p * M + c] = ids[c];
        }
        // CPU reference argmax over candidates
        std::vector<int32_t> ref(n_tokens);
        for (int p = 0; p < n_tokens; p++) {
            float best = -1e30f; int bestid = -1, bestc = -1;
            for (int c = 0; c < M; c++) {
                int id = cand[(size_t)p * M + c];
                const BlockQ6K* row = head.data() + (size_t)id * bpr;
                double acc = 0;
                for (int e = 0; e < n_embd; e++)
                    acc += (double)cpu_dequant(row[e >> 8], e & 255) * hidden[(size_t)p * n_embd + e];
                if ((float)acc > best || ((float)acc == best && c < bestc)) { best = (float)acc; bestid = id; bestc = c; }
            }
            ref[p] = bestid;
        }
        // GPU
        void *d_head, *d_keys; float* d_hidden; int32_t *d_cand, *d_out;
        cudaMalloc(&d_head, head.size() * sizeof(BlockQ6K));
        cudaMalloc(&d_hidden, hidden.size() * sizeof(float));
        cudaMalloc(&d_cand, cand.size() * sizeof(int32_t));
        cudaMalloc(&d_out, n_tokens * sizeof(int32_t));
        cudaMalloc(&d_keys, n_tokens * sizeof(uint64_t));
        cudaMemcpy(d_head, head.data(), head.size() * sizeof(BlockQ6K), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cand, cand.data(), cand.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        bool ok = restricted_lm_head_q6k(d_head, n_embd, n_vocab, d_hidden, n_tokens,
                                         d_cand, M, d_out, d_keys, 0);
        cudaDeviceSynchronize();
        std::vector<int32_t> got(n_tokens);
        cudaMemcpy(got.data(), d_out, n_tokens * sizeof(int32_t), cudaMemcpyDeviceToHost);
        int mism = 0;
        for (int p = 0; p < n_tokens; p++) {
            if (got[p] != ref[p]) { printf("  MISMATCH p=%d gpu=%d cpu=%d\n", p, got[p], ref[p]); mism++; }
        }
        printf("correctness (n_embd=%d n_vocab=%d n_tok=%d M=%d): ok=%d mismatches=%d\n",
               n_embd, n_vocab, n_tokens, M, (int)ok, mism);
        cudaFree(d_head); cudaFree(d_hidden); cudaFree(d_cand); cudaFree(d_out); cudaFree(d_keys);
    }

    // ---- timing at realistic dims: restricted-M vs full-vocab via same kernel ----
    {
        const int n_embd = 5120, n_vocab = 248320, n_tokens = 16;
        const int bpr = n_embd / QK;
        std::mt19937 rng(1);
        std::uniform_int_distribution<int> byte(0, 255);
        std::vector<BlockQ6K> head((size_t)n_vocab * bpr);
        for (auto& b : head) {
            for (auto& x : b.ql) x = byte(rng);
            for (auto& x : b.qh) x = byte(rng);
            for (auto& x : b.scales) x = (int8_t)byte(rng);
            b.d = __half_as_ushort(__float2half(0.005f));
        }
        std::vector<float> hidden((size_t)n_embd * n_tokens, 0.01f);
        void *d_head, *d_keys; float* d_hidden; int32_t *d_cand, *d_out;
        cudaMalloc(&d_head, head.size() * sizeof(BlockQ6K));
        cudaMalloc(&d_hidden, hidden.size() * sizeof(float));
        cudaMalloc(&d_out, n_tokens * sizeof(int32_t));
        cudaMalloc(&d_keys, n_tokens * sizeof(uint64_t));
        cudaMemcpy(d_head, head.data(), head.size() * sizeof(BlockQ6K), cudaMemcpyHostToDevice);

        auto time_M = [&](int M, int iters) -> float {
            std::vector<int32_t> cand((size_t)M * n_tokens);
            for (int p = 0; p < n_tokens; p++)
                for (int c = 0; c < M; c++) cand[(size_t)p * M + c] = (c * 7 + p) % n_vocab;
            cudaMalloc(&d_cand, cand.size() * sizeof(int32_t));
            cudaMemcpy(d_cand, cand.data(), cand.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
            for (int i = 0; i < 5; i++)
                restricted_lm_head_q6k(d_head, n_embd, n_vocab, d_hidden, n_tokens, d_cand, M, d_out, d_keys, 0);
            cudaDeviceSynchronize();
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0);
            for (int i = 0; i < iters; i++)
                restricted_lm_head_q6k(d_head, n_embd, n_vocab, d_hidden, n_tokens, d_cand, M, d_out, d_keys, 0);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
            cudaFree(d_cand);
            return ms / iters;
        };
        printf("\ntiming (n_embd=%d n_vocab=%d n_tok=%d):\n", n_embd, n_vocab, n_tokens);
        printf("  full-vocab head (M=%d) : %.3f ms\n", n_vocab, time_M(n_vocab, 30));
        for (int M : {256, 1024, 2048, 8192})
            printf("  restricted M=%-6d     : %.3f ms\n", M, time_M(M, 100));
        cudaFree(d_head); cudaFree(d_hidden); cudaFree(d_out); cudaFree(d_keys);
    }
    return 0;
}
