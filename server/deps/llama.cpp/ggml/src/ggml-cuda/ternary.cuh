#pragma once
#include "common.cuh"

bool ggml_cuda_ternary_type(ggml_type type);
bool ggml_cuda_ternary_supported(const ggml_tensor * weights, const ggml_tensor * inputs, const ggml_tensor * output);
void ggml_cuda_ternary_mul_mat(const ggml_tensor * weights, const ggml_tensor * inputs,
                              ggml_tensor * output, cudaStream_t stream);
