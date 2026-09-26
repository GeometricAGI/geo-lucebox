#pragma once
#include "common.cuh"
bool ggml_cuda_mxfp4_f32_supported(const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * y);
void ggml_cuda_mxfp4_f32(const ggml_tensor * w, const ggml_tensor * x, ggml_tensor * y, cudaStream_t stream);
