#pragma once
#include "common.cuh"
#include "../ggml-packed-experts.h"
void ggml_cuda_packed_experts(ggml_backend_cuda_context & ctx,ggml_tensor * dst);
