#pragma once
#include <cstdint>
#include <vector>
namespace luce::common {
// Lossless numeric transcode: native adjacent E2M1 nibbles + per-row group32
// E8M0 scales -> GGML MXFP4 blocks (scale byte, low-half/high-half nibbles).
// No requantization, dense decode or weight-value arithmetic. Same 4.25 bpw.
// MXFP4 arithmetic kernels have a separate activation/numerical contract.
std::vector<uint8_t> dsv41_native_fp4_to_mxfp4(const std::vector<uint8_t> & packed,
    const std::vector<uint8_t> & scales,uint64_t rows,uint64_t columns,
    uint64_t output_budget);
}
