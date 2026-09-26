#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
namespace luce::common {
// Canonical NVFP4 safetensors -> GGML block_nvfp4 bodies with a private 5-byte
// prefix (F32 global DIVISOR, reserved zero). Reorders nibbles, preserves scale
// bytes and never rounds/decodes weights. Returned payload bytes are budgeted;
// caller-owned file and bounded JSON metadata/allocator overhead are separate.
std::vector<uint8_t> dsv41_repack_nvfp4(const uint8_t *,size_t,
    uint64_t rows,uint64_t columns,uint64_t output_budget);
}
