#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
namespace luce::common {
struct Dsv41ScalePaletteInfo {
    uint64_t blocks=0, wire_bytes=0, stored_bytes=0;
    uint16_t palette_size=0;
    uint8_t mode=0, index_bits=0;
};
// Validates the canonical GQSP envelope, dimensions, indices and padding without
// allocation. This is a lossless transport codec, not validation of the restored
// quantizer wire (global scale, grid, ternary digits, etc.).
Dsv41ScalePaletteInfo dsv41_inspect_scale_palette(const uint8_t *,size_t,
    uint64_t rows,uint64_t columns,uint32_t block_bytes);
// Expands ONE requested matrix into its original packed wire, including the
// 5-byte global header. New payload is exactly wire_bytes; no decoded weights
// or index arrays. Budget excludes the caller-owned compressed bytes and AWQ
// metadata and allocator overhead. Caller must account those separately and evict this scratch/cache.
std::vector<uint8_t> dsv41_expand_scale_palette(const uint8_t *,size_t,
    uint64_t rows,uint64_t columns,uint32_t block_bytes,uint64_t output_budget);
}
