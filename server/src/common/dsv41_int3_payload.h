#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
namespace luce::common {
// Canonical I32 ten-code words + F32 group scales -> dense 3-bit groups.
// Private prefix F32 1.0 + reserved zero, then each group: F32 scale, LSB-first
// codes (code-4). Group64 uses28B; group128 uses52B. No weight rounding/refitting.
std::vector<uint8_t> dsv41_repack_int3(const uint8_t *,size_t,
    uint64_t rows,uint64_t columns,unsigned group,uint64_t output_budget);
}
