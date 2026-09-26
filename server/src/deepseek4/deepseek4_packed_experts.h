#pragma once
#include <string>
namespace luce::common {
struct DeepSeek4Weights;
struct DeepSeek4PackedExperts;
// Resident, immutable per-expert weights. A failed load publishes no tensors.
bool load_deepseek4_packed_experts(DeepSeek4Weights & weights, const std::string & directory,
    const std::string & source_sha256, const std::string & manifest_sha256,
    const std::string & native_index_sha256, std::string & error);
}
