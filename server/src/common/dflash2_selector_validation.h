#pragma once

#include "geometric_draft_topk_cuda.h"

#include <cstdint>
#include <string>

namespace dflash::common {

// Host-only description of the selector tensors. Keeping validation in terms
// of dimensions makes it usable both while GGUF tensor descriptors are being
// loaded and when a concrete target lm_head is attached to the batched path.
struct DFlash2SelectorLayout {
    int rank = 0;
    int top_k = 0;
    int64_t hproj_rank = 0;
    int64_t pred_rank = 0;
    int64_t pred_vocab = 0;
    int64_t succ_rank = 0;
    int64_t succ_vocab = 0;
    // Zero means that source is unavailable at this validation point. A
    // partial target shard, for example, can declare n_vocab without owning
    // the final output tensor; the concrete lm_head is checked again at use.
    int64_t target_output_vocab = 0;
    int64_t target_declared_vocab = 0;
};

inline bool validate_dflash2_selector_layout(
        const DFlash2SelectorLayout & layout, std::string & error) {
    error.clear();
    if (layout.rank <= 0) {
        error = "DFlash 2 selector rank must be positive (got " +
            std::to_string(layout.rank) + ")";
        return false;
    }
    if (!geometric_draft_topk_cuda_supports_k(layout.top_k)) {
        error = "DFlash 2 selector top_k=" + std::to_string(layout.top_k) +
            " is unsupported; expected one of 1..8, 12, or 16";
        return false;
    }
    if (layout.hproj_rank != layout.rank ||
        layout.pred_rank != layout.rank ||
        layout.succ_rank != layout.rank) {
        error = "DFlash 2 selector rank mismatch: metadata=" +
            std::to_string(layout.rank) + " hproj=" +
            std::to_string(layout.hproj_rank) + " pred_cb=" +
            std::to_string(layout.pred_rank) + " succ_cb=" +
            std::to_string(layout.succ_rank);
        return false;
    }
    if (layout.pred_vocab <= 0 || layout.succ_vocab <= 0) {
        error = "DFlash 2 selector codebook vocab must be positive: pred_cb=" +
            std::to_string(layout.pred_vocab) + " succ_cb=" +
            std::to_string(layout.succ_vocab);
        return false;
    }
    if (layout.pred_vocab != layout.succ_vocab) {
        error = "DFlash 2 selector codebook vocab mismatch: pred_cb=" +
            std::to_string(layout.pred_vocab) + " succ_cb=" +
            std::to_string(layout.succ_vocab);
        return false;
    }
    if (layout.top_k > layout.pred_vocab) {
        error = "DFlash 2 selector top_k=" + std::to_string(layout.top_k) +
            " exceeds codebook vocab=" + std::to_string(layout.pred_vocab);
        return false;
    }
    if (layout.target_output_vocab > 0 &&
        layout.target_declared_vocab > 0 &&
        layout.target_output_vocab != layout.target_declared_vocab) {
        error = "DFlash 2 target vocab mismatch: output/lm_head=" +
            std::to_string(layout.target_output_vocab) + " target.n_vocab=" +
            std::to_string(layout.target_declared_vocab);
        return false;
    }
    if (layout.target_output_vocab > 0 &&
        layout.pred_vocab != layout.target_output_vocab) {
        error = "DFlash 2 selector vocab mismatch: codebook=" +
            std::to_string(layout.pred_vocab) + " target output/lm_head=" +
            std::to_string(layout.target_output_vocab);
        return false;
    }
    if (layout.target_declared_vocab > 0 &&
        layout.pred_vocab != layout.target_declared_vocab) {
        error = "DFlash 2 selector vocab mismatch: codebook=" +
            std::to_string(layout.pred_vocab) + " target.n_vocab=" +
            std::to_string(layout.target_declared_vocab);
        return false;
    }
    return true;
}

}  // namespace dflash::common
