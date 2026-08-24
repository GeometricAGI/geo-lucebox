#pragma once

#include "common/ddtree.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace dflash::common {

inline int chain_decode_bucket_width(int lanes) {
    static constexpr int buckets[] = {
        1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
    };
    if (lanes <= 0) return 0;
    for (int bucket : buckets) {
        if (bucket >= lanes) return bucket;
    }
    return 64;
}

// draft_tokens[0] is the already-pending root; positions 1.. form the
// proposal. DDTree's flat indices then coincide with chain depth.
inline DDTree make_chain_verify_tree(
        const std::vector<int32_t> & draft_tokens) {
    DDTree tree;
    if (draft_tokens.empty()) return tree;

    tree.n_nodes = static_cast<int>(draft_tokens.size()) - 1;
    tree.token_ids.assign(draft_tokens.begin() + 1, draft_tokens.end());
    tree.depths.resize(static_cast<size_t>(tree.n_nodes));
    tree.parents.resize(static_cast<size_t>(tree.n_nodes) + 1);
    tree.child_maps.resize(static_cast<size_t>(tree.n_nodes) + 1);
    tree.parents[0] = -1;
    for (int node = 1; node <= tree.n_nodes; ++node) {
        tree.depths[static_cast<size_t>(node) - 1] = node;
        tree.parents[static_cast<size_t>(node)] = node - 1;
        tree.child_maps[static_cast<size_t>(node) - 1]
                       [tree.token_ids[static_cast<size_t>(node) - 1]] = node;
    }

    const int width = tree.n_nodes + 1;
    tree.visibility.assign(static_cast<size_t>(width) * width, 0);
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col <= row; ++col) {
            tree.visibility[static_cast<size_t>(row) * width + col] = 1;
        }
    }
    return tree;
}

struct ChainLaunchShape {
    int spec_lanes = 0;
    int tree_bucket = 0;
    int tree_rows = 0;
    int ar_lanes = 0;
    int accepted_rows = 0;
    int commit_rows = 0;
};

inline ChainLaunchShape chain_launch_shape(
        const std::vector<uint8_t> & admitted,
        const std::vector<int> & accepted_lengths,
        int tree_width) {
    ChainLaunchShape shape;
    const size_t count = admitted.size();
    for (size_t i = 0; i < count; ++i) {
        if (admitted[i]) {
            ++shape.spec_lanes;
            if (i < accepted_lengths.size()) {
                shape.accepted_rows += std::max(0, accepted_lengths[i]);
            }
        }
    }
    shape.ar_lanes = static_cast<int>(count) - shape.spec_lanes;
    shape.tree_bucket = chain_decode_bucket_width(shape.spec_lanes);
    shape.tree_rows = shape.tree_bucket * std::max(0, tree_width);
    shape.commit_rows = shape.accepted_rows + shape.ar_lanes;
    return shape;
}

// The pending root at path[0] was sampled by the preceding target step, so
// the ordinary sampler has already applied the min-token EOS floor to it.
// Accepted children would bypass that sampler. Stop before an EOS that is
// still below the floor so replay samples a replacement from the kept
// tip's exact logits. Once the floor is met, keep the EOS itself but discard
// deeper accepted tokens that the scheduler would hide after retirement.
template <typename IsEos>
inline size_t chain_min_tokens_safe_prefix(
        const std::vector<int32_t> & path,
        int generated_tokens_before_root,
        int min_tokens,
        IsEos is_eos) {
    const int generated = std::max(0, generated_tokens_before_root);
    for (size_t child = 1; child < path.size(); ++child) {
        if (!is_eos(path[child])) continue;
        return generated + static_cast<int>(child) < min_tokens
            ? child : child + 1;
    }
    return path.size();
}

}  // namespace dflash::common
