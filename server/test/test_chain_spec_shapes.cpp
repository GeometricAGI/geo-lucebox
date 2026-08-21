#include "common/concurrency/chain_spec_shapes.h"
#include "host_check.h"

#include <cstdio>
#include <vector>

using namespace dflash::common;

static int g_checks = 0;

int main() {
    const std::vector<int32_t> draft = {10, 11, 12, 13};
    const DDTree tree = make_chain_verify_tree(draft);
    CHECK(tree.n_nodes == 3);
    CHECK((tree.token_ids == std::vector<int32_t>{11, 12, 13}));
    CHECK((tree.depths == std::vector<int>{1, 2, 3}));
    CHECK((tree.parents == std::vector<int>{-1, 0, 1, 2}));

    const std::vector<int> bucket_inputs = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17,
    };
    const std::vector<int> bucket_expected = {
        1, 2, 3, 4, 6, 6, 8, 8, 12, 12, 16, 16, 24,
    };
    for (size_t i = 0; i < bucket_inputs.size(); ++i) {
        CHECK(chain_decode_bucket_width(bucket_inputs[i]) ==
              bucket_expected[i]);
    }

    int pending = -1;
    const int32_t full_posterior[] = {11, 12, 13, 14};
    std::vector<int> accepted =
        follow_verified_tree(tree, full_posterior, pending);
    CHECK((accepted == std::vector<int>{0, 1, 2, 3}));
    CHECK(pending == 14);

    const int32_t rejected_posterior[] = {11, 99, 13, 14};
    accepted = follow_verified_tree(tree, rejected_posterior, pending);
    CHECK((accepted == std::vector<int>{0, 1}));
    CHECK(pending == 99);

    CHECK(truncate_verified_path(
        accepted, 1, rejected_posterior, pending));
    CHECK((accepted == std::vector<int>{0}));
    CHECK(pending == 11);

    const ChainLaunchShape mixed = chain_launch_shape(
        {1, 0, 1, 0, 0, 0}, {4, 0, 2, 0, 0, 0}, 16);
    CHECK(mixed.spec_lanes == 2);
    CHECK(mixed.tree_bucket == 2);
    CHECK(mixed.tree_rows == 32);
    CHECK(mixed.ar_lanes == 4);
    CHECK(mixed.accepted_rows == 6);
    CHECK(mixed.commit_rows == 10);

    const ChainLaunchShape all_spec = chain_launch_shape(
        {1, 1, 1}, {1, 2, 3}, 16);
    CHECK(all_spec.tree_bucket == 3);
    CHECK(all_spec.ar_lanes == 0);
    CHECK(all_spec.commit_rows == 6);

    const ChainLaunchShape ar_after_spec_failures = chain_launch_shape(
        {0, 0}, {0, 0}, 16);
    CHECK(ar_after_spec_failures.spec_lanes == 0);
    CHECK(ar_after_spec_failures.tree_bucket == 0);
    CHECK(ar_after_spec_failures.tree_rows == 0);
    CHECK(ar_after_spec_failures.ar_lanes == 2);
    CHECK(ar_after_spec_failures.commit_rows == 2);

    const auto eos = [](int32_t token) { return token == 2; };
    const std::vector<int32_t> eos_first_child = {10, 2, 11};
    CHECK(chain_min_tokens_safe_prefix(
        eos_first_child, 0, 3, eos) == 1);
    CHECK(chain_min_tokens_safe_prefix(
        eos_first_child, 2, 3, eos) == 2);
    const std::vector<int32_t> eos_second_child = {10, 11, 2, 12};
    CHECK(chain_min_tokens_safe_prefix(
        eos_second_child, 0, 3, eos) == 2);
    CHECK(chain_min_tokens_safe_prefix(
        eos_second_child, 1, 3, eos) == 3);
    const std::vector<int32_t> eos_root = {2, 11, 12};
    CHECK(chain_min_tokens_safe_prefix(
        eos_root, 0, 3, eos) == eos_root.size());
    CHECK(chain_min_tokens_safe_prefix(
        eos_first_child, 0, 0, eos) == 2);

    std::printf("chain spec shape tests passed: %d checks\n", g_checks);
    return 0;
}
