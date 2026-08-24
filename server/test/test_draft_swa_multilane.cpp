#include "common/dflash_draft_kv.h"
#include "common/draft_swa.h"
#include "internal.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>

using namespace dflash::common;

namespace {

constexpr int N_LANES = 3;
constexpr int CAPACITY = 160;
constexpr int SWA_WINDOW = 64;
constexpr uint16_t F16_ZERO = 0x0000;
constexpr uint16_t F16_NEG_INF = 0xFC00;
constexpr float MAX_ABS_ERROR = 5.0e-4f;

struct Resources {
    ggml_backend_t backend = nullptr;
    DraftWeights weights;
    std::array<DraftKvState, N_LANES> states;
    DraftKvBatchGraph batch;

    ~Resources() {
        draft_kv_batch_free(batch);
        for (DraftKvState & state : states) {
            draft_kv_free(state);
        }
        free_draft_weights(weights);
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};

std::vector<bool> layer_pattern(const DraftWeights & weights) {
    std::vector<bool> pattern;
    pattern.reserve(weights.layers.size());
    for (const DraftLayer & layer : weights.layers) {
        pattern.push_back(layer.is_swa);
    }
    return pattern;
}

bool check_lane_mask(const DraftKvState & state, int committed) {
    const int window_start = committed - SWA_WINDOW;
    for (int query = 0; query < state.q_len; ++query) {
        const size_t row = static_cast<size_t>(query) * state.kv_total;
        for (int slot = 0; slot < state.cap; ++slot) {
            const int position = state.slot_pos[static_cast<size_t>(slot)];
            const bool visible =
                position >= window_start && position < committed;
            const uint16_t expected = visible ? F16_ZERO : F16_NEG_INF;
            if (state.mask_hbuf[row + static_cast<size_t>(slot)] != expected) {
                std::fprintf(stderr,
                    "lane mask mismatch committed=%d query=%d slot=%d pos=%d\n",
                    committed, query, slot, position);
                return false;
            }
        }
        for (int noise = 0; noise < state.q_len; ++noise) {
            const uint16_t expected = noise <= query ? F16_ZERO : F16_NEG_INF;
            if (state.mask_hbuf[
                    row + static_cast<size_t>(state.cap + noise)] != expected) {
                std::fprintf(stderr,
                    "lane causal mask mismatch committed=%d query=%d noise=%d\n",
                    committed, query, noise);
                return false;
            }
        }
    }
    return true;
}

float max_abs_diff(const std::vector<float> & lhs,
                   const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size()) {
        return INFINITY;
    }
    float max_diff = 0.0f;
    for (size_t index = 0; index < lhs.size(); ++index) {
        if (!std::isfinite(lhs[index]) || !std::isfinite(rhs[index])) {
            return INFINITY;
        }
        max_diff = std::max(max_diff, std::fabs(lhs[index] - rhs[index]));
    }
    return max_diff;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 1) {
        std::fprintf(stderr,
            "usage: %s <dflash2.gguf> [gpu]\n", argv[0]);
        return 77;
    }
    if (argc > 3) {
        return 2;
    }

    const int gpu = argc == 3 ? std::atoi(argv[2]) : 0;
    Resources resources;
    resources.backend = ggml_backend_cuda_init(gpu);
    if (!resources.backend) {
        std::fprintf(stderr, "draft SWA qualification: GPU %d unavailable\n", gpu);
        return 1;
    }
    if (!load_draft_gguf(argv[1], resources.backend, resources.weights)) {
        std::fprintf(stderr, "draft SWA qualification: %s\n",
                     dflash27b_last_error());
        return 1;
    }

    const std::vector<bool> trained_pattern = layer_pattern(resources.weights);
    if (!resources.weights.swa_pattern_loaded) {
        std::fprintf(stderr, "draft SWA qualification: GGUF has no SWA pattern\n");
        return 1;
    }
    const DraftSwaOverrideResult swa =
        apply_draft_swa_window_override(resources.weights, SWA_WINDOW);
    if (layer_pattern(resources.weights) != trained_pattern ||
        swa.effective_window != SWA_WINDOW || swa.swa_layers == 0) {
        std::fprintf(stderr,
            "draft SWA qualification: override changed the trained pattern\n");
        return 1;
    }

    const std::array<int, N_LANES> committed = {65, 81, 133};
    const size_t hidden_elements =
        static_cast<size_t>(resources.weights.n_embd) *
        static_cast<size_t>(resources.weights.block_size);
    std::array<std::vector<float>, N_LANES> single_hidden;
    DraftFeatureMirror unused_ring;

    for (int lane = 0; lane < N_LANES; ++lane) {
        DraftKvState & state = resources.states[static_cast<size_t>(lane)];
        if (!draft_kv_init(state, resources.weights, resources.backend,
                           CAPACITY, nullptr)) {
            std::fprintf(stderr, "draft SWA qualification: lane %d init failed\n", lane);
            return 1;
        }
        for (int position = 0; position < committed[static_cast<size_t>(lane)];
             ++position) {
            state.slot_pos[static_cast<size_t>(position % CAPACITY)] = position;
        }
        state.next_pos = committed[static_cast<size_t>(lane)];
        if (!draft_kv_begin_step(
                state, resources.weights, resources.backend, unused_ring,
                committed[static_cast<size_t>(lane)]) ||
            !check_lane_mask(state, committed[static_cast<size_t>(lane)])) {
            return 1;
        }

        std::vector<float> embedding(hidden_elements);
        for (size_t index = 0; index < embedding.size(); ++index) {
            embedding[index] =
                0.01f * static_cast<float>(lane + 1) +
                0.0001f * static_cast<float>(index % 31);
        }
        ggml_backend_tensor_set(state.inp_embed, embedding.data(), 0,
                                embedding.size() * sizeof(float));
        if (ggml_backend_graph_compute(resources.backend, state.gf) !=
            GGML_STATUS_SUCCESS) {
            std::fprintf(stderr,
                         "draft SWA qualification: lane %d compute failed\n", lane);
            return 1;
        }
        single_hidden[static_cast<size_t>(lane)].resize(hidden_elements);
        ggml_backend_tensor_get(
            state.hidden_states,
            single_hidden[static_cast<size_t>(lane)].data(), 0,
            hidden_elements * sizeof(float));
    }

    if (max_abs_diff(single_hidden[0], single_hidden[1]) <= MAX_ABS_ERROR) {
        std::fprintf(stderr,
            "draft SWA qualification: distinct lane inputs collapsed\n");
        return 1;
    }

    std::vector<DraftKvState *> lane_states;
    lane_states.reserve(N_LANES);
    for (DraftKvState & state : resources.states) {
        lane_states.push_back(&state);
    }
    std::vector<std::vector<float>> packed_hidden;
    if (!draft_kv_batch_compute(resources.batch, resources.weights,
                                resources.backend, lane_states, packed_hidden)) {
        return 1;
    }

    for (int lane = 0; lane < N_LANES; ++lane) {
        const float error = max_abs_diff(
            single_hidden[static_cast<size_t>(lane)],
            packed_hidden[static_cast<size_t>(lane)]);
        std::printf("draft SWA lane=%d committed=%d max_abs=%.6g\n",
                    lane, committed[static_cast<size_t>(lane)], error);
        if (error > MAX_ABS_ERROR) {
            std::fprintf(stderr,
                "draft SWA qualification: lane %d exceeds tolerance %.6g\n",
                lane, MAX_ABS_ERROR);
            return 1;
        }
    }

    std::printf("draft SWA post-window three-lane qualification passed\n");
    return 0;
}
