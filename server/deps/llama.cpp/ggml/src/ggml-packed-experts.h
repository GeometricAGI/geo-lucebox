#pragma once
#include "ggml.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Runtime-only descriptors. Pointers refer to immutable weights/scales on the
// executing backend. Owners must outlive graphs; descriptors cannot migrate
// between devices or be serialized as model weights.
struct ggml_packed_expert {
    uint64_t body;
    uint64_t input_scale;
    int32_t type;
    int32_t rows;
    int32_t columns;
    int32_t reserved;
    float global;
    float levels[16];
    float ratios[16];
};
GGML_API bool ggml_packed_expert_init(struct ggml_packed_expert * out, const struct ggml_tensor * weight);
GGML_API struct ggml_tensor * ggml_packed_experts_mul_mat_id(struct ggml_context * ctx,
    struct ggml_tensor * descriptors, struct ggml_tensor * input, struct ggml_tensor * ids, int64_t rows);
GGML_API bool ggml_packed_experts_is_op(const struct ggml_tensor * op);
struct ggml_backend_device;
GGML_API bool ggml_packed_experts_supports_device(const struct ggml_tensor * op, struct ggml_backend_device * device);
#ifdef __cplusplus
}
#endif
