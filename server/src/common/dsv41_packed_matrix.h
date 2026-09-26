#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
#include <memory>
namespace luce::common {
// Owns ONE materialized research GQH/NVFP4/INT3 matrix and its scale metadata. Input
// payload identity is verified by the artifact reader before calling. Payload
// may be raw research wire or canonical GQSP, never the outer AWQ safetensors wrapper.
// NVFP4 accepts dsv41_repack_nvfp4's private wire; its header is a global divisor,
// not a multiplier. It must not be interpreted as a GQH wire or ordinary GGUF.
// INT3 accepts dsv41_repack_int3's private identity prefix and densely packed
// group64/group128 codes with original F32 scales; no AWQ or palette metadata.
// No decoded BF16/F32 weight copy. Backend must outlive this object; serialized
// owner evicts it only when no future graph can reference its tensor.
// Host budget covers expanded wire scratch plus copied AWQ values; it excludes
// caller input buffers, GGML context, registry containers and allocator overhead.
// Device budget covers the backend-reported weight and AWQ buffer allocation.
// An optional arena is borrowed and must outlive this matrix; its range must be
// disjoint from every other live matrix. The caller owns graph/arena lifetime.
// Scalar GQH2_H/3/4, NVFP4 and INT3 require GGML_PREC_RESEARCH_BF16_F32 on MUL_MAT;
// ordinary GQH matmul kernels do not implement this calibrated contract.
class Dsv41PackedMatrix {
public:
    Dsv41PackedMatrix(ggml_backend_t,ggml_type,uint64_t rows,uint64_t columns,
        const uint8_t * payload,size_t payload_bytes,bool palette,
        const float * awq,size_t awq_count,uint64_t host_scratch_budget,
        uint64_t device_budget, ggml_backend_buffer_t arena=nullptr, uint64_t arena_offset=0);
    // Exact aligned backend footprint, including optional conditioning columns.
    static uint64_t allocation_bytes(ggml_backend_t, ggml_type, uint64_t rows, uint64_t columns, bool awq);
    ~Dsv41PackedMatrix();
    Dsv41PackedMatrix(const Dsv41PackedMatrix &)=delete;
    Dsv41PackedMatrix & operator=(const Dsv41PackedMatrix &)=delete;
    ggml_tensor * tensor() const;
    uint64_t device_bytes() const;
    uint64_t metadata_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
