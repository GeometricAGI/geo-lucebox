#pragma once
#include "ggml-backend.h"
#include "dsv41_direct_io.h"
#include "dsv41_payload_cache.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace luce::common {
// Owned dense native tensor (BF16/F32/I32/I64), source rank 0..4.
// Source dimensions are reversed into GGML order. Optional exact BF16->F32
// promotion; no quantization.
// Same backend/graph lifetime rules as Dsv41NativeMatrix apply.
class Dsv41NativeTensor {
public:
    ~Dsv41NativeTensor();
    ggml_tensor * tensor()const;
    uint64_t device_bytes()const;
    Dsv41NativeTensor(const Dsv41NativeTensor &)=delete;
    Dsv41NativeTensor & operator=(const Dsv41NativeTensor &)=delete;
private:
    friend class Dsv41NativeStore;
    Dsv41NativeTensor();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// One native FP4 or FP8 matrix loaded from an indexed safetensors checkpoint.
// FP4 is losslessly repacked to MXFP4; FP8 remains raw I8 plus E8M0 scales.
// Backend must outlive this owner. Destroy all graphs referring to its tensors
// before eviction; synchronization alone does not invalidate captured graphs.
class Dsv41NativeMatrix {
public:
    ~Dsv41NativeMatrix();
    ggml_tensor * weight() const;
    ggml_tensor * scales() const; // null for MXFP4
    uint64_t device_bytes() const;
    Dsv41NativeMatrix(const Dsv41NativeMatrix &)=delete;
    Dsv41NativeMatrix & operator=(const Dsv41NativeMatrix &)=delete;
private:
    friend class Dsv41NativeStore;
    Dsv41NativeMatrix();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Index SHA256 pins tensor-to-shard mapping, not shard contents: source shards
// must be independently authenticated and immutable while the store is used.
// Per-load host budget covers raw weight, scales and FP4 repack simultaneously.
// Bounded JSON metadata, context and allocator overhead need a separate reserve.
struct Dsv41NativeLoadStats {
    uint64_t matrices=0,payload_bytes=0;
    double metadata_seconds=0,read_seconds=0,validation_repack_seconds=0,allocation_seconds=0,upload_seconds=0;
};
struct Dsv41NativeEngram { std::string path; uint64_t weight_offset, scale_offset, rows; };
class Dsv41NativeStore {
public:
    // Linux O_DIRECT payloads: reserve this fixed scratch per concurrent load
    // in addition to payload budgets. Unsupported filesystems fail explicitly;
    // no silent buffered fallback. Index/shard headers use bounded buffered I/O.
    static constexpr size_t io_scratch_bytes=dsv41_io_scratch_bytes;
    Dsv41NativeStore(const std::string & directory,const std::string & expected_index_sha256,
                     std::shared_ptr<Dsv41PayloadCache> payload_cache={});
    ~Dsv41NativeStore();
    // Successful matrix loads only, excluding global dense/embedding loads.
    Dsv41NativeLoadStats load_stats()const;
    Dsv41NativeEngram engram(const std::string & name, uint64_t rows) const;
    // Exact tensor name, dtype and source-order shape are supplied by the model
    // descriptor. Optional exact BF16->F32 promotion for native compressor weights.
    // Host budget covers source+promoted bytes simultaneously (3x source).
    std::unique_ptr<Dsv41NativeTensor> load_tensor(const std::string & name,
        ggml_type type,const std::vector<uint64_t> & source_shape,
        ggml_backend_t,uint64_t host_payload_budget,uint64_t device_budget,
        bool promote_bf16_to_f32=false)const;
    // Gather <=32 BF16 embedding rows directly from disk. Device shape is
    // [columns, row_ids.size()], preserving duplicates and caller order. Host
    // payload cap covers one row; fixed direct-I/O scratch remains separate.
    std::unique_ptr<Dsv41NativeTensor> load_bf16_rows(const std::string & name,
        uint64_t rows,uint64_t columns,const std::vector<uint32_t> & row_ids,
        ggml_backend_t,uint64_t host_payload_budget,uint64_t device_budget)const;
    std::unique_ptr<Dsv41NativeMatrix> load(const std::string & name,
        uint64_t rows,uint64_t columns,ggml_backend_t backend,
        uint64_t host_payload_budget,uint64_t device_budget, bool cache_payload=true) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
