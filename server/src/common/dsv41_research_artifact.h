#pragma once
#include "dsv41_packed_matrix.h"
#include "dsv41_direct_io.h"
#include "dsv41_payload_cache.h"
#include <string>
namespace luce::common {
// Reads one selected replacement from canonical geoquant-v41-research-v1.
// Validates checkpoint identity and hashes the exact bytes bound to the backend.
// This does not load native weights or validate the entire artifact. Unsupported
// selected formats fail explicitly; never substitute a different quantization.
// Caller selects research/research_fp8 according to the ORIGINAL model module,
// independently of storage type. Shape is supplied by that model descriptor.
// Host payload budget covers file bytes, extracted AWQ values, registered AWQ
// values and expanded palette/repacked NVFP4 wire simultaneously. Bounded manifest/header JSON,
// GGML contexts, registry containers and allocator overhead require a separate
// reserve. This per-load budget is not a whole-model residency accountant.
class Dsv41ResearchArtifact {
public:
    // Linux direct payload I/O; additional fixed reserve per concurrent load.
    // Unsupported direct I/O fails; manifest/index reads remain buffered.
    static constexpr size_t io_scratch_bytes=dsv41_io_scratch_bytes;
    Dsv41ResearchArtifact(const std::string & directory,
                         const std::string & expected_source_index_sha256,
                         const std::string & expected_manifest_sha256="",
                         std::shared_ptr<Dsv41PayloadCache> payload_cache={});
    ~Dsv41ResearchArtifact();
    std::unique_ptr<Dsv41PackedMatrix> load(const std::string & name,
        uint64_t rows,uint64_t columns,ggml_backend_t backend,
        uint64_t host_payload_budget,uint64_t device_budget, bool cache_payload=true,
        ggml_backend_buffer_t arena=nullptr,uint64_t arena_offset=0) const;
    uint64_t allocation_bytes(const std::string & name,uint64_t rows,uint64_t columns,ggml_backend_t) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
