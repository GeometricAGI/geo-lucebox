// Muse-Glimmer ModelBackend: autoregressive prefill + decode.
//
// Scope is deliberately narrow for the first cut. The family ships without
// the machinery the older backends grew (speculative decode, layer split,
// expert offload, snapshots), and the honest way to express that is to refuse
// those calls rather than stub them into silent no-ops — a snapshot that
// "succeeds" and restores nothing would corrupt a conversation instead of
// failing a request. The capability table in model_capabilities.h records the
// same facts for the CLI, and backend_factory.cpp cross-checks the two at
// compile time.

#pragma once

#include "common/model_backend.h"
#include "placement/placement_config.h"
#include "muse_internal.h"

#include <memory>
#include <random>
#include <string>

namespace dflash::common {

struct MuseBackendConfig {
    const char *    model_path = nullptr;
    DevicePlacement device;
    int             stream_fd  = -1;
    // Prefill chunk. Also sizes the SWA ring's headroom (ring = window +
    // chunk), so it is a correctness parameter, not only a throughput knob:
    // muse_step refuses a chunk larger than the headroom it was built for.
    int             chunk      = 512;
};

class MuseBackend : public ModelBackend {
public:
    explicit MuseBackend(const MuseBackendConfig & cfg);
    ~MuseBackend() override;

    MuseBackend(const MuseBackend &)             = delete;
    MuseBackend & operator=(const MuseBackend &) = delete;

    bool init();

    void print_ready_banner() const override;

    bool park(ParkTarget target) override;
    bool unpark(ParkTarget target) override;
    bool is_target_parked() const override { return parked_; }

    GenerateResult generate_impl(const GenerateRequest & req,
                                 const DaemonIO & io) override;

    // Snapshots are not implemented for this family yet. Reporting an unused
    // slot and refusing to save keeps callers on the "no snapshot" path
    // instead of handing them a slot that restores garbage.
    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override {}
    bool snapshot_used(int slot) const override { return false; }
    int  snapshot_cur_pos(int slot) const override { return -1; }

    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & req,
                                             const DaemonIO & io) override;

    // Prompt compression (the "compress ..." daemon command) has no muse
    // implementation; refusing keeps the caller on the uncompressed path.
    bool handle_compress(const std::string & line, const DaemonIO & io) override;

    void free_drafter() override {}

    // Release GPU-side state. Idempotent: the destructor calls it too, and
    // the daemon may call it first.
    void shutdown() override;

private:
    MuseBackendConfig cfg_;
    MuseWeights       w_;
    MuseCache         cache_;
    ggml_backend_t    backend_ = nullptr;
    bool              parked_  = false;
    bool              loaded_  = false;
    std::mt19937_64   rng_{0};
};

}  // namespace dflash::common
