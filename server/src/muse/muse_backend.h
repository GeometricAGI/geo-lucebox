// Muse-Glimmer ModelBackend: chunked prefill, autoregressive decode, and
// greedy-chain DFlash speculative decode against the vendor drafter.
//
// Scope is still narrow: layer split, expert offload, conversation snapshots,
// prompt compression and park/unpark have no implementation here, and the
// honest way to express that is to refuse those calls rather than stub them
// into silent no-ops — a snapshot that "succeeds" and restores nothing would
// corrupt a conversation instead of failing a request. The capability table in
// model_capabilities.h records the same facts for the CLI, and
// backend_factory.cpp cross-checks the two at compile time.

#pragma once

#include "common/model_backend.h"
#include "common/dflash_feature_ring.h"
#include "placement/placement_config.h"
#include "muse_internal.h"
#include "muse_dflash_target.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace dflash::common {

struct MuseBackendConfig {
    const char *    model_path = nullptr;
    DevicePlacement device;
    int             stream_fd  = -1;
    // Prefill chunk. Also sizes the SWA ring's headroom (ring = window +
    // chunk), so it is a correctness parameter, not only a throughput knob:
    // muse_step refuses a chunk larger than the headroom it was built for.
    int             chunk      = 512;

    // `--fa-window`: cap how far back the FULL-attention layers look during
    // decode (0 = unlimited, the default). Monolithic only -- there is no
    // layer-split adapter for this family to forward it through.
    //
    // Off by default on purpose: this model has only 13 full-attention layers
    // and they carry its global context, so a finite window there is a real
    // quality risk (dropping the system prompt and tool definitions out of
    // view is how tool calling breaks), not a free speedup.
    int             fa_window  = 0;

    // ── Speculative decode (`--draft`) ─────────────────────────────────
    // The vendor DFlash drafter (`dflash-kquant.gguf`, arch `dflash`, 5
    // blocks, n_embd 6656 matching the target). Monolithic only: there is no
    // layer-split adapter for this family, so the capability row is kMono.
    //
    // Adding these fields is what forces `model_capabilities.h`'s
    // muse-glimmer row to change — the DFLASH_CHECK_ARCH static_asserts in
    // backend_factory.cpp cross-check field presence against the table, so
    // the row cannot claim less (or more) than the struct can carry.
    const char *    draft_path    = nullptr;
    int             draft_gpu     = -1;    // <0 → same GPU as the target
    int             draft_ctx_max = 2048;  // feature-mirror / draft context cap
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

    void free_drafter() override { free_decode_draft(); }

    // Diagnostic hook, not a tuning knob. The drafter emits hidden states that
    // already carry its own final RMS norm, so projecting them through the
    // target's `out_norm` as well is WRONG — and wrong in the quiet way:
    // verification still governs what is emitted, so the only symptom is a
    // lower acceptance rate. Exposing the switch lets a test measure that
    // instead of taking the claim on faith, and gives the spec-decode test its
    // mechanism control (acceptance must move, output must not).
    // Returns false when no drafter is loaded.
    // Diagnostic hook: collect the target's own top-2 logit margin for every
    // token speculative decode commits. Pass nullptr to stop collecting.
    // `test_muse_spec_decode` uses it to separate a broken accept rule from a
    // tied distribution whose tie the batched and single-token matmul kernels
    // break differently — see do_spec_decode. Off on the serving path: it
    // costs a vocab-wide host scan per committed token.
    void spec_collect_margins(std::vector<float> * out) {
        commit_margins_ = out;
        if (out) out->clear();
        // Margins need the verify logits kept around; the greedy serving path
        // otherwise reads back only the per-position argmax.
        if (dflash_target_) dflash_target_->set_keep_verify_logits(out != nullptr);
    }

    bool spec_set_project_out_norm(bool on) {
        if (!dflash_target_) return false;
        dflash_target_->set_apply_out_norm(on);
        return true;
    }

    // Release GPU-side state. Idempotent: the destructor calls it too, and
    // the daemon may call it first.
    void shutdown() override;

private:
    // ── Speculative decode ─────────────────────────────────────────────
    bool load_decode_draft();
    void free_decode_draft();
    bool spec_decode_ready() const {
        return dflash_target_ && draft_backend_ && feature_mirror_.target_feat;
    }

    // Greedy chain speculative decode. Emits into `out_tokens` and returns
    // false only on a hard backend failure — a zero-acceptance round is a
    // performance outcome, not an error.
    // `commit_margins`, when non-null, receives the target's own top-2 logit
    // margin at each committed position. Off by default because it costs a
    // full host-side scan of the verify logits; `test_muse_spec_decode` needs
    // it to tell "spec decode drifted from greedy because it is broken" from
    // "the target was tied there and the two matmul kernels broke the tie
    // differently", which is the one divergence no implementation can avoid.
    bool do_spec_decode(int committed, int n_gen, int32_t seed_tok,
                        std::vector<int32_t> & out_tokens,
                        const DaemonIO & io,
                        bool * visible_emitted,
                        float * accept_rate_out,
                        std::vector<float> * commit_margins = nullptr);

    // Autoregressive greedy/sampled decode. `seed_tok` is the token to place
    // at `committed` (the prefill argmax, or a sample of it).
    bool do_ar_decode(int committed, int n_gen, int32_t seed_tok,
                      const GenerateRequest & req,
                      std::vector<int32_t> & out_tokens,
                      const DaemonIO & io,
                      bool * visible_emitted,
                      GenerateResult & result);

    MuseBackendConfig cfg_;
    MuseWeights       w_;
    MuseCache         cache_;
    ggml_backend_t    backend_ = nullptr;
    bool              parked_  = false;
    bool              loaded_  = false;
    std::mt19937_64   rng_{0};

    // Draft side. All null when no `--draft` was given, which is the only
    // state the AR path needs to distinguish.
    ggml_backend_t     draft_backend_ = nullptr;
    DraftWeights       dw_;
    DraftFeatureMirror feature_mirror_;
    MuseDFlashTarget * dflash_target_ = nullptr;
    std::vector<float> * commit_margins_ = nullptr;   // diagnostic sink, non-owning
};

}  // namespace dflash::common
