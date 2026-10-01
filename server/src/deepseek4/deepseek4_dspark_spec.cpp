// DeepSeek-V4-Flash DSpark speculative decode: DFlashTarget adapter + spec loop.
//
// The drafter (DSparkDrafter, deepseek4_dspark.{h,cpp}) proposes block_size
// candidates conditioned on captured target features; the DS4 target verifies
// them in one batched forward. Feature capture + verify live in
// deepseek4_dspark_verify_forward (deepseek4_graph.cpp). The DSpark Markov head
// (common/dspark_head.cpp) is target-agnostic and reused verbatim.
//
// Fast spec loop (default): ONE batched verify per step, verify width capped
// at DS4_SPEC_Q=4 tokens (seed + 3 candidates). With q <= ratio(4) the verify
// crosses at most one compression boundary and never aliases rolling-state
// rows, so rejection rollback needs no full KV snapshot:
//   - at-risk raw ring rows are saved before the verify and rejected rows are
//     restored after wrap (accepted rows remain committed),
//   - comp rows are index-addressed (pos / ratio)        -> idempotent,
//   - n_comp / n_index_comp are pure functions of commit position,
//   - rejected compressor ring rows are restored in both ratio-4 and
//     ratio-128 layers; a later pool reads these even before replacement,
//   - the ratio-4 prev-half is also restored when a rejected boundary flushed
//     current rows into it. Accepted boundaries and current rows are kept.
// The legacy full-snapshot + double-verify path is kept behind
// LUCE_DS4_FULL_SNAP=1 for A/B validation.

#include "deepseek4_dspark.h"
#include "deepseek4_internal.h"
#include "deepseek4_snapshot.h"
#include "deepseek4_roctx.h"
#include "internal.h"
#include "common/adaptive_spec_width.h"
#include "common/dspark_head.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

namespace luce::common {

// ── DFlashTarget adapter over the DS4 target ────────────────────────────
class DeepSeek4DFlashTarget : public DFlashTarget {
public:
    DeepSeek4DFlashTarget(const DeepSeek4Weights & w, DeepSeek4Cache & cache,
                          ggml_backend_t backend, int device, ggml_backend_t snap_backend,
                          std::vector<int> capture_ids, int mask_tok,
                          MoeHybridStorage * moe_hybrid,
                          MoeExpertComputeRuntime * expert_runtime,
                          MoeHybridRoutingStats * routing_stats)
        : w_(w), cache_(cache), backend_(backend), device_(device), snap_backend_(snap_backend),
          capture_ids_(std::move(capture_ids)), mask_tok_(mask_tok),
          moe_hybrid_(moe_hybrid), expert_runtime_(expert_runtime),
          routing_stats_(routing_stats) {}

    ~DeepSeek4DFlashTarget() override { clear_snapshot(); }

    bool verify_batch(const std::vector<int32_t> & tokens, int base_pos, int & last_tok,
                      std::vector<int32_t> * all_argmax = nullptr,
                      bool capture_ssm_intermediates = false) override {
        (void) capture_ssm_intermediates;
        const int n = (int) tokens.size();
        embed_buf_.resize((size_t) n * w_.n_embd);
        if (!w_.embedder.embed(tokens.data(), n, embed_buf_.data())) {
            std::fprintf(stderr, "[ds4-verify] embed FAILED n=%d tok0=%d tok1=%d vocab=%d\n",
                         n, n > 0 ? tokens[0] : -1, n > 1 ? tokens[1] : -1, w_.n_vocab);
            return false;
        }
        // Sequential verify: q single-token forwards through the same cached
        // graph as ordinary AR decode. This preserves target arithmetic; exact
        // rollback still requires a full snapshot and replay after rejection.
        // LUCE_DS4_SEQ_VERIFY is a diagnostic. The supported reference mode,
        // LUCE_DS4_SPEC_REFERENCE_EXACT, enables both requirements together.
        static const bool seq_verify = [] {
            const char * exact =
                std::getenv("LUCE_DS4_SPEC_REFERENCE_EXACT");
            const char * sequential =
                std::getenv("LUCE_DS4_SEQ_VERIFY");
            return (exact && *exact && *exact != '0') ||
                   (sequential && *sequential && *sequential != '0');
        }();
        if (seq_verify || force_sequential_) {
            boundary_checkpoint_.clear();
            std::vector<int32_t> am_all;
            std::vector<float> feat_all;
            std::vector<float> logits_all;
            am_all.reserve(n);
            for (int t = 0; t < n; t++) {
                std::vector<int32_t> am1;
                std::vector<float> feat1;
                std::vector<float> logits1;
                if (!deepseek4_dspark_verify_forward(backend_, device_, w_, cache_, capture_ids_,
                                                     embed_buf_.data() + (size_t) t * w_.n_embd,
                                                     tokens.data() + t, 1, base_pos + t, am1,
                                                     keep_logits_ ? &logits1 : nullptr,
                                                     feat1, telemetry_,
                                                     /*allow_graph_reuse=*/true,
                                                     moe_hybrid_, expert_runtime_,
                                                     routing_stats_, nullptr)) {
                    return false;
                }
                if (am1.empty()) return false;
                am_all.push_back(am1[0]);
                feat_all.insert(feat_all.end(), feat1.begin(), feat1.end());
                if (keep_logits_) logits_all.insert(logits_all.end(), logits1.begin(), logits1.end());
            }
            verify_features_ = std::move(feat_all);
            if (keep_logits_) verify_logits_ = std::move(logits_all);
            last_tok = am_all.back();
            verify_n_ = n;
            if (all_argmax) *all_argmax = std::move(am_all);
            return true;
        }
        std::vector<int32_t> am;
        if (diag_plain_decode_) {
            // Diagnostic: exactly the call the plain AR decode loop makes (deepseek4_step_layer_range with
            // no verify hooks, decode-graph reuse, the hybrid storage and expert runtime), one token at a time.
            std::vector<int32_t> am_all;
            for (int t = 0; t < n; ++t) {
                std::vector<float> logits, hc_state;
                if (!deepseek4_step_layer_range(backend_, device_, w_, cache_, hc_state,
                                                embed_buf_.data() + (size_t) t * w_.n_embd, 1, base_pos + t,
                                                0, w_.n_layer, &logits, tokens.data() + t, telemetry_,
                                                /*allow_decode_graph_reuse=*/true, nullptr,
                                                moe_hybrid_, expert_runtime_, routing_stats_)) return false;
                if ((int) logits.size() < w_.n_vocab) return false;
                int best = 0;
                for (int i = 1; i < w_.n_vocab; i++) if (logits[(size_t) i] > logits[(size_t) best]) best = i;
                am_all.push_back(best);
            }
            last_tok = am_all.back();
            verify_n_ = n;
            if (all_argmax) *all_argmax = std::move(am_all);
            return true;
        }
        if (diag_plain_ar_) {
            // Diagnostic: the ordinary AR decode step (no verify hooks at all), one token at a time.
            std::vector<int32_t> am_all;
            for (int t = 0; t < n; ++t) {
                std::vector<float> logits;
                if (!deepseek4_step(backend_, device_, w_, cache_, embed_buf_.data() + (size_t) t * w_.n_embd, 1,
                                    base_pos + t, logits, moe_hybrid_, tokens.data() + t, nullptr, telemetry_,
                                    routing_stats_, nullptr, expert_runtime_, true)) return false;
                int best = 0;
                for (int i = 1; i < w_.n_vocab; i++) if (logits[(size_t) i] > logits[(size_t) best]) best = i;
                am_all.push_back(best);
            }
            last_tok = am_all.back();
            verify_n_ = n;
            if (all_argmax) *all_argmax = std::move(am_all);
            return true;
        }
        static const std::vector<int> no_capture;
        // Reuse the normal cached graph for q==1 so reference verification has
        // exactly the same target arithmetic as ordinary AR decode.
        if (!deepseek4_dspark_verify_forward(backend_, device_, w_, cache_, diag_no_capture_ ? no_capture : capture_ids_,
                                             embed_buf_.data(), tokens.data(), n, base_pos, am,
                                             keep_logits_ ? &verify_logits_ : nullptr,
                                             verify_features_, telemetry_,
                                             /*allow_graph_reuse=*/!diag_no_graph_reuse_,
                                             moe_hybrid_, expert_runtime_,
                                             routing_stats_,
                                             diag_no_window_ ? nullptr : &boundary_checkpoint_,
                                             diag_no_window_ ? nullptr : &window_rows_)) {
            return false;
        }
        if (am.empty()) return false;
        last_tok = am.back();
        verify_n_ = n;
        if (all_argmax) *all_argmax = std::move(am);
        return true;
    }

    bool read_verify_logits(int n_tokens, std::vector<float> & out) override {
        if (!keep_logits_ || verify_logits_.empty()) return false;
        const size_t need = (size_t) n_tokens * w_.n_vocab;
        if (verify_logits_.size() < need) return false;
        out.assign(verify_logits_.begin(), verify_logits_.begin() + need);
        return true;
    }

    bool snapshot_kv() override { return deepseek4_snapshot_save(cache_, snap_backend_, snap_); }
    bool restore_kv() override { return deepseek4_snapshot_restore(snap_, cache_); }

    bool is_eos(int token) const override { return deepseek4_is_eos_tok(token, w_); }

    bool embed_tokens(const int32_t * tokens, int n, float * out) const override {
        return w_.embedder.embed(tokens, n, out);
    }

    bool project_hidden_to_tokens(const float * hidden, int n_tokens,
                                  std::vector<int32_t> & tokens_out) override {
        std::vector<float> logits;
        if (!project_hidden_to_logits(hidden, n_tokens, logits)) return false;
        tokens_out.resize(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            const float * row = logits.data() + (size_t) t * w_.n_vocab;
            int best = 0; float bv = row[0];
            for (int i = 1; i < w_.n_vocab; i++) if (row[i] > bv) { bv = row[i]; best = i; }
            tokens_out[t] = best;
        }
        return true;
    }

    // The drafter hidden is already out_norm'd (drafter tail); project with the
    // tied target lm_head only (mul_mat, no norm), matching the reference head.
    bool project_hidden_to_logits(const float * hidden, int n_tokens,
                                  std::vector<float> & logits_out) override {
        if (n_tokens <= 0) return false;
        ggml_init_params ip{};
        ip.mem_size = 32u * 1024 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return false;
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_tensor * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w_.n_embd, n_tokens);
        ggml_set_input(h);
        ggml_tensor * logits = ggml_mul_mat(ctx, w_.output, h);   // [n_vocab, n_tokens]
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
            if (alloc) ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return false;
        }
        ggml_backend_tensor_set(h, hidden, 0, sizeof(float) * (size_t) w_.n_embd * n_tokens);
        const ggml_status st = ggml_backend_graph_compute(backend_, gf);
        if (st != GGML_STATUS_SUCCESS) { ggml_gallocr_free(alloc); ggml_free(ctx); return false; }
        logits_out.resize((size_t) n_tokens * w_.n_vocab);
        ggml_backend_tensor_get(logits, logits_out.data(), 0, sizeof(float) * logits_out.size());
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return true;
    }

    ggml_tensor * lm_head_tensor() override { return w_.output; }
    int hidden_size() const override { return w_.n_embd; }
    int mask_token_id() const override { return mask_tok_; }
    const std::vector<int> & capture_layer_ids() const override { return capture_ids_; }

    void set_keep_logits(bool b) { keep_logits_ = b; }

    static bool diag_tensor_diff(ggml_tensor * a, ggml_tensor * b, size_t & nbytes_diff, float & max_abs) {
        nbytes_diff = 0; max_abs = 0.0f;
        if (!a || !b) { nbytes_diff = (a || b) ? 1 : 0; return nbytes_diff != 0; }
        const size_t na = ggml_nbytes(a), nb = ggml_nbytes(b);
        if (na != nb) { nbytes_diff = na > nb ? na - nb : nb - na; return true; }
        std::vector<uint8_t> x(na), y(nb);
        ggml_backend_tensor_get(a, x.data(), 0, na);
        ggml_backend_tensor_get(b, y.data(), 0, nb);
        for (size_t i = 0; i < na; ++i) if (x[i] != y[i]) ++nbytes_diff;
        if (nbytes_diff && a->type == GGML_TYPE_F32) {
            const float * fx = (const float *) x.data(); const float * fy = (const float *) y.data();
            for (size_t i = 0; i < na / 4; ++i) max_abs = std::max(max_abs, std::fabs(fx[i] - fy[i]));
        }
        return nbytes_diff != 0;
    }

    // Compare two cache snapshots field by field; prints the differing ones.
    void diag_compare_snapshots(const DeepSeek4Snapshot & a, const DeepSeek4Snapshot & b, long step) {
        int printed = 0, differing = 0;
        auto report = [&](const char * field, int layer, ggml_tensor * ta, ggml_tensor * tb) {
            size_t nd; float m;
            if (diag_tensor_diff(ta, tb, nd, m)) {
                ++differing;
                if (printed++ < 16)
                    std::fprintf(stderr, "[ds4-spec-diag-cache] step=%ld layer=%d %s: %zu bytes differ (max_abs %.3g, %zu bytes, type %s)\n",
                                 step, layer, field, nd, m, ta ? ggml_nbytes(ta) : 0, ta ? ggml_type_name(ta->type) : "?");
            }
        };
        if (a.cur_pos != b.cur_pos) std::fprintf(stderr, "[ds4-spec-diag-cache] step=%ld cur_pos %d vs %d\n", step, a.cur_pos, b.cur_pos);
        report("hc_state", -1, a.hc_state_snap, b.hc_state_snap);
        for (size_t l = 0; l < a.layers.size() && l < b.layers.size(); ++l) {
            const auto & x = a.layers[l]; const auto & y = b.layers[l];
            if (x.n_comp != y.n_comp || x.n_index_comp != y.n_index_comp)
                std::fprintf(stderr, "[ds4-spec-diag-cache] step=%ld layer=%zu counters n_comp %d vs %d, n_index_comp %d vs %d\n",
                             step, l, x.n_comp, y.n_comp, x.n_index_comp, y.n_index_comp);
            report("raw_kv", (int) l, x.raw_kv, y.raw_kv);
            report("comp_kv", (int) l, x.comp_kv, y.comp_kv);
            report("index_comp_kv", (int) l, x.index_comp_kv, y.index_comp_kv);
            report("attn_compressor.state_kv", (int) l, x.attn_compressor.state_kv, y.attn_compressor.state_kv);
            report("attn_compressor.state_score", (int) l, x.attn_compressor.state_score, y.attn_compressor.state_score);
            report("indexer_compressor.state_kv", (int) l, x.indexer_compressor.state_kv, y.indexer_compressor.state_kv);
            report("indexer_compressor.state_score", (int) l, x.indexer_compressor.state_score, y.indexer_compressor.state_score);
        }
        std::fprintf(stderr, "[ds4-spec-diag-cache] step=%ld summary: %d differing fields\n", step, differing);
    }

    // Diagnostic (LUCE_DS4_SPEC_DIAG): from the same KV state, run the batched verify and then the same
    // tokens as single-token forwards, capturing every layer's mean-over-HC state and the logits in both,
    // and report per token the first layer that differs and whether the argmax differs. Leaves the cache
    // as it was.
    bool diag_batch_vs_sequential(const std::vector<int32_t> & tokens, int base_pos, long step) {
        if (!snapshot_kv()) return false;
        const std::vector<int> saved_ids = capture_ids_;
        const bool saved_keep = keep_logits_;
        // LUCE_DS4_SPEC_DIAG_RAW: keep the drafter's capture layers and logits setting (the production
        // verify call) instead of capturing every layer; compares argmax only.
        static const bool diag_raw = [] { const char * v = std::getenv("LUCE_DS4_SPEC_DIAG_RAW"); return v && *v && *v != '0'; }();
        // LUCE_DS4_SPEC_DIAG_MODE=logits: production captures but logits kept; =capture: all layers captured, no logits
        static const std::string diag_mode = [] { const char * v = std::getenv("LUCE_DS4_SPEC_DIAG_MODE"); return std::string(v ? v : ""); }();
        const bool variant_mode = diag_mode == "ar" || diag_mode == "plain" || diag_mode == "nocapture" || diag_mode == "nowindow" || diag_mode == "noreuse";
        if (variant_mode) {
            // production sequential verify (force_sequential_) vs the named variant of it, same state
            int lt2 = -1;
            std::vector<int32_t> am_p, am_v;
            // "plain" compares the production verify as the loop runs it (batched); the others the sequential one.
            const bool reference_batched = diag_mode == "plain";
            const char * reference_label = reference_batched ? "production-batched" : "production-seq";
            force_sequential_ = !reference_batched;
            bool ok2 = verify_batch(tokens, base_pos, lt2, &am_p);
            DeepSeek4Snapshot after_p, after_v;
            ok2 = ok2 && deepseek4_snapshot_save(cache_, snap_backend_, after_p);
            ok2 = ok2 && restore_kv();
            if (diag_mode == "ar") diag_plain_ar_ = true;
            if (diag_mode == "plain") diag_plain_decode_ = true;
            if (diag_mode == "nocapture") diag_no_capture_ = true;
            if (diag_mode == "nowindow") diag_no_window_ = true;
            if (diag_mode == "noreuse") diag_no_graph_reuse_ = true;
            ok2 = ok2 && verify_batch(tokens, base_pos, lt2, &am_v);
            diag_plain_ar_ = diag_plain_decode_ = diag_no_capture_ = diag_no_window_ = diag_no_graph_reuse_ = false;
            force_sequential_ = false;
            ok2 = ok2 && deepseek4_snapshot_save(cache_, snap_backend_, after_v);
            if (ok2 && step < 3) {
                std::fprintf(stderr, "[ds4-spec-diag-cache] step=%ld cache after %s verify vs after %s:\n", step, reference_label, diag_mode.c_str());
                diag_compare_snapshots(after_p, after_v, step);
            }
            free_deepseek4_snapshot(after_p);
            free_deepseek4_snapshot(after_v);
            ok2 = ok2 && restore_kv();
            verify_features_.clear(); verify_logits_.clear();
            if (!ok2) return false;
            for (size_t t = 0; t < tokens.size(); ++t)
                std::fprintf(stderr, "[ds4-spec-diag] step=%ld pos=%d slot=%zu tok=%d argmax %s=%d %s=%d%s\n",
                             step, base_pos, t, tokens[t], reference_label, am_p[t], diag_mode.c_str(), am_v[t], am_p[t] == am_v[t] ? "" : " DIFFER");
            return true;
        }
        if (diag_mode == "logits") {
            keep_logits_ = true;
        } else if (diag_mode == "capture") {
            capture_ids_.clear();
            for (int l = 0; l < w_.n_layer; ++l) capture_ids_.push_back(l);
        } else if (!diag_raw) {
            capture_ids_.clear();
            for (int l = 0; l < w_.n_layer; ++l) capture_ids_.push_back(l);
            keep_logits_ = true;
        }
        int lt = -1;
        std::vector<int32_t> am_b, am_s;
        std::vector<float> attn_b, attn_s;
        ds4_set_diag_attn_capture(&attn_b);
        bool ok = verify_batch(tokens, base_pos, lt, &am_b);
        std::vector<float> fb = verify_features_, lb = verify_logits_;
        DeepSeek4Snapshot after_batched, after_sequential;
        ok = ok && deepseek4_snapshot_save(cache_, snap_backend_, after_batched);
        ok = ok && restore_kv();
        force_sequential_ = true;
        ds4_set_diag_attn_capture(&attn_s);
        ok = ok && verify_batch(tokens, base_pos, lt, &am_s);
        force_sequential_ = false;
        ds4_set_diag_attn_capture(nullptr);
        std::vector<float> fs = verify_features_, ls = verify_logits_;
        ok = ok && deepseek4_snapshot_save(cache_, snap_backend_, after_sequential);
        if (ok && step < 4) diag_compare_snapshots(after_batched, after_sequential, step);
        free_deepseek4_snapshot(after_batched);
        free_deepseek4_snapshot(after_sequential);
        ok = ok && restore_kv();
        capture_ids_ = saved_ids;
        keep_logits_ = saved_keep;
        verify_features_.clear();
        verify_logits_.clear();
        if (!ok) return false;
        const int n = (int) tokens.size();
        const size_t nl = (size_t) w_.n_layer, ne = (size_t) w_.n_embd, nv = (size_t) w_.n_vocab;
        if (step < 2) std::fprintf(stderr, "[ds4-spec-diag] step=%ld attention captures: batched %zu floats, sequential %zu floats (expect %zu)\n",
                                   step, attn_b.size(), attn_s.size(), (size_t) n * nl * ne);
        if (diag_raw || !diag_mode.empty()) {
            for (int t = 0; t < n; ++t)
                std::fprintf(stderr, "[ds4-spec-diag] step=%ld pos=%d slot=%d tok=%d argmax batch=%d seq=%d%s (raw production verify)\n",
                             step, base_pos, t, tokens[t], am_b[t], am_s[t], am_b[t] == am_s[t] ? "" : " MISMATCH");
            return true;
        }
        if (fb.size() != fs.size() || fb.size() != (size_t) n * nl * ne || lb.size() < (size_t) n * nv || ls.size() < (size_t) n * nv) {
            std::fprintf(stderr, "[ds4-spec-diag] step=%ld size mismatch fb=%zu fs=%zu lb=%zu ls=%zu\n", step, fb.size(), fs.size(), lb.size(), ls.size());
            return false;
        }
        for (int t = 0; t < n; ++t) {
            int first_layer = -1; float first_max = 0.0f, last_max = 0.0f;
            for (size_t l = 0; l < nl; ++l) {
                float m = 0.0f;
                for (size_t d = 0; d < ne; ++d) m = std::max(m, std::fabs(fb[((size_t) t * nl + l) * ne + d] - fs[((size_t) t * nl + l) * ne + d]));
                if (m > 0.0f && first_layer < 0) { first_layer = (int) l; first_max = m; }
                if (l + 1 == nl) last_max = m;
            }
            float lmax = 0.0f;
            for (size_t v = 0; v < nv; ++v) lmax = std::max(lmax, std::fabs(lb[(size_t) t * nv + v] - ls[(size_t) t * nv + v]));
            std::fprintf(stderr, "[ds4-spec-diag] step=%ld pos=%d slot=%d tok=%d argmax batch=%d seq=%d%s first_diff_layer=%d (max_abs %.3g) last_layer_max_abs=%.3g logits_max_abs=%.3g\n",
                         step, base_pos, t, tokens[t], am_b[t], am_s[t], am_b[t] == am_s[t] ? "" : " MISMATCH",
                         first_layer, first_max, last_max, lmax);
            if (step < 2 && attn_b.size() == (size_t) n * nl * ne && attn_s.size() == attn_b.size()) {
                // batched capture is layer-major ([layer][token]); the sequential one is token-major ([token][layer])
                std::fprintf(stderr, "[ds4-spec-diag-attn] step=%ld slot=%d attention-output max_abs per layer:", step, t);
                for (size_t l = 0; l < nl; ++l) {
                    float m = 0.0f;
                    for (size_t d = 0; d < ne; ++d)
                        m = std::max(m, std::fabs(attn_b[(l * (size_t) n + (size_t) t) * ne + d] - attn_s[((size_t) t * nl + l) * ne + d]));
                    std::fprintf(stderr, " L%zu=%.2g", l, m);
                }
                std::fprintf(stderr, "\n");
            }
            if (step < 2) {
                std::fprintf(stderr, "[ds4-spec-diag-layers] step=%ld slot=%d max_abs per layer:", step, t);
                for (size_t l = 0; l < nl; ++l) {
                    float m = 0.0f, ref = 0.0f;
                    for (size_t d = 0; d < ne; ++d) {
                        m = std::max(m, std::fabs(fb[((size_t) t * nl + l) * ne + d] - fs[((size_t) t * nl + l) * ne + d]));
                        ref = std::max(ref, std::fabs(fs[((size_t) t * nl + l) * ne + d]));
                    }
                    std::fprintf(stderr, " L%zu=%.2g/%.2g", l, m, ref);
                }
                std::fprintf(stderr, "\n");
            }
        }
        return true;
    }
    void set_telemetry(DeepSeek4StepTelemetry * t) { telemetry_ = t; }
    const std::vector<float> & last_features() const { return verify_features_; }
    int last_verify_n() const { return verify_n_; }
    bool restore_first_boundary_checkpoint() {
        if (!boundary_checkpoint_.available ||
            boundary_checkpoint_.layers.size() != cache_.layers.size()) {
            return false;
        }
        auto valid_pair = [](ggml_tensor * src, ggml_tensor * dst) {
            if (!src && !dst) return true;
            return src && dst && src->type == dst->type &&
                ggml_nelements(src) == ggml_nelements(dst) &&
                src->buffer && dst->buffer;
        };
        for (const DeepSeek4SpecBoundaryCheckpointLayer & layer :
             boundary_checkpoint_.layers) {
            if (!valid_pair(layer.attn_kv_src, layer.attn_kv_dst) ||
                !valid_pair(layer.attn_score_src, layer.attn_score_dst) ||
                !valid_pair(layer.index_kv_src, layer.index_kv_dst) ||
                !valid_pair(layer.index_score_src, layer.index_score_dst)) {
                return false;
            }
        }
        bool copied = false;
        auto copy_pair = [&](ggml_tensor * src, ggml_tensor * dst) {
            if (!src) return;
            ggml_backend_tensor_copy_async(backend_, backend_, src, dst);
            copied = true;
        };
        for (const DeepSeek4SpecBoundaryCheckpointLayer & layer :
             boundary_checkpoint_.layers) {
            copy_pair(layer.attn_kv_src, layer.attn_kv_dst);
            copy_pair(layer.attn_score_src, layer.attn_score_dst);
            copy_pair(layer.index_kv_src, layer.index_kv_dst);
            copy_pair(layer.index_score_src, layer.index_score_dst);
        }
        return copied;
    }
    void clear_snapshot() { free_deepseek4_snapshot(snap_); }
    const Ds4VerifyWindowRows & last_window_rows() const { return window_rows_; }

private:
    const DeepSeek4Weights & w_;
    DeepSeek4Cache & cache_;
    ggml_backend_t backend_;
    int device_;
    ggml_backend_t snap_backend_;
    std::vector<int> capture_ids_;
    int mask_tok_;
    DeepSeek4Snapshot snap_{};
    DeepSeek4StepTelemetry * telemetry_ = nullptr;
    bool keep_logits_ = false;
    bool force_sequential_ = false;   // diag_batch_vs_sequential
    bool diag_plain_ar_ = false, diag_no_capture_ = false, diag_no_window_ = false, diag_no_graph_reuse_ = false;
    bool diag_plain_decode_ = false;  // the production AR decode call, no verify hooks
    int verify_n_ = 0;
    std::vector<float> embed_buf_;
    std::vector<float> verify_logits_;
    std::vector<float> verify_features_;
    DeepSeek4SpecBoundaryCheckpoint boundary_checkpoint_;
    Ds4VerifyWindowRows window_rows_;
    MoeHybridStorage * moe_hybrid_ = nullptr;
    MoeExpertComputeRuntime * expert_runtime_ = nullptr;
    MoeHybridRoutingStats * routing_stats_ = nullptr;
};

namespace {

// Build the DraftWeights shim the target-agnostic dspark head expects (only the
// DSpark head fields + n_embd are read).
DraftWeights make_dspark_shim(const DSparkDrafter & d) {
    DraftWeights dw{};
    dw.n_embd = d.core.n_embd;
    dw.dspark.enabled = d.dspark_enabled;
    dw.dspark.markov_rank = d.markov_rank;
    dw.dspark.vocab_size = d.vocab_size;
    dw.dspark.confidence_dim = d.confidence_dim;
    dw.dspark.markov_w1 = d.markov_w1;
    dw.dspark.markov_w2 = d.markov_w2;
    dw.dspark.confidence_w = d.confidence_w;
    dw.dspark.confidence_b = d.confidence_b;
    return dw;
}

bool spec_env_flag(const char * name) {
    const char * v = std::getenv(name);
    return v && *v && *v != '0';
}

// True unless the variable is set to an explicit "0": for switches whose
// default is on.
bool spec_env_default_on(const char * name) {
    const char * value = std::getenv(name);
    return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

// Adaptive verify width policy. The controller may pick any seed-inclusive
// width from kDs4AdaptiveMinWidth up to the request cap (DS4_Q5_VERIFY_TOKENS
// on the q5 path). kDs4VerifyWidthCostMs[q] is the total target-step cost of
// width q measured with the six-expert gfx1151 fused verifier, including the
// fixed drafter/head work (DSpark computes its whole proposal block whatever
// the verify width). Only the ratios matter: q3 costs almost as much as q4,
// so high-acceptance text settles on q4/q5 while low-acceptance prose drops
// to q2/q3. observe() refines the table online, so other backends start from
// this curve instead of importing a DS4 policy. The fused verify cache is
// sized so every width stays resident across the ratio-4 phases.
constexpr int kDs4AdaptiveMinWidth = 2;
// Depths whose confidence-head score decides the width. The head was
// calibrated on q<=4 traffic (three candidates); the fourth candidate of a q5
// verify is decided by the learned conditional acceptance until its own
// score is shown calibrated (see the per-depth calibration line under
// LUCE_DS4_TIMING).
constexpr int kDs4ConfidenceDepths = 3;
constexpr float kDs4VerifyWidthCostMs[DS4_Q5_VERIFY_TOKENS + 1] = {
    0.0f, 0.0f, 75.0f, 100.0f, 122.0f, 123.0f};

// ── Light rollback state ────────────────────────────────────────────────
// Save the ratio-4 rolling state, HC state, and the ring rows a speculative
// verify may overwrite. Pinned host storage lets the GPU copy this compact
// rollback state on its stream before the verifier without a host fence.
// prev-half = first 4 rows of a [comp_width, 8] ratio-4 rolling state.
// Ratio-128 states need only the touched ring rows, not the full 128 rows.
constexpr int kRollbackMaxTokens = 6; // Preserve the existing raw-ring staging capacity.
// Default V4.1 verify width (seed + two drafts): q=3 beat q=4..6 on the
// hybrid tier (count 13.0, code 15.0 tok/s against 13.4/13.7 at q=4).
constexpr int kDs4TokenwiseVerifyDefaultWidth = 3;

size_t prev_half_bytes(const ggml_tensor * t) {
    return t && t->ne[1] == 8 ? (size_t) t->nb[1] * 4 : 0;
}

size_t rollback_state_bytes(const ggml_tensor * t) {
    if (!t) return 0;
    if (t->ne[1] == 8) return 2 * prev_half_bytes(t);
    return t->ne[1] == 128 ? t->nb[1] * kRollbackMaxTokens : 0;
}

int rollback_ring_row(const ggml_tensor * t, int pos) {
    const int period = t->ne[1] == 8 ? 4 : (int) t->ne[1];
    const int row = pos % period;
    return row < 0 ? row + period : row;
}

size_t align_up_rollback(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void assign_pinned_span(DeepSeek4SpecRollback::PinnedSpan & span, size_t bytes,
                        size_t & total) {
    if (bytes == 0) {
        span = {};
        return;
    }
    total = align_up_rollback(total, 64);
    span.offset = total;
    span.size = bytes;
    total += bytes;
}

size_t assign_rollback_spans(const DeepSeek4Cache & cache, DeepSeek4SpecRollback & rb) {
    rb.layers.resize(cache.layers.size());
    size_t total = 0;
    for (size_t il = 0; il < cache.layers.size(); ++il) {
        const DeepSeek4LayerCache & lc = cache.layers[il];
        DeepSeek4SpecRollback::Layer & s = rb.layers[il];
        assign_pinned_span(
            s.pinned_attn_kv, rollback_state_bytes(lc.attn_compressor.state_kv), total);
        assign_pinned_span(
            s.pinned_attn_sc, rollback_state_bytes(lc.attn_compressor.state_score), total);
        assign_pinned_span(
            s.pinned_idx_kv, rollback_state_bytes(lc.indexer_compressor.state_kv), total);
        assign_pinned_span(
            s.pinned_idx_sc, rollback_state_bytes(lc.indexer_compressor.state_score), total);
        s.raw_row_bytes = lc.raw_kv
            ? ggml_row_size(lc.raw_kv->type, lc.raw_kv->ne[0]) : 0;
        // The DSpark artifact exposes five proposal rows, so the widest
        // verifier batch is seed + five candidates.
        assign_pinned_span(s.pinned_raw_rows, s.raw_row_bytes * kRollbackMaxTokens, total);
    }
    assign_pinned_span(
        rb.pinned_hc, cache.hc_state ? ggml_nbytes(cache.hc_state) : 0, total);
    return total;
}

bool init_pinned_rollback(const DeepSeek4Cache & cache, DeepSeek4SpecRollback & rb,
                          ggml_backend_t backend) {
    if (rb.pinned_buf) return true;
    const size_t total = assign_rollback_spans(cache, rb);
    if (total == 0) return false;

    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    ggml_backend_buffer_type_t buft =
        device ? ggml_backend_dev_host_buffer_type(device) : nullptr;
    if (!buft) return false;

    rb.pinned_buf = ggml_backend_buft_alloc_buffer(buft, total);
    if (!rb.pinned_buf) return false;
    rb.pinned_base =
        static_cast<uint8_t *>(ggml_backend_buffer_get_base(rb.pinned_buf));
    if (!rb.pinned_base) {
        ggml_backend_buffer_free(rb.pinned_buf);
        rb.pinned_buf = nullptr;
        return false;
    }
    return true;
}

bool device_rollback_enabled() {
    static const bool enabled = [] {
        const char * e = std::getenv("LUCE_DS4_DEVICE_ROLLBACK");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    return enabled;
}

bool rollback_tensor_on(const ggml_tensor * t, ggml_backend_buffer_type_t buft) {
    if (!t) return true;
    const ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    return buf && ggml_backend_buffer_get_type(buf) == buft;
}

// Device staging needs every saved tensor in the backend's own device memory,
// because the batched copy runs as one kernel on that device.
bool init_device_rollback(const DeepSeek4Cache & cache, DeepSeek4SpecRollback & rb,
                          ggml_backend_t backend) {
    if (!backend || !ggml_backend_is_cuda(backend)) return false;
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    if (!rollback_tensor_on(cache.hc_state, buft)) return false;
    for (const DeepSeek4LayerCache & lc : cache.layers) {
        if (!rollback_tensor_on(lc.attn_compressor.state_kv, buft) ||
            !rollback_tensor_on(lc.attn_compressor.state_score, buft) ||
            !rollback_tensor_on(lc.indexer_compressor.state_kv, buft) ||
            !rollback_tensor_on(lc.indexer_compressor.state_score, buft) ||
            !rollback_tensor_on(lc.raw_kv, buft)) {
            return false;
        }
    }
    const size_t total = assign_rollback_spans(cache, rb);
    if (total == 0) return false;
    if (rb.device_buf && rb.device_backend == backend && rb.device_bytes == total) {
        return true;
    }
    // A different backend or cache layout: retire the old staging once its
    // copies have drained, then allocate for the new one.
    if (rb.device_buf) {
        if (rb.device_backend) ggml_backend_synchronize(rb.device_backend);
        ggml_backend_buffer_free(rb.device_buf);
        rb.device_buf = nullptr;
        rb.device_base = nullptr;
        rb.device_backend = nullptr;
        rb.device_bytes = 0;
    }
    rb.device_buf = ggml_backend_alloc_buffer(backend, total);
    if (!rb.device_buf) return false;
    rb.device_base =
        static_cast<uint8_t *>(ggml_backend_buffer_get_base(rb.device_buf));
    if (!rb.device_base) {
        ggml_backend_buffer_free(rb.device_buf);
        rb.device_buf = nullptr;
        return false;
    }
    rb.device_backend = backend;
    rb.device_bytes = total;
    return true;
}

// Collects the copies of one save or apply for a single batched launch. A
// later copy to the same destination replaces the earlier one, matching the
// last-writer-wins order of the per-row path.
struct RollbackCopies {
    std::vector<ggml_cuda_copy_desc> descs;
    void add(const void * src, void * dst, size_t nbytes) {
        for (ggml_cuda_copy_desc & d : descs) {
            if (d.dst == dst && d.nbytes == nbytes) {
                d.src = src;
                return;
            }
        }
        descs.push_back({src, dst, nbytes});
    }
    void flush(ggml_backend_t backend) {
        ggml_backend_cuda_copy_batch_async(backend, descs.data(), (int) descs.size());
        descs.clear();
    }
};

void save_rollback_state(ggml_backend_t backend, ggml_tensor * t,
                         uint8_t * dst, bool async_copy, int pos, int count,
                         RollbackCopies * copies = nullptr) {
    if (rollback_state_bytes(t) == 0) return;
    const bool rolling = t->ne[1] == 8;
    const size_t bytes = rolling ? rollback_state_bytes(t) : t->nb[1];
    for (int i = 0; i < (rolling ? 1 : count); ++i) {
        const size_t offset = rolling ? 0 : rollback_ring_row(t, pos + i) * t->nb[1];
        if (copies) {
            copies->add((const uint8_t *) t->data + offset, dst + i * bytes, bytes);
        } else if (async_copy) {
            ggml_backend_tensor_get_async(backend, t, dst + i * bytes, offset, bytes);
        } else {
            ggml_backend_tensor_get(t, dst + i * bytes, offset, bytes);
        }
    }
}

void restore_rollback_state(ggml_backend_t backend, ggml_tensor * t,
                            const uint8_t * src, bool async_copy,
                            int pos, int count, int first_rejected, bool restore_prev,
                            RollbackCopies * copies = nullptr) {
    if (rollback_state_bytes(t) == 0) return;
    const bool rolling = t->ne[1] == 8;
    if (rolling && restore_prev) {
        const size_t bytes = prev_half_bytes(t);
        if (copies) copies->add(src, t->data, bytes);
        else if (async_copy) ggml_backend_tensor_set_async(backend, t, src, 0, bytes);
        else            ggml_backend_tensor_set(t, src, 0, bytes);
    }
    // A rejected row whose ring slot is shared with a committed row of the
    // same batch keeps the committed write: with five tokens the fifth
    // token's slot is the seed's. In an undone flush the seed row is put
    // back by deepseek4_spec_restore_seed_row before this runs; in a kept
    // flush the slot belongs to a rejected post-boundary position and is
    // rewritten before it is ever pooled.
    const int period = rolling ? 4 : (int) t->ne[1];
    for (int i = first_rejected; i < count; ++i) {
        if (i >= period && i - period < first_rejected) continue;
        const int row = (rolling ? 4 : 0) + rollback_ring_row(t, pos + i);
        const size_t offset = row * t->nb[1];
        const uint8_t * saved = src + (rolling ? offset : i * t->nb[1]);
        if (copies) {
            copies->add(saved, (uint8_t *) t->data + offset, t->nb[1]);
        } else if (async_copy) {
            ggml_backend_tensor_set_async(backend, t, saved, offset, t->nb[1]);
        } else {
            ggml_backend_tensor_set(t, saved, offset, t->nb[1]);
        }
    }
}

void spec_rollback_save(const DeepSeek4Cache & cache, DeepSeek4SpecRollback & rb,
                        ggml_backend_t backend, bool async_copy,
                        bool pinned_copy, int raw_pos, int raw_count) {
    // Device staging does not depend on the legacy async/pinned switches:
    // without them every row would otherwise be a blocking host copy.
    const bool use_device = backend && device_rollback_enabled() &&
        init_device_rollback(cache, rb, backend);
    ggml_backend_t saved_backend =
        (use_device || async_copy || pinned_copy) ? backend : nullptr;
    if (rb.async_backend && rb.async_backend != saved_backend) {
        ggml_backend_synchronize(rb.async_backend);
    }
    rb.async_backend = saved_backend;
    rb.raw_pos = raw_pos;
    rb.raw_count = std::clamp(raw_count, 0, kRollbackMaxTokens);
    rb.layers.resize(cache.layers.size());
    const bool use_pinned =
        !use_device && pinned_copy && init_pinned_rollback(cache, rb, backend);
    rb.uses_pinned_copy = use_pinned;
    rb.uses_device_copy = use_device;
    const bool use_staged = use_device || use_pinned;
    uint8_t * staged_base = use_device ? rb.device_base : rb.pinned_base;
    RollbackCopies device_copies;
    RollbackCopies * copies = use_device ? &device_copies : nullptr;
    for (size_t il = 0; il < cache.layers.size(); ++il) {
        const DeepSeek4LayerCache & lc = cache.layers[il];
        DeepSeek4SpecRollback::Layer & s = rb.layers[il];
        auto save_state = [&](ggml_tensor * t, std::vector<uint8_t> & buf,
                              const DeepSeek4SpecRollback::PinnedSpan & span) {
            const size_t bytes = rollback_state_bytes(t);
            if (bytes == 0) { buf.clear(); return; }
            if (use_staged) {
                GGML_ASSERT(span.size == bytes);
            } else {
                buf.resize(bytes);
            }
            save_rollback_state(backend, t,
                use_staged ? staged_base + span.offset : buf.data(),
                use_staged || async_copy, rb.raw_pos, rb.raw_count, copies);
        };
        save_state(lc.attn_compressor.state_kv, s.attn_kv, s.pinned_attn_kv);
        save_state(lc.attn_compressor.state_score, s.attn_sc, s.pinned_attn_sc);
        save_state(lc.indexer_compressor.state_kv, s.idx_kv, s.pinned_idx_kv);
        save_state(lc.indexer_compressor.state_score, s.idx_sc, s.pinned_idx_sc);

        s.raw_row_bytes = lc.raw_kv
            ? ggml_row_size(lc.raw_kv->type, lc.raw_kv->ne[0]) : 0;
        if (!lc.raw_kv || lc.raw_kv->ne[1] <= 0 ||
            s.raw_row_bytes == 0 || rb.raw_count == 0) {
            s.raw_rows.clear();
            continue;
        }
        if (!use_staged) {
            s.raw_rows.resize(s.raw_row_bytes * kRollbackMaxTokens);
        }
        for (int t = 0; t < rb.raw_count; ++t) {
            int row = (rb.raw_pos + t) % (int) lc.raw_kv->ne[1];
            if (row < 0) row += (int) lc.raw_kv->ne[1];
            uint8_t * dst = use_staged
                ? staged_base + s.pinned_raw_rows.offset +
                      (size_t) t * s.raw_row_bytes
                : s.raw_rows.data() + (size_t) t * s.raw_row_bytes;
            if (copies) {
                copies->add((const uint8_t *) lc.raw_kv->data +
                                (size_t) row * lc.raw_kv->nb[1],
                            dst, s.raw_row_bytes);
            } else if (use_pinned || async_copy) {
                ggml_backend_tensor_get_async(
                    backend, lc.raw_kv, dst,
                    (size_t) row * lc.raw_kv->nb[1], s.raw_row_bytes);
            } else {
                ggml_backend_tensor_get(
                    lc.raw_kv, dst,
                    (size_t) row * lc.raw_kv->nb[1], s.raw_row_bytes);
            }
        }
    }
    if (cache.hc_state) {
        const size_t bytes = ggml_nbytes(cache.hc_state);
        if (copies) {
            copies->add(cache.hc_state->data, staged_base + rb.pinned_hc.offset, bytes);
        } else if (use_pinned) {
            ggml_backend_tensor_get_async(
                backend, cache.hc_state,
                rb.pinned_base + rb.pinned_hc.offset, 0, bytes);
        } else {
            if (rb.hc_state.size() != bytes) rb.hc_state.resize(bytes);
            if (async_copy) {
                ggml_backend_tensor_get_async(
                    backend, cache.hc_state, rb.hc_state.data(), 0, bytes);
            } else {
                ggml_backend_tensor_get(cache.hc_state, rb.hc_state.data(), 0, bytes);
            }
        }
    }
    if (copies) copies->flush(backend);
}

// Truncate the cache to commit_pos. restore_prev is set when the verify
// crossed a ratio-4 boundary at-or-past commit_pos: that flush filled the
// prev-half rows with a chunk containing rejected tokens, so put the
// pre-verify rows back. (A boundary strictly inside the committed range is a
// legitimate flush and must be kept.) Rejected current/ring rows are restored
// regardless of whether a ratio-4 boundary was crossed.
void spec_rollback_apply(const DeepSeek4SpecRollback & rb, const DeepSeek4Weights & w,
                         DeepSeek4Cache & cache, int commit_pos, bool restore_prev) {
    ggml_backend_t backend = rb.async_backend;
    const bool async_copy = backend != nullptr;
    const bool use_pinned = rb.uses_pinned_copy;
    const bool use_device = rb.uses_device_copy;
    GGML_ASSERT(!use_pinned || (backend && rb.pinned_buf && rb.pinned_base));
    GGML_ASSERT(!use_device || (backend && rb.device_buf && rb.device_base));
    const bool use_staged = use_device || use_pinned;
    const uint8_t * staged_base = use_device ? rb.device_base : rb.pinned_base;
    RollbackCopies device_copies;
    RollbackCopies * copies = use_device ? &device_copies : nullptr;
    cache.cur_pos = commit_pos;
    for (size_t il = 0; il < cache.layers.size(); ++il) {
        DeepSeek4LayerCache & lc = cache.layers[il];
        const uint32_t ratio = il < w.compress_ratios.size() ? w.compress_ratios[il] : 0;
        // Only the owner of the compressed rows counts them (V4.1 readers own none).
        if (ratio > 0 && deepseek4_is_kv_source(w, (int) il)) lc.n_comp = commit_pos / (int) ratio;
        if (lc.index_comp_kv) lc.n_index_comp = commit_pos / (int) ratio;
        const int first_rejected = std::clamp(commit_pos - rb.raw_pos, 0, rb.raw_count);
        if (il < rb.layers.size()) {
            const DeepSeek4SpecRollback::Layer & s = rb.layers[il];
            auto restore_state = [&](ggml_tensor * t, const std::vector<uint8_t> & buf,
                                     const DeepSeek4SpecRollback::PinnedSpan & span) {
                const size_t bytes = rollback_state_bytes(t);
                if (bytes == 0) return;
                GGML_ASSERT((use_staged ? span.size : buf.size()) == bytes);
                restore_rollback_state(backend, t,
                    use_staged ? staged_base + span.offset : buf.data(),
                    use_staged || async_copy, rb.raw_pos, rb.raw_count,
                    first_rejected, restore_prev, copies);
            };
            restore_state(lc.attn_compressor.state_kv, s.attn_kv, s.pinned_attn_kv);
            restore_state(lc.attn_compressor.state_score, s.attn_sc, s.pinned_attn_sc);
            restore_state(lc.indexer_compressor.state_kv, s.idx_kv, s.pinned_idx_kv);
            restore_state(lc.indexer_compressor.state_score, s.idx_sc, s.pinned_idx_sc);
        }
        if (il < rb.layers.size() && lc.raw_kv && lc.raw_kv->ne[1] > 0) {
            const DeepSeek4SpecRollback::Layer & s = rb.layers[il];
            for (int t = first_rejected;
                 t < rb.raw_count && s.raw_row_bytes > 0;
                 ++t) {
                int row = (rb.raw_pos + t) % (int) lc.raw_kv->ne[1];
                if (row < 0) row += (int) lc.raw_kv->ne[1];
                const uint8_t * src = nullptr;
                if (use_staged &&
                    s.pinned_raw_rows.size >=
                        (size_t) (t + 1) * s.raw_row_bytes) {
                    src = staged_base + s.pinned_raw_rows.offset +
                          (size_t) t * s.raw_row_bytes;
                } else if (s.raw_rows.size() >=
                           (size_t) (t + 1) * s.raw_row_bytes) {
                    src = s.raw_rows.data() + (size_t) t * s.raw_row_bytes;
                }
                if (!src) continue;
                if (copies) {
                    copies->add(src, (uint8_t *) lc.raw_kv->data +
                                         (size_t) row * lc.raw_kv->nb[1],
                                s.raw_row_bytes);
                } else if (use_pinned || async_copy) {
                    ggml_backend_tensor_set_async(
                        backend, lc.raw_kv, src,
                        (size_t) row * lc.raw_kv->nb[1], s.raw_row_bytes);
                } else {
                    ggml_backend_tensor_set(
                        lc.raw_kv, src,
                        (size_t) row * lc.raw_kv->nb[1], s.raw_row_bytes);
                }
            }
        }
    }
    if (restore_prev && cache.hc_state) {
        if (copies && rb.pinned_hc.size > 0) {
            copies->add(staged_base + rb.pinned_hc.offset, cache.hc_state->data,
                        rb.pinned_hc.size);
        } else if (use_pinned && rb.pinned_hc.size > 0) {
            ggml_backend_tensor_set_async(
                backend, cache.hc_state,
                rb.pinned_base + rb.pinned_hc.offset, 0, rb.pinned_hc.size);
        } else if (!rb.hc_state.empty()) {
            if (async_copy) {
                ggml_backend_tensor_set_async(
                    backend, cache.hc_state,
                    rb.hc_state.data(), 0, rb.hc_state.size());
            } else {
                ggml_backend_tensor_set(
                    cache.hc_state, rb.hc_state.data(), 0, rb.hc_state.size());
            }
        }
    }
    if (copies) copies->flush(backend);
}

// V4.1 pooled windows (ratio 2): after a rejection at commit_pos, the rows
// of the window commit_pos falls in that belong to accepted tokens must hold
// those tokens' projections, but a later (rejected) token of the batch may
// have overwritten them. Put them back from the rows the verify kept. The
// window starts at or after verify_pos because the seed is always accepted
// and a window is at most two tokens; its remaining rows are rewritten by
// the next tokens before the window is pooled.
void spec_restore_window_rows(const Ds4VerifyWindowRows & rows, const DeepSeek4Weights & w,
                              DeepSeek4Cache & cache, int verify_pos, int commit_pos) {
    for (size_t il = 0; il < cache.layers.size() && il < rows.kv.size(); ++il) {
        DeepSeek4LayerCache & lc = cache.layers[il];
        const int ratio = il < w.compress_ratios.size() ? (int) w.compress_ratios[il] : 0;
        if (!deepseek4_is_window_state(lc.attn_compressor, ratio)) continue;
        const size_t row_bytes = lc.attn_compressor.state_kv->nb[1];
        const int window_start = commit_pos / ratio * ratio;
        for (int p = std::max(window_start, verify_pos); p < commit_pos; ++p) {
            const size_t t = (size_t) (p - verify_pos);
            if (rows.kv[il].size() < (t + 1) * row_bytes ||
                rows.score[il].size() < (t + 1) * row_bytes) {
                continue;
            }
            const size_t offset = (size_t) (p % ratio) * row_bytes;
            ggml_backend_tensor_set(lc.attn_compressor.state_kv,
                                    rows.kv[il].data() + t * row_bytes, offset, row_bytes);
            ggml_backend_tensor_set(lc.attn_compressor.state_score,
                                    rows.score[il].data() + t * row_bytes, offset, row_bytes);
        }
    }
}

using SpecClock = std::chrono::steady_clock;

double spec_ms_since(SpecClock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(SpecClock::now() - t0).count() / 1000.0;
}

}  // namespace

bool deepseek4_spec_restore_seed_row(ggml_backend_t backend, ggml_tensor * state,
                                     int seed_pos) {
    if (!state || !state->buffer || state->ne[1] != 8) return false;
    const int slot = seed_pos & (DS4_SPEC_ROLLING_RATIO - 1);
    // Two views on the same device buffer: the rotated copy of the seed row
    // in the previous half and the seed's current-window slot. A backend
    // copy between them is one device-side row copy on the compute stream,
    // ordered before the previous-half restore that follows it.
    ggml_init_params params = {};
    params.mem_size = 2 * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * src = ggml_view_2d(ctx, state, state->ne[0], 1, state->nb[1],
                                     (size_t) slot * state->nb[1]);
    ggml_tensor * dst = ggml_view_2d(ctx, state, state->ne[0], 1, state->nb[1],
                                     (size_t) (DS4_SPEC_ROLLING_RATIO + slot) * state->nb[1]);
    bool ok = ggml_backend_view_init(src) == GGML_STATUS_SUCCESS &&
              ggml_backend_view_init(dst) == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_tensor_copy_async(backend, backend, src, dst);
    }
    ggml_free(ctx);
    return ok;
}

DeepSeek4SpecRollback::~DeepSeek4SpecRollback() {
    if (async_backend) {
        ggml_backend_synchronize(async_backend);
    }
    if (pinned_buf) {
        ggml_backend_buffer_free(pinned_buf);
    }
    if (device_buf) {
        ggml_backend_buffer_free(device_buf);
    }
}

void deepseek4_spec_rollback_save(const DeepSeek4Cache & cache,
                                  DeepSeek4SpecRollback & rollback,
                                  int raw_pos,
                                  int raw_count,
                                  ggml_backend_t backend,
                                  bool pinned_copy) {
    GGML_ASSERT(!pinned_copy || backend);
    spec_rollback_save(cache, rollback, backend,
                       /*async_copy=*/backend != nullptr, pinned_copy,
                       raw_pos, raw_count);
}

void deepseek4_spec_rollback_apply(const DeepSeek4SpecRollback & rollback,
                                   const DeepSeek4Weights & weights,
                                   DeepSeek4Cache & cache,
                                   int commit_pos,
                                   bool restore_prev) {
    spec_rollback_apply(rollback, weights, cache, commit_pos, restore_prev);
}

// Batched target verify + capture: wraps the existing multi-token
// deepseek4_step_layer_range (dynamic attention + batched HC), which never
// touches the fused single-token 23 tok/s path, with the Ds4VerifyHooks that
// add per-layer mean-over-HC capture and full per-position logits.
bool deepseek4_dspark_verify_forward(ggml_backend_t backend,
                                     int device,
                                     const DeepSeek4Weights & w,
                                     DeepSeek4Cache & cache,
                                     const std::vector<int> & capture_layer_ids,
                                     const float * embed,
                                     const int32_t * token_ids,
                                     int n_tokens,
                                     int kv_start,
                                     std::vector<int32_t> & argmax_out,
                                     std::vector<float> * logits_out,
                                     std::vector<float> & capture_out,
                                     DeepSeek4StepTelemetry * telemetry,
                                     bool allow_graph_reuse,
                                     MoeHybridStorage * moe_hybrid,
                                     MoeExpertComputeRuntime * expert_runtime,
                                     MoeHybridRoutingStats * routing_stats,
                                     DeepSeek4SpecBoundaryCheckpoint *
                                         boundary_checkpoint_out,
                                     Ds4VerifyWindowRows * window_rows) {
    const DeepSeek4RoctxPhaseScope roctx_phase(InferencePhase::Verify);
    std::vector<float> hc_state;
    std::vector<float> all_logits;
    std::vector<float> last_logits;
    std::vector<int32_t> gpu_argmax;
    Ds4VerifyHooks hooks;
    hooks.capture_layer_ids = &capture_layer_ids;
    hooks.capture_out = &capture_out;
    hooks.all_logits_out = &all_logits;
    hooks.argmax_out = &gpu_argmax;
    if (boundary_checkpoint_out) boundary_checkpoint_out->clear();
    hooks.boundary_checkpoint_out = boundary_checkpoint_out;
    hooks.window_rows = window_rows;
    hooks.prefer_argmax_only =
        spec_env_flag("LUCE_DS4_GPU_ARGMAX_VERIFY") && logits_out == nullptr;
    if (!deepseek4_step_layer_range(backend, device, w, cache, hc_state, embed, n_tokens, kv_start,
                                    0, w.n_layer, &last_logits, token_ids,
                                    telemetry, allow_graph_reuse,
                                    &hooks, moe_hybrid, expert_runtime,
                                    routing_stats)) {
        std::fprintf(stderr, "[ds4-verify] step_layer_range returned false (n_tokens=%d kv_start=%d)\n",
                     n_tokens, kv_start);
        return false;
    }
    if (hooks.prefer_argmax_only &&
        (int) gpu_argmax.size() == n_tokens) {
        argmax_out = std::move(gpu_argmax);
        return true;
    }
    if (n_tokens == 1 && (int) all_logits.size() < w.n_vocab &&
        (int) last_logits.size() >= w.n_vocab) {
        // The reference-exact q1 path reuses the normal AR graph. That graph
        // returns its logits through the regular output vector rather than
        // the verifier's q-wide hook.
        all_logits = last_logits;
    }
    if ((int) all_logits.size() < w.n_vocab * n_tokens) {
        std::fprintf(stderr, "[ds4-verify] all_logits too small: got=%zu need=%d (cap=%zu)\n",
                     all_logits.size(), w.n_vocab * n_tokens, capture_out.size());
        return false;
    }
    argmax_out.resize(n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        const float * row = all_logits.data() + (size_t) t * w.n_vocab;
        int best = 0; float bv = row[0];
        for (int i = 1; i < w.n_vocab; i++) if (row[i] > bv) { bv = row[i]; best = i; }
        argmax_out[t] = best;
    }
    if (logits_out) *logits_out = std::move(all_logits);
    return true;
}

bool run_deepseek4_dspark_spec_decode(
        ggml_backend_t backend,
        int device,
        const DeepSeek4Weights & target_w,
        DeepSeek4Cache & target_cache,
        const DSparkDrafter & drafter,
        int committed,
        int last_tok,
        int n_gen,
        const float * prompt_feature_window,
        int win_len,
        std::vector<int32_t> & out_tokens,
        float * accept_rate_out,
        const std::function<bool(int32_t)> & on_token,
        MoeHybridStorage * moe_hybrid,
        MoeExpertComputeRuntime * expert_runtime,
        MoeHybridRoutingStats * routing_stats) {
    const int n_embd = target_w.n_embd;
    const int n_tgt = drafter.n_target_layers;
    const int block = drafter.block_size;
    const int n_swa = target_w.n_swa;
    const int feat_row = n_tgt * n_embd;

    const bool debug = spec_env_flag("LUCE_DS4_DSPARK_DEBUG");
    const bool timing = spec_env_flag("LUCE_DS4_TIMING");
    const bool reference_exact =
        spec_env_flag("LUCE_DS4_SPEC_REFERENCE_EXACT");
    const bool full_snap = reference_exact ||
        spec_env_flag("LUCE_DS4_FULL_SNAP");
    const bool seq_verify_mode = reference_exact ||
        spec_env_flag("LUCE_DS4_SEQ_VERIFY");
    const InferencePhase roctx_phase = reference_exact
        ? InferencePhase::ReferenceExact
        : (seq_verify_mode ? InferencePhase::Sequential : InferencePhase::Batched);
    const DeepSeek4RoctxRange roctx_range(
        "ds4.spec_decode",
        {roctx_phase, n_gen, 0, target_w.n_layer, device});
    const bool async_rollback = spec_env_flag("LUCE_DS4_ASYNC_ROLLBACK");
    const bool pinned_rollback = spec_env_flag("LUCE_DS4_PINNED_ROLLBACK");
    // Kill switch for the q5 two-boundary checkpoint restore (falls back to
    // restore-and-replay) and the per-token diagnostic trace.
    const bool boundary_checkpoint_enabled =
        !spec_env_flag("LUCE_DS4_DISABLE_BOUNDARY_CHECKPOINT");
    const bool token_trace = spec_env_flag("LUCE_DS4_TOKEN_TRACE");
    if (spec_env_flag("LUCE_DS4_Q6_VERIFY")) {
        std::fprintf(stderr,
            "[ds4-spec] q=6 verification is unsupported; use q=5\n");
        return false;
    }
    const bool q5_verify =
        spec_env_flag("LUCE_DS4_Q5_VERIFY") && block >= 4;
    const bool wide_verify = q5_verify;
    const bool draft_overlap_probe =
        spec_env_flag("LUCE_DS4_DRAFT_OVERLAP_PROBE");
    const bool draft_overlap_reuse_context =
        spec_env_flag("LUCE_DS4_DRAFT_OVERLAP_REUSE_CONTEXT");
    if (reference_exact) {
        std::fprintf(stderr,
            "[ds4-spec] reference-exact verifier: sequential target replay "
            "with full rollback snapshots\n");
    }
    ggml_backend_t drafter_backend =
        drafter.core.backend ? drafter.core.backend : backend;
    const bool draft_overlap_probe_active =
        draft_overlap_probe && drafter_backend != backend;
    bool draft_overlap_probe_enabled = draft_overlap_probe_active;
    if (draft_overlap_probe && !draft_overlap_probe_active) {
        std::fprintf(stderr,
            "[ds4-spec] draft overlap probe requested without an independent "
            "in-process backend; probe disabled\n");
    }
    // Shared adaptive verify width is opt-in. The DS4-specific setting wins
    // over the global setting so one backend can be qualified in isolation.
    bool adaptive_width = adaptive_spec_width_globally_enabled();
    if (const char * raw = std::getenv("LUCE_DS4_ADAPTIVE_WIDTH")) {
        adaptive_width = raw[0] && std::strcmp(raw, "0") != 0;
    }
    // A drafter confidence head scores every candidate of this very step, so
    // the width follows the text as it changes instead of lagging behind an
    // acceptance estimate. The controller extends a head that covers fewer
    // depths than the q5 verifier with the learned conditional acceptance of
    // the remaining depth, and target feedback keeps refining that estimate
    // (kDs4ConfidenceDepths bounds the depths taken from the head).
    // LUCE_DS4_CONFIDENCE_WIDTH=0 is the kill switch back to the learned
    // acceptance policy for A/B runs and drafters with a miscalibrated head.
    const bool use_confidence_width = adaptive_width && !seq_verify_mode &&
        spec_env_default_on("LUCE_DS4_CONFIDENCE_WIDTH") &&
        drafter.confidence_w != nullptr && drafter.confidence_b != nullptr &&
        (drafter.confidence_dim == n_embd ||
         drafter.confidence_dim == n_embd + drafter.markov_rank);

    // The conservative fast path remains capped at the compression ratio.
    // The explicit wide path handles a second ratio-4 boundary in-graph and
    // restores/replays only a rejected prefix, avoiding full snapshots on the
    // overwhelmingly common all-accepted path.
    // V4.1 verifies each token through its decode lane, so no compressor
    // boundary limits its width, only the rollback staging does
    // (LUCE_DS4_SPEC_Q up to six); the default width measured best on the
    // R9700 + Strix Halo hybrid tier.
    const bool tokenwise_verify = target_w.hc_staggered_pre;
    const int fast_cap = tokenwise_verify
        ? std::min(block + 1, kRollbackMaxTokens)
        : std::min(block + 1,
                   q5_verify ? DS4_Q5_VERIFY_TOKENS
                             : DS4_CONSERVATIVE_VERIFY_MAX_TOKENS);
    int q_cap = full_snap ? block + 1
              : tokenwise_verify ? std::min(fast_cap, kDs4TokenwiseVerifyDefaultWidth)
              : fast_cap;
    if (const char * qs = std::getenv("LUCE_DS4_SPEC_Q")) {
        const int v = std::atoi(qs);
        if (v >= 2 && v <= block + 1) {
            q_cap = full_snap ? v : std::min(v, fast_cap);
        }
    }
    if (seq_verify_mode &&
        q_cap > DS4_CONSERVATIVE_VERIFY_MAX_TOKENS) {
        std::fprintf(stderr,
                     "[ds4-spec] sequential verify supports q<=%d; "
                     "capping requested q=%d to %d\n",
                     DS4_CONSERVATIVE_VERIFY_MAX_TOKENS, q_cap,
                     DS4_CONSERVATIVE_VERIFY_MAX_TOKENS);
        q_cap = DS4_CONSERVATIVE_VERIFY_MAX_TOKENS;
    }
    // See kDs4AdaptiveMinWidth / kDs4VerifyWidthCostMs for the policy.
    AdaptiveSpecWidth width_controller(
        q_cap, kDs4AdaptiveMinWidth, adaptive_width && !seq_verify_mode);
    std::vector<float> width_cost_ms((size_t) q_cap + 1, 0.0f);
    for (int width = kDs4AdaptiveMinWidth;
         width <= std::min(q_cap, DS4_Q5_VERIFY_TOKENS); ++width) {
        width_cost_ms[(size_t) width] = kDs4VerifyWidthCostMs[width];
    }
    width_controller.set_relative_costs(width_cost_ms);
    if (timing && width_controller.enabled()) {
        std::fprintf(stderr,
                     "[ds4-spec] adaptive width policy=%s\n",
                     use_confidence_width
                         ? "confidence (acceptance fallback)"
                         : "acceptance-and-cost");
    }

    // Snapshot backend for the legacy full-snapshot rollback path.
    ggml_backend_t snap_backend = ggml_backend_cpu_init();
    if (!snap_backend) { std::fprintf(stderr, "[ds4-spec] no CPU snapshot backend\n"); return false; }

    DeepSeek4DFlashTarget target(target_w, target_cache, backend, device, snap_backend,
                                 drafter.capture_layer_ids, drafter.mask_token_id,
                                 moe_hybrid, expert_runtime, routing_stats);
    DraftWeights dw = make_dspark_shim(drafter);
    DeepSeek4SpecRollback rollback;
    DeepSeek4StepTelemetry tel{};
    if (timing) target.set_telemetry(&tel);

    // Host feature window ring [feat_row, n_swa] of absolute positions
    // [committed-N .. committed-1]. Seed from the prefill window.
    std::vector<float> feat_win((size_t) feat_row * n_swa, 0.0f);
    int win_have = win_len > n_swa ? n_swa : win_len;
    if (prompt_feature_window && win_have > 0) {
        // copy the last win_have columns of the prefill window
        const int src_off = (win_len - win_have);
        std::memcpy(feat_win.data(),
                    prompt_feature_window + (size_t) src_off * feat_row,
                    sizeof(float) * (size_t) feat_row * win_have);
    }
    int feat_count = win_have;   // number of valid feature columns ending at committed-1

    auto push_features = [&](const float * cols, int count) {
        if (!cols || count <= 0) return;
        if (count >= n_swa) {
            std::memcpy(
                feat_win.data(), cols + (size_t) (count - n_swa) * feat_row,
                sizeof(float) * (size_t) feat_row * n_swa);
            feat_count = n_swa;
            return;
        }

        // Shift at most once per accepted speculative block. The old
        // per-column loop moved the full ~7.5 MiB window up to four times per
        // q4 step even though only the final contiguous suffix was observable.
        const int keep = std::min(feat_count, n_swa - count);
        const int drop = feat_count - keep;
        if (drop > 0 && keep > 0) {
            std::memmove(
                feat_win.data(), feat_win.data() + (size_t) drop * feat_row,
                sizeof(float) * (size_t) feat_row * keep);
        }
        std::memcpy(
            feat_win.data() + (size_t) keep * feat_row, cols,
            sizeof(float) * (size_t) feat_row * count);
        feat_count = keep + count;
    };

    int lt = last_tok;
    int pos = committed;      // absolute position of the seed (block slot 0)
    int n_generated = 0;
    long accept_sum = 0, offered_sum = 0, steps = 0;
    std::vector<long> width_steps((size_t) q_cap + 1, 0);
    bool ok = true;
    bool stop_requested = false;

    std::vector<float> noise_embed((size_t) n_embd * block);
    std::vector<int32_t> noise_ids(block);
    std::vector<float> local_hidden, confidence_hidden;
    std::vector<float> padded_hidden((size_t) n_embd * (block + 1), 0.0f);
    std::vector<float> padded_confidence_hidden((size_t) n_embd * (block + 1), 0.0f);
    std::vector<int32_t> draft_tok, tgt_am;
    std::vector<float> draft_confidence;
    std::vector<float> step_confidence;

    // Cumulative phase timings (ms).
    double tm_draft = 0, tm_head = 0, tm_save = 0, tm_verify = 0, tm_apply = 0, tm_feat = 0;
    double tm_probe_submit = 0, tm_probe_wait = 0;
    const SpecClock::time_point run_t0 = SpecClock::now();

    while (n_generated < n_gen) {
        const SpecClock::time_point step_t0 = SpecClock::now();
        const int ctx_len = feat_count < n_swa ? feat_count : n_swa;

        // Noise block = [seed] + [MASK]*(block-1).
        SpecClock::time_point t0 = SpecClock::now();
        if (q_cap >= 2) {
            noise_ids[0] = lt;
            for (int i = 1; i < block; i++) {
                noise_ids[i] = drafter.mask_token_id;
            }
            if (!target.embed_tokens(
                    noise_ids.data(), block, noise_embed.data())) {
                std::fprintf(stderr,
                    "[ds4-spec] draft embedding lookup failed\n");
                ok = false;
                break;
            }

            // Drafter forward -> block normed hidden states.
            const bool draft_ok = deepseek4_dspark_draft_forward(
                drafter_backend,
                drafter, noise_embed.data(),
                ctx_len > 0 ? feat_win.data() : nullptr,
                ctx_len, pos, local_hidden,
                use_confidence_width ? &confidence_hidden : nullptr);
            if (!draft_ok) {
                std::fprintf(stderr, "[ds4-spec] drafter forward failed\n");
                ok = false;
                break;
            }
        }
        tm_draft += spec_ms_since(t0);
        if (debug && wide_verify && steps == 0) {
            std::fprintf(stderr, "[ds4-wide] draft-ready block=%d hidden=%zu\n",
                         block, local_hidden.size());
        }

        if (debug) {
            size_t lh_nan = 0; double lh_ss = 0;
            for (float v : local_hidden) { if (!std::isfinite(v)) lh_nan++; else lh_ss += (double) v * v; }
            std::fprintf(stderr, "[ds4-spec] hidden nnan=%zu/%zu rms=%.4f ctx_len=%d\n",
                         lh_nan, local_hidden.size(),
                         lh_nan < local_hidden.size() ? std::sqrt(lh_ss / (double) local_hidden.size()) : 0.0,
                         ctx_len);
        }

        // DSpark Markov chain over the first q_cap-1 candidates. Reference
        // predicts token i+1 from block slot i, so prepend a dummy row 0 and
        // let the (row-0-skipping) chain use slots 1..q-1.
        t0 = SpecClock::now();
        draft_tok.clear();
        draft_confidence.clear();
        bool ds_ok = false;
        // Batched-verify exactness: the batch must not cross a ratio-4
        // boundary except at its last token (state rows stay distinct and the
        // comp emission matches AR). Boundaries sit at p % 4 == 3. The
        // sequential verify has no such limit.
        // The fused verifier handles a ratio-4 compressor boundary at any
        // position inside the batch.  Keep the legacy boundary clamp only for
        // the dynamic batched path; otherwise "fixed q3" degenerates into q1
        // whenever pos % 4 == 3 and pays a full target verify for one token.
        static const bool fused_verify_mode = [] {
            const char * v = std::getenv("LUCE_DS4_FUSED_VERIFY");
            return v && *v && *v != '0';
        }();
        int q_step_cap = tokenwise_verify ? q_cap
                       : (seq_verify_mode || fused_verify_mode)
                       ? std::min(
                             q_cap,
                             q5_verify ? DS4_Q5_VERIFY_TOKENS
                                       : DS4_CONSERVATIVE_VERIFY_MAX_TOKENS)
                       : std::min(
                             q_cap,
                             DS4_CONSERVATIVE_VERIFY_MAX_TOKENS - (pos & 3));
        // A calibrated confidence head already predicts this individual
        // step. Do not stack the slower acceptance-regime cap on top of it;
        // acceptance feedback remains the fallback and drives q5/artifacts
        // without confidence metadata.
        if (!use_confidence_width) {
            q_step_cap = width_controller.next_width_cost_aware(
                {}, q_step_cap);
        }
        if (q_step_cap >= 2) {
            std::memcpy(padded_hidden.data() + n_embd, local_hidden.data(),
                        sizeof(float) * (size_t) n_embd * block);
            if (use_confidence_width) {
                std::memcpy(padded_confidence_hidden.data() + n_embd,
                            confidence_hidden.data(),
                            sizeof(float) * (size_t) n_embd * block);
            }
            ds_ok = dspark_markov_correct_greedy_chain_fused(
                            dw, backend, target.lm_head_tensor(), padded_hidden.data(),
                            q_step_cap, lt, draft_tok,
                            use_confidence_width ? &draft_confidence : nullptr,
                            use_confidence_width
                                ? padded_confidence_hidden.data() : nullptr,
                            nullptr);
            if (!ds_ok) {
                ds_ok = dspark_markov_correct_greedy_chain(dw, backend, target,
                            padded_hidden.data(), q_step_cap, lt, 0.0f, draft_tok);
            }
            if (!ds_ok || (int) draft_tok.size() < 2) {
                // Fallback: plain projection of the block hiddens.
                std::vector<int32_t> pj;
                if (!target.project_hidden_to_tokens(
                        local_hidden.data(), q_step_cap - 1, pj)) {
                    std::fprintf(stderr,
                        "[ds4-spec] draft projection fallback failed\n");
                    ok = false;
                    break;
                }
                draft_tok.clear();
                draft_tok.push_back(lt);
                for (int i = 0; i < q_step_cap - 1; i++) {
                    draft_tok.push_back(pj[(size_t) i]);
                }
            }
        } else {
            draft_tok.push_back(lt);   // q=1: seed only, no speculation
        }
        // Confidence estimates are conditional per candidate. Select the
        // width with the best predicted committed-tokens/step-cost ratio;
        // this avoids treating a narrower verifier as proportionally cheaper
        // when q3 and q4 are nearly the same cost on gfx1151.
        if (use_confidence_width && !draft_confidence.empty()) {
            if (draft_confidence.size() > (size_t) kDs4ConfidenceDepths) {
                step_confidence.assign(draft_confidence.begin(),
                                       draft_confidence.begin() + kDs4ConfidenceDepths);
            } else {
                step_confidence = draft_confidence;
            }
            const int selected_q = width_controller.next_width_cost_aware(
                step_confidence, (int) draft_tok.size());
            if ((int) draft_tok.size() > selected_q) draft_tok.resize((size_t) selected_q);
        } else if (use_confidence_width && !seq_verify_mode) {
            const int selected_q = width_controller.next_width((int)draft_tok.size());
            if ((int)draft_tok.size() > selected_q) {
                draft_tok.resize((size_t)selected_q);
            }
        }
        if ((int) draft_tok.size() > q_step_cap) draft_tok.resize(q_step_cap);
        const int q = (int) draft_tok.size();   // seed + candidates
        if (q >= 0 && q <= q_cap) width_steps[(size_t) q]++;
        tm_head += spec_ms_since(t0);
        if (debug && wide_verify && steps == 0) {
            std::fprintf(stderr, "[ds4-wide] head-ready q=%d\n", q);
        }

        if (debug) {
            std::fprintf(stderr, "[ds4-spec] dbg ds_ok=%d q=%d lt=%d draft=[%d %d %d %d]\n",
                         (int) ds_ok, q, lt,
                         q > 0 ? draft_tok[0] : -1, q > 1 ? draft_tok[1] : -1,
                         q > 2 ? draft_tok[2] : -1, q > 3 ? draft_tok[3] : -1);
        }

        // Feasibility-only control: duplicate the current draft and discard it.
        bool probe_inflight = false;
        if (draft_overlap_probe_enabled && q_cap >= 2) {
            const SpecClock::time_point probe_t0 = SpecClock::now();
            probe_inflight = draft_overlap_reuse_context
                ? deepseek4_dspark_draft_forward_async_reuse_context(
                    drafter_backend, drafter, noise_embed.data(), ctx_len, pos)
                : deepseek4_dspark_draft_forward_async(
                    drafter_backend, drafter, noise_embed.data(),
                    ctx_len > 0 ? feat_win.data() : nullptr, ctx_len, pos);
            tm_probe_submit += spec_ms_since(probe_t0);
            if (!probe_inflight) {
                draft_overlap_probe_enabled = false;
                std::fprintf(stderr,
                    "[ds4-spec] draft overlap probe launch failed; disabling\n");
            }
        }

        static const bool spec_diag = spec_env_flag("LUCE_DS4_SPEC_DIAG");
        if (spec_diag && q >= 2 && !target.diag_batch_vs_sequential(draft_tok, pos, steps)) {
            std::fprintf(stderr, "[ds4-spec-diag] step=%ld diagnostic failed\n", steps);
        }

        // ── Rollback state save (cheap) or legacy full snapshot ──
        t0 = SpecClock::now();
        if (full_snap) {
            if (!target.snapshot_kv()) {
                std::fprintf(stderr, "[ds4-spec] snapshot failed\n");
                ok = false;
                break;
            }
        } else {
            spec_rollback_save(
                target_cache, rollback, backend,
                async_rollback || pinned_rollback, pinned_rollback,
                pos, q);
        }
        tm_save += spec_ms_since(t0);
        if (debug && wide_verify && steps == 0) {
            std::fprintf(stderr, "[ds4-wide] rollback-ready rows=%d\n",
                         rollback.raw_count);
        }

        // First ratio-4 boundary position at or after the seed (p % 4 == 3).
        const int first_boundary = deepseek4_first_ratio4_boundary(pos);
        const bool boundary_crossed = first_boundary <= pos + q - 1;
        const bool multiple_boundaries_crossed =
            deepseek4_verify_crosses_multiple_ratio4_boundaries(pos, q);

        // ── ONE batched verify (writes cache + captures features for all q) ──
        t0 = SpecClock::now();
        int verify_last = -1;
        if (debug && wide_verify && steps == 0) {
            std::fprintf(stderr, "[ds4-wide] verify-begin q=%d pos=%d\n", q, pos);
        }
        const bool verify_ok =
            target.verify_batch(draft_tok, pos, verify_last, &tgt_am);
        tm_verify += spec_ms_since(t0);
        if (probe_inflight) {
            const SpecClock::time_point probe_t0 = SpecClock::now();
            deepseek4_dspark_draft_wait(drafter_backend);
            tm_probe_wait += spec_ms_since(probe_t0);
        }
        if (!verify_ok) {
            if (full_snap) {
                if (!target.restore_kv()) {
                    std::fprintf(stderr, "[ds4-spec] restore after verify failure failed\n");
                }
            } else {
                spec_rollback_apply(
                    rollback, target_w, target_cache, pos, boundary_crossed);
            }
            std::fprintf(stderr, "[ds4-spec] verify failed\n");
            ok = false;
            break;
        }

        // Accept the longest matching prefix. accept counts the seed (slot 0)
        // plus each candidate the target agrees with.
        int accept = 1;
        for (int i = 0; i < q - 1; i++) {
            if (draft_tok[i + 1] == tgt_am[i]) accept++;
            else break;
        }
        const int matched = accept - 1;                       // accepted candidates
        const int bonus = tgt_am[accept - 1];                 // target's token at the accept point
        const int commit_pos = pos + accept;                  // seed + accepted candidates in KV

        if (timing && steps < 8 && q >= 2) {
            // Alignment probe: draft candidate i should match tgt_am[i-1]. A
            // consistent draft[i]==tgt_am[i] pattern instead = off-by-one.
            std::fprintf(stderr, "[ds4-spec-cmp] step=%ld pos=%d draft=[%d %d %d] tgt=[%d %d %d %d] acc=%d\n",
                         steps, pos,
                         q > 1 ? draft_tok[1] : -1, q > 2 ? draft_tok[2] : -1, q > 3 ? draft_tok[3] : -1,
                         tgt_am[0], q > 1 ? tgt_am[1] : -1, q > 2 ? tgt_am[2] : -1, q > 3 ? tgt_am[3] : -1,
                         accept);
        }

        // ── Rollback: truncate to the committed prefix ──
        // The bonus token is DEFERRED: it becomes the next step's seed, whose
        // KV is written then.
        t0 = SpecClock::now();
        if (full_snap && accept < q) {
            // Legacy: full restore + replay the committed tokens through the
            // target so ring/compressor/n_comp advance exactly.
            std::vector<int32_t> kv_toks;
            kv_toks.push_back(lt);
            for (int i = 1; i < accept; i++) kv_toks.push_back(draft_tok[i]);
            if (!target.restore_kv()) {
                std::fprintf(stderr, "[ds4-spec] snapshot restore failed\n");
                ok = false;
                break;
            }
            int replay_last = -1;
            std::vector<int32_t> replay_am;
            if (!target.verify_batch(kv_toks, pos, replay_last, &replay_am)) {
                std::fprintf(stderr, "[ds4-spec] replay verify failed\n");
                ok = false;
                break;
            }
        } else if (!full_snap && accept < q && tokenwise_verify) {
            // Raw rows and compressed-row counters as below; the V4.1
            // compressors keep no ratio-4 halves, only pooled windows.
            spec_rollback_apply(rollback, target_w, target_cache, commit_pos, false);
            spec_restore_window_rows(target.last_window_rows(), target_w, target_cache,
                                     pos, commit_pos);
        } else if (!full_snap && accept < q &&
                   q > DS4_CONSERVATIVE_VERIFY_MAX_TOKENS) {
            // Rejected wide (q5) verify over positions [pos, pos + 4].
            // Recoveries, cheapest first:
            // (1) two ratio-4 boundaries touched (pos % 4 == 3, see
            //     deepseek4_verify_crosses_multiple_ratio4_boundaries): the
            //     fused graph checkpointed the ratio-4 state right after its
            //     first flush, so copy those rows back on-device and truncate
            //     the counters/ring instead of replaying every target layer;
            // (2) one boundary touched: truncate directly like the q<=4
            //     verifier (gfx1151 at 123K: 169.8 -> 160.1 ms/step). The
            //     fifth token writes the seed's rolling slot (4 + pos % 4).
            //     If the accepted prefix reaches past the boundary the flush
            //     stays and that slot belongs to a rejected post-boundary
            //     position, rewritten before it is ever pooled. If the flush
            //     must be undone the seed's row is still on the device in
            //     the rotated previous half, so it is copied back into its
            //     slot before the previous half is restored.
            // (3) two boundaries without a usable checkpoint: restore the
            //     compact pre-verify state and replay the accepted prefix.
            if (multiple_boundaries_crossed && boundary_checkpoint_enabled &&
                target.restore_first_boundary_checkpoint()) {
                spec_rollback_apply(
                    rollback, target_w, target_cache, commit_pos, false);
            } else if (!multiple_boundaries_crossed) {
                const bool undo_flush = commit_pos <= first_boundary;
                if (undo_flush) {
                    for (DeepSeek4LayerCache & lc : target_cache.layers) {
                        deepseek4_spec_restore_seed_row(
                            backend, lc.attn_compressor.state_kv, pos);
                        deepseek4_spec_restore_seed_row(
                            backend, lc.attn_compressor.state_score, pos);
                        deepseek4_spec_restore_seed_row(
                            backend, lc.indexer_compressor.state_kv, pos);
                        deepseek4_spec_restore_seed_row(
                            backend, lc.indexer_compressor.state_score, pos);
                    }
                }
                spec_rollback_apply(
                    rollback, target_w, target_cache, commit_pos, undo_flush);
            } else {
                spec_rollback_apply(
                    rollback, target_w, target_cache, pos, true);
                std::vector<int32_t> kv_toks;
                kv_toks.reserve((size_t) accept);
                kv_toks.push_back(lt);
                for (int i = 1; i < accept; ++i) {
                    kv_toks.push_back(draft_tok[i]);
                }
                int replay_last = -1;
                std::vector<int32_t> replay_am;
                if (!target.verify_batch(
                        kv_toks, pos, replay_last, &replay_am)) {
                    std::fprintf(stderr,
                                 "[ds4-spec] wide rollback replay failed\n");
                    ok = false;
                    break;
                }
            }
        } else if (!full_snap && accept < q) {
            // The prev-half flush is bad only if the boundary sits at-or-past
            // the commit point (its chunk then contains rejected tokens).
            const bool restore_prev = boundary_crossed && first_boundary >= commit_pos;
            spec_rollback_apply(
                rollback, target_w, target_cache, commit_pos, restore_prev);
        }
        // accept == q on the fast path: cur_pos/n_comp already exact, keep.
        tm_apply += spec_ms_since(t0);

        // Push the committed positions' features (slots 0..accept-1 = positions
        // pos..pos+accept-1) into the drafter's context window.
        t0 = SpecClock::now();
        const std::vector<float> & feats = target.last_features();
        const int fN = full_snap ? target.last_verify_n() : accept;
        push_features(feats.data(), fN);
        tm_feat += spec_ms_since(t0);
        width_controller.observe(
            accept, q, (float) spec_ms_since(step_t0));
        if (use_confidence_width) {
            width_controller.observe_confidence(draft_confidence, accept, q);
        }

        // Output tokens this step = accepted candidates + bonus.
        bool hit_eos = false;
        for (int i = 1; i <= accept; i++) {
            const int t = (i < accept) ? draft_tok[i] : bonus;
            if (token_trace) {
                std::fprintf(stderr,
                             "[ds4-spec-token] out=%d step=%ld slot=%d token=%d%s\n",
                             n_generated, steps, i, t,
                             i == accept ? " bonus" : "");
            }
            out_tokens.push_back(t);
            n_generated++;
            if (on_token && !on_token(t)) {
                stop_requested = true;
                break;
            }
            if (target.is_eos(t)) { hit_eos = true; break; }
            if (n_generated >= n_gen) break;
        }
        pos = commit_pos;              // seed + accepted candidates now in KV
        lt = bonus;                    // deferred bonus becomes next seed
        accept_sum += matched;
        offered_sum += q - 1;
        steps++;
        if (timing && (steps <= 4 || (steps & 31) == 0)) {
            std::fprintf(stderr,
                "[ds4-spec-t] step=%ld q=%d acc=%d | draft=%.1f head=%.1f save=%.1f "
                "verify=%.1f probe(submit/wait)=%.1f/%.1f apply=%.1f feat=%.1f "
                "ms (cum means)\n",
                steps, q, accept,
                tm_draft / steps, tm_head / steps, tm_save / steps,
                tm_verify / steps, tm_probe_submit / steps,
                tm_probe_wait / steps, tm_apply / steps, tm_feat / steps);
        }
        if (hit_eos || stop_requested) break;
    }

    const double total_ms = spec_ms_since(run_t0);
    if (accept_rate_out) {
        *accept_rate_out = offered_sum > 0
            ? (float) accept_sum / (float) offered_sum
            : 0.0f;
    }
    std::fprintf(stderr,
                 "[ds4-spec] gen=%d steps=%ld mean_accept=%.2f/%.2f "
                 "q_cap=%d full_snap=%d\n",
                 n_generated, steps,
                 steps ? (double) accept_sum / steps : 0.0,
                 steps ? (double) offered_sum / steps : 0.0, q_cap,
                 (int) full_snap);
    if (width_controller.enabled()) {
        std::fprintf(
            stderr,
            "[ds4-spec] adaptive widths q2=%ld q3=%ld q4=%ld q5=%ld\n",
            q_cap >= 2 ? width_steps[2] : 0,
            q_cap >= 3 ? width_steps[3] : 0,
            q_cap >= 4 ? width_steps[4] : 0,
            q_cap >= 5 ? width_steps[5] : 0);
        if (use_confidence_width) {
            // Recent predicted against observed acceptance per depth and
            // the scale the controller applies to the head's scores.
            std::string line = "[ds4-spec] confidence calibration";
            char buf[64];
            for (int d = 1; d < q_cap; ++d) {
                std::snprintf(buf, sizeof(buf), " d%d pred=%.2f actual=%.2f scale=%.2f",
                              d, width_controller.confidence_predicted(d),
                              width_controller.confidence_observed(d),
                              width_controller.confidence_scale(d));
                line += buf;
            }
            std::fprintf(stderr, "%s\n", line.c_str());
        }
    }
    if (steps > 0) {
        std::fprintf(stderr,
            "[ds4-spec-t] TOTAL %.1f ms, %ld steps (%.1f ms/step), %d tok (%.1f tok/s) | "
            "means: draft=%.1f head=%.1f save=%.1f verify=%.1f "
            "probe_submit=%.1f probe_wait=%.1f apply=%.1f feat=%.1f ms\n",
            total_ms, steps, total_ms / steps, n_generated,
            total_ms > 0 ? n_generated * 1000.0 / total_ms : 0.0,
            tm_draft / steps, tm_head / steps, tm_save / steps,
            tm_verify / steps, tm_probe_submit / steps,
            tm_probe_wait / steps, tm_apply / steps, tm_feat / steps);
    }
    if (timing && steps > 0) {
        const double s = 1000.0 * steps;   // us -> ms per-step means
        std::fprintf(stderr,
            "[ds4-spec-t] verify tel/step: hc_pre_a=%.1f attn_b=%.1f attn_c=%.1f attn_r=%.1f "
            "hc_post_a=%.1f hc_pre_f=%.1f route(b/c/r/s)=%.1f/%.1f/%.1f/%.1f "
            "ffn(b/c/r)=%.1f/%.1f/%.1f eval=%.1f hot=%.1f cold=%.1f comb=%.1f part=%.1f "
            "full(b/s/c/r)=%.1f/%.1f/%.1f/%.1f engram(read/apply)=%.1f/%.1f ghits=%llu gbuilds=%llu ms\n",
            tel.hc_pre_attn_us / s, tel.attn_build_us / s, tel.attn_compute_us / s,
            tel.attn_read_us / s, tel.hc_post_attn_us / s, tel.hc_pre_ffn_us / s,
            tel.route_build_us / s, tel.route_compute_us / s, tel.route_read_us / s,
            tel.route_select_us / s,
            tel.ffn_build_us / s, tel.ffn_compute_us / s, tel.ffn_read_us / s,
            tel.ffn_eval_us / s, tel.ffn_hot_us / s, tel.ffn_cold_us / s,
            tel.ffn_combine_us / s, tel.ffn_partition_us / s,
            tel.full_graph_build_us / s, tel.full_graph_set_us / s,
            tel.full_graph_compute_us / s, tel.full_graph_read_us / s,
            tel.engram_read_us / s, tel.engram_apply_us / s,
            (unsigned long long) tel.ffn_hot_graph_hits,
            (unsigned long long) tel.ffn_hot_graph_builds);
    }
    // Snapshot buffers must be released while their backend is still alive.
    // clear_snapshot() is idempotent, so the target destructor remains a
    // safety net for future exits that are added above this point.
    target.clear_snapshot();
    ggml_backend_free(snap_backend);
    return ok;
}

}  // namespace luce::common
