// End-to-end harness for the GQH qtypes (108/109/110/111) through the ggml backend.
//
// The standalone kernel test (test-gqh-decode.cu) proves the arithmetic. This one
// proves the REGISTRATION: that ggml's type traits size a GQH tensor correctly,
// that the header registry the loader fills reaches the CUDA converters, and that
// a MUL_MAT node on a GQH src0 lands on the CUDA dequant->cuBLAS path instead of
// the CPU backend (which refuses these types for lack of a vec_dot).
//
// It also pins the GGUF contract: the tensor data staged here is the wire file
// with its 5-byte per-tensor header STRIPPED, and the header travels separately
// through the registry -- exactly what the exporter must emit.
//
// Two paths, both driven by MUL_MAT against basis vectors so each decoded weight
// is accumulated exactly once against an exact 1.0:
//
//   1. dequant->cuBLAS: x is the full cols x cols identity, wider than the fused
//      hook's batch cap, so dst[i,j] is fp16(decode(W[i,j])) widened to f32 and is
//      compared BITWISE against the reference decode rounded to fp16.
//   2. fused matvec: x is the first nvec basis vectors, inside the cap, so the kernel
//      accumulates decode(W) * x in f32 and dst[i,j] IS the raw f32 decode,
//      compared BITWISE against the f32 reference. gqh2_c has no fused kernel yet
//      and falls back to path 1, so it skips this check.
//   3. MMQ (the "mmqN" flag): the batched quantised matmul that REPLACES path 1
//      for wide batches once GGML_GQH_MMQ=1. Not bit-exact and cannot be: the
//      grid is curved and int8 perturbs every level. The bar is still tight,
//      because one-hot columns make the ACTIVATION quantisation exact -- amax 1
//      gives d = 1/127 and q = 127, so the dot has a single term and the only
//      error left is one weight step. See the mmq block below.
//
// usage: test-gqh-backend <gqh3|gqh2_h|gqh2_c|gqh4> <rows> <cols> <wire.bin> <decode.f32>
//                        [nvec] [i8] [mmqN]
//   nvec (default 8) must be <= GQH_MAX_COLS (16), or the hook declines and the
//   fallback answers in fp16, failing every fused comparison. GQH is not gated
//   on luce_mmvq_max_ncols; 8 is the DFlash2 default verify width.
//   mmqN adds a SECOND wide multiply at ncols = N (N > 16) and requires that MMQ
//   actually dispatched, which is what makes these cases a gate rather than a
//   re-run of the fallback. Needs GGML_GQH_MMQ=1.
//   gridN overrides the grid code the wire header carries, and decodes the
//   reference on the HOST for that grid instead of reading the .decode.f32 file
//   (ggml_get_type_traits()->to_float, the same CPU decoder test-gqh-cpu-decode
//   checks). That is what makes the int8 range guard testable: no artifact and no
//   vector file uses a grid wide enough to trip it.
//   refuse says MMQ must DECLINE this grid. Then the answer has to come back
//   BITWISE equal to the fp16-rounded reference, which only the dequant path can
//   produce -- so a guard that failed to refuse cannot pass by accident, it comes
//   back int8-perturbed and fails loudly. See the guard in gqh.cu.
// exit:  0 = bit-identical, 1 = mismatch/error, 77 = skipped (no GPU)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"   // ggml_backend_cuda_get_mmq_launch_count

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#define GQH_SUPERBLOCK   256
#define GQH_HEADER_BYTES   5

// The dequant->BLAS path yields fp16(decode(W)) widened to f32. Where the f32
// reference sits EXACTLY halfway between two fp16 values, which of the pair comes
// back is backend-defined: NVIDIA and the host both pick ties-to-even, the ROCm
// path returns the other neighbour. gqh2_c hits this constantly because its
// codebook is k/64 -- dyadic -- so products land on ties; gqh3's (k/4)^gamma grid
// essentially never does, which is why only this rung shows it.
//
// Accept ONLY that: an exact tie whose result is one of the two bracketing fp16
// values. A 1-ULP difference anywhere else, or any larger difference, still fails.
// This is not a tolerance -- the f32 decode itself is checked bit-exactly by the
// fused path below, on both platforms.
static bool fp16_tie_equivalent(float ref, float got) {
    const uint16_t h = ggml_fp32_to_fp16(ref);
    const float    a = ggml_fp16_to_fp32(h);
    for (int d = -1; d <= 1; d += 2) {
        const float b = ggml_fp16_to_fp32((uint16_t) (h + d));
        if (b != got) {
            continue;
        }
        // exact in double: both neighbours are fp16 and ref is f32
        return (double) ref == ((double) a + (double) b) / 2.0;
    }
    return false;
}

// GQH_TEST_BOUND_N: build the per-grid weight-LUT bound at THIS denominator,
// whatever the kernel is actually using.
//
//   unset / empty              -> follow the kernel, i.e. ggml_gqh_q8_denom_eff.
//                                 An ordinary case is then SELF-CONSISTENT and
//                                 passes whichever way the shipping default is set.
//   1..127                     -> that flat denominator.
//   0 / "opt" / "derived"      -> the per-grid derived optimum, ggml_gqh_q8_denom.
//
// This is the ONLY way to get a deliberate mismatch, and a deliberate mismatch is
// the whole negative control: set the kernel to one denominator and the bound to a
// tighter one and the case MUST fail. The bound used to be pinned to the derived
// table unconditionally, which made the negative control a side effect of what the
// default happened to be -- so the moment the default moved to the flat 127, every
// ordinary i8 case failed and the control passed for the wrong reason. Returns -1
// for "follow the kernel" and 0 for "the per-grid derived optimum".
static int bound_denom_request() {
    const char * e = getenv("GQH_TEST_BOUND_N");
    if (!e || !*e) {
        return -1;
    }
    if (!strcmp(e, "opt") || !strcmp(e, "derived") || !strcmp(e, "auto")) {
        return 0;
    }
    const int v = atoi(e);
    if (v == 0) {
        return 0;
    }
    if (v < 1 || v > 127) {
        fprintf(stderr, "GQH_TEST_BOUND_N=%s is not 1..127, 0 or derived\n", e);
        exit(1);
    }
    return v;
}

static std::vector<uint8_t> read_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t) n);
    if (n > 0 && fread(buf.data(), 1, (size_t) n, f) != (size_t) n) {
        fprintf(stderr, "short read on %s\n", path);
        exit(1);
    }
    fclose(f);
    return buf;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <gqh3|gqh2_h|gqh2_c|gqh4> <rows> <cols> <wire.bin> <decode.f32> "
                        "[nvec] [i8] [mmqN]\n", argv[0]);
        return 1;
    }
    const std::string rung = argv[1];
    const int rows = atoi(argv[2]);
    const int cols = atoi(argv[3]);
    if (rows <= 0 || cols <= 0 || cols % GQH_SUPERBLOCK) {
        fprintf(stderr, "bad dims (cols must be a multiple of %d)\n", GQH_SUPERBLOCK);
        return 1;
    }
    ggml_type wtype;
    if      (rung == "gqh3")   { wtype = GGML_TYPE_GQH3;   }
    else if (rung == "gqh2_h") { wtype = GGML_TYPE_GQH2_H; }
    else if (rung == "gqh4")   { wtype = GGML_TYPE_GQH4; }
    else if (rung == "gqh2_c") { wtype = GGML_TYPE_GQH2_C; }
    else { fprintf(stderr, "unknown rung %s\n", rung.c_str()); return 1; }
    // gqh2_c carries no per-tensor header: fp16 scale in-block, frozen codebook.
    const bool has_header = wtype != GGML_TYPE_GQH2_C;

    // Trailing tokens in any order: a bare integer is nvec, "i8" selects the int8
    // activation arm's tolerances, "mmqN" adds the wide MMQ multiply at ncols = N.
    int  nvec     = 8;   // must be <= the fused hook's column cap
    bool i8_flag  = false;
    int  nwide    = 0;
    int  grid_force = -1;
    bool expect_refuse = false;
    for (int a = 6; a < argc; ++a) {
        const std::string tok = argv[a];
        if (tok == "i8") {
            i8_flag = true;
        } else if (tok == "refuse") {
            expect_refuse = true;
        } else if (tok.rfind("mmq", 0) == 0) {
            nwide = atoi(tok.c_str() + 3);
        } else if (tok.rfind("grid", 0) == 0) {
            grid_force = atoi(tok.c_str() + 4);
        } else {
            nvec = atoi(tok.c_str());
        }
    }
    if (nwide < 0 || nwide > cols) {
        fprintf(stderr, "mmq width %d must be in 1..cols (%d)\n", nwide, cols);
        return 1;
    }
    const std::vector<uint8_t> wire = read_file(argv[4]);
    const std::vector<uint8_t> ref  = read_file(argv[5]);
    const size_t n_elem = (size_t) rows * cols;
    if (ref.size() != n_elem * sizeof(float)) {
        fprintf(stderr, "decode.f32 is %zu B, want %zu B\n", ref.size(), n_elem * sizeof(float));
        return 1;
    }

    float tensor_scale = 1.0f;
    int   grid_code    = 0;
    if (has_header) {
        memcpy(&tensor_scale, wire.data(), sizeof(float));
        grid_code = wire[4];
    }
    if (grid_force >= 0) {
        if (!has_header) {
            fprintf(stderr, "gridN needs a header-bearing rung\n");
            return 1;
        }
        grid_code = grid_force;
    }

    // "i8" selects the tolerances the int8 activation arm needs. The arm is NOT
    // bit-exact: both sides carry one symmetric int8 round (weights bake to int8
    // against a grid whose amax is exactly 1.0, activations quantise per group by
    // 127/amax), and the two 1/127s fold into GQH_Q8_XSCALE. Reaching it from this
    // harness also needs GGML_GQH_I8_MINWORK=0, because gqh_i8_shape_ok() floors the
    // arm at 16 Mi of in*rows and the largest case here is 64*1024 = 64 Ki -- 256x
    // under it. That floor, not the I8DOT knob, is why this path had no coverage.
    // A forced grid makes the shipped .decode.f32 the wrong reference, so decode
    // on the host for the grid actually in force. The CPU decoder is registry
    // driven like the CUDA one, so the host copy of the wire body has to be
    // registered too -- the registry is a pointer-range map, so both can be.
    std::vector<float> hostref;
    if (grid_force >= 0) {
        const size_t body_bytes = wire.size() - GQH_HEADER_BYTES;
        const uint8_t * body = wire.data() + GQH_HEADER_BYTES;
        if (body_bytes % (size_t) rows) {
            fprintf(stderr, "wire body %zu B is not a whole number of rows\n", body_bytes);
            return 1;
        }
        const size_t row_bytes = body_bytes / (size_t) rows;
        hostref.resize(n_elem);
        ggml_gqh_register(body, body_bytes, tensor_scale, grid_code);
        const struct ggml_type_traits * tt = ggml_get_type_traits(wtype);
        for (int i = 0; i < rows; ++i) {
            tt->to_float(body + (size_t) i * row_bytes,
                         hostref.data() + (size_t) i * cols, cols);
        }
        ggml_gqh_unregister(body);
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        printf("SKIP: no GPU backend device\n");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        printf("SKIP: GPU backend failed to initialize\n");
        return 77;
    }

    ggml_init_params ip = { ggml_tensor_overhead() * 12 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const bool i8_mode  = i8_flag;
    const bool mmq_mode  = nwide > 0;
    const bool has_fused = true;

    ggml_tensor * W  = ggml_new_tensor_2d(ctx, wtype, cols, rows);
    ggml_tensor * X  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, cols);
    ggml_tensor * X2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, nvec);
    ggml_set_name(W, "W_gqh");

    // The type traits must size the tensor to the wire minus its header.
    const size_t want_data = wire.size() - (has_header ? GQH_HEADER_BYTES : 0);
    if (ggml_nbytes(W) != want_data) {
        fprintf(stderr, "ggml_nbytes(%s %dx%d) = %zu, wire body is %zu B\n",
                ggml_type_name(wtype), rows, cols, ggml_nbytes(W), want_data);
        return 1;
    }

    // A THIRD multiply at a width that is not a tile multiple. D above already
    // sweeps every one-hot position at ncols = cols, but cols is 256/512/1024 --
    // all multiples of the mmq_x granularity, so the partial-tile j_max path never
    // runs. 17/33 do run it, and this is also where the dense wide pass lives.
    ggml_tensor * X3 = mmq_mode ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, nwide) : nullptr;

    ggml_tensor * D  = ggml_mul_mat(ctx, W, X);  // (rows, cols): dequant->cuBLAS, or MMQ
    ggml_tensor * D2 = ggml_mul_mat(ctx, W, X2); // (rows, nvec): fused matvec
    ggml_tensor * D3 = X3 ? ggml_mul_mat(ctx, W, X3) : nullptr;
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, D);
    ggml_build_forward_expand(gf, D2);
    if (D3) {
        ggml_build_forward_expand(gf, D3);
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "graph allocation failed\n");
        return 1;
    }

    ggml_backend_tensor_set(W, wire.data() + (has_header ? GQH_HEADER_BYTES : 0), 0, want_data);
    {
        std::vector<float> ident((size_t) cols * cols, 0.0f);
        for (int j = 0; j < cols; ++j) {
            ident[(size_t) j * cols + j] = 1.0f;
        }
        ggml_backend_tensor_set(X, ident.data(), 0, ident.size() * sizeof(float));
        std::vector<float> onehot((size_t) cols * nvec, 0.0f);
        for (int j = 0; j < nvec; ++j) {
            onehot[(size_t) j * cols + j] = 1.0f;
        }
        ggml_backend_tensor_set(X2, onehot.data(), 0, onehot.size() * sizeof(float));
        if (X3) {
            std::vector<float> w0((size_t) cols * nwide, 0.0f);
            for (int j = 0; j < nwide; ++j) {
                w0[(size_t) j * cols + j] = 1.0f;
            }
            ggml_backend_tensor_set(X3, w0.data(), 0, w0.size() * sizeof(float));
        }
    }

    // Register AFTER allocation: W->data is the device pointer the decode kernels
    // look up, the same ordering llama-gqh.cpp uses at load.
    if (has_header) {
        ggml_gqh_register(W->data, ggml_nbytes(W), tensor_scale, grid_code);
    }

    // Prove which path ran. Without this the mmq cases would pass just as happily
    // on the dequant fallback they are supposed to be replacing -- the tolerances
    // below are looser than the fallback's error, so a silent decline reads as a
    // pass. This is the only thing that makes them a gate.
    const size_t mmq_before = ggml_backend_cuda_get_mmq_launch_count();
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        return 1;
    }
    const size_t mmq_launches = ggml_backend_cuda_get_mmq_launch_count() - mmq_before;

    // Which of the two wide multiplies took MMQ. The graph holds exactly two of
    // them -- D at ncols = cols and D3 at ncols = nwide -- so the count is 0, 1
    // or 2, and because the width gate is an UPPER bound and nwide <= cols, a
    // count of 1 can only be the narrower one. That lets each numeric
    // expectation below follow the path that actually ran instead of assuming
    // one, which matters now that a gate can split the two nodes apart.
    const bool wide_took_mmq  = mmq_launches >= 2;
    const bool probe_took_mmq = mmq_launches >= 1;
    if (mmq_mode && expect_refuse && mmq_launches != 0) {
        fprintf(stderr, "REFUSAL FAILED: MMQ dispatched %zu times at ncols %d / %d "
                        "on grid %d, but this case requires it to decline (either "
                        "the int8 range guard or the width gate) and leave the "
                        "multiply to the dequant path\n",
                mmq_launches, cols, nwide, grid_code);
        return 1;
    }
    if (mmq_mode && !expect_refuse && !probe_took_mmq) {
        fprintf(stderr, "MMQ did not dispatch at ncols %d: %zu launches (needs "
                        "GGML_GQH_MMQ=1, an int8-representable grid, and "
                        "GGML_GQH_MMQ_MAX_NE11 >= %d)\n",
                nwide, mmq_launches, nwide);
        return 1;
    }

    std::vector<float> got(n_elem);
    std::vector<float> got2((size_t) rows * nvec);
    ggml_backend_tensor_get(D,  got.data(),  0, got.size()  * sizeof(float));
    ggml_backend_tensor_get(D2, got2.data(), 0, got2.size() * sizeof(float));

    const float * want = hostref.empty() ? (const float *) ref.data() : hostref.data();

    // Single-term I8 error is bounded by the weight LUT step, which is per row:
    // the grid amax is 1.0, so the step is max|w_row|/127. Dense error scales with
    // the dot's own magnitude, so that pass normalises by ||w_row|| * ||x_col||
    // rather than max(1, |ref|), which would turn a quantisation-sized absolute
    // error on a near-zero accumulator into a meaningless "relative" number.
    std::vector<float>  row_max(rows, 0.0f);
    std::vector<double> row_nrm(rows, 0.0);
    for (int i = 0; i < rows; ++i) {
        double sq = 0.0;
        float  mx = 0.0f;
        for (int k = 0; k < cols; ++k) {
            const float w = want[(size_t) i * cols + k];
            mx = std::max(mx, std::fabs(w));
            sq += (double) w * (double) w;
        }
        row_max[i] = mx;
        row_nrm[i] = std::sqrt(sq);
    }
    // Single-term tolerance, and the dense relative bar. Both are inert at I8DOT=0
    // and with no mmq flag.
    //
    // Why row_max/127 is a real bound and not a shrug: every gqh3/gqh4/gqh2_h grid
    // has amax exactly 1.0, and the ratio quantiser gives the row's largest
    // sub-block r = 15, so row_max IS the superblock scale d_real. A weight step is
    // then d_real/127 and the rounding error at most half of it -- this bar carries
    // 2x margin, while a mis-mapped tile is off by a whole weight (127 steps) or
    // returns zero. It is a tight bar for the failure mode it guards.
    const double I8_DENSE_REL = 2e-2;
    // MMQ's dense bar is its own and much tighter than the I8 activation arm's.
    // Measured 5.3e-4 to 6.9e-4 across gqh3 1x256/4x512/64x1024 at n17/n33/n64,
    // so 3e-3 is roughly 4x margin -- deliberately close, because a wide dense
    // pass is the only thing here that sees accumulation order.
    const double MMQ_DENSE_REL = 3e-3;
    auto int8_step = [&](int i) {
        return std::max((double) row_max[i] / 127.0, 1e-6);
    };
    auto onehot_tol = [&](int i) {
        return i8_mode ? int8_step(i) : 0.0;
    };
    // ---- the per-grid weight-LUT bound (int8 arm) ---------------------------
    // int8_step above stays LITERALLY max|w_row|/127 and must. It bounds "did a
    // tile land on the wrong weight", and reparameterising it by whatever
    // denominator is in force would make the tolerance move with the very thing it
    // measures. It stays a fixed bar that the measurement now clears by a wide
    // margin, which is what turns it into a real measurement.
    //
    // This is a SECOND, much tighter bar, and it is the one that certifies
    // optimal-s. A one-hot dot on the int8 arm has exactly ONE term and its
    // activation side is exact (amax 1 -> d = 1/127, q = 127), so what comes back
    // differs from the reference by the weight LUT error alone: d_real times
    // |q(L,N)/N - L|. Every grid's amax is exactly 1.0 and the uint4 ratio
    // quantiser puts the row's largest sub-block at r = 15, so d_real there IS
    // max|w_row| -- and the bound is row_max times the largest absolute level error
    // the grid leaves AT THE DENOMINATOR IN FORCE, which is exactly
    // ggml_gqh_q8_maxe_at(type, code, that denominator).
    //
    // WHICH denominator the bound is built at is decided in exactly one place, and
    // it is NOT "whatever the shipping default happens to be". Two numbers:
    //
    //   kern_denom  = ggml_gqh_q8_denom_eff -- the denominator the kernel IS using.
    //                 One definition, in ggml/src/gqh.cpp, read by the int8 arms and
    //                 by this harness, so it already accounts for GGML_GQH_Q8N and
    //                 for the shipping default (currently the flat 127).
    //   bound_denom = the same thing UNLESS GQH_TEST_BOUND_N says otherwise.
    //
    // With GQH_TEST_BOUND_N unset the two agree and the bound is a real statement
    // about the quantiser that ran: an ordinary i8 case is self-consistent and
    // passes with the default set either way. With GQH_TEST_BOUND_N set they can be
    // made to disagree ON PURPOSE, and a bound tighter than the kernel can meet is
    // the negative control -- see the negctl cases in server/CMakeLists.txt. A gate
    // that only ever passes proves nothing about whether it can see the thing it
    // claims to measure, so that control is not optional.
    //
    // This bar is also the only thing here that proves the int8 arm DISPATCHED at
    // all: if it silently declined, the f32 arm would answer bit-exactly, err would
    // be 0, and every bound in this file would pass.
    const int q8_denom = i8_mode ? ggml_gqh_q8_denom_eff(wtype, grid_code) : 0;
    const int bound_req = bound_denom_request();
    const int bound_denom = !i8_mode ? 0
                          : bound_req <  0 ? q8_denom
                          : bound_req == 0 ? ggml_gqh_q8_denom(wtype, grid_code)
                          :                  bound_req;
    const double q8_maxe = i8_mode
        ? (double) ggml_gqh_q8_maxe_at(wtype, grid_code, bound_denom) : 0.0;
    const bool bound_mismatch = i8_mode && bound_denom != q8_denom;
    // Slack over the ideal, and it is not a shrug: it covers the ratio quantiser
    // rounding a sub-block scale ABOVE its own amax (d_real can then exceed
    // row_max), plus the fp32 rounding of the d_real * ratio * qxs chain. MEASURED
    // at 1.000 or below on all nine vector shapes, and attained EXACTLY (1.000) on
    // five of them, so this is a cushion for fp32 rounding rather than a fudge
    // factor -- the bound is the real quantity. The "slack used" number printed in
    // every OK line IS that measurement, so a drift shows up without anyone
    // re-deriving anything, and the GQH_TEST_BOUND_N negative control fails at 1.18
    // on the narrowest case.
    const double GRID_MAXE_SLACK = 1.05;
    auto grid_step = [&](int i) {
        return std::max((double) row_max[i] * q8_maxe * GRID_MAXE_SLACK, 1e-7);
    };
    const bool grid_bound_live = i8_mode && bound_denom > 0 && q8_maxe > 0.0;
    size_t bad_grid       = 0;
    double max_grid_slack = 0.0;   // observed |e| / (row_max * maxe): the slack USED
    auto grid_check = [&](int i, int j, double gv, double wv,
                          const char * where, int off) {
        if (!grid_bound_live) {
            return;
        }
        const double err   = std::fabs(gv - wv);
        const double ideal = (double) row_max[i] * q8_maxe;
        if (ideal > 0.0 && err / ideal > max_grid_slack) {
            max_grid_slack = err / ideal;
        }
        if (err > grid_step(i)) {
            if (bad_grid < 5) {
                fprintf(stderr, "  %s off=%d [%d,%d] OVER THE PER-GRID BOUND: got %a "
                                "want %a (err %.4g > %.4g = row_max %.4g * maxe %.4g "
                                "* slack %.2f; grid %d, bound N=%d, kernel N=%d%s)\n",
                        where, off, i, j, gv, wv, err, grid_step(i),
                        (double) row_max[i], q8_maxe, GRID_MAXE_SLACK,
                        grid_code, bound_denom, q8_denom,
                        bound_mismatch ? " -- DELIBERATE MISMATCH" : "");
            }
            ++bad_grid;
        }
    };
    // Path 1 at ncols = cols. With GGML_GQH_MMQ=1 this node IS the MMQ path -- the
    // dequant->cuBLAS route it used to take is exactly what MMQ replaces -- so the
    // expectation changes with it: no longer fp16(decode(W)) bitwise, but the
    // int8-MMQ product within one weight step. The identity columns keep it a
    // single-term dot, so nothing else enters the error.
    //
    // This is also the widest one-hot sweep in the harness: EVERY position 0..cols-1
    // carries the nonzero in some column, so a tile that mis-maps any k is caught
    // here at ncols up to 1024, not just past 15.
    size_t bad = 0;
    size_t ties = 0;
    size_t collapsed = 0;
    double max_onehot_abs = 0.0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const float ref = want[(size_t) i * cols + j];
            const float g = got[(size_t) j * rows + i];
            if (wide_took_mmq) {
                const double err = std::fabs((double) g - (double) ref);
                if (err > max_onehot_abs) { max_onehot_abs = err; }
                // A NON-ZERO weight must not come back as EXACTLY zero. This is
                // the one failure the absolute bound above cannot see, and it is
                // the failure the int8 range guard exists to prevent: on a grid
                // past ~127:1 the innermost levels round to zero, and because
                // those levels are SMALL the error stays well inside row_max/127.
                // Measured on gqh4_64x1024: grid 11 zeroes 17.1% of all weights
                // and still passes the bound. On an accepted grid this cannot
                // fire -- a grid within 127:1 quantises every level to |q| >= 1 --
                // so there are no false positives by construction.
                if (ref != 0.0f && g == 0.0f) {
                    if (collapsed < 5) {
                        fprintf(stderr, "  COLLAPSED TO ZERO at [%d,%d]: want %a, "
                                        "got exactly 0 -- the grid's innermost level "
                                        "does not survive int8\n", i, j, ref);
                    }
                    ++collapsed;
                }
                if (err > int8_step(i)) {
                    if (bad < 5) {
                        fprintf(stderr, "  mmq mismatch at [%d,%d]: got %a want %a "
                                        "(err %.4g > step %.4g)\n",
                                i, j, g, ref, err, int8_step(i));
                    }
                    ++bad;
                }
                continue;
            }
            const float w = ggml_fp16_to_fp32(ggml_fp32_to_fp16(ref));
            if (memcmp(&g, &w, 4) != 0) {
                if (fp16_tie_equivalent(ref, g)) {
                    ++ties;
                    continue;
                }
                if (bad < 5) {
                    fprintf(stderr, "  mismatch at [%d,%d]: got %a want %a\n", i, j, g, w);
                }
                ++bad;
            }
        }
    }


    // ---- Wide-lane coverage -------------------------------------------------
    // The one-hot pass above puts every nonzero activation at position j < nvec,
    // so with cols >= 256 it NEVER drives an activation past position 15. A wide
    // lane map that mis-indexes the reduction beyond 15 -- exactly what the wide
    // verify arms introduce -- is invisible to it. Two further passes close that,
    // and neither weakens the bitwise claim above:
    //
    //   A. offset-swept one-hot. Column j is e_(off+j), so the dot product still
    //      has a SINGLE term and stays bit-exact, but the nonzero lands anywhere
    //      in the reduction. A kernel that drops or mis-maps k >= 16 returns 0.
    //   B. dense activations. Every k is nonzero, so term count and accumulation
    //      order are exercised -- things no single-term probe can see. Summation
    //      makes this one tolerance-based rather than bitwise.
    size_t bad_off = 0;
    size_t bad_dense = 0;
    double max_rel = 0.0;
    int n_off = 0;
    if (has_fused) {
        std::vector<int> offs;
        for (int o : { nvec, 16, cols / 2, cols - nvec }) {
            if (o > 0 && o + nvec <= cols &&
                std::find(offs.begin(), offs.end(), o) == offs.end()) {
                offs.push_back(o);
            }
        }
        n_off = (int) offs.size();
        std::vector<float> xoff((size_t) cols * nvec);
        std::vector<float> g((size_t) rows * nvec);
        for (int off : offs) {
            std::fill(xoff.begin(), xoff.end(), 0.0f);
            for (int j = 0; j < nvec; ++j) {
                xoff[(size_t) j * cols + off + j] = 1.0f;
            }
            ggml_backend_tensor_set(X2, xoff.data(), 0, xoff.size() * sizeof(float));
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "graph compute failed at activation offset %d\n", off);
                return 1;
            }
            ggml_backend_tensor_get(D2, g.data(), 0, g.size() * sizeof(float));
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < nvec; ++j) {
                    float gv = g[(size_t) j * rows + i];
                    float wv = want[(size_t) i * cols + off + j];
                    if (gv == 0.0f) { gv = 0.0f; }   // normalise -0.0, as above
                    if (wv == 0.0f) { wv = 0.0f; }
                    if (i8_mode) {
                        grid_check(i, j, (double) gv, (double) wv, "offset-onehot", off);
                    }
                    const bool differs = i8_mode
                        ? (std::fabs((double) gv - (double) wv) > onehot_tol(i))
                        : (memcmp(&gv, &wv, 4) != 0);
                    if (differs) {
                        if (bad_off < 5) {
                            fprintf(stderr, "  offset-onehot mismatch off=%d [%d,%d]: "
                                            "got %a want %a\n", off, i, j, gv, wv);
                        }
                        ++bad_off;
                    }
                }
            }
        }

        std::vector<float> xd((size_t) cols * nvec);
        uint32_t rs = 0x9e3779b9u;                     // deterministic, no <random>
        for (size_t t = 0; t < xd.size(); ++t) {
            rs = rs * 1664525u + 1013904223u;
            xd[t] = (float) ((int32_t) ((rs >> 8) % 2001) - 1000) / 1024.0f;
        }
        ggml_backend_tensor_set(X2, xd.data(), 0, xd.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed on dense activations\n");
            return 1;
        }
        ggml_backend_tensor_get(D2, g.data(), 0, g.size() * sizeof(float));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < nvec; ++j) {
                double acc = 0.0;
                for (int k = 0; k < cols; ++k) {
                    acc += (double) want[(size_t) i * cols + k] *
                           (double) xd[(size_t) j * cols + k];
                }
                const double gv  = (double) g[(size_t) j * rows + i];
                double den;
                if (i8_mode) {
                    double xsq = 0.0;
                    for (int k = 0; k < cols; ++k) {
                        const double xv = xd[(size_t) j * cols + k];
                        xsq += xv * xv;
                    }
                    den = std::max(1e-12, row_nrm[i] * std::sqrt(xsq));
                } else {
                    den = std::max(1.0, std::fabs(acc));
                }
                const double rel = std::fabs(gv - acc) / den;
                if (rel > max_rel) { max_rel = rel; }
                if (rel > (i8_mode ? I8_DENSE_REL : 1e-4)) {
                    if (bad_dense < 5) {
                        fprintf(stderr, "  dense mismatch [%d,%d]: got %.9g want %.9g "
                                        "rel %.3g\n", i, j, gv, acc, rel);
                    }
                    ++bad_dense;
                }
            }
        }
    }

    // ---- MMQ at a width that is not a tile multiple ------------------------
    // D above sweeps every one-hot position, but only at ncols = cols, and
    // 256/512/1024 are all multiples of the mmq_x granularity, so the partial-tile
    // write-back (the j_max bound) never executes there. nwide of 17 or 33 does
    // execute it. The dense pass is here too: one-hot dots have a single term, so
    // nothing before this point exercises accumulation ORDER at a wide batch.
    size_t bad_mmq_off   = 0;
    size_t bad_mmq_dense = 0;
    double mmq_max_rel   = 0.0;
    double mmq_max_abs   = 0.0;
    int    n_mmq_off     = 0;
    if (mmq_mode) {
        std::vector<int> offs;
        for (int o : { 0, nwide, 16, cols / 2, cols - nwide }) {
            if (o >= 0 && o + nwide <= cols &&
                std::find(offs.begin(), offs.end(), o) == offs.end()) {
                offs.push_back(o);
            }
        }
        n_mmq_off = (int) offs.size();
        std::vector<float> x3((size_t) cols * nwide);
        std::vector<float> g3((size_t) rows * nwide);
        for (int off : offs) {
            std::fill(x3.begin(), x3.end(), 0.0f);
            for (int j = 0; j < nwide; ++j) {
                x3[(size_t) j * cols + off + j] = 1.0f;
            }
            ggml_backend_tensor_set(X3, x3.data(), 0, x3.size() * sizeof(float));
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "graph compute failed at mmq offset %d\n", off);
                return 1;
            }
            ggml_backend_tensor_get(D3, g3.data(), 0, g3.size() * sizeof(float));
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < nwide; ++j) {
                    const float  gf  = g3[(size_t) j * rows + i];
                    const float  wf  = want[(size_t) i * cols + off + j];
                    const double gv  = (double) gf;
                    const double wv  = (double) wf;
                    const double err = std::fabs(gv - wv);
                    if (err > mmq_max_abs) { mmq_max_abs = err; }
                    bool differs;
                    if (probe_took_mmq) {
                        if (wf != 0.0f && gf == 0.0f) {
                            if (collapsed < 5) {
                                fprintf(stderr, "  COLLAPSED TO ZERO off=%d [%d,%d]: "
                                                "want %a, got exactly 0\n", off, i, j, wf);
                            }
                            ++collapsed;
                        }
                        differs = err > int8_step(i);
                    } else {
                        // The guard refused, so this came off the dequant path and
                        // must be fp16-exact. An int8-perturbed answer here means the
                        // guard let the grid through.
                        const float w16 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(wf));
                        differs = memcmp(&gf, &w16, 4) != 0 && !fp16_tie_equivalent(wf, gf);
                    }
                    if (differs) {
                        if (bad_mmq_off < 5) {
                            fprintf(stderr, "  mmq offset-onehot mismatch off=%d [%d,%d]: "
                                            "got %.9g want %.9g (err %.4g, step %.4g, %s)\n",
                                    off, i, j, gv, wv, err, int8_step(i),
                                    probe_took_mmq ? "int8 bound" : "fp16 bitwise");
                        }
                        ++bad_mmq_off;
                    }
                }
            }
        }

        uint32_t rs3 = 0x85ebca6bu;                    // deterministic, no <random>
        for (size_t t = 0; t < x3.size(); ++t) {
            rs3 = rs3 * 1664525u + 1013904223u;
            x3[t] = (float) ((int32_t) ((rs3 >> 8) % 2001) - 1000) / 1024.0f;
        }
        ggml_backend_tensor_set(X3, x3.data(), 0, x3.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed on mmq dense activations\n");
            return 1;
        }
        ggml_backend_tensor_get(D3, g3.data(), 0, g3.size() * sizeof(float));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < nwide; ++j) {
                double acc = 0.0;
                double xsq = 0.0;
                for (int k = 0; k < cols; ++k) {
                    const double xv = x3[(size_t) j * cols + k];
                    acc += (double) want[(size_t) i * cols + k] * xv;
                    xsq += xv * xv;
                }
                const double gv  = (double) g3[(size_t) j * rows + i];
                // Normalise by the dot's own scale. max(1, |acc|) would turn a
                // quantisation-sized error on a near-zero accumulator into a
                // meaningless ratio; both operands carry one int8 round.
                const double den = std::max(1e-12, row_nrm[i] * std::sqrt(xsq));
                const double rel = std::fabs(gv - acc) / den;
                if (rel > mmq_max_rel) { mmq_max_rel = rel; }
                if (rel > MMQ_DENSE_REL) {
                    if (bad_mmq_dense < 5) {
                        fprintf(stderr, "  mmq dense mismatch [%d,%d]: got %.9g want %.9g "
                                        "rel %.3g\n", i, j, gv, acc, rel);
                    }
                    ++bad_mmq_dense;
                }
            }
        }
    }

    if (has_header) {
        ggml_gqh_unregister(W->data);
    }
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_free(backend);

    // Summation cannot preserve the sign of zero -- a -0.0 weight comes back as
    // +0.0 once it is added into an accumulator -- so normalise it on both sides.
    auto norm = [](float v) { return v == 0.0f ? 0.0f : v; };

    size_t bad2 = 0;
    if (has_fused) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < nvec; ++j) {
                const float g = norm(got2[(size_t) j * rows + i]);
                const float w = norm(want[(size_t) i * cols + j]);
                if (i8_mode) {
                    grid_check(i, j, (double) g, (double) w, "fused-onehot", 0);
                }
                const bool differs = i8_mode
                    ? (std::fabs((double) g - (double) w) > onehot_tol(i))
                    : (memcmp(&g, &w, 4) != 0);
                if (differs) {
                    if (bad2 < 5) {
                        fprintf(stderr, "  fused mismatch at [%d,%d]: got %a want %a\n", i, j, g, w);
                    }
                    ++bad2;
                }
            }
        }
    }

    if (bad || bad2 || bad_off || bad_dense || bad_mmq_off || bad_mmq_dense ||
            collapsed || bad_grid) {
        printf("FAIL %s %dx%d [%s]: wide %zu/%zu differ, fused %zu/%d differ, "
               "offset-onehot %zu differ over %d offsets, dense %zu differ "
               "(max rel %.3g), %zu over the per-grid weight-LUT bound "
               "(grid %d, bound N=%d, kernel N=%d%s, maxe %.4g, "
               "worst level error %.4g, slack used %.3f vs %.2f allowed)",
               rung.c_str(), rows, cols, i8_mode ? "i8" : "f32",
               bad, n_elem, bad2, rows * nvec,
               bad_off, n_off, bad_dense, max_rel,
               bad_grid, grid_code, bound_denom, q8_denom,
               bound_mismatch ? " DELIBERATE MISMATCH" : "", q8_maxe,
               max_grid_slack * q8_maxe, max_grid_slack, GRID_MAXE_SLACK);
        if (mmq_mode) {
            printf(", mmq n%d: offset-onehot %zu differ over %d offsets "
                   "(max abs %.4g), dense %zu differ (max rel %.3g), "
                   "%zu non-zero weights COLLAPSED to zero",
                   nwide, bad_mmq_off, n_mmq_off, mmq_max_abs,
                   bad_mmq_dense, mmq_max_rel, collapsed);
        }
        printf("\n");
        return 1;
    }
    if (mmq_mode) {
        if (expect_refuse) {
            printf("OK   %s %dx%d [mmq-refused]: MMQ DECLINED (0 launches) at ncols %d "
                   "and %d on grid %d, and the dequant path answered fp16-exact over "
                   "%d offsets%s (scale %.9g)\n",
                   rung.c_str(), rows, cols, cols, nwide, grid_code, n_mmq_off,
                   ties ? (", " + std::to_string(ties) + " exact fp16 ties").c_str() : "",
                   tensor_scale);
            return 0;
        }
        printf("OK   %s %dx%d [mmq]: MMQ dispatched (%zu launches, wide node on %s), "
               "wide n%d one-hot "
               "within one weight step (max abs %.4g), n%d one-hot over %d offsets "
               "(max abs %.4g), n%d dense max rel %.3g, no level collapsed to "
               "zero, fused matvec n%d bit-identical (scale %.9g, grid %d%s)\n",
               rung.c_str(), rows, cols, mmq_launches,
               wide_took_mmq ? "MMQ" : "dequant (width-gated)", cols, max_onehot_abs,
               nwide, n_mmq_off, mmq_max_abs, nwide, mmq_max_rel, nvec,
               tensor_scale, grid_code, grid_force >= 0 ? ", forced" : "");
        return 0;
    }
    printf("OK   %s %dx%d [%s]: dequant->BLAS matches the fp16-rounded reference%s, "
           "fused matvec %s the f32 reference over %d cols at %d "
           "activation offsets, dense max rel %.3g (scale %.9g, grid %d)",
           rung.c_str(), rows, cols, i8_mode ? "i8" : "f32",
           ties ? (" apart from " + std::to_string(ties) + " exact fp16 ties").c_str() : "",
           i8_mode ? "within the int8 bound of" : "bit-identical to",
           nvec, n_off + 1, max_rel, tensor_scale, grid_code);
    if (grid_bound_live) {
        // slack used is measured AGAINST the bound, so it moves with the
        // denominator; max|e|/row_max is the level error itself and is the number
        // to compare across denominators. maxe is constant within a run, so the
        // argmax is the same item and the product is exact.
        printf(", per-grid weight-LUT bound held: bound N=%d, kernel N=%d%s, "
               "grid maxe %.4g, worst level error %.4g, slack used %.3f of %.2f",
               bound_denom, q8_denom,
               bound_mismatch ? " (MISMATCHED ON PURPOSE)" : "", q8_maxe,
               max_grid_slack * q8_maxe, max_grid_slack, GRID_MAXE_SLACK);
    }
    printf("\n");
    return 0;
}
