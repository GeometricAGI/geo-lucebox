#include "gqh.h"
#include "ggml-impl.h"

#include <cmath>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace {
struct gqh_entry {
    const void * base;
    size_t       nbytes;
    float        tensor_scale;
    int          grid_code;
    std::vector<float> input_scale;
    const void * backend_input_scale = nullptr;
};
std::mutex                  g_gqh_mtx;
std::vector<gqh_entry>      g_gqh_registry;
}  // namespace

void ggml_gqh_register(const void * base, size_t nbytes, float tensor_scale, int grid_code) {
    std::lock_guard<std::mutex> lk(g_gqh_mtx);
    const gqh_entry ne{base, nbytes, tensor_scale, grid_code};
    for (auto & e : g_gqh_registry) {
        if (e.base == base) { e = ne; return; }
    }
    g_gqh_registry.push_back(ne);
}

void ggml_gqh_unregister(const void * base) {
    std::lock_guard<std::mutex> lk(g_gqh_mtx);
    for (size_t i = 0; i < g_gqh_registry.size(); ++i) {
        if (g_gqh_registry[i].base == base) {
            g_gqh_registry.erase(g_gqh_registry.begin() + i);
            return;
        }
    }
}

bool ggml_gqh_lookup(const void * p, float * tensor_scale, int * grid_code) {
    std::lock_guard<std::mutex> lk(g_gqh_mtx);
    for (const auto & e : g_gqh_registry) {
        const uint8_t * b = (const uint8_t *) e.base;
        if ((const uint8_t *) p >= b && (const uint8_t *) p < b + e.nbytes) {
            *tensor_scale = e.tensor_scale;
            *grid_code    = e.grid_code;
            return true;
        }
    }
    return false;
}

static bool gqh_register_input_scale(const ggml_tensor * w, const float * host,
                                         int64_t columns, const void * backend) {
    if (!w || !host || columns <= 0 || columns != w->ne[0] || columns % 256 ||
        w->ne[2] != 1 || w->ne[3] != 1 || !ggml_is_contiguous(w)) return false;
    for (int64_t i = 0; i < columns; ++i) {
        if (!std::isfinite(host[i]) || host[i] <= 0) return false;
    }
    std::lock_guard<std::mutex> lk(g_gqh_mtx);
    for (auto & e : g_gqh_registry) {
        if (e.base == w->data && e.nbytes == ggml_nbytes(w)) {
            e.input_scale.assign(host, host + columns);
            e.backend_input_scale = backend;
            return true;
        }
    }
    return false;
}

bool ggml_gqh_register_ternary_input_scale(const ggml_tensor * w,const float * host,int64_t columns,const void * backend) {
    return w && (w->type==GGML_TYPE_GQH_T || w->type==GGML_TYPE_GQH_T_G32_R4 || w->type==GGML_TYPE_GQH_T_G32_R3)
        && gqh_register_input_scale(w,host,columns,backend);
}
bool ggml_gqh_register_research_input_scale(const ggml_tensor * w,const float * host,int64_t columns,const void * backend) {
    return w && (w->type==GGML_TYPE_GQH2_H || w->type==GGML_TYPE_GQH3 || w->type==GGML_TYPE_GQH4)
        && gqh_register_input_scale(w,host,columns,backend);
}

bool ggml_gqh_lookup_input_scale(const void * p, const float ** host, const void ** backend,
                                 int64_t * columns, size_t * offset) {
    std::lock_guard<std::mutex> lk(g_gqh_mtx);
    const uintptr_t address = reinterpret_cast<uintptr_t>(p);
    for (const auto & e : g_gqh_registry) {
        const uintptr_t base = reinterpret_cast<uintptr_t>(e.base);
        if (address >= base && address - base < e.nbytes && !e.input_scale.empty()) {
            *host = e.input_scale.data(); *backend = e.backend_input_scale;
            *columns = e.input_scale.size(); *offset = address - base;
            return true;
        }
    }
    return false;
}

// The tables hold raw float32 bit patterns so no compiler can re-round them.
static inline float gqh_f32(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// ---- optimal-s for the int8 weight LUT ------------------------------------
//
// The int8 arms bake a rung's level grid to int8 once per dispatch. The obvious
// denominator is 127 -- every grid's amax is exactly 1.0, so 127 lands on the
// extreme level with no clamping -- but 127 is only the WIDEST choice, not the
// most accurate one. The levels in between land wherever the k/127 lattice
// happens to fall, and for a curved grid that is a poor fit: gqh3 code 10 sits
// so badly against it that its rms level error is 12.7x what a better
// denominator gives, on the very same 8 levels.
//
// So pick N per grid. The reconstruction is q/N against a chain that divides by
// 127, and the caller folds the compensating 127/N into its PER-TENSOR WEIGHT
// SCALE -- so a different N costs nothing at all at runtime. It is one host-side
// multiply per dispatch and not one extra instruction in any kernel.
//
// THE OBJECTIVE IS OCCUPANCY-WEIGHTED RMS ABSOLUTE ERROR:
//
//     eps(N) = sqrt( sum_c p_c * (q_c/N - level_c)^2 ),   q_c = clamp(rn(level_c*N))
//
// not worst-case relative error. A dot product accumulates sum_k w_k x_k, so what
// reaches the output is the ABSOLUTE perturbation of each weight; a level's
// relative error never appears anywhere in that sum. Minimising relative error
// instead chases the innermost level of a curved grid, which is the level that
// contributes least to any dot, and an earlier pass that did exactly that
// reported a 20.7x "gain" that does not exist.
//
// p_c is UNIFORM, not measured occupancy. Occupancy-optimal N was only about 3%
// better on the artifact it was measured against, and this table is shared by
// every GQH tensor of every model -- baking one histogram into it buys a rounding
// error and costs the property that makes the table trustworthy.
//
// Search range 1..127 because N > 127 would clamp the extreme level, which both
// costs accuracy and breaks the invariant the device-side LUT check relies on
// (amax == 1.0 exactly, so q(1.0, N) == N exactly).
//
// Derived values for the grids the shipping artifact uses -- gqh3 {3,4} and
// gqh4 {2,3,4} -- with the rms gain over N=127 on that grid's levels:
//
//   gqh3 code 3   N=120   6.483e-04 vs 2.565e-03   3.96x
//   gqh3 code 4   N=120   1.037e-03 vs 2.401e-03   2.32x
//   gqh4 code 2   N=114   1.526e-03 vs 2.062e-03   1.35x
//   gqh4 code 3   N=122   1.594e-03 vs 2.378e-03   1.49x
//   gqh4 code 4   N=110   1.587e-03 vs 2.662e-03   1.68x
//
// Ties (a dyadic grid is exact at several N) resolve to the LARGEST N, which
// keeps the compensating factor closest to 1 and the int8 codes closest to the
// range the arms were tuned on.
//
// The device recomputes rn(level*N) from its own LDS copy of the same levels
// rather than reading a baked table, so this function and the kernel must agree.
// They do, and not by luck: a tie (level*N exactly k+0.5) is EXACT in float32 for
// every |level*N| <= 127, so the host's float multiply and the device's
// v_mul_f32 cannot disagree about which side of a rounding boundary a level sits
// on. The kernel still checks the two extreme LUT entries against N and traps on
// a mismatch, so a future divergence fails loudly instead of quietly rescaling
// every weight in the tensor.
namespace {
struct gqh_q8_fit {
    int   n    = 127;
    float maxe = 0.0f;
    float rms  = 0.0f;
};

int gqh_q8_round(float level, int n) {
    // Deliberately a float32 product and lrintf's round-to-nearest-even, to match
    // the device's __float2int_rn(level * (float) n).
    const float p = level * (float) n;
    long  q = lrintf(p);
    if (q >  127) { q =  127; }
    if (q < -127) { q = -127; }
    return (int) q;
}

gqh_q8_fit gqh_q8_fit_grid(const uint32_t * grid, int nlev) {
    gqh_q8_fit best;
    double best_rms = -1.0;
    for (int n = 1; n <= 127; ++n) {
        double se = 0.0;
        double mx = 0.0;
        for (int i = 0; i < nlev; ++i) {
            const float  lv = gqh_f32(grid[i]);
            const double e  = (double) gqh_q8_round(lv, n) / (double) n - (double) lv;
            se += e * e;
            if (fabs(e) > mx) { mx = fabs(e); }
        }
        const double rms = sqrt(se / (double) nlev);
        // <=, so a tie takes the larger N (the loop ascends).
        if (best_rms < 0.0 || rms <= best_rms) {
            best_rms  = rms;
            best.n    = n;
            best.maxe = (float) mx;
            best.rms  = (float) rms;
        }
    }
    return best;
}

// The level grid itself, so an error can be evaluated at ANY denominator and not
// only at the derived optimum. Needed by ggml_gqh_q8_maxe_at, which is what lets a
// harness state the bound for the denominator actually in force.
bool gqh_grid_levels(enum ggml_type type, int grid_code,
                     const uint32_t ** grid, int * nlev) {
    if (grid_code < 0 || grid_code >= GQH_GRID_CODES) {
        return false;
    }
    switch (type) {
        case GGML_TYPE_GQH3:   *grid = GQH3_GRID [grid_code]; *nlev =  8; return true;
        case GGML_TYPE_GQH4:   *grid = GQH4_GRID [grid_code]; *nlev = 16; return true;
        case GGML_TYPE_GQH2_H: *grid = GQH2H_GRID[grid_code]; *nlev =  4; return true;
        default:               return false;
    }
}

const gqh_q8_fit * gqh_q8_fit_for(enum ggml_type type, int grid_code) {
    struct table {
        gqh_q8_fit gqh3[GQH_GRID_CODES];
        gqh_q8_fit gqh4[GQH_GRID_CODES];
        gqh_q8_fit gqh2h[GQH_GRID_CODES];
        table() {
            for (int c = 0; c < GQH_GRID_CODES; ++c) {
                gqh3 [c] = gqh_q8_fit_grid(GQH3_GRID [c],  8);
                gqh4 [c] = gqh_q8_fit_grid(GQH4_GRID [c], 16);
                gqh2h[c] = gqh_q8_fit_grid(GQH2H_GRID[c],  4);
            }
        }
    };
    static const table t;   // built once, thread-safe (C++11 magic static)
    if (grid_code < 0 || grid_code >= GQH_GRID_CODES) {
        return nullptr;
    }
    switch (type) {
        case GGML_TYPE_GQH3:   return &t.gqh3 [grid_code];
        case GGML_TYPE_GQH4:   return &t.gqh4 [grid_code];
        case GGML_TYPE_GQH2_H: return &t.gqh2h[grid_code];
        default:               return nullptr;
    }
}
}  // namespace

int ggml_gqh_q8_denom(enum ggml_type type, int grid_code) {
    const gqh_q8_fit * f = gqh_q8_fit_for(type, grid_code);
    return f ? f->n : 0;
}

float ggml_gqh_q8_maxe(enum ggml_type type, int grid_code) {
    const gqh_q8_fit * f = gqh_q8_fit_for(type, grid_code);
    return f ? f->maxe : 0.0f;
}

float ggml_gqh_q8_rms(enum ggml_type type, int grid_code) {
    const gqh_q8_fit * f = gqh_q8_fit_for(type, grid_code);
    return f ? f->rms : 0.0f;
}

// maxe at an ARBITRARY denominator, not only at the derived optimum. Same
// arithmetic as the search above -- gqh_q8_round is the float32 / round-to-nearest
// chain the device runs -- just evaluated at the N it is handed instead of the N
// that minimises rms. ggml_gqh_q8_maxe(type, code) is exactly this at
// ggml_gqh_q8_denom(type, code).
//
// This exists so a bound can be stated for the denominator ACTUALLY IN FORCE.
// A bound pinned to the derived optimum while the kernel runs some other N is not
// a bound at all, it is a comparison of two different quantisers -- useful as a
// deliberate negative control and wrong as an ordinary assertion.
float ggml_gqh_q8_maxe_at(enum ggml_type type, int grid_code, int n) {
    const uint32_t * grid = NULL;
    int              nlev = 0;
    if (n < 1 || n > 127 || !gqh_grid_levels(type, grid_code, &grid, &nlev)) {
        return 0.0f;
    }
    double mx = 0.0;
    for (int i = 0; i < nlev; ++i) {
        const float  lv = gqh_f32(grid[i]);
        const double e  = (double) gqh_q8_round(lv, n) / (double) n - (double) lv;
        if (fabs(e) > mx) { mx = fabs(e); }
    }
    return (float) mx;
}

// ---- which denominator is ACTUALLY IN FORCE --------------------------------
//
// THE DEFAULT IS THE FLAT 127, i.e. per-grid optimal-s is OPT-IN.
//
// The per-grid denominators above are a real fidelity gain (weight rms 1.35x-3.96x
// better across the grids this artifact uses) at zero runtime cost, but they were
// gated on HumanEval+ and did not earn the default. Four readings, order-balanced
// ABBA, canonical drafter, R9700, greedy and byte-identical within each arm:
//
//   derived N   144/164, 144/164     (145/145 once the scorer bug below is fixed)
//   flat 127    145/164, 145/164     (146/146)
//
// THE 1-ITEM GAP IS INSIDE THE FLOOR, and an earlier version of this comment said
// otherwise. Two instrument faults were found after the gate ran:
//
//  1. The harness only extracted a fenced code block when the fence was CLOSED
//     (quality_humaneval_plus.py). An unterminated fence leaked the literal
//     ```python line into the graded script, failing it on SyntaxError. 12 items
//     per reading take that path, and it costs ~1 item systematically -- both arms
//     equally, so the comparison held but the totals were low.
//  2. The GRADER is not deterministic. Re-grading BYTE-IDENTICAL replies flips a
//     verdict: HumanEval/39 (prime_fib) alternates pass/fail across grading passes,
//     an execution-timeout effect. So the "floor 0 items" measured here is REPLY
//     determinism, not VERDICT determinism, and the real floor is +-1 item.
//
// With a +-1 grader floor, a 1-item difference is not a result in either direction.
// The arms are also not statistically separable (discordant on 3 items, 2 to 1
// after the scorer fix; McNemar exact two-sided p ~ 1.0). So this is NOT evidence
// of harm.
//
// What survives, and what the default rests on: the measured throughput gain was
// NIL (193.76/191.21 tok/s derived against 194.33/195.13 flat, inside the arm's own
// 1.33% spread and under the rig's ~1.3% floor). A change that alters generated
// tokens -- 38 of 164 replies differ -- for no measurable benefit does not get to
// be the default. Not "it measured worse"; "it measured no better, and it is not
// free of consequence".
//
// Resolving a 1-item effect here would need ~3700 items at the observed discordance
// rate, and that assumes a deterministic scorer, which this is not.
//
// GGML_GQH_Q8N selects:
//   unset                       -> the flat 127 above
//   1..127                      -> that flat denominator
//   0, "opt", "derived", "auto" -> the per-grid derived optimum (opt in)
//   anything else               -> the per-grid derived optimum
//
// THIS FUNCTION IS THE ONLY DEFINITION OF THAT POLICY. It lives here, in the
// library both the kernel and the test harness link, rather than in the CUDA
// translation unit, precisely so a harness can assert against the denominator in
// force instead of hardcoding a copy of the default and drifting from it. The
// previous arrangement kept the default private to gqh.cu and left
// test-gqh-backend pinned to the derived table, which meant flipping the default
// broke 35 ordinary cases that had nothing wrong with them.
//
// Reconsider the default if a workload appears where the extra weight precision
// shows up in an output metric.
#define GQH_Q8N_FLAT_DEFAULT 127

static int gqh_q8n_env() {
    // Read once. getenv mid-run would let two dispatches in one process disagree
    // about the denominator, and the compensating 127/N rides the weight scale.
    static const int n = []() {
        const char * e = getenv("GGML_GQH_Q8N");
        if (!e || !*e) {
            return GQH_Q8N_FLAT_DEFAULT;
        }
        if (!strcmp(e, "opt") || !strcmp(e, "derived") || !strcmp(e, "auto")) {
            return 0;   // 0 == per-grid derived
        }
        const int v = atoi(e);
        return (v >= 1 && v <= 127) ? v : 0;
    }();
    return n;
}

int ggml_gqh_q8_denom_eff(enum ggml_type type, int grid_code) {
    const int forced = gqh_q8n_env();
    return forced > 0 ? forced : ggml_gqh_q8_denom(type, grid_code);
}

static void gqh_header_or_abort(const void * vx, int64_t k, size_t sb_bytes,
                                float * scale, int * code) {
    if (!ggml_gqh_lookup(vx, scale, code)) {
        GGML_ABORT("gqh: tensor slice %p is not registered -- the per-tensor header KV "
                   "was not read at load time", vx);
    }
    if (k % GQH_SUPERBLOCK != 0) {
        GGML_ABORT("gqh: dequant of %" PRId64 " elements is not a whole number of superblocks", k);
    }
    GGML_UNUSED(sb_bytes);
}

// gqh4, gqh3 and gqh2_h share a head: d_real = e4m3(d) * tensor_scale, then
// s_b = d_real * (ratio/15) per 16-weight sub-block.
static inline float gqh_subblock_scale(const uint8_t * b, int sub, float tensor_scale) {
    const uint8_t d = b[0];
    const float d_real = gqh_f32(GQH_E4M3_LUT[d >> 3][d & 7]) * tensor_scale;
    const uint8_t rb = b[1 + (sub >> 1)];
    const int ratio = (sub & 1) ? (rb >> 4) : (rb & 0x0f);
    return d_real * gqh_f32(GQH_RATIO_Q[ratio][0]);
}

void dequantize_row_gqh3(const void * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    float scale;
    int   code;
    gqh_header_or_abort(vx, k, GQH3_SB_BYTES, &scale, &code);

    const uint8_t * b = (const uint8_t *) vx;
    for (int64_t sb = 0; sb < k / GQH_SUPERBLOCK; ++sb, b += GQH3_SB_BYTES) {
        for (int sub = 0; sub < GQH_N_SUB; ++sub) {
            const float s_b = gqh_subblock_scale(b, sub, scale);
            for (int t = 0; t < GQH_SUBBLOCK; ++t) {
                const int j  = sub * GQH_SUBBLOCK + t;
                const int lo = (b[ 9 + (j >> 2)] >> (2 * (j & 3))) & 0x03;
                const int hi = (b[73 + (j >> 3)] >> (j & 7)) & 0x01;
                y[sb * GQH_SUPERBLOCK + j] = gqh_f32(GQH3_GRID[code][lo | (hi << 2)]) * s_b;
            }
        }
    }
}

// gqh4: 137 B superblock. Same head as gqh3, then 128 bytes of uint4 codes packed
// two per byte -- even weight in the low nibble, odd in the high nibble -- into a
// 16-level grid +-(j/8)^gamma picked per tensor.
void dequantize_row_gqh4(const void * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    float scale;
    int   code;
    gqh_header_or_abort(vx, k, GQH4_SB_BYTES, &scale, &code);

    const uint8_t * b = (const uint8_t *) vx;
    for (int64_t sb = 0; sb < k / GQH_SUPERBLOCK; ++sb, b += GQH4_SB_BYTES) {
        for (int sub = 0; sub < GQH_N_SUB; ++sub) {
            const float s_b = gqh_subblock_scale(b, sub, scale);
            for (int t = 0; t < GQH_SUBBLOCK; ++t) {
                const int j  = sub * GQH_SUBBLOCK + t;
                const uint8_t cb = b[9 + (j >> 1)];
                const int c = (j & 1) ? (cb >> 4) : (cb & 0x0f);
                y[sb * GQH_SUPERBLOCK + j] = gqh_f32(GQH4_GRID[code][c]) * s_b;
            }
        }
    }
}

template<int GROUP, int BITS>
static void dequantize_row_ternary(const void * vx, float * y, int64_t k) {
    constexpr int ratios = GROUP == 16 ? 8 : BITS;
    constexpr int stride = 1 + ratios + 52;
    float scale; int code;
    gqh_header_or_abort(vx, k, stride, &scale, &code);
    GGML_ASSERT(code == 0 && std::isfinite(scale) && scale > 0);
    const float * input_scale = nullptr; const void * backend_scale = nullptr;
    int64_t columns = 0; size_t offset = 0;
    const bool compensated = ggml_gqh_lookup_input_scale(vx, &input_scale, &backend_scale, &columns, &offset);
    GGML_ASSERT(!compensated || offset % stride == 0);
    const size_t first_column = compensated ? ((offset / stride) * 256) % columns : 0;
    const uint8_t * b = static_cast<const uint8_t *>(vx);
    constexpr int powers[5] = {1, 3, 9, 27, 81};
    for (int64_t sb = 0; sb < k/256; ++sb, b += stride) {
        GGML_ASSERT((b[0] & 127) != 127);
        for (int j=0;j<52;++j) GGML_ASSERT(b[1+ratios+j] < (j == 51 ? 3 : 243));
        const int exponent=(b[0]>>3)&15, mantissa=b[0]&7;
        float d = exponent ? std::ldexp(1.0f+float(mantissa)*0.125f,exponent-7) : std::ldexp(float(mantissa),-9);
        if (b[0]&128) d=-d;
        const float real=d*scale;
        for (int j=0;j<256;++j) {
            const int bit=(j/GROUP)*BITS;
            unsigned ratio=b[1+bit/8]>>(bit%8);
            if (bit%8+BITS>8) ratio|=unsigned(b[2+bit/8])<<(8-bit%8);
            ratio&=(1<<BITS)-1;
            const float subscale=real*(float(ratio)/float((1<<BITS)-1));
            const int trit=(b[1+ratios+j/5]/powers[j%5])%3-1;
            y[sb*256+j]=float(trit)*subscale;
            if (compensated) y[sb*256+j] /= input_scale[(first_column + sb*256+j) % columns];
        }
    }
}

void dequantize_row_gqh_t(const void * x, float * y, int64_t k) { dequantize_row_ternary<16,4>(x,y,k); }
void dequantize_row_gqh_t_g32_r4(const void * x, float * y, int64_t k) { dequantize_row_ternary<32,4>(x,y,k); }
void dequantize_row_gqh_t_g32_r3(const void * x, float * y, int64_t k) { dequantize_row_ternary<32,3>(x,y,k); }

void dequantize_row_gqh2_h(const void * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    float scale;
    int   code;
    gqh_header_or_abort(vx, k, GQH2H_SB_BYTES, &scale, &code);

    const uint8_t * b = (const uint8_t *) vx;
    for (int64_t sb = 0; sb < k / GQH_SUPERBLOCK; ++sb, b += GQH2H_SB_BYTES) {
        for (int sub = 0; sub < GQH_N_SUB; ++sub) {
            const float s_b = gqh_subblock_scale(b, sub, scale);
            for (int t = 0; t < GQH_SUBBLOCK; ++t) {
                const int j = sub * GQH_SUBBLOCK + t;
                const int c = (b[9 + (j >> 2)] >> (2 * (j & 3))) & 0x03;
                y[sb * GQH_SUPERBLOCK + j] = gqh_f32(GQH2H_GRID[code][c]) * s_b;
            }
        }
    }
}

// gqh2_c: 66 B superblock = fp16 d, then 8 blocks of 32. Each block is 4 codebook
// indices plus a uint32 holding four 7-bit sign indices and a uint4 ratio.
// No registry: everything it needs is in-block or a frozen table.
void dequantize_row_gqh2_c(const void * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    if (k % GQH_SUPERBLOCK != 0) {
        GGML_ABORT("gqh2_c: dequant of %" PRId64 " elements is not a whole number of superblocks", k);
    }
    const uint8_t * b = (const uint8_t *) vx;
    for (int64_t sb = 0; sb < k / GQH_SUPERBLOCK; ++sb, b += GQH2C_SB_BYTES) {
        ggml_fp16_t dh;
        memcpy(&dh, b, sizeof(dh));
        const float d = GGML_FP16_TO_FP32(dh);

        for (int blk = 0; blk < GQH2C_BLOCKS_PER_SB; ++blk) {
            const uint8_t * p = b + 2 + blk * 8;
            uint32_t u;
            memcpy(&u, p + 4, sizeof(u));
            const float s_blk = d * gqh_f32(GQH_RATIO_Q[(u >> 28) & 0x0f][0]);

            for (int grp = 0; grp < GQH2C_GROUPS_PER_BLOCK; ++grp) {
                const uint8_t mask = GQH2C_SIGN_MASK[(u >> (7 * grp)) & 0x7f];
                const uint32_t * cb = GQH2C_CODEBOOK[p[grp]];
                float * out = y + sb * GQH_SUPERBLOCK + blk * GQH2C_BLOCK + grp * GQH2C_GROUP;
                for (int e = 0; e < GQH2C_GROUP; ++e) {
                    // reference is sign * mag * s_blk, left to right -- keep the
                    // sign on the magnitude so a zero s_blk yields -0.0 the same way
                    const float mag = ((mask >> e) & 1) ? -gqh_f32(cb[e]) : gqh_f32(cb[e]);
                    out[e] = mag * s_blk;
                }
            }
        }
    }
}

// Losslessly repacked research INT3: each group keeps its original F32 scale.
template<int GROUP> static void dequantize_dsv41_int3(const void * vx,float * y,int64_t k){
 GGML_ASSERT(k%GROUP==0);constexpr int stride=4+GROUP*3/8;const auto * x=static_cast<const uint8_t *>(vx);
 for(int64_t g=0;g<k/GROUP;++g){const auto * b=x+g*stride;float scale;memcpy(&scale,b,4);
  for(int j=0;j<GROUP;++j){int bit=j*3,shift=bit%8;unsigned code=b[4+bit/8]>>shift;if(shift+3>8)code|=unsigned(b[5+bit/8])<<(8-shift);y[g*GROUP+j]=(int(code&7)-4)*scale;}
 }
}
void dequantize_row_dsv41_int3_g64(const void * x,float * y,int64_t k){dequantize_dsv41_int3<64>(x,y,k);}
void dequantize_row_dsv41_int3_g128(const void * x,float * y,int64_t k){dequantize_dsv41_int3<128>(x,y,k);}
