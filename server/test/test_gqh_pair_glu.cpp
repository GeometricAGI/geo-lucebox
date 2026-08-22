// GQH3 5120x17408 gate/up pair: shipped ggml HIP entry points, not a reimplementation.
//
// Qwen3.8-GQH-Q3KXL's FFN is {MUL_MAT, MUL_MAT, GLU} over two GQH3 5120x17408
// weights sharing one f32 activation. ggml_cuda_gqh_mul_mat_vec_pair must accept
// that shape (ROWS=4 + 2-wave), match two unpaired fused matvecs bit-exactly, and
// the same trio must compute through ggml_swiglu_split.

#include "ds4_test_gpu_runtime.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

bool ggml_cuda_gqh_mul_mat_vec(
        ggml_type type, const void * vx, const float * x, float * y,
        int in, int out, int ncols, int64_t x_col_stride, int64_t y_col_stride,
        cudaStream_t stream);

bool ggml_cuda_gqh_mul_mat_vec_pair(
        ggml_type type,
        const void * vx_a, float * y_a, const void * vx_b, float * y_b,
        const float * x, int in, int out, int ncols,
        int64_t x_col_stride, int64_t y_col_stride, cudaStream_t stream);

#define GQH_SUPERBLOCK   256
#define GQH_HEADER_BYTES 5
#define GQH3_SB_BYTES    105

static constexpr int kIn  = 5120;
static constexpr int kOut = 17408;

static int g_fails = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_fails;                                                         \
        }                                                                      \
    } while (0)

#define HIP_OK(expr)                                                           \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "FAIL: %s -> %s\n", #expr,                    \
                         cudaGetErrorString(_e));                              \
            ++g_fails;                                                         \
        }                                                                      \
    } while (0)

static std::vector<uint8_t> read_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(1);
    }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t) n);
    if (n > 0 && std::fread(buf.data(), 1, (size_t) n, f) != (size_t) n) {
        std::fprintf(stderr, "short read on %s\n", path);
        std::exit(1);
    }
    std::fclose(f);
    return buf;
}

static std::vector<uint8_t> tile_gqh3(const uint8_t * body, int src_rows, int src_cols,
                                      int dst_rows, int dst_cols, int row_off) {
    const int src_nsb = src_cols / GQH_SUPERBLOCK;
    const int dst_nsb = dst_cols / GQH_SUPERBLOCK;
    std::vector<uint8_t> out((size_t) dst_rows * dst_nsb * GQH3_SB_BYTES);
    for (int r = 0; r < dst_rows; ++r) {
        const int sr = (r + row_off) % src_rows;
        const uint8_t * src_row = body + (size_t) sr * src_nsb * GQH3_SB_BYTES;
        uint8_t * dst_row = out.data() + (size_t) r * dst_nsb * GQH3_SB_BYTES;
        for (int s = 0; s < dst_nsb; ++s) {
            std::memcpy(dst_row + (size_t) s * GQH3_SB_BYTES,
                        src_row + (size_t) (s % src_nsb) * GQH3_SB_BYTES,
                        GQH3_SB_BYTES);
        }
    }
    return out;
}

static bool bits_equal(const float * a, const float * b, size_t n, const char * tag) {
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t ua, ub;
        std::memcpy(&ua, a + i, 4);
        std::memcpy(&ub, b + i, 4);
        if (ua != ub) {
            if (bad < 5) {
                std::fprintf(stderr, "  %s mismatch i=%zu got %a want %a\n",
                             tag, i, a[i], b[i]);
            }
            ++bad;
        }
    }
    if (bad) {
        std::fprintf(stderr, "FAIL %s: %zu / %zu bits differ\n", tag, bad, n);
    }
    return bad == 0;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <gqh3_64x1024.wire.bin>\n", argv[0]);
        return 2;
    }

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no HIP device\n");
        return 77;
    }

    const std::vector<uint8_t> wire = read_file(argv[1]);
    if (wire.size() < GQH_HEADER_BYTES + (size_t) 64 * 4 * GQH3_SB_BYTES) {
        std::fprintf(stderr, "wire too small: %zu\n", wire.size());
        return 1;
    }
    float tensor_scale = 1.0f;
    std::memcpy(&tensor_scale, wire.data(), sizeof(float));
    const int grid_code = wire[4];
    const uint8_t * body = wire.data() + GQH_HEADER_BYTES;

    auto body_a = tile_gqh3(body, 64, 1024, kOut, kIn, /*row_off=*/0);
    auto body_b = tile_gqh3(body, 64, 1024, kOut, kIn, /*row_off=*/7);

    std::vector<float> xh(kIn);
    for (int i = 0; i < kIn; ++i) {
        xh[i] = 0.01f * (float) ((i % 17) + 1);
    }

    uint8_t * d_a = nullptr;
    uint8_t * d_b = nullptr;
    float * d_x = nullptr;
    float * d_ya = nullptr;
    float * d_yb = nullptr;
    float * d_ya1 = nullptr;
    float * d_yb1 = nullptr;
    HIP_OK(cudaMalloc(&d_a, body_a.size()));
    HIP_OK(cudaMalloc(&d_b, body_b.size()));
    HIP_OK(cudaMalloc(&d_x, xh.size() * sizeof(float)));
    HIP_OK(cudaMalloc(&d_ya, (size_t) kOut * sizeof(float)));
    HIP_OK(cudaMalloc(&d_yb, (size_t) kOut * sizeof(float)));
    HIP_OK(cudaMalloc(&d_ya1, (size_t) kOut * sizeof(float)));
    HIP_OK(cudaMalloc(&d_yb1, (size_t) kOut * sizeof(float)));
    HIP_OK(cudaMemcpy(d_a, body_a.data(), body_a.size(), cudaMemcpyHostToDevice));
    HIP_OK(cudaMemcpy(d_b, body_b.data(), body_b.size(), cudaMemcpyHostToDevice));
    HIP_OK(cudaMemcpy(d_x, xh.data(), xh.size() * sizeof(float), cudaMemcpyHostToDevice));
    if (g_fails) {
        return 1;
    }

    ggml_gqh_register(d_a, body_a.size(), tensor_scale, grid_code);
    ggml_gqh_register(d_b, body_b.size(), tensor_scale, grid_code);

    const bool pair_ok = ggml_cuda_gqh_mul_mat_vec_pair(
            GGML_TYPE_GQH3, d_a, d_ya, d_b, d_yb, d_x,
            kIn, kOut, /*ncols=*/1, /*x_col_stride=*/kIn, /*y_col_stride=*/kOut,
            /*stream=*/0);
    CHECK(pair_ok);
    if (pair_ok) {
        std::printf("gqh pair-fused gate/up gqh3 %dx%d\n", kIn, kOut);
    }

    const bool a_ok = ggml_cuda_gqh_mul_mat_vec(
            GGML_TYPE_GQH3, d_a, d_x, d_ya1, kIn, kOut, 1, kIn, kOut, 0);
    const bool b_ok = ggml_cuda_gqh_mul_mat_vec(
            GGML_TYPE_GQH3, d_b, d_x, d_yb1, kIn, kOut, 1, kIn, kOut, 0);
    CHECK(a_ok);
    CHECK(b_ok);
    HIP_OK(cudaDeviceSynchronize());

    std::vector<float> ya(kOut), yb(kOut), ya1(kOut), yb1(kOut);
    HIP_OK(cudaMemcpy(ya.data(), d_ya, ya.size() * sizeof(float), cudaMemcpyDeviceToHost));
    HIP_OK(cudaMemcpy(yb.data(), d_yb, yb.size() * sizeof(float), cudaMemcpyDeviceToHost));
    HIP_OK(cudaMemcpy(ya1.data(), d_ya1, ya1.size() * sizeof(float), cudaMemcpyDeviceToHost));
    HIP_OK(cudaMemcpy(yb1.data(), d_yb1, yb1.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(bits_equal(ya.data(), ya1.data(), ya.size(), "pair vs single gate"));
    CHECK(bits_equal(yb.data(), yb1.data(), yb.size(), "pair vs single up"));

    // DFlash2 verify width: same 5120x17408 GQH3 pair, ncols=8, ROWS=3 arm.
    constexpr int kN8 = 8;
    std::vector<float> xh8((size_t) kIn * kN8);
    for (int c = 0; c < kN8; ++c) {
        for (int i = 0; i < kIn; ++i) {
            xh8[(size_t) c * kIn + i] = 0.01f * (float) ((i % 17) + 1 + c);
        }
    }
    float * d_x8 = nullptr;
    float * d_ya8 = nullptr;
    float * d_yb8 = nullptr;
    float * d_ya8s = nullptr;
    float * d_yb8s = nullptr;
    HIP_OK(cudaMalloc(&d_x8, xh8.size() * sizeof(float)));
    HIP_OK(cudaMalloc(&d_ya8, (size_t) kOut * kN8 * sizeof(float)));
    HIP_OK(cudaMalloc(&d_yb8, (size_t) kOut * kN8 * sizeof(float)));
    HIP_OK(cudaMalloc(&d_ya8s, (size_t) kOut * kN8 * sizeof(float)));
    HIP_OK(cudaMalloc(&d_yb8s, (size_t) kOut * kN8 * sizeof(float)));
    HIP_OK(cudaMemcpy(d_x8, xh8.data(), xh8.size() * sizeof(float), cudaMemcpyHostToDevice));
    const bool pair8 = ggml_cuda_gqh_mul_mat_vec_pair(
            GGML_TYPE_GQH3, d_a, d_ya8, d_b, d_yb8, d_x8,
            kIn, kOut, kN8, /*x_col_stride=*/kIn, /*y_col_stride=*/kOut, 0);
    CHECK(pair8);
    if (pair8) {
        std::printf("gqh pair-fused gate/up gqh3 %dx%d ncols=%d\n", kIn, kOut, kN8);
    }
    CHECK(ggml_cuda_gqh_mul_mat_vec(
            GGML_TYPE_GQH3, d_a, d_x8, d_ya8s, kIn, kOut, kN8, kIn, kOut, 0));
    CHECK(ggml_cuda_gqh_mul_mat_vec(
            GGML_TYPE_GQH3, d_b, d_x8, d_yb8s, kIn, kOut, kN8, kIn, kOut, 0));
    HIP_OK(cudaDeviceSynchronize());
    std::vector<float> ya8((size_t) kOut * kN8), yb8((size_t) kOut * kN8);
    std::vector<float> ya8s((size_t) kOut * kN8), yb8s((size_t) kOut * kN8);
    HIP_OK(cudaMemcpy(ya8.data(), d_ya8, ya8.size() * sizeof(float), cudaMemcpyDeviceToHost));
    HIP_OK(cudaMemcpy(yb8.data(), d_yb8, yb8.size() * sizeof(float), cudaMemcpyDeviceToHost));
    HIP_OK(cudaMemcpy(ya8s.data(), d_ya8s, ya8s.size() * sizeof(float), cudaMemcpyDeviceToHost));
    HIP_OK(cudaMemcpy(yb8s.data(), d_yb8s, yb8s.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(bits_equal(ya8.data(), ya8s.data(), ya8.size(), "N=8 pair vs single gate"));
    CHECK(bits_equal(yb8.data(), yb8s.data(), yb8.size(), "N=8 pair vs single up"));
    cudaFree(d_x8); cudaFree(d_ya8); cudaFree(d_yb8); cudaFree(d_ya8s); cudaFree(d_yb8s);

    ggml_gqh_unregister(d_a);
    ggml_gqh_unregister(d_b);

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    CHECK(dev != nullptr);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    CHECK(backend != nullptr);

    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * Wg = ggml_new_tensor_2d(ctx, GGML_TYPE_GQH3, kIn, kOut);
    ggml_tensor * Wu = ggml_new_tensor_2d(ctx, GGML_TYPE_GQH3, kIn, kOut);
    ggml_tensor * X  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kIn, 1);
    ggml_tensor * gate = ggml_mul_mat(ctx, Wg, X);
    ggml_tensor * up   = ggml_mul_mat(ctx, Wu, X);
    ggml_tensor * glu  = ggml_swiglu_split(ctx, gate, up);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, glu);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    CHECK(ggml_gallocr_alloc_graph(galloc, gf));

    ggml_backend_tensor_set(Wg, body_a.data(), 0, body_a.size());
    ggml_backend_tensor_set(Wu, body_b.data(), 0, body_b.size());
    ggml_backend_tensor_set(X, xh.data(), 0, xh.size() * sizeof(float));
    ggml_gqh_register(Wg->data, ggml_nbytes(Wg), tensor_scale, grid_code);
    ggml_gqh_register(Wu->data, ggml_nbytes(Wu), tensor_scale, grid_code);

    CHECK(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> g_gate(kOut), g_up(kOut), g_glu(kOut);
    ggml_backend_tensor_get(gate, g_gate.data(), 0, g_gate.size() * sizeof(float));
    ggml_backend_tensor_get(up,   g_up.data(),   0, g_up.size()   * sizeof(float));
    ggml_backend_tensor_get(glu,  g_glu.data(),  0, g_glu.size()  * sizeof(float));
    CHECK(bits_equal(g_gate.data(), ya.data(), g_gate.size(), "graph gate vs pair"));
    CHECK(bits_equal(g_up.data(),   yb.data(), g_up.size(),   "graph up vs pair"));

    size_t glu_bad = 0;
    for (int i = 0; i < kOut; ++i) {
        const float silu = ya[i] / (1.0f + expf(-ya[i]));
        const float want = silu * yb[i];
        if (!(std::fabs(g_glu[i] - want) <= 1e-4f * (1.0f + std::fabs(want)))) {
            if (glu_bad < 5) {
                std::fprintf(stderr, "  glu mismatch i=%d got %a want %a\n", i, g_glu[i], want);
            }
            ++glu_bad;
        }
    }
    CHECK(glu_bad == 0);

    ggml_gqh_unregister(Wg->data);
    ggml_gqh_unregister(Wu->data);
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_x);
    cudaFree(d_ya); cudaFree(d_yb); cudaFree(d_ya1); cudaFree(d_yb1);

    if (g_fails) {
        std::printf("FAIL test_gqh_pair_glu (%d checks)\n", g_fails);
        return 1;
    }
    std::printf("OK   test_gqh_pair_glu: pair dispatch + bit-exact vs two singles + SWIGLU graph\n");
    return 0;
}
