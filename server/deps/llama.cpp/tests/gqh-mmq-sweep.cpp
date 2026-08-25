// Per-width MMQ vs dequant->cuBLAS sweep for the GQH qtypes.
//
// This is the tool the GQH_MMQ_MAX_NE11 gate threshold comes from. MMQ and the
// dequant path pull in opposite directions for GQH -- MMQ deletes the fp16
// materialisation but re-decodes the weight tile once per output column tile, so
// it wins narrow and loses wide -- and the crossover is a property of the KERNEL,
// not of any workload. Measuring it in the server would tangle it with prefill
// chunking, spec-decode tree width and prefix caching; here it is one mul_mat.
//
// Weights: a real GQH wire body from tests/gqh-vectors, TILED to the requested
// shape. Any byte pattern is a valid GQH code stream, so this exercises the real
// decode work at the real shape without needing a multi-GB artifact. The numbers
// are kernel timings, not model quality, so the tiling is irrelevant to them.
//
// Arms are selected by GGML_GQH_MMQ in the environment (the library reads it
// once per process), so a sweep runs one arm and the caller interleaves. Each
// width reports whether MMQ actually dispatched, read from the launch counter --
// a sweep that silently measured the same path twice would otherwise look like a
// dead-flat crossover.
//
// usage: gqh-mmq-sweep <gqh3|gqh4> <K> <M> <wire.bin> <ncols,...> [iters]
// exit:  0 ok, 1 error, 77 no GPU

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define GQH_SUPERBLOCK   256
#define GQH_HEADER_BYTES   5

static std::vector<uint8_t> read_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b((size_t) n);
    if (n > 0 && fread(b.data(), 1, (size_t) n, f) != (size_t) n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    return b;
}

int main(int argc, char ** argv) {
    if (argc < 6 || argc > 7) {
        fprintf(stderr, "usage: %s <gqh3|gqh4> <K> <M> <wire.bin> <ncols,...> [iters]\n", argv[0]);
        return 1;
    }
    const std::string rung = argv[1];
    const int64_t K = atoll(argv[2]);          // input dim  (ne0 of src0)
    const int64_t M = atoll(argv[3]);          // output dim (ne1 of src0)
    const int iters = argc == 7 ? atoi(argv[6]) : 30;
    ggml_type wtype;
    if      (rung == "gqh3") wtype = GGML_TYPE_GQH3;
    else if (rung == "gqh4") wtype = GGML_TYPE_GQH4;
    else { fprintf(stderr, "rung must be gqh3 or gqh4\n"); return 1; }
    if (K <= 0 || M <= 0 || K % GQH_SUPERBLOCK) { fprintf(stderr, "K must be a multiple of %d\n", GQH_SUPERBLOCK); return 1; }

    std::vector<int> widths;
    { char * s = strdup(argv[5]); for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) widths.push_back(atoi(t)); free(s); }
    if (widths.empty()) { fprintf(stderr, "no widths\n"); return 1; }

    const std::vector<uint8_t> wire = read_file(argv[4]);
    float tensor_scale; memcpy(&tensor_scale, wire.data(), sizeof(float));
    const int grid_code = wire[4];
    const uint8_t * body = wire.data() + GQH_HEADER_BYTES;
    const size_t body_bytes = wire.size() - GQH_HEADER_BYTES;

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { printf("SKIP: no GPU backend device\n"); return 77; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { printf("SKIP: backend init failed\n"); return 77; }

    const char * arm = getenv("GGML_GQH_MMQ");
    printf("# %s %lldx%lld  GGML_GQH_MMQ=%s  iters=%d  scale=%.6g grid=%d\n",
           rung.c_str(), (long long) K, (long long) M, arm ? arm : "(unset)", iters, tensor_scale, grid_code);
    printf("# %-8s %12s %12s %10s %s\n", "ncols", "ms/iter", "GB/s(src0)", "mmq/iter", "path");

    for (int ncols : widths) {
        ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * W = ggml_new_tensor_2d(ctx, wtype, K, M);
        ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, ncols);
        ggml_tensor * D = ggml_mul_mat(ctx, W, X);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, D);

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) { fprintf(stderr, "alloc failed at ncols=%d\n", ncols); return 1; }

        // Tile the wire body across the whole tensor.
        {
            const size_t nb = ggml_nbytes(W);
            std::vector<uint8_t> buf(nb);
            for (size_t o = 0; o < nb; o += body_bytes) {
                memcpy(buf.data() + o, body, std::min(body_bytes, nb - o));
            }
            ggml_backend_tensor_set(W, buf.data(), 0, nb);
            std::vector<float> x((size_t) K * ncols);
            uint32_t rs = 0x9e3779b9u;
            for (size_t t = 0; t < x.size(); ++t) { rs = rs*1664525u + 1013904223u; x[t] = (float)((int32_t)((rs>>8)%2001)-1000)/1024.0f; }
            ggml_backend_tensor_set(X, x.data(), 0, x.size()*sizeof(float));
        }
        ggml_gqh_register(W->data, ggml_nbytes(W), tensor_scale, grid_code);

        for (int w = 0; w < 3; ++w) {
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
        }
        ggml_backend_synchronize(backend);

        const size_t mmq0 = ggml_backend_cuda_get_mmq_launch_count();
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) {
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
        }
        ggml_backend_synchronize(backend);
        const auto t1 = std::chrono::steady_clock::now();
        const size_t mmq_n = ggml_backend_cuda_get_mmq_launch_count() - mmq0;

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        const double gbs = (double) ggml_nbytes(W) / (ms * 1e-3) / 1e9;
        printf("  %-8d %12.4f %12.1f %10.2f %s\n", ncols, ms, gbs,
               (double) mmq_n / iters, mmq_n ? "MMQ" : "dequant+BLAS");
        fflush(stdout);

        ggml_gqh_unregister(W->data);
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return 0;
}
