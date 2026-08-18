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
//   2. fused matvec: x is the first 8 basis vectors, inside the cap, so the kernel
//      accumulates decode(W) * x in f32 and dst[i,j] IS the raw f32 decode,
//      compared BITWISE against the f32 reference. gqh2_c has no fused kernel yet
//      and falls back to path 1, so it skips this check.
//
// usage: test-gqh-backend <gqh3|gqh2_h|gqh2_c|gqh4> <rows> <cols> <wire.bin> <decode.f32> [nvec]
//   nvec (default 8) must be <= the tree's fused-matvec column cap, or the hook declines
//   and the fallback answers in fp16, failing every fused comparison. llama.cpp caps at
//   MMVQ_MAX_BATCH_SIZE (8); lucebox caps at luce_mmvq_max_ncols (default 3).
// exit:  0 = bit-identical, 1 = mismatch/error, 77 = skipped (no GPU)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    if (argc != 6 && argc != 7) {
        fprintf(stderr, "usage: %s <gqh3|gqh2_h|gqh2_c|gqh4> <rows> <cols> <wire.bin> <decode.f32> [nvec]\n", argv[0]);
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

    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int nvec = argc == 7 ? atoi(argv[6]) : 8; // must be <= the fused hook's column cap
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

    ggml_tensor * D  = ggml_mul_mat(ctx, W, X);  // (rows, cols): dequant->cuBLAS
    ggml_tensor * D2 = ggml_mul_mat(ctx, W, X2); // (rows, 8):    fused matvec
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, D);
    ggml_build_forward_expand(gf, D2);

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
    }

    // Register AFTER allocation: W->data is the device pointer the decode kernels
    // look up, the same ordering llama-gqh.cpp uses at load.
    if (has_header) {
        ggml_gqh_register(W->data, ggml_nbytes(W), tensor_scale, grid_code);
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        return 1;
    }

    std::vector<float> got(n_elem);
    std::vector<float> got2((size_t) rows * nvec);
    ggml_backend_tensor_get(D,  got.data(),  0, got.size()  * sizeof(float));
    ggml_backend_tensor_get(D2, got2.data(), 0, got2.size() * sizeof(float));

    const float * want = (const float *) ref.data();
    size_t bad = 0;
    size_t ties = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const float ref = want[(size_t) i * cols + j];
            const float g = got[(size_t) j * rows + i];
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
                if (memcmp(&g, &w, 4) != 0) {
                    if (bad2 < 5) {
                        fprintf(stderr, "  fused mismatch at [%d,%d]: got %a want %a\n", i, j, g, w);
                    }
                    ++bad2;
                }
            }
        }
    }

    if (bad || bad2) {
        printf("FAIL %s %dx%d: dequant %zu/%zu differ, fused %zu/%d differ\n",
               rung.c_str(), rows, cols, bad, n_elem, bad2, rows * nvec);
        return 1;
    }
    printf("OK   %s %dx%d: dequant->BLAS matches the fp16-rounded reference%s, "
           "fused matvec bit-identical to the f32 reference over %d cols (scale %.9g, grid %d)\n",
           rung.c_str(), rows, cols,
           ties ? (" apart from " + std::to_string(ties) + " exact fp16 ties").c_str() : "",
           nvec, tensor_scale, grid_code);
    return 0;
}
