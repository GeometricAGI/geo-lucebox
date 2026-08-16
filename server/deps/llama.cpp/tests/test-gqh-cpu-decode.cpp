// Bit-exactness harness for the GQH CPU decoders, driven through the public type
// traits -- ggml_get_type_traits(type)->to_float -- so it checks the registration
// as well as the arithmetic. No GPU needed.
//
// gqh3/gqh2_h resolve their per-tensor header from the ggml_gqh_register registry,
// which is what the loader fills from GGUF KV; this test registers the same way.
// gqh2_c needs no registration at all.
//
// usage: test-gqh-cpu-decode <gqh3|gqh2_h|gqh2_c> <rows> <cols> <wire.bin> <decode.f32>
// exit:  0 = bit-identical, 1 = mismatch or error

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool read_file(const char * path, std::vector<uint8_t> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize((size_t) n);
    const bool ok = n == 0 || fread(out.data(), 1, (size_t) n, f) == (size_t) n;
    fclose(f);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <gqh3|gqh2_h|gqh2_c> <rows> <cols> <wire.bin> <decode.f32>\n", argv[0]);
        return 1;
    }
    const std::string rung = argv[1];
    const int64_t rows = atoll(argv[2]);
    const int64_t cols = atoll(argv[3]);

    ggml_type type;
    if      (rung == "gqh3")   { type = GGML_TYPE_GQH3;   }
    else if (rung == "gqh2_h") { type = GGML_TYPE_GQH2_H; }
    else if (rung == "gqh2_c") { type = GGML_TYPE_GQH2_C; }
    else { fprintf(stderr, "unknown rung %s\n", rung.c_str()); return 1; }

    const bool has_header = type != GGML_TYPE_GQH2_C;
    const int64_t n = rows * cols;

    std::vector<uint8_t> wire, ref;
    if (!read_file(argv[4], wire) || !read_file(argv[5], ref)) {
        return 1;
    }
    if (ref.size() != (size_t) n * sizeof(float)) {
        fprintf(stderr, "decode.f32 is %zu B, want %zu B\n", ref.size(), (size_t) n * sizeof(float));
        return 1;
    }

    const size_t hdr  = has_header ? 5 : 0;
    const size_t body = wire.size() - hdr;
    const size_t want = (size_t) (n / ggml_blck_size(type)) * ggml_type_size(type);
    if (body != want) {
        fprintf(stderr, "wire body is %zu B, type traits say %zu B for %lldx%lld %s\n",
                body, want, (long long) rows, (long long) cols, ggml_type_name(type));
        return 1;
    }

    const uint8_t * blocks = wire.data() + hdr;
    float tensor_scale = 1.0f;
    int   grid_code    = 0;
    if (has_header) {
        memcpy(&tensor_scale, wire.data(), sizeof(float));
        grid_code = wire[4];
        ggml_gqh_register(blocks, body, tensor_scale, grid_code);
    }

    std::vector<float> got((size_t) n, -1.0f);
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (!traits->to_float) {
        fprintf(stderr, "%s has no to_float\n", ggml_type_name(type));
        return 1;
    }
    // Decode row by row, the way ggml calls it -- a whole-tensor call would not
    // exercise the registry's row-slice resolution.
    for (int64_t r = 0; r < rows; ++r) {
        traits->to_float(blocks + (size_t) r * (body / rows), got.data() + r * cols, cols);
    }

    if (has_header) {
        ggml_gqh_unregister(blocks);
    }

    const float * wantf = (const float *) ref.data();
    int64_t bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        uint32_t a, b;
        memcpy(&a, &got[i],   sizeof(a));
        memcpy(&b, &wantf[i], sizeof(b));
        if (a == b) { continue; }
        if (bad < 8) {
            fprintf(stderr, "  [%lld] (row %lld col %lld) got 0x%08x (%.9g) want 0x%08x (%.9g)\n",
                    (long long) i, (long long) (i / cols), (long long) (i % cols),
                    a, got[i], b, wantf[i]);
        }
        ++bad;
    }
    if (bad) {
        printf("FAIL %s %lldx%lld: %lld/%lld elements differ\n", rung.c_str(),
               (long long) rows, (long long) cols, (long long) bad, (long long) n);
        return 1;
    }
    printf("OK   %s %lldx%lld: CPU to_float bit-identical (%lld elements)\n", rung.c_str(),
           (long long) rows, (long long) cols, (long long) n);
    return 0;
}
