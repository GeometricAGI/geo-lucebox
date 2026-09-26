#include "ggml.h"
#include "../deps/llama.cpp/ggml/src/gqh.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>

static void require(bool pass, const char * message) {
    if (!pass) throw std::runtime_error(message);
}

int main(int argc, char ** argv) try {
    require(argc == 3, "Usage: test RUNG WIRE_64x1024");
    const std::string rung = argv[1];
    const ggml_type type = rung == "gqh_t" ? GGML_TYPE_GQH_T :
        rung == "gqh_t_g32_r4" ? GGML_TYPE_GQH_T_G32_R4 :
        rung == "gqh_t_g32_r3" ? GGML_TYPE_GQH_T_G32_R3 : GGML_TYPE_COUNT;
    require(type != GGML_TYPE_COUNT, "Invalid rung");
    std::ifstream file(argv[2], std::ios::binary | std::ios::ate);
    require(bool(file), "Cannot read fixture");
    std::vector<char> wire(static_cast<size_t>(file.tellg()));
    file.seekg(0); file.read(wire.data(), wire.size());
    auto ctx = ggml_init({ggml_tensor_overhead()*2, nullptr, true});
    auto w = ggml_new_tensor_2d(ctx, type, 1024, 64);
    require(wire.size() == ggml_nbytes(w)+5, "Wire shape mismatch");
    w->data = wire.data()+5;
    float global; std::memcpy(&global, wire.data(), 4);
    std::vector<float> divisors(1024), original(64*1024), got(64*1024);
    for (size_t c=0; c<divisors.size(); ++c) divisors[c] = 0.5f + float(c%17)/16;
    require(!ggml_gqh_register_ternary_input_scale(w, divisors.data(), 1024, nullptr),
            "Unregistered tensor accepted");
    char other_tensor;
    ggml_gqh_register(&other_tensor, 1, global, 0);
    ggml_gqh_register(w->data, ggml_nbytes(w), global, wire[4]);
    auto decode = ggml_get_type_traits(type)->to_float;
    decode(w->data, original.data(), original.size());
    require(!ggml_gqh_register_ternary_input_scale(w, divisors.data(), 512, nullptr), "Wrong columns accepted");
    require(ggml_gqh_register_ternary_input_scale(w, divisors.data(), 1024, nullptr), "Valid scales rejected");
    const float * borrowed = nullptr; const void * device = nullptr;
    int64_t columns = 0; size_t offset = 0;
    require(ggml_gqh_lookup_input_scale(w->data, &borrowed, &device, &columns, &offset), "Lookup failed");
    ggml_gqh_unregister(&other_tensor);
    const float * after_move = nullptr;
    require(ggml_gqh_lookup_input_scale(w->data, &after_move, &device, &columns, &offset), "Moved lookup failed");
    require(after_move == borrowed, "Unrelated unregister invalidated borrowed storage");
    const auto saved = divisors;
    divisors[0] = 0;
    require(!ggml_gqh_register_ternary_input_scale(w, divisors.data(), 1024, nullptr), "Zero accepted");
    divisors[0] = std::numeric_limits<float>::quiet_NaN();
    require(!ggml_gqh_register_ternary_input_scale(w, divisors.data(), 1024, nullptr), "NaN accepted");
    divisors.assign(1024, 99); // Registry owns its copy, and rejection preserves it.
    decode(w->data, got.data(), got.size());
    for (size_t i=0; i<got.size(); ++i) require(got[i] == original[i]/saved[i%1024], "Whole tensor compensation mismatch");
    for (int row=0; row<64; ++row) {
        decode(static_cast<char *>(w->data)+row*w->nb[1], got.data(), 1024);
        for (int col=0; col<1024; ++col) require(got[col] == original[row*1024+col]/saved[col], "Row slice mismatch");
    }
    ggml_gqh_register(w->data, ggml_nbytes(w), global, wire[4]);
    decode(w->data, got.data(), got.size());
    require(got == original, "Re-registration retained stale compensation");
    require(ggml_gqh_register_ternary_input_scale(w, saved.data(), 1024, nullptr), "Reattach failed");
    ggml_gqh_unregister(w->data);
    require(!ggml_gqh_register_ternary_input_scale(w, saved.data(), 1024, nullptr), "Unregister retained entry");
    ggml_gqh_register(w->data, ggml_nbytes(w), global, wire[4]);
    decode(w->data, got.data(), got.size());
    require(got == original, "Pointer reuse retained compensation");
    ggml_gqh_unregister(w->data); ggml_free(ctx);
    return 0;
} catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
