// The "geoquant.gqh.headers" KV parser, tested without a model.
//
// gqh4/gqh3/gqh2_h carry a 5-byte per-tensor header out of band, and
// gqh_qtype_has_header() in gqh_headers.cpp is the ONLY place that decides which
// qtypes require one. Until this test existed that decision was reachable only
// through a real model load, so adding a rung meant either trusting the one-line
// change or building a multi-GB artifact to exercise it.
//
// register_gqh_headers() reads the KV out of the GGUF by path and matches it
// against the tensors resident in a ggml_context, so a KV-ONLY GGUF is enough --
// it never touches tensor data on disk. Runs on the CPU: the registry lives in
// ggml-base (ggml/src/gqh.cpp), not in a backend.
//
// Note the interface difference from the fork's llama_gqh_register_tensors, which
// takes a caller-filtered tensor vector: this one walks the context itself, so
// ignoring gqh2_c (110) is ITS job, not the caller's. That is checked below.

#include "common/gqh_headers.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Declared in ggml/src/gqh.h, internal to ggml and not on the public include
// path. It is GGML_API, so declaring it here is enough to link.
extern "C" bool ggml_gqh_lookup(const void * p, float * tensor_scale, int * grid_code);

static const char * GQH_KV_KEY = "geoquant.gqh.headers";

// ctest runs this in the build directory, so clean up after ourselves.
static std::vector<std::string> tmp_files;

static int failures = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        failures++;
    }
}

// A negative case must be refused for ITS OWN reason. This parser reports to
// stderr and returns false, so the run log carries the reasons inline.
static void check_refused(bool registered, const char * what) {
    check(!registered, what);
    if (!registered) {
        std::printf("  refused as expected: %s\n", what);
    }
}

struct entry {
    std::string name;
    uint32_t    qtype;
    float       scale;
    uint8_t     code;
};

// Mirrors scripts/gqh/make_gqh_probe.py: build_header_kv.
static std::vector<uint8_t> build_kv(const std::vector<entry> & entries,
                                     uint32_t count_override = 0,
                                     const char * magic = "GQHh1\0\0\0") {
    std::vector<uint8_t> b;
    auto put = [&](const void * p, size_t n) {
        const uint8_t * q = (const uint8_t *) p;
        b.insert(b.end(), q, q + n);
    };
    put(magic, 8);
    const uint32_t count = count_override ? count_override : (uint32_t) entries.size();
    const uint32_t reserved = 0;
    put(&count, 4);
    put(&reserved, 4);
    for (const entry & e : entries) {
        const uint32_t len = (uint32_t) e.name.size();
        put(&len, 4);
        put(e.name.data(), len);
        put(&e.qtype, 4);
        put(&e.scale, 4);
        put(&e.code, 1);
        const uint8_t pad[3] = { 0, 0, 0 };
        put(pad, 3);
    }
    return b;
}

// A GGUF carrying nothing but the header KV. The parser reopens the file with
// no_alloc and looks up one key, so tensor info on disk is irrelevant.
static std::string write_kv_gguf(const std::vector<uint8_t> & blob, const char * stem) {
    const std::string path = std::string("./") + stem + ".gguf";
    gguf_context * g = gguf_init_empty();
    gguf_set_arr_data(g, GQH_KV_KEY, GGUF_TYPE_UINT8, blob.data(), (int64_t) blob.size());
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/ true);
    gguf_free(g);
    tmp_files.push_back(path);
    return path;
}

static std::string write_empty_gguf(const char * stem) {
    const std::string path = std::string("./") + stem + ".gguf";
    gguf_context * g = gguf_init_empty();
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/ true);
    gguf_free(g);
    tmp_files.push_back(path);
    return path;
}

// 2-D GQH tensors with real (allocated) data, so the registry has pointer ranges
// to resolve -- the row-slice lookup the decoders actually use.
struct model {
    ggml_context * ctx = nullptr;

    model(const std::vector<std::pair<const char *, ggml_type>> & spec) {
        ggml_init_params ip = {};
        ip.mem_size = 64u * 1024 * 1024;
        ctx = ggml_init(ip);
        for (const auto & nt : spec) {
            ggml_tensor * t = ggml_new_tensor_2d(ctx, nt.second, 256, 4);
            ggml_set_name(t, nt.first);
        }
    }
    ~model() {
        if (ctx) {
            dflash::common::unregister_gqh_headers(ctx);
            ggml_free(ctx);
        }
    }
    ggml_tensor * get(const char * name) const { return ggml_get_tensor(ctx, name); }
};

int main() {
    using dflash::common::register_gqh_headers;
    using dflash::common::unregister_gqh_headers;
    using dflash::common::ggml_context_has_gqh;
    using dflash::common::gqh_cap_spec_ddtree_budget;

    // ---- 1. happy path: one tensor per header-bearing rung, plus a gqh2_c ----
    // The gqh2_c tensor is resident and has NO entry. This parser must ignore it.
    {
        model m({ { "blk.0.ffn_down.weight", GGML_TYPE_GQH3   },
                  { "blk.0.ffn_up.weight",   GGML_TYPE_GQH2_H },
                  { "blk.0.ffn_gate.weight", GGML_TYPE_GQH4   },
                  { "blk.0.attn_q.weight",   GGML_TYPE_GQH2_C } });
        const std::string p = write_kv_gguf(build_kv({
            { "blk.0.ffn_down.weight", 108, 0.125f,  3 },
            { "blk.0.ffn_up.weight",   109, 2.5f,    7 },
            { "blk.0.ffn_gate.weight", 111, 8.0f,   11 } }), "gqh_hdr_ok");
        check(register_gqh_headers(p, m.ctx), "happy path registers");

        struct expect { const char * name; float scale; int code; bool registered; };
        const expect want[] = { { "blk.0.ffn_down.weight", 0.125f,  3, true  },
                                { "blk.0.ffn_up.weight",   2.5f,    7, true  },
                                { "blk.0.ffn_gate.weight", 8.0f,   11, true  },
                                { "blk.0.attn_q.weight",   0.0f,    0, false } };
        for (const expect & w : want) {
            ggml_tensor * t = m.get(w.name);
            float scale = -1.0f;
            int   code  = -1;
            check(ggml_gqh_lookup(t->data, &scale, &code) == w.registered,
                  (std::string("registration of ") + w.name).c_str());
            if (w.registered) {
                check(scale == w.scale && code == w.code,
                      (std::string("header values of ") + w.name).c_str());
                // The decoders look up by ROW SLICE, not just the base pointer.
                const size_t row = ggml_row_size(t->type, 256);
                float s2 = -1.0f;
                int   c2 = -1;
                check(ggml_gqh_lookup((const char *) t->data + 2 * row, &s2, &c2)
                          && s2 == w.scale && c2 == w.code,
                      (std::string("row-slice lookup of ") + w.name).c_str());
            }
        }
        unregister_gqh_headers(m.ctx);
        float s_gone = -1.0f;
        int   c_gone = -1;
        check(!ggml_gqh_lookup(m.get("blk.0.ffn_down.weight")->data, &s_gone, &c_gone),
              "unregister clears");
    }

    // ---- 2. a context with no GQH tensor at all: nothing to do, so true ----
    {
        model m({ { "blk.0.attn_k.weight", GGML_TYPE_Q4_0 } });
        const std::string p = write_empty_gguf("gqh_hdr_none");
        check(register_gqh_headers(p, m.ctx), "no GQH tensors -> true, no KV needed");
    }

    // ---- 3. a resident gqh4 tensor with no entry must FAIL ------------------
    {
        model m({ { "blk.0.ffn_gate.weight", GGML_TYPE_GQH4 } });
        const std::string p = write_kv_gguf(build_kv({}), "gqh_hdr_uncovered");
        check_refused(register_gqh_headers(p, m.ctx), "uncovered gqh4 tensor");
    }

    // ---- 4. an entry naming no resident tensor must be ACCEPTED -------------
    // MTP-block weights are skipped on a normal load, so a correct artifact
    // carries entries that will never match. The cover check is one-directional.
    {
        model m({ { "blk.0.ffn_gate.weight", GGML_TYPE_GQH4 } });
        const std::string p = write_kv_gguf(build_kv({
            { "blk.0.ffn_gate.weight",       111, 8.0f, 11 },
            { "blk.64.nextn.ffn_down.weight", 111, 1.0f, 0 } }), "gqh_hdr_phantom");
        check(register_gqh_headers(p, m.ctx), "phantom MTP entry is accepted");
    }

    // ---- 5. an entry whose qtype carries no header must FAIL ---------------
    {
        model m({ { "blk.0.ffn_gate.weight", GGML_TYPE_GQH4   },
                  { "blk.0.attn_q.weight",   GGML_TYPE_GQH2_C } });
        const std::string p = write_kv_gguf(build_kv({
            { "blk.0.ffn_gate.weight", 111, 8.0f, 11 },
            { "blk.0.attn_q.weight",   110, 1.0f,  0 } }), "gqh_hdr_qtype110");
        check_refused(register_gqh_headers(p, m.ctx), "entry for headerless qtype 110");
    }

    // ---- 6. qtype mismatch between entry and resident tensor must FAIL -----
    {
        model m({ { "blk.0.ffn_gate.weight", GGML_TYPE_GQH4 } });
        const std::string p = write_kv_gguf(build_kv({
            { "blk.0.ffn_gate.weight", 108, 8.0f, 11 } }), "gqh_hdr_mismatch");
        check_refused(register_gqh_headers(p, m.ctx), "qtype mismatch (entry 108, tensor 111)");
    }

    // ---- 7. structural corruption must FAIL, not read past the blob --------
    {
        model m({ { "blk.0.ffn_gate.weight", GGML_TYPE_GQH4 } });
        const std::vector<entry> one = { { "blk.0.ffn_gate.weight", 111, 8.0f, 11 } };

        check_refused(register_gqh_headers(
            write_kv_gguf(build_kv(one, 0, "GQHh2\0\0\0"), "gqh_hdr_magic"), m.ctx),
            "wrong magic");
        check_refused(register_gqh_headers(
            write_kv_gguf(build_kv(one, 9), "gqh_hdr_count"), m.ctx),
            "entry_count over-declared");
        {
            std::vector<uint8_t> b = build_kv(one);
            b.resize(b.size() - 5);
            check_refused(register_gqh_headers(
                write_kv_gguf(b, "gqh_hdr_trunc"), m.ctx), "truncated entry");
        }
        check_refused(register_gqh_headers(
            write_kv_gguf({}, "gqh_hdr_empty"), m.ctx), "empty blob");
        check_refused(register_gqh_headers(
            write_empty_gguf("gqh_hdr_nokv"), m.ctx), "KV missing with resident GQH");
    }

    // ---- 8. a grid code outside GAMMA_GRID must FAIL -----------------------
    // 12 codes, so 12 is out of range; the decoders index the table unchecked.
    {
        model m({ { "blk.0.ffn_gate.weight", GGML_TYPE_GQH4 } });
        const std::string p = write_kv_gguf(build_kv({
            { "blk.0.ffn_gate.weight", 111, 1.0f, 12 } }), "gqh_hdr_gridcode");
        check_refused(register_gqh_headers(p, m.ctx), "grid code 12 is out of range");
    }

    // ---- 9. SpecLA budget cap: GQH stays on ncols == native_block ----------
    {
        check(gqh_cap_spec_ddtree_budget(nullptr, 8, 8) == 8,
              "no ctx: budget 8 stays 8");
        model gqh({ { "blk.0.ffn_gate.weight", GGML_TYPE_GQH3 } });
        check(ggml_context_has_gqh(gqh.ctx), "GQH3 context is detected");
        check(gqh_cap_spec_ddtree_budget(gqh.ctx, 8, 8) == 7,
              "GQH budget 8 caps to 7 (ncols=8)");
        check(gqh_cap_spec_ddtree_budget(gqh.ctx, 7, 8) == 7,
              "GQH budget 7 is already at the cap");
        check(gqh_cap_spec_ddtree_budget(gqh.ctx, 22, 8) == 7,
              "GQH budget 22 caps to 7");
        model q4({ { "blk.0.ffn_gate.weight", GGML_TYPE_Q4_K } });
        check(!ggml_context_has_gqh(q4.ctx), "Q4_K context is not GQH");
        check(gqh_cap_spec_ddtree_budget(q4.ctx, 8, 8) == 8,
              "non-GQH budget 8 is unchanged");
        model c2({ { "blk.0.attn_q.weight", GGML_TYPE_GQH2_C } });
        check(ggml_context_has_gqh(c2.ctx), "gqh2_c (no KV header) still counts");
        check(gqh_cap_spec_ddtree_budget(c2.ctx, 8, 8) == 7,
              "gqh2_c budget 8 caps to 7");
    }

    for (const std::string & f : tmp_files) {
        std::remove(f.c_str());
    }

    if (failures) {
        std::fprintf(stderr, "\ntest_gqh_headers: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("OK   test_gqh_headers: KV parse, header values, cover rules, "
                "gqh2_c ignored, 6 malformed inputs, SpecLA budget cap\n");
    return 0;
}
