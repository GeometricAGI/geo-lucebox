#pragma once

// GQH (Geo-Quant Hierarchical) decode support shared by every backend.
//
// Holds the per-tensor header registry and the CPU decoders. The registry lives
// here, in ggml-base, rather than in a backend: gqh3/gqh2_h scale every weight by
// a 5-byte per-tensor header (float32 tensor_scale + uint8 grid code) that a
// fixed-size ggml block cannot hold, and BOTH the CPU to_float hooks and the CUDA
// converters need it. One registration at load serves both.
//
// gqh2_c needs nothing out of band: its scale is fp16 in-block and its codebook
// and sign table are frozen constants in gqh-tables.h. Its decoder is
// unconditional.
//
// Bit-exactness against geoquant/formats/gqh.py (decode3/decode/decode_c) is the
// acceptance gate, so these decoders match the reference's float32 operation
// ORDER. Do not reassociate them.

#include "ggml.h"
#include "gqh-tables.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolve a registered tensor's header. `p` may address a ROW SLICE inside the
// tensor, not just its base -- to_float is called per row and the CUDA op splits
// src0 by rows. Returns false if `p` falls in no registered tensor.
GGML_API bool ggml_gqh_lookup(const void * p, float * tensor_scale, int * grid_code);

// As above, plus the registered tensor's row length (ne[0]) and base pointer. The
// planar device layout needs both: plane offsets are a function of nsb =
// ne0/GQH_SUPERBLOCK, and the row index has to be recovered from (p - base) against
// the PADDED row stride rather than ggml_row_size(). `ne0` is 0 for entries
// registered through ggml_gqh_register (tight layout, which needs neither).
// `ne0` and `base` may be NULL.
// `planar` reports whether THIS tensor's device image is the 4-plane layout. It is a
// per-tensor property, not a global mode: model weights can be planar while a test or
// scratch tensor registered through ggml_gqh_register stays tight in the same process.
// Any of `ne0`, `base`, `planar` may be NULL.
GGML_API bool ggml_gqh_lookup_ex(const void * p, float * tensor_scale, int * grid_code,
                                 int64_t * ne0, const void ** base, int * planar);

// Register with the row length recorded. `nbytes` must span the whole device buffer
// (the PADDED planar size when the planar layout is in use), since lookup resolves
// interior row slices by pointer range.
GGML_API void ggml_gqh_register_ex(const void * base, size_t nbytes, float tensor_scale,
                                   int grid_code, int64_t ne0, int planar);

// CPU decoders behind the type traits. gqh3/gqh2_h abort on an unregistered
// pointer rather than guess a scale; see the to_float comment in ggml.c.
void dequantize_row_gqh3  (const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_gqh2_h(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_gqh2_c(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

#ifdef __cplusplus
}
#endif
