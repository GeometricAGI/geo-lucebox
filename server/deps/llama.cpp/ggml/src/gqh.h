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

// CPU decoders behind the type traits. gqh3/gqh2_h abort on an unregistered
// pointer rather than guess a scale; see the to_float comment in ggml.c.
void dequantize_row_gqh3  (const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_gqh2_h(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_gqh2_c(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

#ifdef __cplusplus
}
#endif
