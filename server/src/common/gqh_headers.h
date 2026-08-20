#pragma once

// GQH (108/109/111) per-tensor header registration, shared by every model family.
//
// gqh4/gqh3/gqh2_h scale every weight by a 5-byte per-tensor header (float32
// tensor_scale + uint8 grid code) that a fixed-size ggml block cannot hold, so it
// travels in the "geoquant.gqh.headers" GGUF KV. An unregistered GQH tensor aborts
// at decode, so the KV is read, validated and registered as part of the load.
//
// Model-agnostic on purpose: it walks the weights context and reads one KV, so it
// needs nothing from a family's weight struct. GQH is not tied to a surface the way
// the ROCmFPX mix qtypes are, which is exactly why it cannot live in one loader.
// gqh2_c (110) needs no entry -- its scale is fp16 in-block.

#include <string>

struct ggml_context;

namespace dflash {
namespace common {

// Register every resident gqh4/gqh3/gqh2_h tensor in `ctx` from the KV in `gguf_path`.
// Returns true when there is nothing to do (no GQH tensors), so callers can invoke
// it unconditionally. Returns false, having registered nothing, on any violation:
// a missing/malformed KV, a resident tensor with no entry, a qtype disagreement, a
// non-2-D tensor, a row length off the 256 superblock grid, a non-contiguous
// tensor, or a non-finite/non-positive scale. Diagnostics go to stderr.
//
// Call AFTER the weights are uploaded -- the registry is keyed by device pointer.
bool register_gqh_headers(const std::string & gguf_path, ggml_context * ctx);

// Drop the entries for `ctx`. Call BEFORE ggml_free, while the tensors are still
// valid: the registry resolves by pointer range, so a stale entry would shadow a
// later load that reuses the address. Safe on a context that registered nothing.
void unregister_gqh_headers(ggml_context * ctx);

}  // namespace common
}  // namespace dflash
