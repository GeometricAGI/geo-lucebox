#pragma once

// Dense ROCmFPX mix (qtype 105/106) registration from the embedded
// "geoquant.dmix2.sidecar" GGUF KV. Same wire as llama.cpp's
// llama-rocmfpx-mix.cpp. Unregistered 105/106 tensors abort at decode.

#include <string>

struct ggml_context;

namespace dflash {
namespace common {

// Register every resident 105/106 tensor in `ctx` from the KV in `gguf_path`.
// No-op (true) when the context has no mix tensors. False, having registered
// nothing, on a missing/malformed KV, a resident tensor with no entry, a
// qtype disagreement, or a non-2-D / misaligned / non-contiguous tensor.
// Extra sidecar names with no resident tensor are allowed (MTP skip).
// Call AFTER weights are uploaded — the registry is keyed by device pointer.
bool register_dmix2_sidecar(const std::string & gguf_path, ggml_context * ctx);

// Drop entries for `ctx`. Call BEFORE ggml_free. Safe if nothing registered.
void unregister_dmix2_sidecar(ggml_context * ctx);

}  // namespace common
}  // namespace dflash
