#pragma once

// Runtime toggle for the batched (MMQ) path on the mix qtypes 105/106.
//
// Plain C++ so the host-compiled tests can include it and stay in sync with
// mmq.cu; keeping the declarations here is what makes a signature change a
// compile error rather than a link-time surprise.

// Whether the batched path is active. Defaults to GGML_CUDA_MIX_MMQ_DEFAULT
// (on), overridable per-process with DFLASH_MIX_MMQ=0.
bool ggml_cuda_mix_mmq_enabled();

// Force the toggle, ignoring the environment, so one process can A/B both
// paths. test_rocmfp_mix_mmq uses this to compare MMQ against the validated
// matvec kernel.
void ggml_cuda_set_mix_mmq_enabled(bool enabled);

// Drop the forced value and fall back to the environment/default.
void ggml_cuda_clear_mix_mmq_override();
