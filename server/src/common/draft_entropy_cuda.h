// GPU temperature-agnostic entropy of the draft's per-position logit
// distribution. Used to gate the candidate-restricted greedy head at
// runtime: low draft entropy (confident) -> safe to use the cheap
// restricted head; high entropy (uncertain) -> fall back to the exact
// full-vocab head. See docs/topk-head-optimization.md §5d/§5e for the
// calibration this backs.
//
// Deliberately a standalone kernel (not folded into geometric_draft_topk_cuda.cu)
// so it stays independent of that kernel's temperature scaling and doesn't
// risk perturbing the DDTree tree-construction / sampling paths that already
// depend on it. Cost: one extra full-vocab-wide (bandwidth-bound) GPU pass
// over logits already resident on-device -- no additional host D2H beyond
// the tiny n_positions-sized entropy output, so it does not reintroduce the
// vocab-wide D2H the fast decode path is built to avoid.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace dflash::common {

// d_logits: device pointer, [vocab] contiguous per position, n_positions rows.
// out_entropy: HOST pointer, n_positions floats (nats), temperature-agnostic
// (raw logits, as if temperature=1 regardless of any sampling temperature).
// Returns false on any CUDA error (out_entropy left untouched).
bool extract_draft_entropy_cuda(const float * d_logits, int vocab, int n_positions,
                                float * out_entropy, cudaStream_t stream = nullptr);

}  // namespace dflash::common
