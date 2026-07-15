// Candidate-restricted greedy LM head (Q6_K weight). See the .cu for the idea.
//
// Computes, for each of n_tokens positions, the argmax vocab id over that
// position's M candidate ids — using a fused gather + Q6_K dequant + dot +
// argmax kernel that reads only the M candidate rows of the head, not all
// n_vocab. Approximate-greedy: exact iff the true argmax ∈ the candidate set
// (calibrated; see docs/topk-head-optimization.md).

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace dflash::common {

// d_head_q6k : device pointer to the Q6_K output.weight, row-major
//              [n_embd × n_vocab] (n_embd is the contiguous/block dim; one
//              vocab row is n_embd/256 Q6_K blocks).
// d_hidden   : device [n_embd × n_tokens] f32 pre-head hidden states.
// d_cand_ids : device [M × n_tokens] i32, column p = position p's M candidate
//              vocab ids.
// d_out_tokens : device [n_tokens] i32, written with the per-position argmax id.
// d_scratch_keys : device scratch of >= n_tokens * sizeof(uint64_t) bytes.
// Returns false on bad args or a launch error (caller falls back to full head).
bool restricted_lm_head_q6k(const void * d_head_q6k,
                            int n_embd, int n_vocab,
                            const float * d_hidden, int n_tokens,
                            const int32_t * d_cand_ids, int M,
                            int32_t * d_out_tokens,
                            void * d_scratch_keys,
                            cudaStream_t stream);

}  // namespace dflash::common
