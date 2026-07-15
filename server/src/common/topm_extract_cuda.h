// Fast GPU top-M candidate extractor. See the .cu for the method.
//
// d_logits  : device [vocab × n_tokens] f32 (column p contiguous = position p).
// d_cand_ids: device [M × n_tokens] i32, written with position p's M candidate
//             vocab ids (unordered; contains the true top-(M) approximately,
//             exactly above the threshold bin).
// d_scratch : device scratch of >= extract_topm_scratch_bytes(n_tokens).
// Returns false on bad args / launch error.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace dflash::common {

bool   extract_topm_cuda(const float * d_logits, int vocab, int n_tokens, int M,
                         int32_t * d_cand_ids, void * d_scratch, cudaStream_t stream);
size_t extract_topm_scratch_bytes(int n_tokens);

}  // namespace dflash::common
