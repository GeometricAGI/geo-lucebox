#include "common.cuh"
#include "ggml-cuda.h"

#include <cstddef>
#include <vector>

#if defined(GGML_USE_HIP)
#ifndef cudaPointerAttributes
#define cudaPointerAttributes    hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaMemoryTypeDevice     hipMemoryTypeDevice
#define cudaMemoryTypeManaged    hipMemoryTypeManaged
#endif
#endif

namespace {

__global__ void gdn_transition_journal_commit_kernel(
        const float * journal,
        float * state,
        const int32_t * accepted_prefixes,
        const int32_t * active_slot_ids,
        int state_size,
        int n_heads,
        int n_tokens,
        int n_seqs,
        int n_state_slots,
        int journal_width,
        int gate_values) {
    const int sequence = blockIdx.z;
    const int head = blockIdx.y;
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    const int state_elements = state_size * state_size;
    if (sequence >= n_seqs || head >= n_heads ||
        element >= state_elements) {
        return;
    }

    const int slot = active_slot_ids[sequence];
    const int accepted = accepted_prefixes[sequence];
    if (slot < 0 || slot >= n_state_slots ||
        accepted < 0 || accepted > n_tokens) {
        return;
    }

    const int row = element % state_size;
    const int col = element / state_size;
    const size_t state_offset =
        (((size_t) slot*n_heads + head)*state_size + col)*state_size + row;
    float current = state[state_offset];

    for (int token = 0; token < accepted; ++token) {
        const float * transition = journal +
            (((size_t) sequence*n_tokens + token)*n_heads + head) *
                journal_width;
        const float gate = gate_values == 1
            ? transition[0]
            : transition[row];
        const float key = transition[gate_values + row];
        const float delta =
            transition[gate_values + state_size + col];
        current = fmaf(key, delta, gate * current);
    }

    state[state_offset] = current;
}

bool device_pointer(const void * pointer, int & device) {
    if (pointer == nullptr) return false;
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice &&
        attributes.type != cudaMemoryTypeManaged) {
        return false;
    }
    device = attributes.device;
    return true;
}

} // namespace

namespace {

__global__ void gdn_conv_journal_commit_kernel(
        const float * conv_input,
        float * conv_state,
        const int32_t * accepted_prefixes,
        const int32_t * active_slot_ids,
        int window,
        int channels,
        int n_tokens,
        int n_seqs,
        int n_state_slots) {
    const int sequence = blockIdx.y;
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = window * channels;
    if (sequence >= n_seqs || element >= count) return;
    const int slot = active_slot_ids[sequence];
    const int accepted = accepted_prefixes[sequence];
    if (slot < 0 || slot >= n_state_slots ||
        accepted < 0 || accepted > n_tokens) return;
    const int k = element % window;
    const int channel = element / window;
    const size_t source =
        ((size_t) sequence * channels + channel) * (window + n_tokens) +
        accepted + k;
    const size_t destination =
        ((size_t) slot * channels + channel) * window + k;
    conv_state[destination] = conv_input[source];
}

__global__ void tree_cache_commit_kernel(
        uint8_t * cache,
        const int64_t * commit_rows,
        const int32_t * active_slot_ids,
        size_t row_bytes,
        size_t head_stride,
        int n_heads,
        int tree_width,
        int n_seqs,
        int n_cache_rows,
        int scratch_base,
        int scratch_stride) {
    const int byte = blockIdx.x * blockDim.x + threadIdx.x;
    const int flat = blockIdx.y;
    const int head = blockIdx.z;
    if ((size_t) byte >= row_bytes || flat >= tree_width*n_seqs ||
        head >= n_heads) return;
    const int lane = flat / tree_width;
    const int node = flat % tree_width;
    const int slot = active_slot_ids[lane];
    const int64_t destination_row = commit_rows[flat];
    if (slot < 0 || destination_row < 0 ||
        destination_row >= n_cache_rows) return;
    const int64_t source_row =
        (int64_t) scratch_base + (int64_t) slot*scratch_stride + node;
    if (source_row < 0 || source_row >= n_cache_rows) return;
    const size_t source =
        (size_t) head*head_stride + (size_t) source_row*row_bytes + byte;
    const size_t destination =
        (size_t) head*head_stride + (size_t) destination_row*row_bytes + byte;
    cache[destination] = cache[source];
}

__global__ void tree_feature_commit_kernel(
        const uint8_t * source,
        uint8_t * destination,
        const int32_t * destination_rows,
        size_t row_bytes,
        int n_rows,
        int destination_capacity) {
    const int byte = blockIdx.x * blockDim.x + threadIdx.x;
    const int source_row = blockIdx.y;
    if ((size_t) byte >= row_bytes || source_row >= n_rows) return;
    const int destination_row = destination_rows[source_row];
    if (destination_row < 0 || destination_row >= destination_capacity) return;
    destination[(size_t) destination_row*row_bytes + byte] =
        source[(size_t) source_row*row_bytes + byte];
}

bool same_device_pointer(const void * pointer, int expected_device) {
    int pointer_device = -1;
    return device_pointer(pointer, pointer_device) &&
        pointer_device == expected_device;
}

} // namespace

extern "C" bool ggml_backend_cuda_gdn_transition_journal_commit_many(
        const ggml_tensor * const * journals,
        ggml_tensor * const * states,
        const ggml_tensor * const * conv_inputs,
        ggml_tensor * const * conv_states,
        int n_layers,
        const ggml_tensor * accepted_prefixes,
        const ggml_tensor * active_slot_ids) {
    if (!journals || !states || !conv_inputs || !conv_states ||
        n_layers <= 0 || !accepted_prefixes || !active_slot_ids ||
        accepted_prefixes->type != GGML_TYPE_I32 ||
        active_slot_ids->type != GGML_TYPE_I32 ||
        !ggml_is_contiguous(accepted_prefixes) ||
        !ggml_is_contiguous(active_slot_ids)) return false;

    const int64_t n_seqs = ggml_nelements(accepted_prefixes);
    if (n_seqs < 1 || ggml_nelements(active_slot_ids) != n_seqs) return false;
    int device = -1;
    if (!device_pointer(accepted_prefixes->data, device) ||
        !same_device_pointer(active_slot_ids->data, device)) return false;

    std::vector<int32_t> accepted((size_t) n_seqs);
    std::vector<int32_t> slots((size_t) n_seqs);
    const size_t map_bytes = (size_t) n_seqs*sizeof(int32_t);
    if (cudaMemcpy(accepted.data(), accepted_prefixes->data, map_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(slots.data(), active_slot_ids->data, map_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;

    int common_tokens = -1;
    int common_state_slots = -1;
    std::vector<uint8_t> seen;
    for (int layer = 0; layer < n_layers; ++layer) {
        const ggml_tensor * journal = journals[layer];
        ggml_tensor * state = states[layer];
        const ggml_tensor * conv_input = conv_inputs[layer];
        ggml_tensor * conv_state = conv_states[layer];
        if (!journal || !state || !conv_input || !conv_state ||
            journal->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
            conv_input->type != GGML_TYPE_F32 || conv_state->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(journal) || !ggml_is_contiguous(state) ||
            !ggml_is_contiguous(conv_input) || !ggml_is_contiguous(conv_state)) return false;
        const int64_t state_size = state->ne[0];
        const int64_t heads = state->ne[2];
        const int64_t tokens = journal->ne[2];
        const int64_t state_slots = state->ne[3];
        if ((state_size != 16 && state_size != 32 &&
             state_size != 64 && state_size != 128) ||
            state->ne[1] != state_size || heads < 1 ||
            journal->ne[1] != heads || journal->ne[3] != n_seqs ||
            (journal->ne[0] != 2*state_size + 1 &&
             journal->ne[0] != 3*state_size) || tokens < 1 ||
            conv_state->ne[0] < 1 || conv_state->ne[1] < 1 ||
            conv_state->ne[2] != state_slots || conv_state->ne[3] != 1 ||
            conv_input->ne[0] != conv_state->ne[0] + tokens ||
            conv_input->ne[1] != conv_state->ne[1] ||
            conv_input->ne[2] != n_seqs || conv_input->ne[3] != 1) return false;
        if (common_tokens < 0) {
            common_tokens = (int) tokens;
            common_state_slots = (int) state_slots;
            seen.assign((size_t) state_slots, 0);
        } else if (tokens != common_tokens || state_slots != common_state_slots) {
            return false;
        }
        const void * pointers[] = {
            journal->data, state->data, conv_input->data, conv_state->data,
        };
        for (const void * pointer : pointers) {
            if (!same_device_pointer(pointer, device)) return false;
        }
    }
    for (int64_t lane = 0; lane < n_seqs; ++lane) {
        if (accepted[(size_t) lane] < 0 ||
            accepted[(size_t) lane] > common_tokens) return false;
        const int slot = slots[(size_t) lane];
        if (slot == -1) continue;
        if (slot < 0 || slot >= common_state_slots) return false;
        if (seen[(size_t) slot]) return false;
        seen[(size_t) slot] = 1;
    }

    ggml_cuda_set_device(device);
    constexpr int threads = 256;
    (void) cudaGetLastError();
    for (int layer = 0; layer < n_layers; ++layer) {
        const ggml_tensor * journal = journals[layer];
        ggml_tensor * state = states[layer];
        const int state_size = (int) state->ne[0];
        const int heads = (int) state->ne[2];
        const int tokens = (int) journal->ne[2];
        const int journal_width = (int) journal->ne[0];
        const int64_t state_elements = (int64_t) state_size*state_size;
        const dim3 state_grid(
            (unsigned int) ((state_elements + threads - 1)/threads),
            (unsigned int) heads, (unsigned int) n_seqs);
        gdn_transition_journal_commit_kernel<<<state_grid, threads>>>(
            (const float *) journal->data, (float *) state->data,
            (const int32_t *) accepted_prefixes->data,
            (const int32_t *) active_slot_ids->data,
            state_size, heads, tokens, (int) n_seqs,
            (int) state->ne[3], journal_width,
            journal_width == 2*state_size + 1 ? 1 : state_size);

        const ggml_tensor * conv_input = conv_inputs[layer];
        ggml_tensor * conv_state = conv_states[layer];
        const int conv_elements =
            (int) (conv_state->ne[0]*conv_state->ne[1]);
        const dim3 conv_grid(
            (unsigned int) ((conv_elements + threads - 1)/threads),
            (unsigned int) n_seqs, 1);
        gdn_conv_journal_commit_kernel<<<conv_grid, threads>>>(
            (const float *) conv_input->data, (float *) conv_state->data,
            (const int32_t *) accepted_prefixes->data,
            (const int32_t *) active_slot_ids->data,
            (int) conv_state->ne[0], (int) conv_state->ne[1], tokens,
            (int) n_seqs, (int) conv_state->ne[2]);
    }
    if (cudaGetLastError() != cudaSuccess) return false;
    return cudaDeviceSynchronize() == cudaSuccess;
}

extern "C" bool ggml_backend_cuda_tree_cache_commit_many(
        ggml_tensor * const * caches,
        int n_caches,
        const ggml_tensor * commit_rows,
        const ggml_tensor * active_slot_ids,
        int tree_scratch_base,
        int tree_scratch_stride) {
    if (!caches || n_caches <= 0 || !commit_rows || !active_slot_ids ||
        commit_rows->type != GGML_TYPE_I64 ||
        active_slot_ids->type != GGML_TYPE_I32 ||
        !ggml_is_contiguous(commit_rows) ||
        !ggml_is_contiguous(active_slot_ids) ||
        commit_rows->ne[0] < 1 || commit_rows->ne[1] < 1 ||
        ggml_nelements(active_slot_ids) != commit_rows->ne[1] ||
        tree_scratch_base < 0 || tree_scratch_stride < commit_rows->ne[0]) return false;
    const int tree_width = (int) commit_rows->ne[0];
    const int n_seqs = (int) commit_rows->ne[1];
    const int n_rows = tree_width*n_seqs;
    int device = -1;
    if (!device_pointer(commit_rows->data, device) ||
        !same_device_pointer(active_slot_ids->data, device)) return false;

    std::vector<int64_t> destinations((size_t) n_rows);
    std::vector<int32_t> slots((size_t) n_seqs);
    if (cudaMemcpy(destinations.data(), commit_rows->data,
                   destinations.size()*sizeof(int64_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(slots.data(), active_slot_ids->data,
                   slots.size()*sizeof(int32_t), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    int cache_rows = -1;
    for (int index = 0; index < n_caches; ++index) {
        ggml_tensor * cache = caches[index];
        if (!cache || !ggml_is_contiguous(cache) || cache->ne[0] < 1 ||
            cache->ne[1] < 1 || cache->ne[2] < 1 || cache->ne[3] != 1 ||
            cache->nb[1] < ggml_row_size(cache->type, cache->ne[0]) ||
            !same_device_pointer(cache->data, device)) return false;
        if (cache_rows < 0) cache_rows = (int) cache->ne[1];
        else if (cache->ne[1] != cache_rows) return false;
    }
    for (int lane = 0; lane < n_seqs; ++lane) {
        const int slot = slots[(size_t) lane];
        if (slot == -1) continue;
        if (slot < 0) return false;
        const int64_t source_end = (int64_t) tree_scratch_base +
            (int64_t) slot*tree_scratch_stride + tree_width;
        if (source_end > cache_rows) return false;
        for (int node = 0; node < tree_width; ++node) {
            const int64_t destination =
                destinations[(size_t) lane*tree_width + node];
            if (destination < -1 || destination >= cache_rows) return false;
        }
    }

    ggml_cuda_set_device(device);
    constexpr int threads = 256;
    (void) cudaGetLastError();
    for (int index = 0; index < n_caches; ++index) {
        ggml_tensor * cache = caches[index];
        const dim3 grid(
            (unsigned int) ((cache->nb[1] + threads - 1)/threads),
            (unsigned int) n_rows, (unsigned int) cache->ne[2]);
        tree_cache_commit_kernel<<<grid, threads>>>(
            (uint8_t *) cache->data,
            (const int64_t *) commit_rows->data,
            (const int32_t *) active_slot_ids->data,
            cache->nb[1], cache->nb[2], (int) cache->ne[2],
            tree_width, n_seqs, cache_rows,
            tree_scratch_base, tree_scratch_stride);
    }
    if (cudaGetLastError() != cudaSuccess) return false;
    return cudaDeviceSynchronize() == cudaSuccess;
}

extern "C" bool ggml_backend_cuda_tree_feature_commit(
        const ggml_tensor * source,
        ggml_tensor * destination,
        const ggml_tensor * destination_rows) {
    if (!source || !destination || !destination_rows ||
        source->type != destination->type || source->type != GGML_TYPE_BF16 ||
        destination_rows->type != GGML_TYPE_I32 ||
        !ggml_is_contiguous(source) || !ggml_is_contiguous(destination) ||
        !ggml_is_contiguous(destination_rows) ||
        source->ne[0] != destination->ne[0] || source->ne[2] != 1 ||
        source->ne[3] != 1 || destination->ne[2] != 1 ||
        destination->ne[3] != 1 ||
        ggml_nelements(destination_rows) != source->ne[1] ||
        source->nb[1] != destination->nb[1]) return false;
    int device = -1;
    if (!device_pointer(source->data, device) ||
        !same_device_pointer(destination->data, device) ||
        !same_device_pointer(destination_rows->data, device)) return false;
    const int n_rows = (int) source->ne[1];
    std::vector<int32_t> rows((size_t) n_rows);
    if (cudaMemcpy(rows.data(), destination_rows->data,
                   rows.size()*sizeof(int32_t), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (int row : rows) {
        if (row < -1 || row >= destination->ne[1]) return false;
    }
    ggml_cuda_set_device(device);
    constexpr int threads = 256;
    const dim3 grid(
        (unsigned int) ((source->nb[1] + threads - 1)/threads),
        (unsigned int) n_rows, 1);
    (void) cudaGetLastError();
    tree_feature_commit_kernel<<<grid, threads>>>(
        (const uint8_t *) source->data, (uint8_t *) destination->data,
        (const int32_t *) destination_rows->data,
        source->nb[1], n_rows, (int) destination->ne[1]);
    if (cudaGetLastError() != cudaSuccess) return false;
    return cudaDeviceSynchronize() == cudaSuccess;
}
