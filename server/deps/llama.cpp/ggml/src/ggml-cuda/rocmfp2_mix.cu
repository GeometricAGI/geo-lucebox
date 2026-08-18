// Runtime decode for GGML_TYPE_Q2_1_ROCMFP2_MIX (106).
// A per-tensor registry supplies the per-expert codebook/mode that the ggml
// to_fp16 converter signature cannot carry; the deepseek4 loader registers each
// mixed tensor after staging its decode tables to device memory.
#include "rocmfp2_mix.cuh"
#include "convert.cuh"
// For ggml_cuda_op_swiglu_ds4_single: the fused path must apply the EXACT function the
// standalone swiglu_ds4 kernel applies, not a re-derivation of the formula.
#include "unary.cuh"
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

// MSVC's host toolchain gives nvcc device code neither __builtin_memcpy nor
// __builtin_assume_aligned. memcpy lowers to the same register moves under
// every device compiler, and the alignment hint is an optimization-only
// contract that the callers' address arithmetic already guarantees -- so the
// MSVC fallback changes nothing except compiling. GCC/Clang (every HIP build,
// where the wide-load numbers were measured) keep the builtins.
#if defined(_MSC_VER) && !defined(__clang__)
#define MIX_MEMCPY(dst, src, n)   memcpy((dst), (src), (n))
#define MIX_ASSUME_ALIGNED(p, a)  (p)
#else
#define MIX_MEMCPY(dst, src, n)   __builtin_memcpy((dst), (src), (n))
#define MIX_ASSUME_ALIGNED(p, a)  __builtin_assume_aligned((p), (a))
#endif

#if defined(GGML_USE_HIP)
#ifndef cudaPointerAttributes
// Same local-mapping pattern topk-rows.cu already uses in this directory: the vendor shim
// does not map the pointer-attribute API, and these are needed to find which device owns an
// expert tensor. hipPointerAttribute_t is layout-compatible for the fields read here.
#define cudaPointerAttributes    hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaMemoryTypeDevice     hipMemoryTypeDevice
#define cudaMemoryTypeManaged    hipMemoryTypeManaged
#endif
#endif


#define MIX_QK 32
#define MIX_QS 8
#define MIX_BLOCK_BYTES 10
// Learned levels per codebook. A qtype-106 entry carries TWO codebooks (the 7s1c
// layout's 1-bit select picks one per 16-weight half-block), so an expert's table
// is 2 * MIX_K bf16 = 16 B.
#define MIX_K 4

namespace {
struct MixEntry {
    const void * base;
    size_t nb02;          // byte stride between experts
    size_t expert_bytes;  // payload bytes inside each expert stride
    int n_experts, out, in;
    const nv_bfloat16 * codebooks;  // n_experts * 2 * 4
    const uint8_t * modes;          // n_experts
    int  device;          // device the side-data lives on; frees must happen in that context
};
// Dispatch wrappers keep this lock from lookup through kernel enqueue. It is
// recursive because MMQ takes the public dispatch lock and then calls the
// ordinary lookup helper, which also protects itself.
std::recursive_mutex g_mix_mtx;
std::vector<MixEntry> g_mix_registry;

// Resolve the device that owns `p`. The mix side-data must be allocated on the SAME device as
// the expert tensor it describes: the kernel dereferences both together, and cudaMalloc uses
// the CURRENT device, which is not necessarily the one the model was loaded onto. Returns the
// current device when the pointer cannot be attributed (single-device hosts, or host memory),
// which preserves the previous behaviour exactly.
static int mix_device_of(const void * p) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    cudaPointerAttributes attr{};
    if (p && cudaPointerGetAttributes(&attr, p) == cudaSuccess) {
        // cudaMemoryTypeUnregistered/Host leave `device` meaningless; only trust device memory.
        if ((attr.type == cudaMemoryTypeDevice ||
             attr.type == cudaMemoryTypeManaged) && attr.device >= 0) {
            return attr.device;
        }
    }
    cudaGetLastError();   // clear the error a non-device pointer sets
    return dev;
}

// RAII device switch: restores the previous device even on the early-return error paths.
struct MixDeviceGuard {
    int prev = -1;
    bool active = false;
    bool valid = false;
    explicit MixDeviceGuard(int dev) {
        if (cudaGetDevice(&prev) != cudaSuccess) return;
        if (dev == prev) {
            valid = true;
            return;
        }
        if (cudaSetDevice(dev) == cudaSuccess) active = valid = true;
    }
    ~MixDeviceGuard() { if (active) cudaSetDevice(prev); }
};

// Free an entry's device side-data. Caller holds g_mix_mtx, so no new launch
// can acquire these pointers. Synchronizing here drains launches that released
// the lock after enqueueing but are still running on the device.
void mix_free_entry_device(MixEntry & e) {
    MixDeviceGuard guard(e.device);   // free where it was allocated
    if (!guard.valid) {
        GGML_ABORT("rocmfp2_mix: failed to select the side-data device");
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const cudaError_t codebook_err = e.codebooks
        ? cudaFree((void *) e.codebooks) : cudaSuccess;
    const cudaError_t mode_err = e.modes
        ? cudaFree((void *) e.modes) : cudaSuccess;
    e.codebooks = nullptr;
    e.modes = nullptr;
    if (codebook_err != cudaSuccess) CUDA_CHECK(codebook_err);
    if (mode_err != cudaSuccess) CUDA_CHECK(mode_err);
}

// Enforce the wide-load invariant for EVERY registration path. mix_block_accum reads the
// two 8-aligned u64 bracketing each 10-byte block -- a 16 B window from (addr & ~7). For
// the last block of a row that window ends (6 - (10*(nb-1) & 7)) B past the row end, which
// is 0 only when nb % 4 == 0, i.e. in % 128 == 0. Any other `in` reads up to 6 B past the
// tensor allocation, and for the last tensor in a buffer that is a fault.
//
bool mix_validate_shape(int in) {
    if (in % 128 != 0) {
        GGML_LOG_ERROR("rocmfp2_mix: in=%d must be a multiple of 128\n", in);
        return false;
    }
    return true;
}

bool mix_validate_registration(
        const void * base, size_t nb02, int n_experts, int out, int in,
        const void * codebooks, const void * modes) {
    if (!base || nb02 == 0 || n_experts <= 0 || out <= 0 || in <= 0 ||
        !codebooks || !modes) {
        GGML_LOG_ERROR("rocmfp2_mix: invalid registration metadata\n");
        return false;
    }
    if (!mix_validate_shape(in)) return false;

    const size_t row_bytes = (size_t) (in / MIX_QK) * MIX_BLOCK_BYTES;
    if ((size_t) out > std::numeric_limits<size_t>::max() / row_bytes) {
        GGML_LOG_ERROR("rocmfp2_mix: expert tensor size overflows\n");
        return false;
    }
    const size_t expert_bytes = (size_t) out * row_bytes;
    if (nb02 < expert_bytes) {
        GGML_LOG_ERROR("rocmfp2_mix: expert stride is smaller than the tensor shape\n");
        return false;
    }
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(base);
    const std::uintptr_t stride = (std::uintptr_t) nb02;
    const std::uintptr_t expert_span = (std::uintptr_t) expert_bytes;
    if ((size_t) stride != nb02 || (size_t) expert_span != expert_bytes) {
        GGML_LOG_ERROR("rocmfp2_mix: expert span does not fit an address\n");
        return false;
    }
    const std::uintptr_t available =
        std::numeric_limits<std::uintptr_t>::max() - address;
    if ((std::uintptr_t) (n_experts - 1) > available / stride) {
        GGML_LOG_ERROR("rocmfp2_mix: registered expert address range overflows\n");
        return false;
    }
    const std::uintptr_t last_offset =
        (std::uintptr_t) (n_experts - 1) * stride;
    if (expert_span - 1 > available - last_offset) {
        GGML_LOG_ERROR("rocmfp2_mix: final expert address range overflows\n");
        return false;
    }
    constexpr size_t table_bytes = 2 * MIX_K * sizeof(nv_bfloat16);
    if ((size_t) n_experts >
        std::numeric_limits<size_t>::max() / table_bytes) {
        GGML_LOG_ERROR("rocmfp2_mix: codebook allocation size overflows\n");
        return false;
    }
    return true;
}

void mix_register_impl(const void * base, size_t nb02, int n_experts, int out, int in,
                       const nv_bfloat16 * codebooks, const uint8_t * modes,
                       int device) {
    std::lock_guard<std::recursive_mutex> lk(g_mix_mtx);
    const size_t expert_bytes =
        (size_t) out * (size_t) (in / MIX_QK) * MIX_BLOCK_BYTES;
    MixEntry ne{base, nb02, expert_bytes, n_experts, out, in, codebooks, modes,
                device};
    for (auto & e : g_mix_registry) {
        if (e.base == base) {  // update in place — free the old owned buffers first
            mix_free_entry_device(e);
            e = ne;
            return;
        }
    }
    g_mix_registry.push_back(ne);
}
}  // namespace

// Host-side convenience for the deepseek4 loader: stage per-expert codebooks
// (bf16) and modes from host memory into device buffers, then register. The
// registry owns these buffers and frees them on unregister/update. On a
// cudaMalloc failure, allocated buffers are freed before the error propagates.
extern "C" bool ggml_cuda_rocmfp2_mix_register_host(
        const void * base, size_t nb02, int n_experts, int out, int in,
        const void * codebooks_bf16_host, const uint8_t * modes_host) {
    if (!mix_validate_registration(
            base, nb02, n_experts, out, in,
            codebooks_bf16_host, modes_host)) {
        return false;
    }
    for (int i = 0; i < n_experts; ++i) {
        if (modes_host[i] > 1) {
            GGML_LOG_ERROR("rocmfp2_mix: unsupported mode %u\n",
                           (unsigned) modes_host[i]);
            return false;
        }
    }
    const size_t cb_bytes = (size_t) n_experts * 2 * MIX_K * sizeof(nv_bfloat16);
    void * cb_dev = nullptr; void * modes_dev = nullptr;
    // Allocate where the WEIGHTS are. Without this the side-data lands on the current
    // device while the kernel runs on the model's device, and the first request
    // segfaults on a multi-GPU host.
    const int device = mix_device_of(base);
    MixDeviceGuard guard(device);
    if (!guard.valid) {
        GGML_LOG_ERROR("rocmfp2_mix: failed to select the tensor's device\n");
        return false;
    }
    cudaError_t err = cudaMalloc(&cb_dev, cb_bytes);
    if (err == cudaSuccess) err = cudaMemcpy(cb_dev, codebooks_bf16_host, cb_bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMalloc(&modes_dev, (size_t) n_experts);
    if (err == cudaSuccess) err = cudaMemcpy(modes_dev, modes_host, (size_t) n_experts, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        if (cb_dev)    cudaFree(cb_dev);
        if (modes_dev) cudaFree(modes_dev);
        GGML_LOG_ERROR("rocmfp2_mix: decode-table upload failed: %s\n",
                       cudaGetErrorString(err));
        return false;
    }
    mix_register_impl(base, nb02, n_experts, out, in, (const nv_bfloat16 *) cb_dev,
                      (const uint8_t *) modes_dev, device);
    return true;
}

extern "C" void ggml_cuda_rocmfp2_mix_unregister(const void * base) {
    std::lock_guard<std::recursive_mutex> lk(g_mix_mtx);
    for (size_t i = 0; i < g_mix_registry.size(); ++i) {
        if (g_mix_registry[i].base == base) {
            mix_free_entry_device(g_mix_registry[i]);
            g_mix_registry.erase(g_mix_registry.begin() + i);
            return;
        }
    }
}

static bool mix_lookup(
        const void * vx, MixEntry & out_e, int & out_expert,
        size_t * out_byte_offset = nullptr) {
    std::lock_guard<std::recursive_mutex> lk(g_mix_mtx);
    if (!vx) return false;
    const std::uintptr_t address =
        reinterpret_cast<std::uintptr_t>(vx);
    for (const auto & e : g_mix_registry) {
        if (!e.base || e.nb02 == 0 || e.n_experts <= 0) continue;
        const std::uintptr_t base =
            reinterpret_cast<std::uintptr_t>(e.base);
        if (address < base) continue;
        const std::uintptr_t offset = address - base;
        const std::uintptr_t expert = offset / e.nb02;
        const std::uintptr_t within_expert = offset % e.nb02;
        if (expert < static_cast<std::uintptr_t>(e.n_experts) &&
            within_expert < e.expert_bytes) {
            out_e = e;
            out_expert = (int) expert;
            if (out_byte_offset) {
                *out_byte_offset = (size_t) within_expert;
            }
            return true;
        }
    }
    return false;
}

static bool mix_lookup_expert_base(
        const void * vx, MixEntry & out_e, int & out_expert) {
    size_t byte_offset = 0;
    return mix_lookup(vx, out_e, out_expert, &byte_offset) && byte_offset == 0;
}

// Branchless decode. Different lanes decode different meta bytes, so `e` is
// per-lane data-dependent and the two guards (`e > 0x7E`, `exp == 0`) diverge
// within a warp -- the branched form pays exec-mask save/restore + v_cmpx per
// call AND still runs both arms under divergence. Computing both arms straight
// and selecting is bit-identical (the selected value equals the branch result
// for every input; both arms are always finite for uint8 `e`, so no
// NaN/inf contaminates the unselected path) while dropping the control flow.
__device__ __forceinline__ float mix_ue4m3(uint8_t e) {
    int exp = e >> 3, mant = e & 7;
    float normal = ldexpf((float) (8 + mant), exp - 11);
    float sub = (float) mant * 0.0009765625f;  // exp==0 subnormal branch, 2^-10
    float r = (exp == 0) ? sub : normal;
    return (e > 0x7E) ? 0.0f : r;
}

// 2-bit codes pack four to a byte and never straddle a boundary -- so unlike
// qtype-105's 3-bit codes this needs no multi-byte gather, no shift arithmetic
// across bytes, and no MIX_QS bounds test. Least-significant pair first.
__device__ __forceinline__ uint32_t mix_fp2_code(const uint8_t * qs, int i) {
    return (uint32_t) (qs[i >> 2] >> (2 * (i & 3))) & 3u;
}

// Same 2-bit code, read directly out of a 64-bit register holding the block's 8
// code bytes (byte k of `codes` == qs[k]). Bit-identical to mix_fp2_code(qs, i):
// (codes >> (8*(i>>2) + 2*(i&3))) & 3. Max shift for i=31 is 62 < 64. Lets the
// wide load-from-floor staging keep the codes in a register instead of a stack buf.
__device__ __forceinline__ uint32_t mix_fp2_code_u64(uint64_t codes, int i) {
    return (uint32_t) (codes >> (8 * (i >> 2) + 2 * (i & 3))) & 3u;
}

// Fixed levels for mode 0 (uniform qtype-107 fallback): {-1, 0, 1, 2}, code order.
// A 4-entry lookup beats qtype-105's sign/magnitude arithmetic and is exact.
__device__ __forceinline__ float mix_fp2_fixed(uint32_t code) {
    return (float) ((int) code - 1);   // 0->-1, 1->0, 2->1, 3->2
}

// One thread per element of a single expert slice (k = out*in elements).
__global__ void dequantize_rocmfp2_mix_kernel(
        const uint8_t * __restrict__ data, const nv_bfloat16 * __restrict__ book,
        const uint8_t * __restrict__ mode_ptr, int in, int64_t k, half * __restrict__ y) {
    // Same defect, same fix as the matvec path: `book` was read from global once per
    // ELEMENT (one thread per element), for a 16 B workgroup-invariant table. Stage it
    // in LDS above the bounds early-return so every thread reaches __syncthreads().
    // The dense (non-MoE) fallback runs through this kernel, so it has to be fixed
    // before any dense adaptive timing number is quoted.
    __shared__ float s_lut[2 * MIX_K];
    if ((int) threadIdx.x < 2 * MIX_K) {
        s_lut[threadIdx.x] = __bfloat162float(book[threadIdx.x]);
    }
    __syncthreads();
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;
    const int mode = (int) mode_ptr[0];
    const int nb  = in / MIX_QK;
    const int row = idx / in;
    const int col = idx % in;
    const int b   = row * nb + (col / MIX_QK);
    const int j   = col % MIX_QK;
    const int half = (j >= MIX_QK / 2) ? 1 : 0;
    const uint8_t * blk = data + (int64_t) b * MIX_BLOCK_BYTES;
    const uint8_t meta = blk[MIX_QS + half];
    const uint32_t code = mix_fp2_code(blk, j);
    float val;
    if (mode == 0) {
        val = mix_ue4m3(meta) * mix_fp2_fixed(code);
    } else {
        const float scale = mix_ue4m3(meta & 0x7F);
        const int bk = meta >> 7;
        val = scale * s_lut[bk * MIX_K + (int) code];
    }
    y[idx] = __float2half(val);
}

void dequantize_rocmfp2_mix_to_fp16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    std::lock_guard<std::recursive_mutex> dispatch_lock(g_mix_mtx);
    MixEntry e;
    int expert;
    size_t byte_offset = 0;
    if (!mix_lookup(vx, e, expert, &byte_offset)) {
        GGML_ABORT("rocmfp2_mix: tensor slice %p not registered", vx);
    }
    const int64_t registered_elements = (int64_t) e.in * e.out;
    if (k < 0 || k > registered_elements || (k > 0 && !y)) {
        GGML_ABORT("rocmfp2_mix: invalid dequantization range");
    }
    if (k == 0) return;
    const int64_t requested_blocks = (k + MIX_QK - 1) / MIX_QK;
    const size_t available_blocks =
        (e.expert_bytes - byte_offset) / MIX_BLOCK_BYTES;
    if (byte_offset % MIX_BLOCK_BYTES != 0 ||
        static_cast<uint64_t>(requested_blocks) > available_blocks) {
        GGML_ABORT("rocmfp2_mix: invalid dequantization range");
    }
    const nv_bfloat16 * book = e.codebooks + (size_t) expert * 2 * 4;
    const uint8_t * mode_ptr = e.modes + expert;
    const int threads = 256;
    const int64_t block_count = (k + threads - 1) / threads;
    if (block_count > std::numeric_limits<int>::max()) {
        GGML_ABORT("rocmfp2_mix: dequantization grid is too large");
    }
    const int blocks = (int) block_count;
    // Portable launch: triple-chevron compiles under both nvcc and hipcc; the
    // hipLaunchKernelGGL macro is HIP-only and breaks the default CUDA build,
    // which still globs this *.cu file.
    dequantize_rocmfp2_mix_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(
        (const uint8_t *) vx, book, mode_ptr, e.in, k, y);
}

// ---- fused quantized matvec (MMVQ-style decode) ----
// One warp per output row; lanes stride over the row's blocks, decode 32 weights
// each (bit-identical to dequantize_rocmfp2_mix_kernel), multiply by x, f32
// accumulate, warp-reduce. blockIdx.y selects the column (token). Reading the
// quantized blocks once avoids the ~10x f16 round-trip of the dequant fallback.
#define MIX_WARP 32
#define MIX_UNROLL 8

// Output rows folded per warp in the MoE matvec (the fused form doubles the FOLDS:
// each row carries an `up` and a `gate` fold). Every row keeps its own accumulator
// and its own fixed ascending-j left fold over the same block sequence, so this is a
// pure work-per-warp knob -- no value of it changes any row's arithmetic or summation
// order, and the output is bit-identical for any choice.
//
// What it trades (measured on H200, decode geometry in=7168/out=2048/n_used=4/ntok=1,
// standalone harness, medians of 7x200 launches, adaptive-codebook mode):
//   + FEWER SECTORS. The block's 32 activations are staged ONCE per warp-iteration
//     regardless of ROWS, and activations are ~76% of this kernel's global-load
//     sectors (lane L owns blocks 32 floats apart, so every float4 activation request
//     touches 32 distinct sectors and uses 16 B of each). Doubling ROWS cuts
//     sectors/work by 1.76x unfused, 1.6x fused.
//   - FEWER WARPS. Total warps = out*n_used*ntok/ROWS. At ROWS=4 that is 2048 warps
//     for 132 SMs, and this kernel is latency-bound (66.6% L1, 7.7% DRAM), so
//     halving the warp count removes the latency hiding that paid for the sectors.
//
// The two instantiations land on opposite sides of that trade, so they get different
// values rather than one compromise (all bit-identical output):
//   unfused (1 fold/row): ROWS 2 -> 4 -> 8 = 47.58 -> **36.13** -> 35.21 us. Its
//     sector saving is the larger one. ROWS=8 edges it here but collapses on the
//     fixed-codebook mode (38.81 -> 29.95 -> 34.11 us) and leaves only 512 blocks
//     = 3.9/SM, so 4.
//   fused (2 folds/row): ROWS 2 -> 4 = 66.44 -> **68.96** us, so it keeps 2. It had
//     already amortised the stage over 4 folds, so it buys less, and it pays the
//     warp-count loss in full. Confirmed concurrency-bound rather than fold-bound by
//     an ntok sweep (`bench <mode> <fuse> <ntok>`): the ROWS=4 penalty shrinks away
//     as ntok 1 -> 8 restores the warp count.
// Note ROWS=4 for the fused kernel ALSO fixes the wave quantisation (2048 -> 1024
// blocks, one wave) and is still slower. Resident warps, not wave count, is the
// objective at ntok=1.
// Must stay in sync with `rows_per_block` in the two mul_mat_id wrappers below.
// The dense and 3-D slice kernels keep their own 2-row blocking and are unaffected.
#define MIX_MOE_ROWS_FUSED   2
#define MIX_MOE_ROWS_UNFUSED 4

// Down-shift warp shuffle confined to a 32-lane logical group. width=MIX_WARP
// keeps the reduction self-contained on wave64 (GFX8/9, physical wave = 64) and
// is a no-op vs the default warp width on wave32 (gfx1151) / NVIDIA, so the
// reduced value — and thus the greedy output hash — is bit-identical there.
// HIP keeps the bare (mask-free) __shfl_down; modern CUDA only has the _sync
// form (and HIP's vendor shim doesn't cover __shfl_down_sync), so branch.
__device__ __forceinline__ float mix_warp_shfl_down(float v, int off) {
#if defined(__HIP_PLATFORM_AMD__)
    return __shfl_down(v, off, MIX_WARP);
#else
    return __shfl_down_sync(0xffffffffu, v, off, MIX_WARP);
#endif
}

// Stage a block's 32 activations into registers with float4 loads.
//
// Measured motivation (H200, muse-glimmer dense decode): the per-block j loop
// below reads xc[col0 + 0 .. 31], and nvcc emitted one 32-bit LDG per element —
// the overwhelming majority of the kernel's global load instructions, against a
// handful for the (already wide-staged) weights. ncu put the sibling qtype-105
// kernel at 1.65% DRAM throughput, 97% L1 throughput and 98.4% L1 hit rate with
// 56% of warp cycles stalled on LG throttle: not bandwidth bound, bound purely
// on the NUMBER of narrow global load instructions. The 32 floats are contiguous
// (128 B), so four at a time costs one instruction instead of four.
//
// Bit-exact: this changes only how the SAME 32 f32 values reach registers. The
// j loop still consumes them in ascending order into the same acc chain.
//
// The alignment test is warp- AND block-uniform (col0 is always a multiple of
// MIX_QK, i.e. 128 B, so xc+col0 has xc's alignment), so the branch costs no
// divergence. The scalar arm is not dead code: ggml hands this kernel a src1
// row pointer, and a strided/offset column would land unaligned.
__device__ __forceinline__ void mix_load_x32(
        const float * __restrict__ xp, float (&xv)[MIX_QK]) {
    if ((((uintptr_t) xp) & 15u) == 0) {
        const float4 * __restrict__ p4 = (const float4 *) xp;
        #pragma unroll
        for (int k = 0; k < MIX_QK / 4; ++k) {
            const float4 v = p4[k];
            xv[4*k + 0] = v.x; xv[4*k + 1] = v.y;
            xv[4*k + 2] = v.z; xv[4*k + 3] = v.w;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < MIX_QK; ++j) xv[j] = xp[j];
    }
}

// A block's 10 packed bytes, staged in registers: the 32 fp2 codes plus the two
// scale/bank bytes. Split out of mix_block_accum_x so a caller folding several rows
// against one activation stage can issue EVERY row's weight load before any of the
// FMA work (see mix_moe_block_fold). That matters because this kernel is
// latency-bound, not throughput-bound: at the decode geometry a block spends most of
// its time waiting on dependent global loads, so the number of loads in flight is the
// lever. Splitting the phases changes no value and no accumulation order.
struct MixBlock {
    uint64_t codes;
    uint8_t  m0;   // block byte MIX_QS+0
    uint8_t  m1;   // block byte MIX_QS+1
};

__device__ __forceinline__ MixBlock mix_block_load(const uint8_t * __restrict__ b) {
    const uintptr_t addr  = (uintptr_t) b;
    const uint8_t * base8 = (const uint8_t *) MIX_ASSUME_ALIGNED(
            (const void *) (addr & ~(uintptr_t) 7), 8);
    const int sh = (int) (addr & 7) * 8;                 // 0, 16, 32, 48
    uint64_t lo, hi;
    MIX_MEMCPY(&lo, base8, 8);
    MIX_MEMCPY(&hi, base8 + 8, 8);
    MixBlock r;
    r.codes = (sh == 0) ? lo : ((lo >> sh) | (hi << (64 - sh)));
    r.m0    = (uint8_t) (hi >> sh);
    r.m1    = (uint8_t) (hi >> (sh + 8));
    return r;
}

// Fold one already-staged block's 32 terms into acc. Body lifted verbatim out of
// mix_block_accum_x, which is now this plus mix_block_load.
__device__ __forceinline__ void mix_block_fma(
        const MixBlock & blk, const float (&xv)[MIX_QK],
        int mode, const float * __restrict__ lut, float & acc) {
    const uint64_t codes = blk.codes;
    const uint8_t  m0 = blk.m0, m1 = blk.m1;
    if (mode == 0) {
        const float s0 = mix_ue4m3(m0), s1 = mix_ue4m3(m1);
        #pragma unroll
        for (int j = 0; j < MIX_QK; ++j) {
            const float s = (j < MIX_QK/2) ? s0 : s1;
            acc += s * mix_fp2_fixed(mix_fp2_code_u64(codes, j)) * xv[j];
        }
    } else {
        const float s0 = mix_ue4m3(m0 & 0x7F), s1 = mix_ue4m3(m1 & 0x7F);
        const float * bk0 = lut + (m0 >> 7) * MIX_K;
        const float * bk1 = lut + (m1 >> 7) * MIX_K;
        #pragma unroll
        for (int j = 0; j < MIX_QK; ++j) {
            const float s = (j < MIX_QK/2) ? s0 : s1;
            const float * bk = (j < MIX_QK/2) ? bk0 : bk1;
            acc += s * bk[mix_fp2_code_u64(codes, j)] * xv[j];
        }
    }
}

// Accumulate one block's 32 terms directly into acc, in fixed j order, exactly
// as the un-refactored loop did (acc += s * w_j * x_j). Adding each term into
// the shared running acc — rather than forming a per-block partial sum first —
// preserves the flat left-fold summation order, so the f32 result is bit-for-bit
// identical to the original. (Correctness hashes the greedy output; a per-block
// tree reduction changes the rounding and flips tokens.) The block's byte loads
// do not depend on acc, so unrolling the caller over several blocks lets the
// compiler overlap their loads even though the acc-add chain stays serial.
__device__ __forceinline__ void mix_block_accum_x(
        const uint8_t * __restrict__ b, const float (&xv)[MIX_QK],
        int mode, const float * __restrict__ lut, float & acc) {
    // Exactly mix_block_load + mix_block_fma, and deliberately not a second copy of
    // the 32-term fold: the dense/slice kernels and the MoE kernel must not be able
    // to drift apart numerically. Both callees are __forceinline__, so this composes
    // to the same code the one-piece version emitted (verified: dense 64 regs, slice
    // 70, harness hashes unchanged).
    mix_block_fma(mix_block_load(b), xv, mode, lut, acc);
}

// Original (col0-addressed) entry point, kept for the MoE and 3-D slice kernels.
// Stages the activations itself; identical arithmetic.
__device__ __forceinline__ void mix_block_accum(
        const uint8_t * __restrict__ b, const float * __restrict__ xc, int col0,
        int mode, const float * __restrict__ lut, float & acc) {
    float xv[MIX_QK];
    mix_load_x32(xc + col0, xv);
    mix_block_accum_x(b, xv, mode, lut, acc);
}

// Two output rows, ONE activation stage. The 32 activations a block consumes are
// the same for every row, so loading them once and folding both rows against the
// registers halves the (dominant) activation load issue per unit work. This is
// the same register-blocking the MoE kernel below already does — the dense 2-D
// kernel never got it because the model this family was built for is MoE and
// never took this path.
//
// Bit-exact: each row keeps its own accumulator and its own ascending-j fold over
// the same block sequence, so acc0/acc1 match the one-row-per-warp kernel element
// for element.
__device__ __forceinline__ void mix_block_accum2(
        const uint8_t * __restrict__ b0, const uint8_t * __restrict__ b1,
        const float * __restrict__ xp, int mode, const float * __restrict__ lut,
        float & acc0, float & acc1) {
    float xv[MIX_QK];
    mix_load_x32(xp, xv);
    mix_block_accum_x(b0, xv, mode, lut, acc0);
    mix_block_accum_x(b1, xv, mode, lut, acc1);
}

// ---- LDS activation staging for the MoE fold ----
//
// The problem this solves. Lane L owns blocks L, L+MIX_WARP, ..., so lane L consumes
// activations xcol[32*L .. 32*L+31] -- consecutive lanes are 32 floats = 128 B apart.
// A warp-wide float4 activation request therefore touches 32 DISTINCT 128 B lines and
// uses only 16 B of each: 8 requests x 32 lines = 256 L1 line-lookups per
// warp-iteration, which is why ncu put this kernel at 89.7% (pre-iter-1) / 66.6%
// (post) L1/TEX throughput with DRAM at <8%. It is bound on L1 tag-lookup
// wavefronts, not bandwidth.
//
// But the warp's 32 lanes together consume a CONTIGUOUS 1024-float (4 KB) window per
// iteration -- only the lane->element map is a permutation. So load that window
// COALESCED (each request 32 consecutive float4 = 512 B = 4 lines: 8 x 4 = 32
// line-lookups, an 8x cut) into LDS, then let each lane read its own 32 floats back.
//
// The swizzle. A block's 32 floats are 8 float4 chunks; chunk c of the window's local
// block b lands at word `b*MIX_QK + 4*((c + b) & 7)`. A 128-bit LDS access is serviced
// in phases of 8 lanes (8 x 16 B = 128 B = all 32 banks), so conflict-freedom only
// needs the 8 lanes of a phase to hit disjoint bank quads:
//   read-back  (b = lane, c fixed): bank = 4*((c + lane) & 7); over any 8 consecutive
//     lanes (c + lane) & 7 takes all 8 values -> 8 disjoint quads = all 32 banks.
//   fill (b = g>>3, c = g&7, g = lane + i*MIX_WARP): within an 8-lane phase b is fixed
//     and c runs 0..7 -> again all 8 values.
// A rotate rather than the 36-float pad the obvious fix would use: same
// conflict-freedom, but 4 KB/warp instead of 4.5 KB and no dead words.
__device__ __forceinline__ int mix_swz_word(int b, int c) {
    return b * MIX_QK + 4 * ((c + b) & (MIX_QK / 4 - 1));
}

// Stage `nblk` blocks' worth of a contiguous activation window into this warp's LDS
// buffer with coalesced float4 loads. Bit-exact: the SAME f32 values reach the fold,
// only the route changes.
//
// The trip count is exact, never ragged: mix_validate_shape() forces in % 128 == 0, so
// nb % 4 == 0, and `nblk` is either MIX_WARP or nb % MIX_WARP -- a multiple of 4 either
// way -- hence nblk * (MIX_QK/4) is a multiple of MIX_WARP. No lane sits out a
// half-iteration, and the caller's __syncwarp() sees a fully-written buffer.
//
// The alignment test is warp-uniform (the window base is a multiple of MIX_QK floats =
// 128 B, so it inherits xcol's alignment), so the branch costs no divergence. The
// scalar arm is live: ggml can hand this path a strided/offset src1 column.
// FULL says the window is a whole MIX_WARP blocks, which makes `iters` the compile-time
// constant MIX_QK/4 and lets the fill unroll completely. It matters: with a runtime trip
// count nvcc emits a rolled loop with an unroll-remainder epilogue, and the model's shape
// (nb=224 = 7 full windows) would pay that even though it never runs a partial window.
template <bool FULL>
__device__ __forceinline__ void mix_stage_window(
        float * __restrict__ s, const float * __restrict__ xw, int nblk, int lane) {
    const int iters = FULL ? MIX_QK / 4 : nblk * (MIX_QK / 4) / MIX_WARP;
    if ((((uintptr_t) xw) & 15u) == 0) {
        const float4 * __restrict__ p4 = (const float4 *) xw;
        #pragma unroll
        for (int i = 0; i < iters; ++i) {
            const int g = lane + i * MIX_WARP;
            const float4 v = p4[g];
            float * __restrict__ d = s + mix_swz_word(g / (MIX_QK / 4), g & (MIX_QK / 4 - 1));
            d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < iters; ++i) {
            const int g = lane + i * MIX_WARP;
            const float * __restrict__ q = xw + 4 * g;
            float * __restrict__ d = s + mix_swz_word(g / (MIX_QK / 4), g & (MIX_QK / 4 - 1));
            d[0] = q[0]; d[1] = q[1]; d[2] = q[2]; d[3] = q[3];
        }
    }
}

// Read local block `b`'s 32 activations back out of the staged window, ascending j, as
// 8 conflict-free 128-bit LDS loads. `s` is 16 B aligned and mix_swz_word is a multiple
// of 4 words, so every float4 access is aligned.
__device__ __forceinline__ void mix_read_x32(
        const float * __restrict__ s, int b, float (&xv)[MIX_QK]) {
    #pragma unroll
    for (int c = 0; c < MIX_QK / 4; ++c) {
        const float4 v = *(const float4 *) (s + mix_swz_word(b, c));
        xv[4*c + 0] = v.x; xv[4*c + 1] = v.y;
        xv[4*c + 2] = v.z; xv[4*c + 3] = v.w;
    }
}

// One block of the MoE fold: stage the block's 32 activations ONCE, then fold them
// against this warp's ROWS `up` rows and (fused only) its ROWS `gate` rows.
//
// Measured motivation (H200 sm_90, ncu on the real decode geometry
// in=7168/out=2048/top-k 4): the FUSED kernel sat at 89.7% L1/TEX throughput with
// DRAM at 3.9% and issue slots at 21% -- bound on L1 tag-lookup wavefronts, not on
// bandwidth, arithmetic or occupancy. 31.7 M global-load sectors against 1.16 M
// requests is 27.2 sectors/request, i.e. essentially the 32-sector worst case,
// because consecutive lanes own blocks 32 floats apart: each float4 activation
// request touches 32 distinct sectors and uses only 16 of each 32 B.
//
// The caller used to invoke mix_block_accum() once per fold, each of which staged the
// SAME 32 activations from the SAME address. The comment there asserted nvcc would CSE
// those loads to one issue per element; the SASS says otherwise (288 static LDG.E.128
// for a loop that needs 36), so the fused path paid the 32-sector activation request
// four times per block. Staging once cuts activation requests 4x on the fused path and
// 2x on the unfused one. This is the same register-blocking mix_block_accum2() already
// does for the dense 2-D path, extended to the fused gate/up pair.
//
// Bit-exact: identical to the four mix_block_accum() calls it replaces. The same 32
// activation values arrive from the same addresses, and each row is folded by the same
// mix_block_accum_x into its own accumulator over the same fixed ascending-j order, so
// every acc keeps its exact left-fold summation order. Only the load *route* changes,
// not which terms enter which accumulator in what order -- required, because the
// correctness gate hashes the greedy token stream and any reassociation could flip it.
template <bool FUSE_GLU, int ROWS>
__device__ __forceinline__ void mix_moe_block_fold(
        // The row bases are a plain (non-__restrict__) array on purpose: a tail warp
        // whose row index is past `out` has that slot clamped onto row0 by the caller,
        // so asserting no-alias here would be a lie for that shape. Every base is
        // read-only const, so permitting the aliasing costs nothing.
        const uint8_t * const (&rowbase)[ROWS], const uint8_t * const (&growbase)[ROWS],
        const float (&xv)[MIX_QK], int b,
        int mode, int gmode, const float * __restrict__ lut,
        float (&acc)[ROWS], float (&gacc)[ROWS]) {
    const int64_t off = (int64_t) b * MIX_BLOCK_BYTES;
    // Issue all 2*ROWS weight-block loads before any FMA. Leaving the loads inside
    // per-row accumulate calls let nvcc serialise row i's load behind row i-1's
    // 32-term fold, which on a latency-bound kernel is the whole cost. The
    // activations arrive pre-staged in `xv` (the caller reads them out of LDS), so
    // this block issues nothing to L1 but weights.
    MixBlock w[ROWS], gw[ROWS];
    #pragma unroll
    for (int i = 0; i < ROWS; ++i) w[i] = mix_block_load(rowbase[i] + off);
    if (FUSE_GLU) {
        #pragma unroll
        for (int i = 0; i < ROWS; ++i) gw[i] = mix_block_load(growbase[i] + off);
    }
    #pragma unroll
    for (int i = 0; i < ROWS; ++i) mix_block_fma(w[i], xv, mode, lut, acc[i]);
    if (FUSE_GLU) {
        #pragma unroll
        for (int i = 0; i < ROWS; ++i) mix_block_fma(gw[i], xv, gmode, lut + 2 * MIX_K, gacc[i]);
    }
}

// One MIX_WARP-block window of the MoE fold: stage this warp's activation window into
// LDS, then fold the lane's own block against it. FULL drops both the runtime trip count
// in the fill and the `lane < nblk` predicate, so the hot path carries neither.
template <bool FUSE_GLU, int ROWS, bool FULL>
__device__ __forceinline__ void mix_moe_window(
        float * __restrict__ sxw, const float * __restrict__ xcol, int base, int nblk, int lane,
        const uint8_t * const (&rowbase)[ROWS], const uint8_t * const (&growbase)[ROWS],
        int mode, int gmode, const float * __restrict__ lut,
        float (&acc)[ROWS], float (&gacc)[ROWS]) {
    mix_stage_window<FULL>(sxw, xcol + (int64_t) base * MIX_QK, nblk, lane);
    // Required, not warp-synchrony folklore: sm_90 schedules threads independently, so a
    // lane's LDS writes are not ordered against a sibling lane's reads without it.
    // Warp-scoped, so unlike __syncthreads() it does not couple the block's two warps and
    // cost the latency hiding iteration 2 showed this kernel lives on.
    __syncwarp();
    if (FULL || lane < nblk) {
        float xv[MIX_QK];
        mix_read_x32(sxw, lane, xv);
        mix_moe_block_fold<FUSE_GLU, ROWS>(rowbase, growbase, xv, base + lane,
                                           mode, gmode, lut, acc, gacc);
    }
    // WAR: a lane that races ahead to the next window's fill must not overwrite slots a
    // sibling lane is still reading.
    __syncwarp();
}

// The lane's block loop is unrolled by MIX_UNROLL into a SINGLE accumulator kept
// in the exact original block order (acc += dot(blk), stride MIX_WARP), so the
// f32 output is bit-for-bit identical to the un-unrolled path — required because
// the correctness gate hashes the greedy output, which a reassociated reduction
// would flip. This matvec is read-once (fixed DRAM volume) and latency-bound on
// the strided per-block weight loads; the only serial dependency is the cheap
// acc-add chain, while the MIX_UNROLL mix_block_dot() evaluations are mutually
// independent, so unrolling lets the compiler issue several blocks' weight loads
// before consuming them — exposing memory-level parallelism per lane without
// touching the summation order.
__global__ void mix_matvec_rocmfp2_kernel(
        const uint8_t * __restrict__ data, const nv_bfloat16 * __restrict__ book,
        const uint8_t * __restrict__ mode_ptr, const float * __restrict__ x,
        float * __restrict__ y, int in, int out,
        int64_t x_col_stride, int64_t y_col_stride) {
    const int warps_per_block = blockDim.x / MIX_WARP;
    const int warp = blockIdx.x * warps_per_block + (threadIdx.x / MIX_WARP);
    const int row  = warp * 2;                  // two output rows per warp
    const int lane = threadIdx.x % MIX_WARP;
    const int col  = blockIdx.y;
    // Widen the 2 * MIX_K bf16 codebook to f32 in LDS once per workgroup instead of
    // re-loading it from global per weight inside mix_block_accum. HOISTED ABOVE the
    // row early-return: __syncthreads() requires every thread of the workgroup to
    // arrive, and `row >= out` retires whole warps in the tail block.
    __shared__ float s_lut[2 * MIX_K];
    if ((int) threadIdx.x < 2 * MIX_K) {
        s_lut[threadIdx.x] = __bfloat162float(book[threadIdx.x]);
    }
    __syncthreads();
    if (row >= out) return;
    const int mode = (int) mode_ptr[0];
    const int nb   = in / MIX_QK;
    // `two` is warp-uniform, so the hot loop never diverges. It is false only for
    // the tail warp of an odd `out`; that warp reuses row0's base so every load
    // stays in bounds and acc1 is simply never stored.
    const bool two = (row + 1) < out;
    const uint8_t * rowbase0 = data + (int64_t) row * nb * MIX_BLOCK_BYTES;
    const uint8_t * rowbase1 = two ? data + (int64_t) (row + 1) * nb * MIX_BLOCK_BYTES
                                   : rowbase0;
    const float   * xc       = x + (int64_t) col * x_col_stride;
    // f32 accumulate of f32-dequantized weights * f32 activations. The dequant
    // is bit-exact vs the reference (validated in ~/p4-validate/hip). This is
    // slightly higher precision than the f16 dequant->cuBLAS fallback; on the
    // (underpowered, N=10) smoke suite the two land within a +/-2 greedy-flip
    // noise band, with every divergence a shared model-limited miss, a harness
    // answer-extraction artifact, or an HE formatting coin-flip -- no reasoning
    // regression. Kept f32 for simplicity/speed (no per-weight rounding ops).
    float acc0 = 0.0f, acc1 = 0.0f;
    int blk = lane;
    // Main body: MIX_UNROLL blocks per iteration, each accumulated in stride
    // order into the single acc. The blocks' byte loads are independent (only
    // the acc-add chain is serial), so unrolling overlaps their loads while the
    // summation order stays identical. Guard keeps all MIX_UNROLL in range.
    for (; blk + (MIX_UNROLL - 1) * MIX_WARP < nb; blk += MIX_UNROLL * MIX_WARP) {
        #pragma unroll
        for (int u = 0; u < MIX_UNROLL; ++u) {
            const int b = blk + u * MIX_WARP;
            mix_block_accum2(rowbase0 + (int64_t) b * MIX_BLOCK_BYTES,
                             rowbase1 + (int64_t) b * MIX_BLOCK_BYTES,
                             xc + b * MIX_QK, mode, s_lut, acc0, acc1);
        }
    }
    // Remainder: fewer than MIX_UNROLL strided blocks left for this lane.
    for (; blk < nb; blk += MIX_WARP) {
        mix_block_accum2(rowbase0 + (int64_t) blk * MIX_BLOCK_BYTES,
                         rowbase1 + (int64_t) blk * MIX_BLOCK_BYTES,
                         xc + blk * MIX_QK, mode, s_lut, acc0, acc1);
    }
    #pragma unroll
    for (int off = MIX_WARP/2; off > 0; off >>= 1) {
        acc0 += mix_warp_shfl_down(acc0, off);
        acc1 += mix_warp_shfl_down(acc1, off);
    }
    if (lane == 0) {
        const int64_t o = (int64_t) col * y_col_stride + row;
        y[o] = acc0;
        if (two) y[o + 1] = acc1;
    }
}

// ---- stream-sync-free fused MoE matvec (mul_mat_id) ----
// One warp per (output row, expert-slot, token). The expert index is read from
// the routing `ids` tensor ON DEVICE, so the whole qtype-106 mul_mat_id runs
// without the generic ggml_cuda_mul_mat_id fallback's host id-sort +
// cudaStreamSynchronize (which serialises decode AND blocks CUDA-graph capture
// of the FFN subgraph — the dominant cost of the wall-clock-timed ffn_compute).
// The per-output-row math is the SAME flat fold as mix_matvec_rocmfp2_kernel
// (identical mix_block_accum, identical summation order) for the resolved
// expert, so every output element is bit-for-bit identical to the per-expert
// slice path the fallback would take (the correctness gate hashes the greedy
// output, so any reassociation would flip a token).
//
// Pin occupancy to the seed's proven-optimal 12 waves/SIMD. The branchless
// mix_ue4m3 dropped VGPR 98->96, which would otherwise let the allocator raise
// occupancy to 16 waves -- the tier iter-3 measured at -5% (extra resident waves
// thrash the reused activation column out of L1). The pin is a bit-exact
// occupancy hint (no math change) that keeps the arithmetic win at the 12-wave
// optimum instead of accidentally tripping into the known-bad 16-wave tier.
#if defined(__HIP_PLATFORM_AMD__)
__attribute__((amdgpu_waves_per_eu(12, 12)))
#endif

// ---- fused DENSE 3-D-slice matvec (batched mul_mat over src0->ne[2]) ----
// The target's attention output projection reshapes attn_output_a to
// [group_dim, n_lora_o, n_out_group] and mul_mats it against a matching 3-D src1
// (deepseek4_graph.cpp:2122), so src1->ne[2] > 1 and the 2-D fused hook's
// `src1->ne[2] == 1` gate rejects it -- sending ~31% of the attention weight read to
// dequantize->cuBLAS, which reads MORE bytes than the f16 it replaces.
//
// This is the MoE kernel with one change: the slice index comes from blockIdx.y
// instead of a routing `ids[]` lookup. Everything else -- the data + slice*nb02
// stride, codebooks + slice*2*K, modes[slice], the two-rows-per-warp register
// blocking, the fixed ascending-j fold -- is unchanged, so each output element is
// bit-identical to the MoE path for the same weights.
__global__ void mix_matvec_rocmfp2_slice_kernel(
        const uint8_t * __restrict__ data, size_t nb02,
        const nv_bfloat16 * __restrict__ codebooks, const uint8_t * __restrict__ modes,
        const float * __restrict__ src1,
        float * __restrict__ dst, int in, int out, int ne11,
        int64_t src1_s1, int64_t src1_s2,     // element strides (float) over ne11, token
        int64_t dst_s1, int64_t dst_s2) {     // element strides (float) over slot, token
    const int warps_per_block = blockDim.x / MIX_WARP;
    const int warp  = blockIdx.x * warps_per_block + (threadIdx.x / MIX_WARP);
    const int row0  = warp * 2;                 // two output rows per warp
    const int lane  = threadIdx.x % MIX_WARP;
    const int slot  = blockIdx.y;               // DENSE: the slice index. It indexes
                                                // src1 AND dst as well as the weights --
                                                // zeroing it made every slice read slice-0
                                                // activations and overwrite slice-0 output.
    const int token = blockIdx.z;
    // slot = blockIdx.y and token = blockIdx.z, so `expert` -- and hence the codebook,
    // mode and expert data base -- is WORKGROUP-UNIFORM. That is what makes staging the
    // table in LDS legal: one table serves every warp in the block. Read the expert and
    // stage BEFORE the row0 early-return so all threads reach the __syncthreads()
    // (`row0 >= out` retires whole warps in the tail block).
    const int expert = slot;                    // DENSE: the weight slice is the grid slice,
                                                // not a routing lookup. This is the ONLY
                                                // change from the MoE kernel.
    __shared__ float s_lut[2 * MIX_K];
    if ((int) threadIdx.x < 2 * MIX_K) {
        s_lut[threadIdx.x] =
            __bfloat162float(codebooks[(int64_t) expert * 2 * MIX_K + threadIdx.x]);
    }
    __syncthreads();
    if (row0 >= out) return;
    const bool two  = (row0 + 1) < out;         // false only for an odd-out tail warp
    const uint8_t     * edata   = data + (int64_t) expert * nb02;
    const int           mode    = (int) modes[expert];
    const int           nb      = in / MIX_QK;
    const uint8_t     * rowbase0 = edata + (int64_t) row0 * nb * MIX_BLOCK_BYTES;
    // For an odd tail warp (no row1) reuse row0's base so the loads stay in-bounds;
    // acc1 is simply never written. out=hidden is even here so `two` is uniformly
    // true across the whole warp (no divergence in the hot loop).
    const uint8_t     * rowbase1 = two ? edata + (int64_t) (row0 + 1) * nb * MIX_BLOCK_BYTES
                                       : rowbase0;
    // src1 is [in, ne11, ntok]; the get_rows-equivalent row for (slot, token)
    // is token*ne11 + slot%ne11 — i.e. token column + the slot%ne11 broadcast.
    const float * xcol = src1 + (int64_t) token * src1_s2 + (int64_t) (slot % ne11) * src1_s1;
    // Two output rows in one warp. Each row is folded by the SAME mix_block_accum
    // that the single-row path uses (byte-identical inlined body, same fixed j
    // order, same acc-add chain) so acc0/acc1 are bit-for-bit identical to the
    // single-row kernel's output for those rows. The two calls per block share the
    // same __restrict__ xcol + col0, so the compiler CSEs the strided activation
    // loads to one issue per element — halving activation LSU issue on this partly
    // load-instruction-bound matvec — WITHOUT reordering either row's summation.
    float acc0 = 0.0f, acc1 = 0.0f;
    int blk = lane;
    for (; blk + (MIX_UNROLL - 1) * MIX_WARP < nb; blk += MIX_UNROLL * MIX_WARP) {
        #pragma unroll
        for (int u = 0; u < MIX_UNROLL; ++u) {
            const int b = blk + u * MIX_WARP;
            mix_block_accum(rowbase0 + (int64_t) b * MIX_BLOCK_BYTES, xcol, b * MIX_QK, mode, s_lut, acc0);
            mix_block_accum(rowbase1 + (int64_t) b * MIX_BLOCK_BYTES, xcol, b * MIX_QK, mode, s_lut, acc1);
        }
    }
    for (; blk < nb; blk += MIX_WARP) {
        mix_block_accum(rowbase0 + (int64_t) blk * MIX_BLOCK_BYTES, xcol, blk * MIX_QK, mode, s_lut, acc0);
        mix_block_accum(rowbase1 + (int64_t) blk * MIX_BLOCK_BYTES, xcol, blk * MIX_QK, mode, s_lut, acc1);
    }
    #pragma unroll
    for (int off = MIX_WARP/2; off > 0; off >>= 1) {
        acc0 += mix_warp_shfl_down(acc0, off);
        acc1 += mix_warp_shfl_down(acc1, off);
    }
    if (lane == 0) {
        dst[(int64_t) token * dst_s2 + (int64_t) slot * dst_s1 + row0] = acc0;
        if (two) dst[(int64_t) token * dst_s2 + (int64_t) slot * dst_s1 + row0 + 1] = acc1;
    }
}

// FUSE_GLU folds the SECOND mul_mat_id of a DeepSeek4 gate/up pair plus the SwiGLU into this
// launch. The unfused shape is two matvec launches writing two [n_ff_exp, n_used, ntok]
// intermediates, then a third kernel reading both back to apply the GLU. qtype 107 never paid
// that: ggml_cuda_try_fuse_mul_mat_glu collapses the trio into one mul_mat_vec_q, which is why
// the profile showed 107 at 15050 launches against 106's 30100 plus a 28 ms swiglu_ds4 pass.
//
// TEMPLATED rather than a second kernel on purpose: both instantiations run the SAME
// accumulation over the SAME fixed block order, so each dot product is bit-identical to what
// the unfused path computes, and the fused result is bit-identical to
// swiglu_ds4(unfused_gate, unfused_up). A copy-pasted kernel would only *probably* stay that way.
//
// Naming follows the mmvq fusion convention: the PRIMARY tensor is `up` (src0 of the surviving
// mul_mat_id) and `gate` arrives as the extra operand, because
// ggml_cuda_op_swiglu_ds4_single(gate, up, limit) is not symmetric -- silu() is applied to gate.
template <bool FUSE_GLU, int MIX_MOE_ROWS, int WARPS>
__global__ void mix_matvec_rocmfp2_moe_kernel(
        const uint8_t * __restrict__ data, size_t nb02,
        const nv_bfloat16 * __restrict__ codebooks, const uint8_t * __restrict__ modes,
        const float * __restrict__ src1, const int32_t * __restrict__ ids,
        float * __restrict__ dst, int in, int out, int n_experts, int ne11,
        int64_t ids_s0, int64_t ids_s1,       // element strides (int32) over slot, token
        int64_t src1_s1, int64_t src1_s2,     // element strides (float) over ne11, token
        int64_t dst_s1, int64_t dst_s2,       // element strides (float) over slot, token
        // FUSE_GLU only. The gate tensor's own registry entry -- separate codebooks and modes,
        // NOT assumed equal to up's. Producers may emit identical codebooks for the two
        // halves, but the kernel does not rely on that and staging both costs 32 B of LDS.
        const uint8_t * __restrict__ gdata, size_t gnb02,
        const nv_bfloat16 * __restrict__ gcodebooks, const uint8_t * __restrict__ gmodes,
        float glu_limit) {
    // WARPS is a template parameter, not blockDim.x/MIX_WARP, because it also sizes the
    // per-warp LDS activation buffer below -- a runtime warp count could index past it.
    // Both mul_mat_id wrappers launch WARPS * MIX_WARP threads.
    const int wib   = threadIdx.x / MIX_WARP;   // warp index within the block
    const int warp  = blockIdx.x * WARPS + wib;
    const int row0  = warp * MIX_MOE_ROWS;      // MIX_MOE_ROWS output rows per warp
    const int lane  = threadIdx.x % MIX_WARP;
    const int slot  = blockIdx.y;
    const int token = blockIdx.z;
    // slot = blockIdx.y and token = blockIdx.z, so `expert` -- and hence the codebook,
    // mode and expert data base -- is WORKGROUP-UNIFORM. That is what makes staging the
    // table in LDS legal: one table serves every warp in the block. Read the expert and
    // stage BEFORE the row0 early-return so all threads reach the __syncthreads()
    // (`row0 >= out` retires whole warps in the tail block).
    const int expert = ids[(int64_t) token * ids_s1 + (int64_t) slot * ids_s0];
    // Routed ids come from the router's top-k, which is in-range by construction -- but this
    // kernel deliberately bypasses the generic host-side id sort that used to sit between the
    // routing tensor and the weights, so it must not turn a sentinel (-1), a padded batch
    // slot, or corrupted routing into an unchecked OOB read of codebooks/modes/data. Degrade
    // an invalid id to a ZERO contribution for this (token, slot): deterministic, and the
    // same thing a masked-out slot means. All threads still reach the barrier below.
    const bool bad_expert = expert < 0 || expert >= n_experts;
    // Both tables in one array so the staging stays a single guarded write per thread and one
    // barrier. Gate's table occupies [2*MIX_K, 4*MIX_K).
    __shared__ float s_lut[FUSE_GLU ? 4 * MIX_K : 2 * MIX_K];
    // One MIX_WARP-block activation window per warp (4 KB), swizzled -- see
    // mix_swz_word. PER-WARP rather than shared by the whole block on purpose: both
    // warps do walk the identical block sequence, so one buffer would serve both, but
    // that needs __syncthreads() in the hot loop and this block has only 2 warps.
    // Iteration 2 measured that trading warp-level independence for fewer loads loses
    // here (MIX_MOE_ROWS=4 on the fused kernel: 1.6x fewer sectors, 0.96x the speed),
    // and cross-warp sharing would only take activation line-lookups 32 -> 16 after
    // coalescing has already taken them 256 -> 32. Not worth a block barrier.
    __shared__ __align__(16) float s_xw[WARPS][MIX_WARP * MIX_QK];
    if (!bad_expert && (int) threadIdx.x < 2 * MIX_K) {
        s_lut[threadIdx.x] =
            __bfloat162float(codebooks[(int64_t) expert * 2 * MIX_K + threadIdx.x]);
    }
    if (FUSE_GLU) {
        const int gt = (int) threadIdx.x - 2 * MIX_K;
        if (!bad_expert && gt >= 0 && gt < 2 * MIX_K) {
            s_lut[2 * MIX_K + gt] =
                __bfloat162float(gcodebooks[(int64_t) expert * 2 * MIX_K + gt]);
        }
    }
    __syncthreads();
    if (row0 >= out) return;
    if (bad_expert) {
        if (lane == 0) {
            const int64_t o = (int64_t) token * dst_s2 + (int64_t) slot * dst_s1 + row0;
            #pragma unroll
            for (int i = 0; i < MIX_MOE_ROWS; ++i) {
                if (row0 + i < out) dst[o + i] = 0.0f;
            }
        }
        return;
    }

    const uint8_t     * edata   = data + (int64_t) expert * nb02;
    const int           mode    = (int) modes[expert];
    const int           nb      = in / MIX_QK;
    // Gate shares the shape, the expert and the row indices -- only the bytes and the table
    // differ -- so it reuses `nb`, `row0` and the same activation column below.
    const uint8_t * gedata = FUSE_GLU ? gdata + (int64_t) expert * gnb02 : nullptr;
    const int       gmode  = FUSE_GLU ? (int) gmodes[expert] : 0;
    // Row bases for this warp's MIX_MOE_ROWS rows. A tail warp whose row index is past
    // `out` CLAMPS onto row0 so its loads stay in bounds; that accumulator is simply
    // never stored. `out` is a multiple of MIX_MOE_ROWS for this model (2048), so the
    // clamp is uniform across the warp and costs no divergence in the hot loop -- but it
    // must stay correct for a ragged `out`, which neither the model nor the microbench
    // exercises.
    const uint8_t * rowbase[MIX_MOE_ROWS];
    const uint8_t * growbase[MIX_MOE_ROWS];
    #pragma unroll
    for (int i = 0; i < MIX_MOE_ROWS; ++i) {
        const int r = (row0 + i) < out ? row0 + i : row0;
        rowbase[i]  = edata + (int64_t) r * nb * MIX_BLOCK_BYTES;
        growbase[i] = FUSE_GLU ? gedata + (int64_t) r * nb * MIX_BLOCK_BYTES : nullptr;
    }
    // src1 is [in, ne11, ntok]; the get_rows-equivalent row for (slot, token)
    // is token*ne11 + slot%ne11 — i.e. token column + the slot%ne11 broadcast.
    const float * xcol = src1 + (int64_t) token * src1_s2 + (int64_t) (slot % ne11) * src1_s1;
    // Two output rows in one warp. Each row is folded by the SAME mix_block_accum_x
    // that the single-row path uses (byte-identical inlined body, same fixed j
    // order, same acc-add chain) so acc0/acc1 are bit-for-bit identical to the
    // single-row kernel's output for those rows. The folds share ONE explicit
    // activation stage per block (mix_moe_block_fold) instead of relying on the
    // compiler to CSE a separate stage per fold: it does not (the SASS showed 288
    // static LDG.E.128 for a loop that needs 36), and on this L1-wavefront-bound
    // kernel that stage is the most expensive request in the loop. Sharing it does
    // NOT reorder either row's summation.
    float acc[MIX_MOE_ROWS];
    // Gate accumulates in its own registers over the SAME block order, so gacc is bit-identical
    // to what the separate gate launch produced. ALL folds consume ONE staged copy of the
    // block's activations, so the fused form reads the activation column once for
    // 2*MIX_MOE_ROWS rows instead of once per row -- on a launch measured at 89.7% L1/TEX
    // throughput with DRAM at 3.9%, that is the second saving after the launch itself, and
    // the larger of the two.
    float gacc[MIX_MOE_ROWS];
    #pragma unroll
    for (int i = 0; i < MIX_MOE_ROWS; ++i) { acc[i] = 0.0f; gacc[i] = 0.0f; }
    // Walk the row in MIX_WARP-block windows. Lane L still folds blocks L, L+MIX_WARP,
    // ... in ascending order into its own accumulator, exactly as the strided loop this
    // replaces did -- so every acc keeps its bit-identical left-fold summation order.
    // What changed is only how the activations reach `xv`: staged once per window,
    // coalesced, through LDS instead of 8 32-line-wide global requests per block.
    //
    // Windowing (rather than the old `blk += MIX_WARP` stride) is what makes the
    // staging legal at all: with nb % MIX_WARP != 0 the strided loop lets lanes exit at
    // different trip counts, so a departed lane would stop contributing to the
    // cooperative fill while its neighbours read slots nobody wrote.
    //
    // This also retires the MIX_UNROLL path, which was dead code: its guard
    // `blk + 7*MIX_WARP < nb` is false on the first test at the model's nb=224, so every
    // block already went through the tail loop. Ordering is unaffected either way --
    // the unrolled body visited the same ascending b sequence.
    float * __restrict__ sxw = s_xw[wib];
    const int nfull = nb / MIX_WARP;
    for (int w = 0; w < nfull; ++w) {
        mix_moe_window<FUSE_GLU, MIX_MOE_ROWS, true>(
                sxw, xcol, w * MIX_WARP, MIX_WARP, lane,
                rowbase, growbase, mode, gmode, s_lut, acc, gacc);
    }
    // At most ONE partial window, and only for shapes with nb % MIX_WARP != 0 (the model's
    // nb=224 has none). Its block count is a multiple of 4 -- mix_validate_shape forces
    // in % 128 == 0 -- so the fill's trip count stays exact there too.
    if (nb % MIX_WARP) {
        mix_moe_window<FUSE_GLU, MIX_MOE_ROWS, false>(
                sxw, xcol, nfull * MIX_WARP, nb % MIX_WARP, lane,
                rowbase, growbase, mode, gmode, s_lut, acc, gacc);
    }
    #pragma unroll
    for (int off = MIX_WARP/2; off > 0; off >>= 1) {
        #pragma unroll
        for (int i = 0; i < MIX_MOE_ROWS; ++i) {
            acc[i] += mix_warp_shfl_down(acc[i], off);
            if (FUSE_GLU) gacc[i] += mix_warp_shfl_down(gacc[i], off);
        }
    }
    if (lane == 0) {
        const int64_t o = (int64_t) token * dst_s2 + (int64_t) slot * dst_s1 + row0;
        #pragma unroll
        for (int i = 0; i < MIX_MOE_ROWS; ++i) {
            if (row0 + i >= out) continue;      // ragged-`out` tail row: clamped, not owned
            // The SAME function the standalone swiglu_ds4 kernel applies, on inputs
            // bit-identical to the ones it would have read back from the two intermediates.
            dst[o + i] = FUSE_GLU ? ggml_cuda_op_swiglu_ds4_single(gacc[i], acc[i], glu_limit)
                                  : acc[i];
        }
    }
}

// Launch the sync-free MoE matvec for a qtype-106 mul_mat_id. Resolves the
// tensor's registry entry (base/nb02/codebooks/modes for ALL experts) from vx;
// the per-expert index is read on device. Returns false if the tensor is not
// registered (caller keeps the generic fallback). ne11 is the src1 broadcast
// dim (1 for decode). All *_s* are element strides (see kernel).
bool ggml_cuda_rocmfp2_mix_mul_mat_vec_3d(
        const void * vx, const float * src1, float * dst,
        int in, int out, int nslices, int ntokens,
        int64_t src1_token_stride, int64_t src1_slice_stride,
        int64_t dst_token_stride,  int64_t dst_slice_stride,
        cudaStream_t stream) {
    std::lock_guard<std::recursive_mutex> dispatch_lock(g_mix_mtx);
    MixEntry e;
    int slice0;
    if (!mix_lookup_expert_base(vx, e, slice0)) {
        return false;  // not registered -> caller keeps the dequant fallback
    }
    // The registry must cover every slice this launch will index via blockIdx.y.
    // Registering a 3-D dense tensor with n_experts < ne02 would silently read a
    // codebook belonging to another tensor, so refuse rather than corrupt.
    if (slice0 != 0 || in != e.in || out != e.out || nslices <= 0 ||
        e.n_experts < nslices || ntokens <= 0) {
        return false;
    }
    const int warps_per_block = 2;
    const int threads = warps_per_block * MIX_WARP;
    const int rows_per_block = 2 * warps_per_block;
    dim3 grid((out + rows_per_block - 1) / rows_per_block, nslices, ntokens);
    // Stride mapping, and the reason this wrapper names its arguments by MEANING.
    // The kernel inherits the MoE indexing `src1 + token*src1_s2 + (slot%ne11)*src1_s1`
    // and `dst[token*dst_s2 + slot*dst_s1]`, i.e. **_s1 is the SLOT stride and _s2 the
    // TOKEN stride** -- the opposite of the ne[1]/ne[2] reading the names suggest. Passing
    // ggml's nb[1]/nb[2] straight through (and ne11=1) silently dropped the slice offset
    // entirely, because `slot % 1 == 0`, and produced completely wrong values that the
    // per-slice comparison test caught. ne11 must be >= nslices for `slot % ne11` to be
    // the identity on the slice index.
    mix_matvec_rocmfp2_slice_kernel<<<grid, dim3(threads), 0, stream>>>(
        (const uint8_t *) e.base, e.nb02, e.codebooks, e.modes,
        src1, dst, in, out, /*ne11=*/nslices,
        /*src1_s1=slot  */ src1_slice_stride, /*src1_s2=token*/ src1_token_stride,
        /*dst_s1 =slot  */ dst_slice_stride,  /*dst_s2 =token*/ dst_token_stride);
    return true;
}

bool ggml_cuda_rocmfp2_mix_mul_mat_id(
        const void * vx, const float * src1, const int32_t * ids, float * dst,
        int in, int out, int n_expert_used, int n_tokens, int ne11,
        int64_t ids_s0, int64_t ids_s1,
        int64_t src1_s1, int64_t src1_s2,
        int64_t dst_s1, int64_t dst_s2, cudaStream_t stream) {
    std::lock_guard<std::recursive_mutex> dispatch_lock(g_mix_mtx);
    MixEntry e;
    int expert0;
    if (!mix_lookup_expert_base(vx, e, expert0)) {
        return false;  // not registered -> caller falls back to sort + dequant
    }
    if (expert0 != 0 || in != e.in || out != e.out ||
        n_expert_used <= 0 || n_tokens <= 0 || ne11 <= 0) {
        return false;
    }
    const int warps_per_block = 2;               // 64 threads (mirror the mmvq path)
    const int threads = warps_per_block * MIX_WARP;
    // MIX_MOE_ROWS_UNFUSED output rows per warp (register-blocked activation reuse),
    // so a workgroup of `warps_per_block` warps covers that many times as many rows.
    const int rows_per_block = MIX_MOE_ROWS_UNFUSED * warps_per_block;
    dim3 grid((out + rows_per_block - 1) / rows_per_block, n_expert_used, n_tokens);
    mix_matvec_rocmfp2_moe_kernel<false, MIX_MOE_ROWS_UNFUSED, warps_per_block><<<grid, dim3(threads), 0, stream>>>(
        (const uint8_t *) e.base, e.nb02, e.codebooks, e.modes,
        src1, ids, dst, in, out, e.n_experts, ne11,
        ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2,
        nullptr, 0, nullptr, nullptr, 0.0f);
    return true;
}

// Fused gate/up + SwiGLU for a DeepSeek4 qtype-106 expert pair. `vx_up` is the surviving
// mul_mat_id's src0 and `vx_gate` the collapsed one, matching the mmvq fusion convention
// (swiglu_ds4 applies silu to GATE, so the two are not interchangeable).
//
// Refuses -- leaving the caller's unfused path intact -- unless BOTH tensors are registered and
// agree on shape. A registry miss on one half would otherwise fuse a decoded tensor with a
// garbage one, and the artifact loads either way, so this is the "fluent garbage" failure mode
// rather than a crash.
bool ggml_cuda_rocmfp2_mix_mul_mat_id_glu(
        const void * vx_up, const void * vx_gate,
        const float * src1, const int32_t * ids, float * dst,
        int in, int out, int n_expert_used, int n_tokens, int ne11,
        int64_t ids_s0, int64_t ids_s1,
        int64_t src1_s1, int64_t src1_s2,
        int64_t dst_s1, int64_t dst_s2,
        float glu_limit, cudaStream_t stream) {
    std::lock_guard<std::recursive_mutex> dispatch_lock(g_mix_mtx);
    MixEntry eu, eg;
    int expert0_u, expert0_g;
    if (!mix_lookup_expert_base(vx_up, eu, expert0_u) ||
        !mix_lookup_expert_base(vx_gate, eg, expert0_g)) {
        return false;
    }
    if (expert0_u != 0 || expert0_g != 0 || in != eu.in || out != eu.out ||
        eu.in != eg.in || eu.out != eg.out || eu.n_experts != eg.n_experts ||
        n_expert_used <= 0 || n_tokens <= 0 || ne11 <= 0) {
        return false;   // not a matched pair; the caller's two-launch path is still correct
    }
    const int warps_per_block = 2;
    const int threads = warps_per_block * MIX_WARP;
    const int rows_per_block = MIX_MOE_ROWS_FUSED * warps_per_block;
    dim3 grid((out + rows_per_block - 1) / rows_per_block, n_expert_used, n_tokens);
    mix_matvec_rocmfp2_moe_kernel<true, MIX_MOE_ROWS_FUSED, warps_per_block><<<grid, dim3(threads), 0, stream>>>(
        (const uint8_t *) eu.base, eu.nb02, eu.codebooks, eu.modes,
        src1, ids, dst, in, out, eu.n_experts, ne11,
        ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2,
        (const uint8_t *) eg.base, eg.nb02, eg.codebooks, eg.modes, glu_limit);
    return true;
}

bool ggml_cuda_rocmfp2_mix_registered(const void * vx) {
    MixEntry e;
    int expert;
    return mix_lookup_expert_base(vx, e, expert);
}

void ggml_cuda_rocmfp2_mix_registry_lock() {
    g_mix_mtx.lock();
}

void ggml_cuda_rocmfp2_mix_registry_unlock() {
    g_mix_mtx.unlock();
}

bool ggml_cuda_rocmfp2_mix_mmq_info(
        const void * vx, const void ** codebooks, const uint8_t ** modes) {
    if (!codebooks || !modes) {
        return false;
    }
    MixEntry e;
    int expert;
    size_t byte_offset = 0;
    if (!mix_lookup(vx, e, expert, &byte_offset) ||
        byte_offset % MIX_BLOCK_BYTES != 0) {
        return false;
    }
    *codebooks = e.codebooks + (size_t) expert * 2 * MIX_K;
    *modes = e.modes + expert;
    return true;
}

bool ggml_cuda_rocmfp2_mix_mul_mat_vec(
        const void * vx, const float * x, float * y,
        int in, int out, int ncols,
        int64_t x_col_stride, int64_t y_col_stride, cudaStream_t stream) {
    std::lock_guard<std::recursive_mutex> dispatch_lock(g_mix_mtx);
    MixEntry e;
    int expert;
    if (!mix_lookup_expert_base(vx, e, expert)) {
        return false;  // not registered -> caller falls back to dequant->cuBLAS
    }
    if (in != e.in || out != e.out || ncols <= 0) {
        return false;
    }
    const nv_bfloat16 * book = e.codebooks + (size_t) expert * 2 * 4;
    const uint8_t * mode_ptr = e.modes + expert;
    // Launch-config-only occupancy lever (bit-exact: one warp still owns one
    // row, per-row summation order unchanged). 2 warps/block (64 threads) is the
    // finest grouping that keeps a sibling warp per block for latency hiding
    // (1 warp/block regressed — attempt #1), while halving the workgroup size to
    // give the scheduler finer packing/tail balance on this BW-bound matvec.
    const int warps_per_block = 2;               // 64 threads
    const int threads = warps_per_block * MIX_WARP;
    // Each warp now owns TWO rows, so a block covers 2 * warps_per_block of them.
    const int rows_per_block = 2 * warps_per_block;
    dim3 grid((out + rows_per_block - 1) / rows_per_block, ncols, 1);
    mix_matvec_rocmfp2_kernel<<<grid, dim3(threads), 0, stream>>>(
        (const uint8_t *) vx, book, mode_ptr, x, y, in, out, x_col_stride, y_col_stride);
    return true;
}
