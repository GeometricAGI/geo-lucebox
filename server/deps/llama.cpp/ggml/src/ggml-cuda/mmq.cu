#include "common.cuh"
#include "ggml-cuda.h"
#include "mmq.cuh"
#include "mmq_big.h"
#include "quantize.cuh"
#include "mmid.cuh"
#include "rocmfp2_mix.cuh"
#include "rocmfp3_mix.cuh"
#include "gqh.cuh"

static thread_local size_t g_mmq_launch_count = 0;

extern "C" size_t ggml_backend_cuda_get_mmq_launch_count(void) {
    return g_mmq_launch_count;
}

namespace {

class mix_registry_dispatch_guard {
public:
    explicit mix_registry_dispatch_guard(ggml_type type) : type_(type) {
        if (type_ == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
            ggml_cuda_rocmfp2_mix_registry_lock();
        } else if (type_ == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
            ggml_cuda_rocmfp3_mix_registry_lock();
        }
    }

    ~mix_registry_dispatch_guard() {
        if (type_ == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
            ggml_cuda_rocmfp2_mix_registry_unlock();
        } else if (type_ == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
            ggml_cuda_rocmfp3_mix_registry_unlock();
        }
    }

    mix_registry_dispatch_guard(const mix_registry_dispatch_guard &) = delete;
    mix_registry_dispatch_guard & operator=(
        const mix_registry_dispatch_guard &) = delete;

private:
    ggml_type type_;
};

}  // namespace

// Big-tile dispatch (see mmq_big.h): the default instances for the dense
// hybrid types are 64x64 (GGML_CUDA_MMQ_SMALL_TILE, tuned for spec-decode
// verify widths); at prefill widths the narrow x-tile re-streams the weights,
// so wide batches take the 128x128 twin instances instead. RDNA4 only: the
// measurement is from gfx1201, and gfx1151 keeps its existing behavior.
// LUCE_MMQ_BIG_PREFILL=0 disables.
static bool lucebox_mmq_big_tile_take(const ggml_type type, const int64_t ncols_dst) {
    static const bool enabled = []() {
        const char * e = getenv("LUCE_MMQ_BIG_PREFILL");
        return !(e && e[0] == '0' && e[1] == '\0');
    }();
    // Measured crossover on gfx1201 (iq4_xs 17408x5120): small tile wins to
    // N=64, tie at 128, big wins 16-18% at 512. Take big only where it is a
    // clear win.
    if (!enabled || ncols_dst < 256) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        return false;
    }
    switch (type) {
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    const bool is_mix_type =
        args.type_x == GGML_TYPE_Q2_1_ROCMFP2_MIX ||
        args.type_x == GGML_TYPE_Q3_1_ROCMFP3_MIX;
    GGML_ASSERT(!is_mix_type || (args.mix_codebooks && args.mix_modes));
    ++g_mmq_launch_count;
    if (lucebox_mmq_big_tile_take(args.type_x, args.ncols_dst)) {
        switch (args.type_x) {
            case GGML_TYPE_IQ4_XS: mul_mat_q_case_big_iq4_xs(ctx, &args, stream); return;
            case GGML_TYPE_Q4_K:   mul_mat_q_case_big_q4_k  (ctx, &args, stream); return;
            case GGML_TYPE_Q5_K:   mul_mat_q_case_big_q5_k  (ctx, &args, stream); return;
            case GGML_TYPE_Q6_K:   mul_mat_q_case_big_q6_k  (ctx, &args, stream); return;
            case GGML_TYPE_Q8_0:   mul_mat_q_case_big_q8_0  (ctx, &args, stream); return;
            default: break;
        }
    }
    switch (args.type_x) {
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
            mul_mat_q_case<GGML_TYPE_Q4_0_ROCMFP4_FAST>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0_ROCMFP2:
            mul_mat_q_case<GGML_TYPE_Q2_0_ROCMFP2>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_1_ROCMFP2_MIX:
            mul_mat_q_case<GGML_TYPE_Q2_1_ROCMFP2_MIX>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_1_ROCMFP3_MIX:
            mul_mat_q_case<GGML_TYPE_Q3_1_ROCMFP3_MIX>(ctx, args, stream);
            break;
        case GGML_TYPE_GQH3:
            mul_mat_q_case<GGML_TYPE_GQH3>(ctx, args, stream);
            break;
        case GGML_TYPE_GQH4:
            mul_mat_q_case<GGML_TYPE_GQH4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_0_ROCMFPX:
            mul_mat_q_case<GGML_TYPE_Q3_0_ROCMFPX>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
#ifndef GGML_CUDA_BLACKWELL_CONSUMER
            if (args.type_x == GGML_TYPE_MXFP4) {
                mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            } else {
                mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            }
#else
            GGML_ABORT("FP4 quantization requires sm_120a, not supported on consumer Blackwell (SM 12.0)");
#endif
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_mul_mat_q_impl(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src0_pair,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst,
        ggml_tensor * dst_pair) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.
    GGML_ASSERT((src0_pair == nullptr) == (dst_pair == nullptr));
    if (src0_pair) {
        GGML_ASSERT(src0_pair->type == src0->type);
        GGML_ASSERT(dst_pair->type == dst->type);
        GGML_ASSERT(ggml_are_same_shape(src0_pair, src0));
        GGML_ASSERT(ggml_are_same_stride(src0_pair, src0));
        GGML_ASSERT(ggml_are_same_shape(dst_pair, dst));
        GGML_ASSERT(ggml_are_same_stride(dst_pair, dst));
    }

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // Keep mix side data alive until every MMQ launch using it is enqueued.
    // Registry teardown takes the same lock and drains the owning device before
    // freeing those buffers.
    mix_registry_dispatch_guard mix_guard(src0->type);
    const void * mix_codebooks_raw = nullptr;
    const uint8_t * mix_modes = nullptr;
    if (src0->type == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
        GGML_ASSERT(ggml_cuda_rocmfp2_mix_mmq_info(
            src0->data, &mix_codebooks_raw, &mix_modes));
    } else if (src0->type == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
        GGML_ASSERT(ggml_cuda_rocmfp3_mix_mmq_info(
            src0->data, &mix_codebooks_raw, &mix_modes));
    }
    // GQH per-tensor header -> int8 grid LUT + tile-scale prefactor. Unlike the
    // mix registries this needs no dispatch lock: the result is 20 bytes copied
    // into the kernarg struct here, so nothing the kernel reads can be torn down
    // underneath it. should_use_mmq plus ggml_cuda_gqh_mmq_eligible already
    // refused every type/grid this cannot resolve, hence the assert.
    gqh_mmq_params gqh_params = {};
    if (ggml_cuda_gqh_mmq_type(src0->type)) {
        GGML_ASSERT(ggml_cuda_gqh_mmq_info(
            src0->type, src0->data, gqh_params.lut, &gqh_params.dscale));
    }
    const nv_bfloat16 * mix_codebooks =
        reinterpret_cast<const nv_bfloat16 *>(mix_codebooks_raw);

    // Temporary weight tensors may have an allocation tail read by tiled MMQ.
    // Clear it once for each projection before launching either multiply.
    const ggml_tensor * weights[] = {src0, src0_pair};
    for (const ggml_tensor * weight : weights) {
        if (!weight ||
            ggml_backend_buffer_get_usage(weight->buffer) !=
                GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            continue;
        }
        const size_t size_data  = ggml_nbytes(weight);
        const size_t size_alloc =
            ggml_backend_buffer_get_alloc_size(weight->buffer, weight);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(weight));
            GGML_ASSERT(!weight->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) weight->data + size_data, 0,
                                       size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) &&
         ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);
    // Keep the established small-batch tuning hook from #503. The default
    // remains stream-k; positive values opt small verify widths into the
    // lower-overhead data-parallel path.
    static const int luce_mmq_dp_max_ne1 = []() {
        const char * value = getenv("LUCE_MMQ_DP_MAX_NE1");
        return value ? atoi(value) : 0;
    }();
    if (use_stream_k && ne11 <= luce_mmq_dp_max_ne1) {
        use_stream_k = false;
    }

    // TODO: tighter pool buffer size vs q8 path
    const bool use_native_mxfp4 = blackwell_mma_available(cc) && src0->type == GGML_TYPE_MXFP4;
    const bool grouped_src = !ids && ggml_mul_mat_is_grouped_src(dst);
    const ggml_tensor * grouped_physical = grouped_src ? src1->view_src : nullptr;
    int64_t grouped_width = 0;
    int64_t grouped_row_stride = 0;
    int64_t grouped_plane_stride = 0;
    if (grouped_src) {
        const int64_t groups = ggml_mul_mat_grouped_src_groups(dst);
        GGML_ASSERT(groups > 1 && ne10 % groups == 0);
        grouped_width = ne10 / groups;

        if (grouped_physical) {
            GGML_ASSERT(grouped_physical->type == GGML_TYPE_F32);
            GGML_ASSERT(grouped_physical->ne[0] == grouped_width);
            GGML_ASSERT(grouped_physical->ne[1] == ne11);
            GGML_ASSERT(grouped_physical->ne[2] == groups);
            GGML_ASSERT(grouped_physical->ne[3] == 1);
            grouped_row_stride = grouped_physical->nb[1] / ts_src1;
            grouped_plane_stride = grouped_physical->nb[2] / ts_src1;
        } else {
            // A scheduler copy between unlike backends materializes the raw
            // bytes of the logical 2-D view as a standalone tensor. The bytes
            // retain the grouped physical ordering, while view_src metadata
            // cannot cross the allocation boundary. ggml_mul_mat_grouped_src
            // guarantees a contiguous [width, rows, groups] physical source,
            // so reconstruct those two strides from the op's group count.
            grouped_row_stride = grouped_width;
            grouped_plane_stride = grouped_width * ne11;
        }
        GGML_ASSERT(!use_native_mxfp4);
    }

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (grouped_src) {
                quantize_mmq_q8_1_grouped_cuda(
                    src1_d, src1_q8_1.get(), src0->type,
                    ne10, grouped_width,
                    grouped_row_stride, grouped_plane_stride,
                    ne10_padded, ne11, stream);
            } else if (use_native_mxfp4) {
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_mxfp4_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_mxfp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) /
                                    (8 * QK_MXFP4 * sizeof(int))  // block_fp4_mmq holds 256 values (8 blocks of 32)
                                :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr,
            mix_codebooks, mix_modes, gqh_params, dst_d,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            use_stream_k, ne1};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        if (src0_pair) {
            mmq_args pair_args = args;
            pair_args.x = (const char *) src0_pair->data;
            pair_args.dst = (float *) dst_pair->data;
            if (src0_pair->type == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
                const void * pair_codebooks = nullptr;
                const uint8_t * pair_modes = nullptr;
                GGML_ASSERT(ggml_cuda_rocmfp2_mix_mmq_info(
                    src0_pair->data, &pair_codebooks, &pair_modes));
                pair_args.mix_codebooks =
                    reinterpret_cast<const nv_bfloat16 *>(pair_codebooks);
                pair_args.mix_modes = pair_modes;
            } else if (src0_pair->type == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
                const void * pair_codebooks = nullptr;
                const uint8_t * pair_modes = nullptr;
                GGML_ASSERT(ggml_cuda_rocmfp3_mix_mmq_info(
                    src0_pair->data, &pair_codebooks, &pair_modes));
                pair_args.mix_codebooks =
                    reinterpret_cast<const nv_bfloat16 *>(pair_codebooks);
                pair_args.mix_modes = pair_modes;
            } else if (ggml_cuda_gqh_mmq_type(src0_pair->type)) {
                // The pair shares nothing per-tensor: each projection has its own
                // header, so re-resolve rather than reuse src0's.
                GGML_ASSERT(ggml_cuda_gqh_mmq_info(
                    src0_pair->type, src0_pair->data,
                    pair_args.gqh.lut, &pair_args.gqh.dscale));
            }
            ggml_cuda_mul_mat_q_switch_type(ctx, pair_args, stream);
        }
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    // Pad ids_dst by mmq_x to prevent OOB reads in the stream-k kernel.
    // The kernel loads a full mmq_x-wide tile from ids_dst (line 3709 in mmq.cuh)
    // without bounds-checking the load, only the write-back is bounded.
    const int64_t mmq_x_pad = (int64_t)get_mmq_x_max_host(cc);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows + mmq_x_pad);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    {
        // The sorter compacts negative (masked) owner routes out of the expert
        // ranges. Quantization still visits these fixed-capacity arrays, so
        // make the unused tail point at token zero instead of stale pool data.
        CUDA_CHECK(cudaMemsetAsync(ids_src1.get(), 0,
                                   ne_get_rows * sizeof(int32_t), stream));
        CUDA_CHECK(cudaMemsetAsync(ids_dst.get(), 0,
                                   (ne_get_rows + mmq_x_pad) * sizeof(int32_t),
                                   stream));
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    // The helper's fixed-capacity layout is part of MMQ's addressing contract.
    // A future compact route count must be produced and owned by the graph
    // builder; until then, quantize the complete initialized route array.
    const int64_t n_routes_quantized = ne_get_rows;
    const size_t nbytes_src1_q8_1 = n_routes_quantized*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    const int64_t ne11_flat = n_routes_quantized;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_mxfp4) {
            quantize_mmq_mxfp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t s12 = use_native_mxfp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (8 * QK_MXFP4 * sizeof(int)) :
                                           ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    const int64_t ncols_max = ne12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(),
        mix_codebooks, mix_modes, gqh_params, dst_d,
        ne00, ne01, ne_get_rows, s01, n_routes_quantized, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        use_stream_k, ncols_max};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
    if (src0_pair) {
        mmq_args pair_args = args;
        pair_args.x = (const char *) src0_pair->data;
        pair_args.dst = (float *) dst_pair->data;
        if (src0_pair->type == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
            const void * pair_codebooks = nullptr;
            const uint8_t * pair_modes = nullptr;
            GGML_ASSERT(ggml_cuda_rocmfp2_mix_mmq_info(
                src0_pair->data, &pair_codebooks, &pair_modes));
            pair_args.mix_codebooks =
                reinterpret_cast<const nv_bfloat16 *>(pair_codebooks);
            pair_args.mix_modes = pair_modes;
        } else if (src0_pair->type == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
            const void * pair_codebooks = nullptr;
            const uint8_t * pair_modes = nullptr;
            GGML_ASSERT(ggml_cuda_rocmfp3_mix_mmq_info(
                src0_pair->data, &pair_codebooks, &pair_modes));
            pair_args.mix_codebooks =
                reinterpret_cast<const nv_bfloat16 *>(pair_codebooks);
            pair_args.mix_modes = pair_modes;
        } else if (ggml_cuda_gqh_mmq_type(src0_pair->type)) {
            GGML_ASSERT(ggml_cuda_gqh_mmq_info(
                src0_pair->type, src0_pair->data,
                pair_args.gqh.lut, &pair_args.gqh.dscale));
        }
        ggml_cuda_mul_mat_q_switch_type(ctx, pair_args, stream);
    }
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst) {
    ggml_cuda_mul_mat_q_impl(ctx, src0, nullptr, src1, ids, dst, nullptr);
}

void ggml_cuda_mul_mat_q_pair(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0_a,
        const ggml_tensor * src0_b,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst_a,
        ggml_tensor * dst_b) {
    ggml_cuda_mul_mat_q_impl(
        ctx, src0_a, src0_b, src1, ids, dst_a, dst_b);
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool use_stream_k = ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc))
                            && src1_ncols == ne11;
    mix_registry_dispatch_guard mix_guard(src0->type);
    const void * mix_codebooks_raw = nullptr;
    const uint8_t * mix_modes = nullptr;
    if (src0->type == GGML_TYPE_Q2_1_ROCMFP2_MIX) {
        GGML_ASSERT(ggml_cuda_rocmfp2_mix_mmq_info(
            src0_dd_i, &mix_codebooks_raw, &mix_modes));
    } else if (src0->type == GGML_TYPE_Q3_1_ROCMFP3_MIX) {
        GGML_ASSERT(ggml_cuda_rocmfp3_mix_mmq_info(
            src0_dd_i, &mix_codebooks_raw, &mix_modes));
    }
    // The split-buffer op receives a ROW SLICE, which ggml_gqh_lookup resolves to
    // its owning tensor (that is why it takes an interior pointer at all).
    gqh_mmq_params gqh_params = {};
    if (ggml_cuda_gqh_mmq_type(src0->type)) {
        GGML_ASSERT(ggml_cuda_gqh_mmq_info(
            src0->type, src0_dd_i, gqh_params.lut, &gqh_params.dscale));
    }
    const mmq_args args = {
        src0_dd_i, src0->type, (const int *) src1_ddq_i, nullptr, nullptr,
        reinterpret_cast<const nv_bfloat16 *>(mix_codebooks_raw), mix_modes,
        gqh_params, dst_dd_i,
        ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, src1_ncols};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_padded_row_size);
}

#ifndef GGML_CUDA_MIX_MMQ_DEFAULT
// OFF unless a model backend opts in (MuseBackend::init does). The measured
// win — muse-glimmer r1 at 103/122 and 81.0 tok/s, against 101/122 and 43.5
// with it off — is muse's, and muse's mix tensors are dense ggml_mul_mat.
// A process-wide default would extend that result to models it was never
// measured on, which is how the "inert on DS4" mistake happened.
#define GGML_CUDA_MIX_MMQ_DEFAULT false
#endif

// ── Batched path for the mix qtypes (105/106) ───────────────────────────
// Without it, ne11 > 1 falls back to dequantize-to-bf16 + dense GEMM, which
// discards the whole point of a sub-4 bpw artifact for the duration of the
// multiply (measured: 48% of a 16-token speculative verify's GPU time inside
// dequantize_rocmfp{2,3}_mix_kernel).
//
// Runtime-settable rather than a function-local static so a single process can
// A/B both paths; test_rocmfp_mix_mmq compares them against the validated
// matvec kernel that way. Env var still provides the default.
static bool g_mix_mmq_forced     = false;
static bool g_mix_mmq_forced_val = false;

static const char * mix_mmq_env_value() {
    const char * value = getenv("DFLASH_MIX_MMQ");
    if (value == nullptr) {
        // Legacy spelling; the flag predates its use outside DS4 prefill.
        value = getenv("DFLASH_DS4_MIX_MMQ_PREFILL");
    }
    return value;
}

bool ggml_cuda_mix_mmq_env_pinned() {
    static const bool pinned = mix_mmq_env_value() != nullptr;
    return pinned;
}

// Precedence: an explicit set_ call (the test's A/B harness) beats the
// environment, which beats the compiled default. Backends opting in for their
// own model must therefore check ggml_cuda_mix_mmq_env_pinned() first, so an
// operator running DFLASH_MIX_MMQ=0 is not silently overridden.
bool ggml_cuda_mix_mmq_enabled() {
    if (g_mix_mmq_forced) {
        return g_mix_mmq_forced_val;
    }
    if (ggml_cuda_mix_mmq_env_pinned()) {
        static const bool from_env = []() {
            const char * value = mix_mmq_env_value();
            return !(value[0] == '0' && value[1] == '\0');
        }();
        return from_env;
    }
    return GGML_CUDA_MIX_MMQ_DEFAULT;
}

void ggml_cuda_set_mix_mmq_enabled(bool enabled) {
    g_mix_mmq_forced     = true;
    g_mix_mmq_forced_val = enabled;
}

void ggml_cuda_clear_mix_mmq_override() {
    g_mix_mmq_forced = false;
}

// Widest ne11 at which the mix-qtype MMQ path still beats dequantize-to-bf16 +
// dense GEMM. MMQ is not uniformly better: it wins by a lot when the batch is
// narrow and loses when it is wide, because the dequant path hands a wide N to
// a well-tiled dense GEMM while the MMQ kernel does not tile as far. Measured
// on muse-lowbpw-r1 (71 dense mix tensors), prefill tok/s, MMQ off -> on:
//
//   ne11        8      16      64     256    1024    2048
//   gfx1151  1.80x   1.77x   1.65x   1.05x   0.89x   0.86x
//   gfx1201  5.11x   4.02x   3.26x   1.83x   1.13x   0.98x
//
// so the crossover sits between 256 and 1024 on RDNA 3.5 and between 1024 and
// 2048 on RDNA 4. Decline MMQ past it rather than making the operator choose:
// decode (ne11 == 1) never reaches this gate at all, and a served request wants
// the narrow-batch win on its verify steps AND the wide-batch win on its
// prefill, which one process-wide boolean cannot deliver.
//
// NVIDIA has no width sweep yet — the H200 result behind this path (1.20x) was
// measured on ne11 4..16 verify batches. It therefore takes the conservative
// RDNA 3.5 bound, which keeps every measured NVIDIA win and declines only the
// widths nothing has measured there. Raise it with the env var once swept.
static int64_t mix_mmq_max_ne11(int cc) {
    static const int64_t override_value = []() -> int64_t {
        const char * value = getenv("DFLASH_MIX_MMQ_MAX_NE11");
        return value ? strtoll(value, nullptr, 10) : -1;
    }();
    if (override_value >= 0) {
        return override_value;
    }
    return GGML_CUDA_CC_IS_RDNA4(cc) ? 1024 : 256;
}

// GQH's two effects point opposite ways, so this is a width gate, not a switch.
// MMQ deletes the fp16 materialisation but re-decodes the weight tile once per
// OUTPUT COLUMN TILE, and GQH's unpack is expensive (bit-sliced planes or uint4
// codes, plus a curved-grid select per code, off an odd superblock stride that
// forbids aligned loads). So it wins by 2-4x narrow and loses by up to 1.8x wide.
//
// The threshold is measured, not guessed: gqh-mmq-sweep on the three shapes that
// carry the artifact, 60 iterations, warm-up pass discarded, spreads under 1%
// except where noted. MMQ/dequant time ratio, so under 1.00 is an MMQ win:
//
//   ncols                          160    192    256    320
//   GQH3 5120x17408 (130 tensors)  0.56   0.60   0.74   1.15
//   GQH4 17408x5120 ( 61 tensors)  0.89   1.03   1.10   1.22
//   GQH4 6144x5120  ( 61 tensors)  0.86   0.89   0.95   1.02
//
// GQH4 17408x5120 crosses first, between 160 and 192, so 160 is the widest bound
// at which EVERY shape in the artifact is still a win. It is not a round number
// chosen for looking tidy - 192 already costs that shape 3%.
//
// Re-measured on a second R9700 / gfx1201 box the curve keeps its shape but
// shifts OUTWARD: GQH4 17408x5120, still the shape that crosses first, reads
// 0.83 at 160 and 0.87 at 192 there and does not cross until between 256 (0.94)
// and 320 (1.05); the other two cross around 512. So 160 is conservative on
// that box rather than wrong - it is the widest bound that is a win on BOTH.
//
// The other leg of the justification is the WORKLOAD, which the kernel curve
// cannot supply: a crossover says where MMQ stops paying, not whether anything
// ever dispatches there. GGML_GQH_NE11_LOG=1 (the census in ggml-cuda.cu) over a
// DFlash2 verify at --draft-block-size 16 plus a 7.3k-token prefill:
//
//   ne11        GQH mul_mat calls   past the matvec (gate-visible)
//   16                     40280                                0
//   39..104                 4323                             4323
//   512                     5502                             5502
//
// Verify at block 16 never reaches the gate -- the matvec owns it and returns
// first. Every call the gate CAN see is either <= 104, where MMQ wins 1.5-2.7x,
// or exactly the 512-wide prefill chunk, where it loses ~1.2x. Nothing lands in
// 105..511, so every threshold in that window dispatches identically and the
// exact value is not load-bearing. What IS load-bearing is that the gate
// declines 512: lifting it costs ~7% of prefill throughput end to end.
//
// Widths below 17 never reach here: ggml_cuda_gqh_mul_mat_vec owns 1..16 and
// returns before this is consulted, so the gate is purely an upper bound.
static int64_t gqh_mmq_max_ne11(int cc) {
    static const int64_t override_value = []() -> int64_t {
        const char * value = getenv("GGML_GQH_MMQ_MAX_NE11");
        return value ? strtoll(value, nullptr, 10) : -1;
    }();
    if (override_value >= 0) {
        return override_value;
    }
    GGML_UNUSED(cc);
    return 160;
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

#ifdef ROCMFP2_AFFINE
    // The affine Q8_1 tile loader carries scale and -offset and uses the
    // activation sum correction. Keep it opt-in until model-level validation.
    if (type == GGML_TYPE_Q2_0_ROCMFP2 &&
        std::getenv("DFLASH_CUDA_MMQ_FP2_AFFINE") == nullptr) {
        return false;
    }
    // Batched expert MMQ is finite with the affine correction, but its
    // gather/quantize path is slower than grouped MMVQ at DS4 verify width.
    if (type == GGML_TYPE_Q2_0_ROCMFP2 && n_experts > 1) {
        return false;
    }
    if (type == GGML_TYPE_Q2_0_ROCMFP2) {
        const char * runtime_disable = std::getenv(
            "DFLASH_CUDA_MMQ_FP2_AFFINE_RUNTIME_DISABLE");
        if (runtime_disable && *runtime_disable &&
            std::strcmp(runtime_disable, "0") != 0) {
            return false;
        }
    }
    if (type == GGML_TYPE_Q2_0_ROCMFP2) {
        // Owner-isolation/qualification switch. AMD architecture codes are
        // accepted in decimal or 0x form (for example 0x1100 or 0x1151).
        static const int required_cc = [] {
            const char * raw = std::getenv(
                "DFLASH_CUDA_MMQ_FP2_AFFINE_CC");
            if (!raw || !*raw) return 0;
            char * end = nullptr;
            const long parsed = std::strtol(raw, &end, 0);
            return end && end != raw && *end == '\0' && parsed > 0 &&
                           parsed <= INT_MAX
                ? (int) parsed
                : 0;
        }();
        if (required_cc > 0 && cc != required_cc) {
            return false;
        }
    }
    if (type == GGML_TYPE_Q2_0_ROCMFP2) {
        static const int min_ncols = [] {
            const char * raw = std::getenv(
                "DFLASH_CUDA_MMQ_FP2_AFFINE_MIN_NCOLS");
            if (!raw || !*raw) return 0;
            char * end = nullptr;
            const long parsed = std::strtol(raw, &end, 10);
            return end && end != raw && *end == '\0' && parsed > 0 &&
                           parsed <= INT_MAX
                ? (int) parsed
                : 0;
        }();
        if (ne11 < min_ncols) {
            return false;
        }
    }
#endif // ROCMFP2_AFFINE

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
#ifdef GGML_CUDA_BLACKWELL_CONSUMER
            mmq_supported = false;
            break;
#endif
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        case GGML_TYPE_GQH3:
        case GGML_TYPE_GQH4:
            // GQH prefill used to dequantise the WHOLE weight set to fp16 for a
            // cuBLAS GEMM: a fixed per-request cost set by the artifact, not the
            // prompt (13.44 GB -> ~27 GB of fp16, measured as a steady 0.3 s per
            // request against IQ4_XS's 0.1 s -- section 9 of
            // docs/QWEN38_R9700_REPRO.md). MMQ removes the materialisation.
            //
            // This does not contest the narrow widths: ggml_cuda_gqh_mul_mat_vec
            // owns 1..16 and wins decode there, and declines above GQH_MAX_COLS,
            // so MMQ catches exactly the widths that used to fall to dequant.
            //
            // Arch gate mirrors ROCmFPX: the wave32 WMMA branch is what these
            // tile shapes are validated on. The dp4a arm in both GQH tile
            // loaders is UNVALIDATED -- on every arch reached here the WMMA
            // branch is what compiles and runs, dp4a is dead code, and there is
            // no build switch to force it, so its tile indices have never been
            // exercised by a single test. amd_wmma_available(cc) is what keeps
            // it unreachable: it is not a relaxation of the allowlist below but
            // a hard invariant on top of it, so that adding an arch to the
            // validated list can never silently arm dp4a instead. The allowlist
            // stays the list of arches these tile shapes were measured on --
            // notably NOT RDNA3.0, which amd_wmma_available admits but which
            // was never validated here.
            //
            // n_experts > 1 declines because there is no expert-batched arm
            // (see ggml_cuda_gqh_mmq_eligible), and the caller must ALSO pass
            // that per-tensor check -- this signature has no tensor, so it
            // cannot see an int8-unrepresentable grid.
            // Width-gated: see gqh_mmq_max_ne11 for the measured crossover and
            // why a single all-or-nothing switch cannot serve both effects.
            mmq_supported = ggml_cuda_gqh_mmq_enabled() && n_experts <= 1 &&
                            ne11 <= gqh_mmq_max_ne11(cc) &&
                            amd_wmma_available(cc) &&
                            (GGML_CUDA_CC_IS_RDNA3_5(cc) ||
                             GGML_CUDA_CC_IS_RDNA4(cc));
            break;
        case GGML_TYPE_Q2_0_ROCMFP2:
        case GGML_TYPE_Q3_0_ROCMFPX:
        case GGML_TYPE_Q4_0_ROCMFP4_FAST:
            // ROCmFPX MMQ uses wave32 WMMA plus portable packed-int dot
            // products. Both gfx1151 (RDNA 3.5) and gfx12xx (RDNA 4) provide
            // that contract; the latter is required by the grouped DS4
            // prefill projection because no BLAS path understands its
            // strided [K/group, N, group] activation layout.
            mmq_supported = GGML_CUDA_CC_IS_RDNA3_5(cc) ||
                            GGML_CUDA_CC_IS_RDNA4(cc);
            break;
        case GGML_TYPE_Q2_1_ROCMFP2_MIX: {
            // Batched path for the mix qtypes. Without it, ne11 > 1 falls back
            // to dequantize-to-bf16 + dense GEMM, which throws away the whole
            // point of a 3.3 bpw artifact for the duration of the multiply:
            // measured 48% of a 16-token speculative verify's GPU time inside
            // dequantize_rocmfp{2,3}_mix_kernel.
            //
            // Enabling it on gfx1201 measured 1.7-1.9x on that verify AND
            // halved the batch-vs-sequential logit drift (0.774 -> 0.380),
            // because the dequant path rounds through bf16 where MMQ keeps
            // integer dot products. Faster and more faithful.
            //
            // NVIDIA is opted in behind the same env var so the claim can be
            // measured rather than assumed: the DP4A tile these types declare
            // is portable, but nothing has gated it here yet.
            //
            // Width-gated: see mix_mmq_max_ne11 for why wide batches decline.
            mmq_supported = ggml_cuda_mix_mmq_enabled() &&
                ne11 <= mix_mmq_max_ne11(cc) &&
                (GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4(cc) ||
                 GGML_CUDA_CC_IS_NVIDIA(cc));
            break;
        }
        case GGML_TYPE_Q3_1_ROCMFP3_MIX: {
            // Batched path for the mix qtypes. Without it, ne11 > 1 falls back
            // to dequantize-to-bf16 + dense GEMM, which throws away the whole
            // point of a 3.3 bpw artifact for the duration of the multiply:
            // measured 48% of a 16-token speculative verify's GPU time inside
            // dequantize_rocmfp{2,3}_mix_kernel.
            //
            // Enabling it on gfx1201 measured 1.7-1.9x on that verify AND
            // halved the batch-vs-sequential logit drift (0.774 -> 0.380),
            // because the dequant path rounds through bf16 where MMQ keeps
            // integer dot products. Faster and more faithful.
            //
            // NVIDIA is opted in behind the same env var so the claim can be
            // measured rather than assumed: the DP4A tile these types declare
            // is portable, but nothing has gated it here yet.
            //
            // Width-gated: see mix_mmq_max_ne11 for why wide batches decline.
            mmq_supported = ggml_cuda_mix_mmq_enabled() &&
                ne11 <= mix_mmq_max_ne11(cc) &&
                (GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4(cc) ||
                 GGML_CUDA_CC_IS_NVIDIA(cc));
            break;
        }
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                // Wide Q4_K batches on gfx1151 favor dequantization +
                // hipBLAS; keep MMQ for decode/small-prefill shapes and for
                // the unmeasured RDNA 3.0 family.
                case GGML_TYPE_Q4_K:
                    return !GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 256;
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
