#include "packed-experts.cuh"
#include "../ggml-packed-experts-decode.h"
#include <cstdlib>
#include <cstring>
template<int Type>
static __device__ __forceinline__ float packed_dot(const ggml_packed_expert & d,
        const float * input, int row, int cols, int lane) {
    float sum=0;
    for(int col=lane;col<cols;col+=256) sum+=packed_weight<Type>(d,row,col)*input[col];
    return sum;
}
template<bool Typed>
static __global__ void packed_experts_kernel(const ggml_packed_expert * desc,const float * x,const int32_t * ids,float * y,int cols,int rows,int routes,int input_routes,int experts) {
    int row=blockIdx.x,route=blockIdx.y,token=blockIdx.z,lane=threadIdx.x;
    int id=ids[int64_t(token)*routes+route];
    // Refuse invalid routes without dereferencing arbitrary weight addresses.
    if(id<0||id>=experts){if(lane==0)y[(int64_t(token)*routes+route)*rows+row]=nanf("");return;}
    const auto & d=desc[id];
    if(d.rows!=rows||d.columns!=cols){if(lane==0)y[(int64_t(token)*routes+route)*rows+row]=nanf("");return;}
    const float * input=x+(int64_t(token)*input_routes+route%input_routes)*cols;
    float sum=0;
    if constexpr (Typed) {
        // Routes differ per block; specialize once, outside the weight loop.
        switch (d.type) {
#define PACKED_CASE(TYPE) case TYPE: sum=packed_dot<TYPE>(d,input,row,cols,lane); break;
            PACKED_CASE(GGML_TYPE_GQH_T)
            PACKED_CASE(GGML_TYPE_GQH_T_G32_R4)
            PACKED_CASE(GGML_TYPE_GQH_T_G32_R3)
            PACKED_CASE(GGML_TYPE_DSV41_INT3_G64)
            PACKED_CASE(GGML_TYPE_DSV41_INT3_G128)
            PACKED_CASE(GGML_TYPE_MXFP4)
            PACKED_CASE(GGML_TYPE_NVFP4)
            PACKED_CASE(GGML_TYPE_GQH2_H)
            PACKED_CASE(GGML_TYPE_GQH3)
            PACKED_CASE(GGML_TYPE_GQH4)
            PACKED_CASE(GGML_TYPE_BF16)
            PACKED_CASE(GGML_TYPE_F32)
#undef PACKED_CASE
            default: sum=packed_dot<-1>(d,input,row,cols,lane); break;
        }
    } else sum=packed_dot<-1>(d,input,row,cols,lane);
    __shared__ float sums[256];sums[lane]=sum;__syncthreads();
    for(int step=128;step;step/=2){if(lane<step)sums[lane]+=sums[lane+step];__syncthreads();}
    if(lane==0)y[(int64_t(token)*routes+route)*rows+row]=sums[0];
}
void ggml_cuda_packed_experts(ggml_backend_cuda_context & ctx,ggml_tensor * dst) {
    GGML_ASSERT(ggml_packed_experts_is_op(dst));
    auto * d=dst->src[0];auto * x=dst->src[1];
    const char * mode=std::getenv("LUCE_DS41_TYPED_EXPERTS");
    const bool typed=mode && std::strcmp(mode,"0")!=0;
    auto kernel=typed ? packed_experts_kernel<true> : packed_experts_kernel<false>;
    kernel<<<dim3(dst->ne[0],dst->ne[1],dst->ne[2]),256,0,ctx.stream()>>>(
        static_cast<const ggml_packed_expert *>(d->data),static_cast<const float *>(x->data),static_cast<const int32_t *>(dst->src[2]->data),static_cast<float *>(dst->data),x->ne[0],dst->ne[0],dst->ne[1],x->ne[1],d->ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
