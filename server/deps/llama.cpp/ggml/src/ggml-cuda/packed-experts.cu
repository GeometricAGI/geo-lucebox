#include "packed-experts.cuh"
#include "../ggml-packed-experts-decode.h"
static __global__ void packed_experts_kernel(const ggml_packed_expert * desc,const float * x,const int32_t * ids,float * y,int cols,int rows,int routes,int input_routes,int experts) {
    int row=blockIdx.x,route=blockIdx.y,token=blockIdx.z,lane=threadIdx.x;
    int id=ids[int64_t(token)*routes+route];
    // Refuse invalid routes without dereferencing arbitrary weight addresses.
    if(id<0||id>=experts){if(lane==0)y[(int64_t(token)*routes+route)*rows+row]=nanf("");return;}
    const auto & d=desc[id];
    if(d.rows!=rows||d.columns!=cols){if(lane==0)y[(int64_t(token)*routes+route)*rows+row]=nanf("");return;}
    const float * input=x+(int64_t(token)*input_routes+route%input_routes)*cols;
    float sum=0;for(int col=lane;col<cols;col+=256)sum+=packed_weight(d,row,col)*input[col];
    __shared__ float sums[256];sums[lane]=sum;__syncthreads();
    for(int step=128;step;step/=2){if(lane<step)sums[lane]+=sums[lane+step];__syncthreads();}
    if(lane==0)y[(int64_t(token)*routes+route)*rows+row]=sums[0];
}
void ggml_cuda_packed_experts(ggml_backend_cuda_context & ctx,ggml_tensor * dst) {
    GGML_ASSERT(ggml_packed_experts_is_op(dst));
    auto * d=dst->src[0];auto * x=dst->src[1];
    packed_experts_kernel<<<dim3(dst->ne[0],dst->ne[1],dst->ne[2]),256,0,ctx.stream()>>>(
        static_cast<const ggml_packed_expert *>(d->data),static_cast<const float *>(x->data),static_cast<const int32_t *>(dst->src[2]->data),static_cast<float *>(dst->data),x->ne[0],dst->ne[0],dst->ne[1],x->ne[1],d->ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
