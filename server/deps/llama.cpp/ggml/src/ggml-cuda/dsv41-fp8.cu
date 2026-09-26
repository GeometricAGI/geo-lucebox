#include "dsv41-fp8.cuh"
#include "../ggml-dsv41-quant.h"
static __global__ void fp8_matmul(const uint8_t * w,const uint8_t * scales,const uint16_t * x,uint16_t * y,int cols,int rows) {
    int row=blockIdx.x,token=blockIdx.y,lane=threadIdx.x;float sum=0;
    for(int64_t col=lane;col<cols;col+=256){
        float v=ds41_decode_fp8(w[int64_t(row)*cols+col],scales[int64_t(row/32)*((cols+31)/32)+col/32]);
        sum+=v*ds41_from_bf16(x[int64_t(token)*cols+col]);
    }
    __shared__ float accum[256];accum[lane]=sum;__syncthreads();
    for(int step=128;step;step/=2){if(lane<step)accum[lane]+=accum[lane+step];__syncthreads();}
    if(lane==0)y[int64_t(token)*rows+row]=ds41_to_bf16(accum[0]);
}
void ggml_cuda_op_dsv41_fp8_matmul(ggml_backend_cuda_context & ctx,ggml_tensor * dst) {
    auto * w=dst->src[0];
    fp8_matmul<<<dim3(w->ne[1],dst->ne[1]),256,0,ctx.stream()>>>((const uint8_t *)w->data,(const uint8_t *)dst->src[1]->data,(const uint16_t *)dst->src[2]->data,(uint16_t *)dst->data,w->ne[0],w->ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
