#include "mxfp4-f32.cuh"
#include <climits>

bool ggml_cuda_mxfp4_f32_supported(const ggml_tensor * w,const ggml_tensor * x,const ggml_tensor * y) {
    return y->op==GGML_OP_MUL_MAT && y->op_params[0]==GGML_PREC_F32 &&
        w->type==GGML_TYPE_MXFP4 && x->type==GGML_TYPE_F32 && y->type==GGML_TYPE_F32 &&
        w->ne[0]>0 && w->ne[0]%32==0 && w->ne[0]<=INT_MAX && w->ne[1]>0 && w->ne[1]<=INT_MAX &&
        w->ne[2]==1 && w->ne[3]==1 && x->ne[0]==w->ne[0] && x->ne[1]>0 && x->ne[1]<=65535 &&
        x->ne[2]==1 && x->ne[3]==1 && y->ne[0]==w->ne[1] && y->ne[1]==x->ne[1] &&
        y->ne[2]==1 && y->ne[3]==1 && ggml_is_contiguous(w) && ggml_is_contiguous(x) && ggml_is_contiguous(y);
}

// Correctness fallback: preserve caller activation values and decode only in
// registers. Explicit F32 precision avoids the normal Q8 activation conversion.
static __global__ void mxfp4_f32(const uint8_t * w,const float * x,float * y,int cols,int rows) {
    const int row=blockIdx.x,token=blockIdx.y,lane=threadIdx.x;
    const float values[8]={0,.5f,1,1.5f,2,3,4,6};
    float sum=0;
    for(int64_t col=lane;col<cols;col+=256) {
        const uint8_t * b=w+((int64_t(row)*cols+col)/32)*17;
        const int j=col%32,code=(b[1+j%16]>>(j<16?0:4))&15;
        float v=ldexpf(values[code&7],int(b[0])-127);
        if(code&8)v=-v;
        sum+=v*x[int64_t(token)*cols+col];
    }
    __shared__ float accum[256];accum[lane]=sum;__syncthreads();
    for(int step=128;step;step/=2){if(lane<step)accum[lane]+=accum[lane+step];__syncthreads();}
    if(lane==0)y[int64_t(token)*rows+row]=accum[0];
}
void ggml_cuda_mxfp4_f32(const ggml_tensor * w,const ggml_tensor * x,ggml_tensor * y,cudaStream_t stream) {
    GGML_ASSERT(ggml_cuda_mxfp4_f32_supported(w,x,y));
    mxfp4_f32<<<dim3(w->ne[1],x->ne[1]),256,0,stream>>>((const uint8_t *)w->data,(const float *)x->data,(float *)y->data,w->ne[0],w->ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
