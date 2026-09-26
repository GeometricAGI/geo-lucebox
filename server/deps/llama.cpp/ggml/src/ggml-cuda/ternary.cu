#include "ternary.cuh"
#include "../gqh.h"
#include <climits>

bool ggml_cuda_ternary_type(ggml_type type) {
    return type==GGML_TYPE_GQH_T || type==GGML_TYPE_GQH_T_G32_R4 || type==GGML_TYPE_GQH_T_G32_R3;
}

bool ggml_cuda_ternary_supported(const ggml_tensor * w,const ggml_tensor * x,const ggml_tensor * y) {
    const float * host=nullptr; const void * backend=nullptr;
    int64_t columns=0; size_t offset=0;
    if(w->data && ggml_gqh_lookup_input_scale(w->data,&host,&backend,&columns,&offset) &&
       (!backend || columns!=w->ne[0] || offset%ggml_row_size(w->type,w->ne[0])!=0)) return false;
    return ggml_cuda_ternary_type(w->type) && x->type==GGML_TYPE_F32 && y->type==GGML_TYPE_F32 &&
        w->ne[0]>0 && w->ne[0]%256==0 && w->ne[1]>0 && w->ne[0]<=INT_MAX/w->ne[1] &&
        w->ne[2]==1 && w->ne[3]==1 && x->ne[0]==w->ne[0] && x->ne[1]>0 && x->ne[1]<=65535 &&
        x->ne[2]==1 && x->ne[3]==1 && y->ne[0]==w->ne[1] && y->ne[1]==x->ne[1] &&
        y->ne[2]==1 && y->ne[3]==1 && ggml_is_contiguous(w) && ggml_is_contiguous(x) && ggml_is_contiguous(y);
}

// Native research contract: reconstruct and round each weight to BF16 in a
// register, then accumulate FP32. No dense decoded weight allocation.
template<int GROUP,int BITS>
static __global__ void ternary_matmul(const uint8_t * body,const float * x,float * y,int cols,int rows,float global,const float * input_scale) {
    constexpr int ratio_bytes=GROUP==16?8:BITS, stride=1+ratio_bytes+52;
    int row=blockIdx.x,token=blockIdx.y,lane=threadIdx.x;
    float sum=0;
    for(int col=lane;col<cols;col+=256) {
        int index=row*cols+col,j=index%256;
        const uint8_t * b=body+(index/256)*stride;
        int exponent=(b[0]>>3)&15,mantissa=b[0]&7;
        float d=exponent?ldexpf(1.0f+mantissa*0.125f,exponent-7):ldexpf(float(mantissa),-9);
        if(b[0]&128)d=-d;
        int bit=(j/GROUP)*BITS;
        unsigned ratio=b[1+bit/8]>>(bit%8);
        if(bit%8+BITS>8)ratio|=unsigned(b[2+bit/8])<<(8-bit%8);
        ratio&=(1<<BITS)-1;
        int place=j%5,divisor=place==0?1:place==1?3:place==2?9:place==3?27:81;
        int trit=(b[1+ratio_bytes+j/5]/divisor)%3-1;
        float value=float(trit)*((d*global)*(float(ratio)/float((1<<BITS)-1)));
        if(input_scale)value/=input_scale[col];
        unsigned u=__float_as_uint(value);
        value=__uint_as_float((u+0x7fff+((u>>16)&1))&0xffff0000u);
        sum+=value*x[int64_t(token)*cols+col];
    }
    __shared__ float accum[256];accum[lane]=sum;__syncthreads();
    for(int step=128;step;step/=2) {if(lane<step)accum[lane]+=accum[lane+step];__syncthreads();}
    if(lane==0)y[int64_t(token)*rows+row]=accum[0];
}

void ggml_cuda_ternary_mul_mat(const ggml_tensor * w,const ggml_tensor * x,ggml_tensor * y,cudaStream_t stream) {
    GGML_ASSERT(ggml_cuda_ternary_supported(w,x,y));
    float scale;int code;
    GGML_ASSERT(ggml_gqh_lookup(w->data,&scale,&code) && code==0 && std::isfinite(scale) && scale>0);
    const float * host_scale=nullptr; const void * input_scale=nullptr;
    int64_t columns=0; size_t offset=0;
    if(ggml_gqh_lookup_input_scale(w->data,&host_scale,&input_scale,&columns,&offset)) {
        GGML_ASSERT(input_scale && columns==w->ne[0] && offset%ggml_row_size(w->type,w->ne[0])==0);
    }
    dim3 grid(w->ne[1],x->ne[1]);
    switch(w->type) {
        case GGML_TYPE_GQH_T: ternary_matmul<16,4><<<grid,256,0,stream>>>((const uint8_t *)w->data,(const float *)x->data,(float *)y->data,w->ne[0],w->ne[1],scale,(const float *)input_scale);break;
        case GGML_TYPE_GQH_T_G32_R4: ternary_matmul<32,4><<<grid,256,0,stream>>>((const uint8_t *)w->data,(const float *)x->data,(float *)y->data,w->ne[0],w->ne[1],scale,(const float *)input_scale);break;
        case GGML_TYPE_GQH_T_G32_R3: ternary_matmul<32,3><<<grid,256,0,stream>>>((const uint8_t *)w->data,(const float *)x->data,(float *)y->data,w->ne[0],w->ne[1],scale,(const float *)input_scale);break;
        default: GGML_ABORT("Unsupported ternary type");
    }
    CUDA_CHECK(cudaGetLastError());
}
