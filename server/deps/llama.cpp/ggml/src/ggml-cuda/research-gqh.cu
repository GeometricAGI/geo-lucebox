#include "research-gqh.cuh"
#include "../gqh.h"
#include <climits>
#include <cstring>
struct research_grid {float level[16];float ratio[16];};
bool ggml_cuda_research_gqh_supported(const ggml_tensor * w,const ggml_tensor * x,const ggml_tensor * y){
 if(w->type!=GGML_TYPE_BF16&&w->type!=GGML_TYPE_GQH2_H&&w->type!=GGML_TYPE_GQH3&&w->type!=GGML_TYPE_GQH4&&w->type!=GGML_TYPE_NVFP4&&w->type!=GGML_TYPE_DSV41_INT3_G64&&w->type!=GGML_TYPE_DSV41_INT3_G128)return false;
 const float * host=nullptr;const void * backend=nullptr;int64_t columns=0;size_t offset=0;
 if(w->data&&ggml_gqh_lookup_input_scale(w->data,&host,&backend,&columns,&offset)&&
    (!backend||columns!=w->ne[0]||offset%ggml_row_size(w->type,w->ne[0])!=0))return false;
 return x->type==GGML_TYPE_F32&&y->type==GGML_TYPE_F32&&w->ne[0]>0&&w->ne[0]%(w->type==GGML_TYPE_BF16?1:w->type==GGML_TYPE_DSV41_INT3_G128?128:(w->type==GGML_TYPE_NVFP4||w->type==GGML_TYPE_DSV41_INT3_G64)?64:256)==0&&
  w->ne[1]>0&&w->ne[0]<=INT_MAX/w->ne[1]&&w->ne[2]==1&&w->ne[3]==1&&
  x->ne[0]==w->ne[0]&&x->ne[1]>0&&x->ne[1]<=65535&&x->ne[2]==1&&x->ne[3]==1&&
  y->ne[0]==w->ne[1]&&y->ne[1]==x->ne[1]&&y->ne[2]==1&&y->ne[3]==1&&
  ggml_is_contiguous(w)&&ggml_is_contiguous(x)&&ggml_is_contiguous(y);
}
// Reconstruct one weight in a register, apply AWQ BEFORE BF16 rounding, then
// accumulate F32. NVFP4 uses its exact global divisor before BF16 rounding. No FP16 weight expansion, Q8 activation proxy or dense buffer.
template<int BITS> static __global__ void research_gqh_matmul(const uint8_t * body,const float * x,float * y,
 int cols,int rows,float global,research_grid grid,const float * awq){
 constexpr int stride=BITS<0?4+(-BITS)*3/8:BITS==0?36:BITS==2?73:BITS==3?105:137;
 constexpr int block=BITS<0?-BITS:BITS==0?64:256;
 int row=blockIdx.x,token=blockIdx.y,lane=threadIdx.x;float sum=0;
 for(int col=lane;col<cols;col+=256){
  int index=row*cols+col,j=index%block;const uint8_t * b=body+(index/block)*stride;
  float value;
  if constexpr(BITS<0){
   int bit=j*3,shift=bit%8;unsigned code=b[4+bit/8]>>shift;if(shift+3>8)code|=unsigned(b[5+bit/8])<<(8-shift);
   value=float(int(code&7)-4)*(*reinterpret_cast<const float *>(b));
  }else{
  uint8_t scale=BITS==0?b[j/16]:b[0];
  int exponent=(scale>>3)&15,mantissa=scale&7;
  float d=exponent?ldexpf(1.f+mantissa*.125f,exponent-7):ldexpf(float(mantissa),-9);
  if(scale&128)d=-d;
  if constexpr(BITS==0){
   int code=(b[4+(j/16)*8+j%8]>>(4*((j%16)/8)))&15;
   value=(grid.level[code]*d)/global;
  }else{
   int sub=j/16,ratio=(b[1+sub/2]>>(4*(sub%2)))&15;
   int code;
   if constexpr(BITS==4)code=(b[9+j/2]>>(4*(j%2)))&15;
   else {code=(b[9+j/4]>>(2*(j%4)))&3;if constexpr(BITS==3)code|=((b[73+j/8]>>(j%8))&1)<<2;}
   value=grid.level[code]*((d*global)*grid.ratio[ratio]);
  }
  }
  if(awq)value/=awq[col];
  unsigned u=__float_as_uint(value);value=__uint_as_float((u+0x7fff+((u>>16)&1))&0xffff0000u);
  sum+=value*x[int64_t(token)*cols+col];
 }
 __shared__ float partial[256];partial[lane]=sum;__syncthreads();
 for(int step=128;step;step/=2){if(lane<step)partial[lane]+=partial[lane+step];__syncthreads();}
 if(lane==0)y[int64_t(token)*rows+row]=partial[0];
}
// Dense native head: preserve full F32 activations and logits while reading
// BF16 checkpoint weights directly. No dense F32 weight temporary.
static __global__ void native_bf16_f32_matmul(const uint16_t * w,const float * x,float * y,int cols,int rows){
 int row=blockIdx.x,token=blockIdx.y,lane=threadIdx.x;float sum=0;
 for(int col=lane;col<cols;col+=256){
  float weight=__uint_as_float(uint32_t(w[int64_t(row)*cols+col])<<16);
  sum+=weight*x[int64_t(token)*cols+col];
 }
 __shared__ float partial[256];partial[lane]=sum;__syncthreads();
 for(int step=128;step;step/=2){if(lane<step)partial[lane]+=partial[lane+step];__syncthreads();}
 if(lane==0)y[int64_t(token)*rows+row]=partial[0];
}
void ggml_cuda_research_gqh_mul_mat(const ggml_tensor * w,const ggml_tensor * x,ggml_tensor * y,cudaStream_t stream){
 GGML_ASSERT(ggml_cuda_research_gqh_supported(w,x,y));
 if(w->type==GGML_TYPE_BF16){
  native_bf16_f32_matmul<<<dim3(w->ne[1],x->ne[1]),256,0,stream>>>((const uint16_t *)w->data,(const float *)x->data,(float *)y->data,w->ne[0],w->ne[1]);
  CUDA_CHECK(cudaGetLastError());return;
 }
 float global;int code;
 GGML_ASSERT(ggml_gqh_lookup(w->data,&global,&code)&&std::isfinite(global)&&global>0&&code>=0&&code<GQH_GRID_CODES);
 research_grid grid{};std::memcpy(grid.ratio,GQH_RATIO_Q,sizeof(grid.ratio));
 if(w->type==GGML_TYPE_DSV41_INT3_G64||w->type==GGML_TYPE_DSV41_INT3_G128){GGML_ASSERT(global==1.f&&code==0);}
 else if(w->type==GGML_TYPE_NVFP4){const float levels[]={0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};std::memcpy(grid.level,levels,sizeof(levels));GGML_ASSERT(code==0);}
 else if(w->type==GGML_TYPE_GQH2_H)std::memcpy(grid.level,GQH2H_GRID[code],4*sizeof(float));
 else if(w->type==GGML_TYPE_GQH3)std::memcpy(grid.level,GQH3_GRID[code],8*sizeof(float));
 else std::memcpy(grid.level,GQH4_GRID[code],16*sizeof(float));
 const float * host=nullptr;const void * awq=nullptr;int64_t columns=0;size_t offset=0;
 ggml_gqh_lookup_input_scale(w->data,&host,&awq,&columns,&offset);
 dim3 launch(w->ne[1],x->ne[1]);
 #define LAUNCH(bits) research_gqh_matmul<bits><<<launch,256,0,stream>>>((const uint8_t *)w->data,(const float *)x->data,(float *)y->data,w->ne[0],w->ne[1],global,grid,(const float *)awq)
 if(w->type==GGML_TYPE_DSV41_INT3_G64){LAUNCH(-64);}else if(w->type==GGML_TYPE_DSV41_INT3_G128){LAUNCH(-128);}else if(w->type==GGML_TYPE_NVFP4){LAUNCH(0);}else if(w->type==GGML_TYPE_GQH2_H){LAUNCH(2);}else if(w->type==GGML_TYPE_GQH3){LAUNCH(3);}else{LAUNCH(4);}
 #undef LAUNCH
 CUDA_CHECK(cudaGetLastError());
}
