#pragma once
#include <math.h>
#include <stdint.h>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define DS41_INLINE static __host__ __device__ __forceinline__
#else
#define DS41_INLINE static inline
#endif
// Scalar arithmetic shared by CPU and GPU; native golden fixtures independently
// verify the rounding. Inputs and outputs use BF16 bits, not half precision.
DS41_INLINE float ds41_from_bf16(uint16_t x) {
    union {uint32_t u;float f;} v;v.u=(uint32_t)x<<16;return v.f;
}
DS41_INLINE uint16_t ds41_to_bf16(float x) {
    union {uint32_t u;float f;} v;v.f=x;
    if((v.u&0x7fffffffU)>0x7f800000U)return (uint16_t)((v.u>>16)|0x40);
    return (uint16_t)((v.u+0x7fff+((v.u>>16)&1))>>16);
}
DS41_INLINE float ds41_e4m3(float x) {
    float a=fminf(fabsf(x),448.0f);
    union {uint32_t u;float f;} bits;bits.f=a;
    int exponent=(int)((bits.u>>23)&255)-127-3;
    if(exponent < -9)exponent=-9;
    float step=ldexpf(1.0f,exponent), scaled=a/step;
    float lo=floorf(scaled),fraction=scaled-lo;
    if(fraction>0.5f || (fraction==0.5f && ((int)lo&1)))lo+=1.0f;
    return copysignf(lo*step,x);
}
DS41_INLINE float ds41_e2m1(float x) {
    const float table[8]={0,0.5f,1,1.5f,2,3,4,6};
    float a=fminf(fabsf(x),6.0f),best_diff=a;int best=0;
    for(int i=1;i<8;++i){float diff=fabsf(a-table[i]);
        if(diff<best_diff || (diff==best_diff && !(i&1) && (best&1))){best=i;best_diff=diff;}}
    return copysignf(table[best],x);
}
DS41_INLINE float ds41_ceil_pow2(float x) {
    union {uint32_t u;float f;} v;v.f=x;
    uint32_t exponent=(v.u>>23)&255;
    exponent+=(v.u&0x7fffffU)!=0;v.u=exponent<<23;return v.f;
}
DS41_INLINE float ds41_scale(float amax,int mode) {
    if(mode==1)return ds41_e4m3(fmaxf(amax,0.01171875f)/6.0f);
    if(mode==2)return ds41_ceil_pow2(fmaxf(amax,1e-4f)*(1.0f/448.0f));
    return ds41_ceil_pow2(fmaxf(amax,0x1.8p-124f)*(1.0f/6.0f));
}
DS41_INLINE uint16_t ds41_quant_value(float value,float scale,int mode) {
    float normalized=value/scale;
    return ds41_to_bf16((mode==2?ds41_e4m3(normalized):ds41_e2m1(normalized))*scale);
}
DS41_INLINE float ds41_decode_fp8(uint8_t code,uint8_t scale) {
    unsigned exponent=(code>>3)&15,mantissa=code&7;
    if(scale==255 || (exponent==15 && mantissa==7))return NAN;
    float v=exponent?ldexpf(1.0f+mantissa*0.125f,(int)exponent-7):ldexpf((float)mantissa,-9);
    v=ldexpf(v,(int)scale-127);
    return code&128?-v:v;
}
#undef DS41_INLINE
