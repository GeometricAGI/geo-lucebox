#pragma once
#include "ggml-packed-experts.h"
#include <math.h>
#include <string.h>
#include "ggml-packed-iq2-grid.h"
#if defined(__CUDACC__) || defined(__HIPCC__)
#define PACKED_HD __host__ __device__
#else
#define PACKED_HD
#endif
static PACKED_HD inline float packed_float(uint32_t bits) { float x; memcpy(&x,&bits,4); return x; }
static PACKED_HD inline float packed_bf16(float x) { uint32_t b;memcpy(&b,&x,4);return packed_float((b+0x7fff+((b>>16)&1))&0xffff0000u); }
static PACKED_HD inline float packed_e4m3(uint8_t b) {
    int e=(b>>3)&15,m=b&7;float x=e?ldexpf(1.f+m*.125f,e-7):ldexpf(float(m),-9);return b&128?-x:x;
}
static PACKED_HD inline float packed_half(const uint8_t * p) {
    unsigned h=unsigned(p[0])|(unsigned(p[1])<<8);int e=(h>>10)&31,m=h&1023;
    float value=e?ldexpf(1.f+float(m)/1024.f,e-15):ldexpf(float(m),-24);
    return h&32768?-value:value;
}
static PACKED_HD inline uint64_t packed_iq2_grid(unsigned index) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return packed_iq2_grid_device[index];
#else
    return packed_iq2_grid_host[index];
#endif
}
template<int Type = -1>
static PACKED_HD inline float packed_weight(const ggml_packed_expert & d, int row, int col) {
    const int type = Type < 0 ? d.type : Type;
    const auto * body=reinterpret_cast<const uint8_t *>(uintptr_t(d.body));
    const int64_t index=int64_t(row)*d.columns+col;
    float v=0;
    if(type==GGML_TYPE_BF16) {uint16_t b;memcpy(&b,body+index*2,2);v=packed_float(uint32_t(b)<<16);}
    else if(type==GGML_TYPE_F32) {memcpy(&v,body+index*4,4);}
    else if(type==GGML_TYPE_Q2_K) {
        int j=index%256;const uint8_t * b=body+(index/256)*84;
        int scale=b[j/16],code=(b[16+(j/128)*32+j%32]>>(2*((j%128)/32)))&3;
        // Match the reference's separate multiply/subtract before BF16 rounding.
        volatile float product=(packed_half(b+80)*float(scale&15))*float(code);
        v=product-packed_half(b+82)*float(scale>>4);
    } else if(type==GGML_TYPE_IQ2_XXS) {
        int j=index%256,within=j%32;const uint8_t * b=body+(index/256)*66;
        const uint8_t * q=b+2+(j/32)*8;
        uint32_t word=uint32_t(q[4])|(uint32_t(q[5])<<8)|(uint32_t(q[6])<<16)|(uint32_t(q[7])<<24);
        unsigned signs=(word>>(7*(within/8)))&127;
        unsigned parity=signs;parity^=parity>>4;parity^=parity>>2;parity^=parity>>1;signs|=(parity&1)<<7;
        float scale=packed_half(b)*(0.5f+float(word>>28))*0.25f;
        unsigned grid=(packed_iq2_grid(q[within/8])>>(8*(within%8)))&255;
        v=scale*float(grid)*((signs&(1u<<(within%8)))?-1.f:1.f);
    }
    else if(type==GGML_TYPE_DSV41_INT3_G64 || type==GGML_TYPE_DSV41_INT3_G128) {
        const int group=type==GGML_TYPE_DSV41_INT3_G64?64:128, j=index%group;
        const uint8_t * b=body+(index/group)*(4+group*3/8);float scale;memcpy(&scale,b,4);
        int bit=j*3,shift=bit%8;unsigned code=b[4+bit/8]>>shift;if(shift+3>8)code|=unsigned(b[5+bit/8])<<(8-shift);
        v=(int(code&7)-4)*scale;
    } else if(type==GGML_TYPE_MXFP4) {
        int j=index%32;const uint8_t * b=body+(index/32)*17;
        int code=(b[1+j%16]>>(4*(j/16)))&15;
        // E8M0 exponent zero is 2^-127 (including subnormal float output).
        v=d.levels[code]*ldexpf(1.f,int(b[0])-127);
    } else if(type==GGML_TYPE_NVFP4) {
        int j=index%64;const uint8_t * b=body+(index/64)*36;
        int code=(b[4+(j/16)*8+j%8]>>(4*((j%16)/8)))&15;
        v=(d.levels[code]*packed_e4m3(b[j/16]))/d.global;
    } else {
        const bool ternary=type==GGML_TYPE_GQH_T||type==GGML_TYPE_GQH_T_G32_R4||type==GGML_TYPE_GQH_T_G32_R3;
        int stride=type==GGML_TYPE_GQH_T?61:type==GGML_TYPE_GQH_T_G32_R4?57:type==GGML_TYPE_GQH_T_G32_R3?56:type==GGML_TYPE_GQH2_H?73:type==GGML_TYPE_GQH3?105:137;
        int j=index%256;const uint8_t * b=body+(index/256)*stride;
        float scale=packed_e4m3(b[0])*d.global;
        if(ternary) {
            int group=type==GGML_TYPE_GQH_T?16:32,bits=type==GGML_TYPE_GQH_T_G32_R3?3:4;
            int bit=(j/group)*bits;unsigned ratio=b[1+bit/8]>>(bit%8);
            if(bit%8+bits>8)ratio|=unsigned(b[2+bit/8])<<(8-bit%8);
            ratio&=(1<<bits)-1;int place=j%5,power=place==0?1:place==1?3:place==2?9:place==3?27:81;
            int trit=(b[stride-52+j/5]/power)%3-1;
            v=float(trit)*(scale*(float(ratio)/float((1<<bits)-1)));
        } else {
            int sub=j/16,ratio=(b[1+sub/2]>>(4*(sub%2)))&15,code;
            if(type==GGML_TYPE_GQH4)code=(b[9+j/2]>>(4*(j%2)))&15;
            else {code=(b[9+j/4]>>(2*(j%4)))&3;if(type==GGML_TYPE_GQH3)code|=((b[73+j/8]>>(j%8))&1)<<2;}
            v=d.levels[code]*(scale*d.ratios[ratio]);
        }
    }
    if(d.input_scale)v/=reinterpret_cast<const float *>(uintptr_t(d.input_scale))[col];
    return packed_bf16(v);
}
#undef PACKED_HD
