#include "deepseek4/deepseek4_hc_cuda.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace luce::common;
static void check(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
int main()try {
    auto backend=ggml_backend_cuda_init(0);check(backend,"backend initialization failed");
    check(deepseek4_cuda_hc_set_device(0),"device selection failed");
    constexpr int embd=5120,hc=4,cols=embd*hc,rows=24;
    constexpr float eps=1e-6f;
    auto ctx=ggml_init({ggml_tensor_overhead()*4+4096,nullptr,true});
    auto f32=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,cols,rows);
    auto f16=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,cols,rows);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer,"allocation failed");
    std::vector<float> weights(cols*rows);
    std::vector<ggml_fp16_t> halves(weights.size());
    for(size_t i=0;i<weights.size();++i){weights[i]=std::sin(float(i)*.017f)*.04713f;halves[i]=ggml_fp32_to_fp16(weights[i]);}
    ggml_backend_tensor_set(f32,weights.data(),0,weights.size()*4);
    ggml_backend_tensor_set(f16,halves.data(),0,halves.size()*2);
    for(int tokens:{1,2,9})for(bool full:{false,true}){
        std::vector<float> state(cols*tokens),mix(rows*tokens),single(rows);
        for(size_t i=0;i<state.size();++i)state[i]=std::cos(float(i)*.00137f)*.23f;
        auto fn=full?f32:f16;
        check(deepseek4_cuda_hc_pre_mix_batch(state.data(),tokens,fn->data,embd,hc,eps,mix.data(),full),"batched mix failed");
        for(int t=0;t<tokens;++t){
            check(deepseek4_cuda_hc_pre_mix(state.data()+t*cols,fn->data,embd,hc,eps,single.data(),full),"single mix failed");
            double ss=0;for(int c=0;c<cols;++c)ss+=double(state[t*cols+c])*state[t*cols+c];
            double inv=1/std::sqrt(ss/cols+eps);
            for(int row=0;row<rows;++row){
                double expected=0;
                for(int c=0;c<cols;++c){float weight=full?weights[row*cols+c]:ggml_fp16_to_fp32(halves[row*cols+c]);expected+=double(weight)*state[t*cols+c]*inv;}
                float got=mix[t*rows+row];
                if(!std::isfinite(got)||std::abs(got-expected)>2e-4*(1+std::abs(expected))){fprintf(stderr,"f32=%d tokens=%d row=%d got=%g want=%g\n",full,tokens,row,got,expected);throw std::runtime_error("controller mix differs from reference");}
                check(got==single[row],"batch/single controller mismatch");
            }
        }
    }
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
    puts("DS4.1 controller F32/F16 projections: finite, CPU reference and single/batched parity passed");return 0;
}catch(const std::exception & e){fprintf(stderr,"%s\n",e.what());return 1;}
