#include "deepseek4/deepseek4_hc_cuda.h"
#include "deepseek4/deepseek4_backend.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace luce::common;
static void check(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
static void full_model_check(const char * model, const char * requests, const char * kind) {
    const bool workspace=std::strcmp(kind,"workspace")==0;
    const bool experts=std::strcmp(kind,"experts")==0;
    if (experts) setenv("LUCE_DS41_REUSE_ATTN_WORKSPACE","1",1);
    DeepSeek4BackendConfig config;
    config.model_path=model;
    config.device.backend=PlacementBackend::Cuda;
    config.device.gpu=0;
    config.max_ctx=512;
    DeepSeek4Backend engine(config);
    check(engine.init(),"model initialization failed");
    std::ifstream input(requests);
    const auto bundle=nlohmann::json::parse(input);
    size_t index=0;
    for (const auto & item : bundle.at("requests")) {
        GenerateRequest request;
        request.prompt=item.at("token_ids").get<std::vector<int32_t>>();
        request.n_gen=128;
        request.force_ar_decode=true;
        std::vector<int32_t> reference;
        // Reverse order on the second prompt to expose warm-cache bias.
        for (int pass=0;pass<2;++pass) {
            const char * mode=((pass+index)%2)==0 ? "0" : "1";
            setenv(experts ? "LUCE_DS41_TYPED_EXPERTS" : workspace ? "LUCE_DS41_REUSE_ATTN_WORKSPACE" : "LUCE_DS4_HC_DEVICE_RMS",mode,1);
            int emitted=0;
            request.on_token=[&](int32_t) {
                ++emitted;
                if (index==0 && std::getenv("LUCE_DS41_PROFILE_COMPARE")) {
                    if (emitted==16) check(cudaProfilerStart()==cudaSuccess,"profiler start failed");
                    if (emitted==80) check(cudaProfilerStop()==cudaSuccess,"profiler stop failed");
                }
                return true;
            };
            auto result=engine.generate_impl(request,DaemonIO{});
            check(!result.error,"model generation failed");
            check(result.tokens.size()==128,"model benchmark ended before fixed token budget");
            if (pass==0) reference=result.tokens;
            else check(result.tokens==reference,"optimization changed full-model tokens");
            std::fprintf(stderr,"MODEL_COMPARE kind=%s prompt=%zu mode=%s tokens=%zu prefill_s=%.6f decode_s=%.6f\n",
                         kind,index,mode,result.tokens.size(),result.prefill_s,result.decode_s);
        }
        ++index;
    }
    unsetenv("LUCE_DS4_HC_DEVICE_RMS");
    unsetenv("LUCE_DS41_REUSE_ATTN_WORKSPACE");
    unsetenv("LUCE_DS41_TYPED_EXPERTS");
    std::fprintf(stderr,"PASS: full-model fixed-token optimization parity\n");
}
int main(int argc, char ** argv)try {
    check(argc==1 || argc==3 || (argc==4 && (std::strcmp(argv[3],"workspace")==0 || std::strcmp(argv[3],"experts")==0)),
          "usage: test_ds41_hc_f32 [model.gguf frozen-token-requests.json [workspace|experts]]");
    if (argc==4 && std::strcmp(argv[3],"experts")==0) setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);
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
        std::vector<float> legacy(mix.size());
        setenv("LUCE_DS4_HC_DEVICE_RMS", "0", 1);
        check(deepseek4_cuda_hc_pre_mix_batch(state.data(),tokens,fn->data,embd,hc,eps,legacy.data(),full),"legacy mix failed");
        setenv("LUCE_DS4_HC_DEVICE_RMS", "1", 1);
        check(deepseek4_cuda_hc_pre_mix_batch(state.data(),tokens,fn->data,embd,hc,eps,mix.data(),full),"batched mix failed");
        check(std::memcmp(mix.data(),legacy.data(),mix.size()*sizeof(float))==0,"device RMS changed controller bits");
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
        if (tokens==1 && full) {
            for (const char * mode : {"0", "1"}) {
                setenv("LUCE_DS4_HC_DEVICE_RMS", mode, 1);
                auto start=std::chrono::steady_clock::now();
                for (int i=0;i<200;++i)
                    check(deepseek4_cuda_hc_pre_mix_batch(state.data(),tokens,fn->data,embd,hc,eps,mix.data(),full),"timed mix failed");
                const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/200;
                std::printf("controller device_rms=%s: %.2f us/call\n",mode,us);
            }
        }
    }
    unsetenv("LUCE_DS4_HC_DEVICE_RMS");
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
    if (argc>=3) full_model_check(argv[1],argv[2],argc==4 ? argv[3] : "rms");
    puts("DS4.1 controller F32/F16 projections: finite, CPU reference and single/batched parity passed");return 0;
}catch(const std::exception & e){fprintf(stderr,"%s\n",e.what());return 1;}
