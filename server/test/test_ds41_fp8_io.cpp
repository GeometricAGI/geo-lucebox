// Native FP8 storage must not silently round a production F32 activation to BF16.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool check(ggml_backend_t backend, ggml_type type, int cols, int rows, int tokens) {
    ggml_init_params params{};
    params.mem_size=16*ggml_tensor_overhead()+ggml_graph_overhead_custom(16,false);
    params.no_alloc=true;
    auto ctx=ggml_init(params);
    auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_I8,cols,rows);
    auto scale=ggml_new_tensor_2d(ctx,GGML_TYPE_I8,(cols+31)/32,(rows+31)/32);
    auto x=ggml_new_tensor_2d(ctx,type,cols,tokens);
    auto y=ggml_dsv41_fp8_matmul(ctx,w,scale,x);
    if(y->type!=type || !ggml_backend_supports_op(backend,y))return false;
    auto graph=ggml_new_graph_custom(ctx,16,false);ggml_build_forward_expand(graph,y);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if(!buffer)return false;
    std::vector<uint8_t> weights(size_t(cols)*rows),scales(size_t((cols+31)/32)*((rows+31)/32));
    std::vector<float> input(size_t(cols)*tokens),reference(size_t(rows)*tokens),output(reference.size());
    std::vector<ggml_bf16_t> input_bf16(input.size()),output_bf16(output.size());
    for(size_t i=0;i<weights.size();++i)weights[i]=uint8_t(((i%3+6)<<3)|(i%8)|((i%5==0)?128:0));
    for(size_t i=0;i<scales.size();++i)scales[i]=uint8_t(126+i%3);
    for(size_t i=0;i<input.size();++i) {
        input[i]=std::sin(float(i)*0.13f)*0.2f+0.00390625f;
        input_bf16[i]=ggml_fp32_to_bf16(input[i]);
        if(type==GGML_TYPE_BF16)input[i]=ggml_bf16_to_fp32(input_bf16[i]);
    }
    if(cols==1 && rows==1) {
        weights[0]=0x38;scales[0]=127;input[0]=1.00390625f;
        input_bf16[0]=ggml_fp32_to_bf16(input[0]);
        if(type==GGML_TYPE_BF16)input[0]=ggml_bf16_to_fp32(input_bf16[0]);
    }
    for(int t=0;t<tokens;++t)for(int r=0;r<rows;++r) {
        double sum=0;
        for(int c=0;c<cols;++c) {
            auto code=weights[size_t(r)*cols+c];
            int exponent=int((code>>3)&15)-7+int(scales[size_t(r/32)*((cols+31)/32)+c/32])-127;
            double value=std::ldexp(1.0+double(code&7)/8,exponent)*(code&128?-1:1);
            sum+=value*input[size_t(t)*cols+c];
        }
        reference[size_t(t)*rows+r]=float(sum);
    }
    ggml_backend_tensor_set(w,weights.data(),0,weights.size());
    ggml_backend_tensor_set(scale,scales.data(),0,scales.size());
    if(type==GGML_TYPE_F32)ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
    else ggml_backend_tensor_set(x,input_bf16.data(),0,input_bf16.size()*sizeof(ggml_bf16_t));
    bool ok=ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    if(type==GGML_TYPE_F32)ggml_backend_tensor_get(y,output.data(),0,output.size()*sizeof(float));
    else {
        ggml_backend_tensor_get(y,output_bf16.data(),0,output_bf16.size()*sizeof(ggml_bf16_t));
        for(size_t i=0;i<output.size();++i)output[i]=ggml_bf16_to_fp32(output_bf16[i]);
    }
    double worst=0;
    for(size_t i=0;i<output.size();++i) {
        double error=std::fabs(double(output[i])-reference[i]);worst=std::max(worst,error);
        double tolerance=type==GGML_TYPE_F32?1e-4+2e-5*std::fabs(reference[i]):1e-4+0.008*std::fabs(reference[i]);
        if(cols==1 && rows==1)tolerance=1e-7;
        if(!std::isfinite(output[i]) || error>tolerance)ok=false;
    }
    std::printf("backend=%s type=%s cols=%d rows=%d tokens=%d max_error=%.9g %s\n",ggml_backend_name(backend),ggml_type_name(type),cols,rows,tokens,worst,ok?"PASS":"FAIL");
    ggml_backend_buffer_free(buffer);ggml_free(ctx);return ok;
}
int main(int argc,char ** argv) {
    bool cpu=argc>1 && !std::strcmp(argv[1],"cpu");
    int count=cpu?1:ggml_backend_cuda_get_device_count();
    if(!count)return 77;
    for(int i=0;i<count;++i) {
        auto backend=cpu?ggml_backend_cpu_init():ggml_backend_cuda_init(i);
        if(!backend)return 2;
        for(auto type:{GGML_TYPE_BF16,GGML_TYPE_F32})
            for(auto shape:std::vector<std::vector<int>>{{1,1,1},{65,35,3},{5120,1280,1},{1280,64,32}})
                if(!check(backend,type,shape[0],shape[1],shape[2]))return 3;
        ggml_backend_free(backend);
    }
    return 0;
}
