#include "ggml-packed-experts.h"
#include "gqh.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>

static void check(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
static float bf16(float x) { return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x)); }
int main(int argc, char ** argv) try {
    check(argc == 3, "usage: test_packed_experts cpu|cuda fixture_directory");
    const bool cpu = std::string(argv[1]) == "cpu";
    ggml_backend_t backend = cpu ? ggml_backend_cpu_init() : ggml_backend_cuda_init(0);
    check(backend, "backend init failed");
    if (cpu) ggml_backend_cpu_set_n_threads(backend, 4);
    constexpr int rows=64, cols=1024, experts=9, routes=6, tokens=3;
    const ggml_type types[experts] = {GGML_TYPE_GQH_T, GGML_TYPE_GQH_T_G32_R4, GGML_TYPE_GQH_T_G32_R3,
        GGML_TYPE_DSV41_INT3_G64, GGML_TYPE_DSV41_INT3_G128, GGML_TYPE_MXFP4, GGML_TYPE_NVFP4,
        GGML_TYPE_BF16, GGML_TYPE_F32};
    const char * names[] = {"gqh_t", "gqh_t_g32_r4", "gqh_t_g32_r3"};
    auto ctx = ggml_init({ggml_tensor_overhead()*128 + ggml_graph_overhead_custom(128,false), nullptr, true});
    ggml_tensor * weights[experts];
    for (int e=0;e<experts;++e) weights[e]=ggml_new_tensor_2d(ctx,types[e],cols,rows);
    auto scale_tensor=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,cols);
    auto descriptors=ggml_new_tensor_2d(ctx,GGML_TYPE_I8,sizeof(ggml_packed_expert),experts);
    auto weight_buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    check(weight_buffer,"weight allocation failed");
    std::vector<float> divisors(cols);
    for(int c=0;c<cols;++c)divisors[c]=.5f+float(c%17)/16;
    ggml_backend_tensor_set(scale_tensor,divisors.data(),0,cols*4);
    std::vector<std::vector<float>> reference(experts,std::vector<float>(rows*cols));
    std::vector<ggml_packed_expert> desc(experts);
    for(int e=0;e<experts;++e){
        auto w=weights[e];std::vector<uint8_t> body(ggml_nbytes(w));float global=1;
        if(e<3){
            std::ifstream f(std::string(argv[2])+"/"+names[e]+"_64x1024.wire.bin",std::ios::binary);
            uint8_t code=0;f.read(reinterpret_cast<char *>(&global),4);f.read(reinterpret_cast<char *>(&code),1);
            check(code==0 && bool(f.read(reinterpret_cast<char *>(body.data()),body.size())),"bad fixture");
        }else if(e==3||e==4){
            int group=e==3?64:128,stride=4+group*3/8;
            for(size_t b=0;b<body.size();b+=stride){float scale=.037f;memcpy(body.data()+b,&scale,4);for(int j=4;j<stride;++j)body[b+j]=uint8_t((b+j)*37);}
        }else if(e==5){
            for(size_t b=0;b<body.size();b+=17){body[b]=123;for(int j=1;j<17;++j)body[b+j]=uint8_t((b+j)*37);}
        }else if(e==6){
            global=2.75f;
            for(size_t b=0;b<body.size();b+=36){for(int j=0;j<4;++j)body[b+j]=uint8_t(0x28+j);for(int j=4;j<36;++j)body[b+j]=uint8_t((b+j)*37);}
        }else{
            for(int i=0;i<rows*cols;++i){float v=std::sin(float(i)*.17f)*.13f;if(e==7){auto b=ggml_fp32_to_bf16(v);memcpy(body.data()+i*2,&b,2);}else memcpy(body.data()+i*4,&v,4);}
        }
        const bool header=e<5||e==6;
        ggml_tensor host=*w;host.buffer=nullptr;host.data=body.data();
        if(header)ggml_gqh_register(host.data,body.size(),global,0);
        if(e<3)check(ggml_gqh_register_ternary_input_scale(&host,divisors.data(),cols,nullptr),"host compensation failed");
        if(e==8)memcpy(reference[e].data(),body.data(),body.size());
        else ggml_get_type_traits(w->type)->to_float(body.data(),reference[e].data(),rows*cols);
        // Native NVFP4 row decoder has no tensor-level research global scale.
        if(e==6)for(auto & v:reference[e])v/=global;
        for(auto & v:reference[e])v=bf16(v);
        if(header)ggml_gqh_unregister(host.data);
        ggml_backend_tensor_set(w,body.data(),0,body.size());
        if(header)ggml_gqh_register(w->data,body.size(),global,0);
        if(e<3)check(ggml_gqh_register_ternary_input_scale(w,divisors.data(),cols,scale_tensor->data),"device compensation failed");
        check(ggml_packed_expert_init(&desc[e],w),"descriptor rejected");
    }
    ggml_backend_tensor_set(descriptors,desc.data(),0,desc.size()*sizeof(desc[0]));
    for(int input_routes:{1,routes}){
        auto graph_ctx=ggml_init({ggml_tensor_overhead()*32+ggml_graph_overhead_custom(32,false),nullptr,true});
        auto x=ggml_new_tensor_3d(graph_ctx,GGML_TYPE_F32,cols,input_routes,tokens);
        auto ids=ggml_new_tensor_2d(graph_ctx,GGML_TYPE_I32,routes,tokens);
        auto y=ggml_packed_experts_mul_mat_id(graph_ctx,descriptors,x,ids,rows);
        check(ggml_backend_supports_op(backend,y),"backend refused packed op");
        if (!cpu) {
            auto other=ggml_backend_cpu_init();
            check(!ggml_backend_supports_op(other,y),"CPU accepted GPU-pointer descriptors");
            ggml_backend_free(other);
        }
        auto buffer=ggml_backend_alloc_ctx_tensors(graph_ctx,backend);check(buffer,"graph allocation failed");
        auto graph=ggml_new_graph_custom(graph_ctx,32,false);ggml_build_forward_expand(graph,y);
        std::vector<float> input(cols*input_routes*tokens),actual(rows*routes*tokens);
        std::vector<int32_t> routing(routes*tokens);
        for(int pass=0;pass<3;++pass){
            for(size_t i=0;i<input.size();++i)input[i]=std::cos(float(i+pass)*.031f)*.2f;
            for(size_t i=0;i<routing.size();++i)routing[i]=int((i/2+pass*3)%experts); // Duplicate and changing GPU routes.
            ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
            ggml_backend_tensor_set(ids,routing.data(),0,routing.size()*4);
            check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute failed");
            ggml_backend_tensor_get(y,actual.data(),0,actual.size()*4);
            for(int t=0;t<tokens;++t)for(int r=0;r<routes;++r)for(int row=0;row<rows;++row){
                double want=0;int e=routing[t*routes+r];
                for(int c=0;c<cols;++c)want+=double(reference[e][row*cols+c])*input[(t*input_routes+r%input_routes)*cols+c];
                float got=actual[(t*routes+r)*rows+row];
                if(!std::isfinite(got)||std::abs(got-want)>2e-4*(1+std::abs(want))){
                    fprintf(stderr,"type=%d pass=%d routes=%d row=%d got=%g expected=%g\n",int(types[e]),pass,input_routes,row,got,want);throw std::runtime_error("routed result mismatch");
                }
            }
        }
        ggml_backend_buffer_free(buffer);ggml_free(graph_ctx);
    }
    for(auto w:weights)ggml_gqh_unregister(w->data);
    ggml_backend_buffer_free(weight_buffer);ggml_free(ctx);ggml_backend_free(backend);
    puts("mixed packed experts: conditioned decode, changing routes, broadcast/per-route inputs and graph reuse passed");return 0;
}catch(const std::exception & e){fprintf(stderr,"%s\n",e.what());return 1;}
