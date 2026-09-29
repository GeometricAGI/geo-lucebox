// Real DS4.1 projection geometry, bounded synthetic routing, no model-TPS claim.
#include "ggml-packed-experts.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gqh.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
static void check(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char ** argv) try {
    check(argc==7||argc==8,"usage: bench_packed_experts fixture_dir q2_k|iq2_xxs|gqh_t|gqh2_h rows cols iterations device [weight_device]");
    std::string type=argv[2];int rows=std::stoi(argv[3]),cols=std::stoi(argv[4]),iterations=std::stoi(argv[5]);
    check(rows>0&&cols>0&&cols%256==0&&iterations>0,"invalid geometry/iterations");
    ggml_type qt=type=="q2_k"?GGML_TYPE_Q2_K:type=="iq2_xxs"?GGML_TYPE_IQ2_XXS:type=="gqh_t"?GGML_TYPE_GQH_T:GGML_TYPE_GQH2_H;
    check(type=="q2_k"||type=="iq2_xxs"||type=="gqh_t"||type=="gqh2_h","unknown type");
    bool header=type=="gqh_t"||type=="gqh2_h";
    std::ifstream f(std::string(argv[1])+"/"+type+(header?"_64x1024.wire.bin":".raw"),std::ios::binary);
    check(bool(f),"missing fixture");float global=1;uint8_t code=0;
    if(header){f.read(reinterpret_cast<char *>(&global),4);f.read(reinterpret_cast<char *>(&code),1);}
    std::vector<uint8_t> seed((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());check(!seed.empty(),"empty fixture");
    setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);setenv("LUCE_DS41_TYPED_EXPERTS","1",1);
    auto backend=ggml_backend_cuda_init(std::stoi(argv[6]));check(backend,"backend init failed");
    int compute_device=std::stoi(argv[6]), weight_device=argc==8?std::stoi(argv[7]):compute_device;
    auto weight_backend=weight_device==compute_device?backend:ggml_backend_cuda_init(weight_device);
    check(weight_backend,"weight backend init failed");
    constexpr int experts=384,routes=6;
    auto ctx=ggml_init({ggml_tensor_overhead()*420+ggml_graph_overhead_custom(32,false),nullptr,true});
    std::vector<ggml_tensor *> weights;
    for(int e=0;e<experts;++e)weights.push_back(ggml_new_tensor_2d(ctx,qt,cols,rows));
    auto scales=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,cols);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,weight_backend);check(buffer,"weight allocation failed");
    auto descriptor_ctx=ggml_init({4*ggml_tensor_overhead()+4096,nullptr,true});
    auto descriptors=ggml_new_tensor_2d(descriptor_ctx,GGML_TYPE_I8,sizeof(ggml_packed_expert),experts);
    auto descriptor_buffer=ggml_backend_alloc_ctx_tensors(descriptor_ctx,backend);
    check(descriptor_buffer,"descriptor allocation failed");
    auto graph_ctx=ggml_init({ggml_tensor_overhead()*16+ggml_graph_overhead_custom(32,false),nullptr,true});
    auto x=ggml_new_tensor_3d(graph_ctx,GGML_TYPE_F32,cols,1,1);
    auto ids=ggml_new_tensor_2d(graph_ctx,GGML_TYPE_I32,routes,1);
    auto y=ggml_packed_experts_mul_mat_id(graph_ctx,descriptors,x,ids,rows);
    auto graph_buffer=ggml_backend_alloc_ctx_tensors(graph_ctx,backend);check(graph_buffer,"graph allocation failed");
    std::vector<uint8_t> body(ggml_nbytes(weights[0]));
    for(size_t pos=0;pos<body.size();pos+=seed.size())memcpy(body.data()+pos,seed.data(),std::min(seed.size(),body.size()-pos));
    std::vector<float> divisors(cols),input(cols),reference(size_t(rows)*cols),actual(size_t(rows)*routes);
    for(int c=0;c<cols;++c){divisors[c]=.5f+float(c%17)/16;input[c]=std::cos(float(c)*.031f)*.2f;}
    ggml_backend_tensor_set(scales,divisors.data(),0,cols*4);
    ggml_tensor host=*weights[0];host.buffer=nullptr;host.data=body.data();
    if(header)ggml_gqh_register(host.data,body.size(),global,code);
    if(type=="gqh_t")check(ggml_gqh_register_ternary_input_scale(&host,divisors.data(),cols,nullptr),"host compensation failed");
    ggml_get_type_traits(qt)->to_float(body.data(),reference.data(),int64_t(rows)*cols);
    for(auto & v:reference)v=ggml_bf16_to_fp32(ggml_fp32_to_bf16(v));
    if(header)ggml_gqh_unregister(host.data);
    std::vector<ggml_packed_expert> desc(experts);
    for(int e=0;e<experts;++e){
        ggml_backend_tensor_set(weights[e],body.data(),0,body.size());
        if(header)ggml_gqh_register(weights[e]->data,body.size(),global,code);
        if(type=="gqh_t")check(ggml_gqh_register_ternary_input_scale(weights[e],divisors.data(),cols,scales->data),"device compensation failed");
        check(ggml_packed_expert_init(&desc[e],weights[e]),"descriptor rejected");
    }
    ggml_backend_synchronize(weight_backend);
    ggml_backend_tensor_set(descriptors,desc.data(),0,desc.size()*sizeof(desc[0]));
    ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
    auto graph=ggml_new_graph_custom(graph_ctx,32,false);ggml_build_forward_expand(graph,y);
    std::vector<int32_t> routing(routes);std::vector<double> times;
    for(int step=-5;step<iterations;++step){
        for(int r=0;r<routes;++r)routing[r]=((step+5)*17+r*53)%experts;
        auto start=std::chrono::steady_clock::now();
        ggml_backend_tensor_set(ids,routing.data(),0,routes*4);
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute failed");ggml_backend_synchronize(backend);
        if(step>=0)times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    ggml_backend_tensor_get(y,actual.data(),0,actual.size()*4);
    for(int r=0;r<routes;++r)for(int row:{0,rows/2,rows-1}){
        double want=0;for(int c=0;c<cols;++c)want+=double(reference[size_t(row)*cols+c])*input[c];
        float got=actual[size_t(r)*rows+row];check(std::isfinite(got)&&std::abs(got-want)<2e-4*(1+std::abs(want)),"projection parity failed");
    }
    std::sort(times.begin(),times.end());double sum=0;for(auto ms:times)sum+=ms;
    printf("{\"format\":\"%s\",\"rows\":%d,\"cols\":%d,\"experts\":%d,\"routes\":%d,\"iterations\":%d,\"weight_bytes\":%zu,\"compute_device\":%d,\"weight_device\":%d,\"mean_ms\":%.6f,\"median_ms\":%.6f,\"p95_ms\":%.6f,\"sampled_parity\":true,\"synthetic_tiled_weights\":true}\n",type.c_str(),rows,cols,experts,routes,iterations,body.size()*experts,compute_device,weight_device,sum/times.size(),times[times.size()/2],times[std::min(times.size()-1,size_t(times.size()*.95))]);
    for(auto w:weights)if(header)ggml_gqh_unregister(w->data);
    ggml_backend_buffer_free(graph_buffer);ggml_free(graph_ctx);
    ggml_backend_buffer_free(descriptor_buffer);ggml_free(descriptor_ctx);
    ggml_backend_buffer_free(buffer);ggml_free(ctx);
    if(weight_backend!=backend)ggml_backend_free(weight_backend);
    ggml_backend_free(backend);return 0;
}catch(const std::exception & e){fprintf(stderr,"%s\n",e.what());return 1;}
