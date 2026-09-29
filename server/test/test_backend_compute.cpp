// A copy-only probe cannot detect a binary missing one GPU's kernel target.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <vector>
int main() {
    const int devices=ggml_backend_cuda_get_device_count();
    if (!devices) return 77;
    for(int device=0;device<devices;++device) {
        auto backend=ggml_backend_cuda_init(device);
        if(!backend)return 2;
        constexpr int n=5120,rows=2;
        ggml_init_params params{};
        params.mem_size=16*ggml_tensor_overhead()+ggml_graph_overhead_custom(16,false);
        params.no_alloc=true;
        auto ctx=ggml_init(params);
        auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,n,rows);
        auto scale=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,n);
        auto y=ggml_mul(ctx,ggml_rms_norm(ctx,x,1e-6f),scale);
        auto graph=ggml_new_graph_custom(ctx,16,false);ggml_build_forward_expand(graph,y);
        auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
        if(!buffer)return 3;
        std::vector<float> input(n*rows),weight(n),output(n*rows);
        for(int i=0;i<n*rows;++i)input[i]=float(i%31-15)/16;
        for(int i=0;i<n;++i)weight[i]=float(i%7+1)/8;
        ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
        ggml_backend_tensor_set(scale,weight.data(),0,weight.size()*sizeof(float));
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 4;
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(y,output.data(),0,output.size()*sizeof(float));
        double worst=0;
        for(int row=0;row<rows;++row) {
            double ss=0;for(int i=0;i<n;++i)ss+=double(input[row*n+i])*input[row*n+i];
            for(int i=0;i<n;++i) {
                double expected=input[row*n+i]/std::sqrt(ss/n+1e-6)*weight[i];
                if(!std::isfinite(output[row*n+i]))return 5;
                worst=std::fmax(worst,std::fabs(output[row*n+i]-expected));
            }
        }
        std::printf("device=%d rms_norm_mul_max_error=%.9g\n",device,worst);std::fflush(stdout);
        ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
        if(worst>1e-5)return 6;
    }
    return 0;
}
