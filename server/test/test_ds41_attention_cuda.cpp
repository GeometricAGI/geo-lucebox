#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

// Independent scalar softmax/value oracle, including the sink denominator.
static void check(ggml_backend_t backend, int rows, int tokens, bool half, bool indexed) {
    constexpr int dim = 512, heads = 4, raw = 128, selected = 32;
    auto * ctx = ggml_init({4u << 20, nullptr, true});
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, tokens, heads);
    auto * kv = ggml_new_tensor_2d(ctx, half ? GGML_TYPE_F16 : GGML_TYPE_F32, dim, rows);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rows, tokens);
    auto * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads);
    auto * indices = indexed ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, selected, tokens) : nullptr;
    const float scale = 1.0f / std::sqrt(float(dim));
    auto * out = ggml_flash_attn_ext(ctx, q, kv, kv, mask, scale, 0, 0);
    ggml_flash_attn_ext_add_sinks(out, sinks);
    ggml_flash_attn_ext_set_ds4_sparse(out, raw, raw, indexed ? -selected : 0, 32);
    if (indices) ggml_flash_attn_ext_set_ds4_indexer_topk(out, indices);
    require(ggml_backend_supports_op(backend, out), "CUDA rejected DS4 attention");
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "allocation failed");
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    std::vector<float> qv(dim*tokens*heads), kvv(dim*rows), sv(heads), actual(dim*tokens*heads);
    std::vector<ggml_fp16_t> kh(kvv.size()), mv(rows*tokens);
    std::vector<int> ids(selected*tokens);
    for (size_t i=0;i<qv.size();++i) qv[i]=std::sin(float(i)*.031f)*.8f;
    for (size_t i=0;i<kvv.size();++i) {
        kvv[i]=std::cos(float(i)*.017f)*.7f;
        kh[i]=ggml_fp32_to_fp16(kvv[i]);
        if (half) kvv[i]=ggml_fp16_to_fp32(kh[i]);
    }
    for (int h=0;h<heads;++h) sv[h]=float(h)-1;
    for (int replay=0;replay<3;++replay) {
        for(int t=0;t<tokens;++t) {
            for(int j=0;j<selected;++j) ids[t*selected+j]=(j*3+t+replay)%(rows-raw);
            for(int r=0;r<rows;++r) {
                bool visible=replay != 2 && (r+t+replay)%7!=0;
                if(indexed && r>=raw) visible=visible && std::find(ids.begin()+t*selected,ids.begin()+(t+1)*selected,r-raw)!=ids.begin()+(t+1)*selected;
                mv[t*rows+r]=ggml_fp32_to_fp16(visible ? 0.f : -INFINITY);
            }
        }
        ggml_backend_tensor_set(q,qv.data(),0,qv.size()*4);
        ggml_backend_tensor_set(kv,half ? (void*)kh.data() : (void*)kvv.data(),0,ggml_nbytes(kv));
        ggml_backend_tensor_set(mask,mv.data(),0,mv.size()*2);
        ggml_backend_tensor_set(sinks,sv.data(),0,sv.size()*4);
        if(indices) ggml_backend_tensor_set(indices,ids.data(),0,ids.size()*sizeof(int));
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"graph compute failed");
        ggml_backend_tensor_get(out,actual.data(),0,actual.size()*4);
        for(int t=0;t<tokens;++t) for(int h=0;h<heads;++h) {
            std::vector<double> scores(rows);
            double max_score=sv[h];
            for(int r=0;r<rows;++r) {
                double score=ggml_fp16_to_fp32(mv[t*rows+r]);
                for(int d=0;d<dim;++d) score+=double(qv[(h*tokens+t)*dim+d])*kvv[r*dim+d]*scale;
                scores[r]=score;max_score=std::max(max_score,score);
            }
            double denom=std::exp(sv[h]-max_score);
            for(auto & score:scores){score=std::exp(score-max_score);denom+=score;}
            for(int d=0;d<dim;++d) {
                double expected=0;
                for(int r=0;r<rows;++r)expected+=scores[r]*kvv[r*dim+d];
                expected/=denom;
                float got=actual[(t*heads+h)*dim+d];
                if(!std::isfinite(got)||std::abs(got-expected)>3e-4*(1+std::abs(expected))) {
                    std::fprintf(stderr,"rows=%d tokens=%d half=%d indexed=%d got=%g expected=%g\n",rows,tokens,half,indexed,got,expected);
                    throw std::runtime_error("attention differs from scalar oracle");
                }
            }
        }
    }
    ggml_backend_buffer_free(buffer);ggml_free(ctx);
}
int main() try {
    auto backend=ggml_backend_cuda_init(0);require(backend,"CUDA initialization failed");
    for(bool half:{false,true})for(bool indexed:{false,true})for(int rows:{255,256,257,2048})for(int tokens:{1,3})check(backend,rows,tokens,half,indexed);
    ggml_backend_free(backend);std::puts("DS4 D=512 attention scalar-oracle parity passed");return 0;
} catch(const std::exception & e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
