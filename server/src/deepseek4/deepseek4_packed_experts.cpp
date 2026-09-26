#include "deepseek4_packed_experts.h"
#include "deepseek4_internal.h"
#include "common/dsv41_research_artifact.h"
#include "common/dsv41_native_matrix.h"
#include "ggml-packed-experts.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <stdexcept>
namespace luce::common {
struct DeepSeek4PackedExperts {
    ggml_context * ctx=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    ggml_backend_t backend=nullptr;
    std::vector<ggml_backend_buffer_t> arenas;
    std::vector<std::unique_ptr<Dsv41PackedMatrix>> packed;
    std::vector<std::unique_ptr<Dsv41NativeMatrix>> native;
    ~DeepSeek4PackedExperts() {
        if(backend)ggml_backend_synchronize(backend);
        packed.clear();native.clear();
        for(auto arena:arenas)ggml_backend_buffer_free(arena);
        if(buffer)ggml_backend_buffer_free(buffer);
        if(ctx)ggml_free(ctx);
    }
};
bool load_deepseek4_packed_experts(DeepSeek4Weights & w,const std::string & root,
 const std::string & source,const std::string & manifest,const std::string & index,std::string & error) try {
    if(w.arch!="deepseek41"||!w.backend||w.n_layer<=0||w.n_layer>128||w.n_expert<=0||w.n_expert>4096||w.n_ff_exp<=0||w.n_embd<=0||w.layers.size()!=size_t(w.n_layer))throw std::runtime_error("invalid packed expert model geometry");
    auto owner=std::make_shared<DeepSeek4PackedExperts>();owner->backend=w.backend;
    Dsv41ResearchArtifact research(root,source,manifest);
    Dsv41NativeStore native(root,index);
    const auto path=std::filesystem::path(root)/"geoquant_artifact.json";
    if(std::filesystem::file_size(path)>(64u<<20))throw std::runtime_error("artifact manifest exceeds bound");
    std::ifstream input(path);nlohmann::json meta;input>>meta;
    const auto & replacements=meta.at("replacements");
    owner->ctx=ggml_init({size_t(w.n_layer*3+1)*ggml_tensor_overhead()+4096,nullptr,true});
    if(!owner->ctx)throw std::runtime_error("packed descriptor context allocation failed");
    std::vector<std::array<ggml_tensor *,3>> tensors(w.n_layer);
    for(auto & layer:tensors)for(auto & t:layer)t=ggml_new_tensor_2d(owner->ctx,GGML_TYPE_I8,sizeof(ggml_packed_expert),w.n_expert);
    owner->buffer=ggml_backend_alloc_ctx_tensors(owner->ctx,w.backend);
    if(!owner->buffer)throw std::runtime_error("packed descriptor allocation failed");
    ggml_backend_buffer_set_usage(owner->buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    size_t free_bytes=0,total=0;auto device=ggml_backend_get_device(w.backend);
    ggml_backend_dev_memory(device,&free_bytes,&total);
    const bool cpu=ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_CPU;
    uint64_t remaining=cpu?UINT64_MAX:free_bytes;
    constexpr uint64_t reserve=3ull<<30;
    if(!cpu){if(remaining<=reserve)throw std::runtime_error("no resident expert budget after 3 GiB workspace reserve");remaining-=reserve;}
    uint64_t bytes=ggml_backend_buffer_get_size(owner->buffer);
    owner->packed.reserve(size_t(w.n_layer)*w.n_expert*3);
    const char * names[]={"w1","w3","w2"};
    for(int layer=0;layer<w.n_layer;++layer){
        uint64_t arena_bytes=0;
        for(int surface=0;surface<3;++surface)for(int expert=0;expert<w.n_expert;++expert) {
            const std::string name="layers."+std::to_string(layer)+".ffn.experts."+std::to_string(expert)+"."+names[surface];
            if(!replacements.contains(name))continue;
            const uint64_t size=research.allocation_bytes(name,surface==2?w.n_embd:w.n_ff_exp,surface==2?w.n_ff_exp:w.n_embd,w.backend);
            if(arena_bytes>remaining||size>remaining-arena_bytes)throw std::runtime_error("resident layer exceeds expert budget");
            arena_bytes+=size;
        }
        ggml_backend_buffer_t arena=nullptr;
        if(arena_bytes) {
            if(!cpu) {
                ggml_backend_dev_memory(device,&free_bytes,&total);
                if(free_bytes<reserve||arena_bytes>free_bytes-reserve)throw std::runtime_error("resident layer would consume workspace reserve");
            }
            arena=ggml_backend_alloc_buffer(w.backend,arena_bytes);
            if(!arena)throw std::runtime_error("resident layer arena allocation failed");
            owner->arenas.push_back(arena);
            ggml_backend_buffer_set_usage(arena,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        uint64_t arena_offset=0;
        for(int surface=0;surface<3;++surface){
            std::vector<ggml_packed_expert> desc(w.n_expert);
            const uint64_t rows=surface==2?w.n_embd:w.n_ff_exp,cols=surface==2?w.n_ff_exp:w.n_embd;
            for(int expert=0;expert<w.n_expert;++expert){
                const std::string name="layers."+std::to_string(layer)+".ffn.experts."+std::to_string(expert)+"."+names[surface];
                ggml_tensor * tensor=nullptr;uint64_t used=0;
                if(replacements.contains(name)){
                    auto p=research.load(name,rows,cols,w.backend,1ull<<30,remaining,false,arena,arena_offset);tensor=p->tensor();used=p->device_bytes();arena_offset+=used;owner->packed.push_back(std::move(p));
                }else{
                    auto p=native.load(name,rows,cols,w.backend,1ull<<30,remaining,false);
                    if(p->scales())throw std::runtime_error("native FP8 experts need an explicit scaled binding: "+name);
                    tensor=p->weight();used=p->device_bytes();owner->native.push_back(std::move(p));
                }
                if(used>remaining)throw std::runtime_error("resident expert allocation exceeded budget");
                remaining-=used;bytes+=used;
                if(!ggml_packed_expert_init(&desc[expert],tensor))throw std::runtime_error("unsupported packed expert descriptor: "+name);
            }
            ggml_backend_tensor_set(tensors[layer][surface],desc.data(),0,desc.size()*sizeof(desc[0]));
        }
        if(arena_offset!=arena_bytes)throw std::runtime_error("resident arena accounting mismatch");
        std::fprintf(stderr,"[deepseek41-packed] resident layer %d/%d: %.3f GiB (no eviction)\n",layer+1,w.n_layer,double(bytes)/(1ull<<30));
    }
    for(int layer=0;layer<w.n_layer;++layer)w.layers[layer].packed_experts=tensors[layer];
    w.packed_expert_owner=std::move(owner);
    w.packed_research_directory=root;w.packed_native_index_sha256=index;return true;
} catch(const std::exception & e){error=e.what();return false;}
}
