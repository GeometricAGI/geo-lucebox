#include "common/dsv41_packed_matrix.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gqh.h"
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>
using namespace luce::common;
static void check(bool ok,const char * message){if(!ok)throw std::runtime_error(message);}
int main(int argc,char ** argv)try {
    check(argc==3,"usage: test_packed_arena cpu|cuda ternary_fixture");
    auto backend=std::string(argv[1])=="cpu"?ggml_backend_cpu_init():ggml_backend_cuda_init(0);
    check(backend,"backend init failed");
    std::ifstream file(argv[2],std::ios::binary|std::ios::ate);check(bool(file),"fixture open failed");
    std::vector<uint8_t> wire(size_t(file.tellg()));file.seekg(0);file.read(reinterpret_cast<char *>(wire.data()),wire.size());
    const auto bytes=Dsv41PackedMatrix::allocation_bytes(backend,GGML_TYPE_GQH_T,64,1024,true);
    auto arena=ggml_backend_alloc_buffer(backend,bytes*3);check(arena,"arena allocation failed");
    std::vector<std::unique_ptr<Dsv41PackedMatrix>> matrices;
    std::vector<std::vector<uint8_t>> expected;
    std::vector<float> awq(1024,1.5f);
    for(int i=0;i<3;++i){
        wire[5]=uint8_t(0x28+i);expected.emplace_back(wire.begin()+5,wire.end());
        matrices.push_back(std::make_unique<Dsv41PackedMatrix>(backend,GGML_TYPE_GQH_T,64,1024,wire.data(),wire.size(),false,awq.data(),awq.size(),1<<20,bytes,arena,bytes*i));
        check(matrices.back()->device_bytes()==bytes,"arena footprint mismatch");
    }
    auto check_matrix=[&](int i){
        std::vector<uint8_t> got(expected[i].size());auto * t=matrices[i]->tensor();
        ggml_backend_tensor_get(t,got.data(),0,got.size());check(got==expected[i],"arena payload overlap");
        const float * host=nullptr;const void * device=nullptr;int64_t columns=0;size_t offset=0;
        check(ggml_gqh_lookup_input_scale(t->data,&host,&device,&columns,&offset)&&columns==1024&&host[0]==1.5f,"arena conditioning mismatch");
    };
    for(int i=0;i<3;++i)check_matrix(i);
    matrices[1].reset();check_matrix(0);check_matrix(2);
    for(auto offset:{uint64_t(1),bytes*3}){
        bool rejected=false;
        try{Dsv41PackedMatrix bad(backend,GGML_TYPE_GQH_T,64,1024,wire.data(),wire.size(),false,awq.data(),awq.size(),1<<20,bytes,arena,offset);}
        catch(const std::runtime_error &){rejected=true;}
        check(rejected,"invalid arena range accepted");check_matrix(0);check_matrix(2);
    }
    matrices.clear();ggml_backend_buffer_free(arena);ggml_backend_free(backend);
    puts("packed arena: disjoint payloads/scales, accounting, borrowed lifetime and invalid ranges passed");return 0;
}catch(const std::exception & e){fprintf(stderr,"%s\n",e.what());return 1;}
