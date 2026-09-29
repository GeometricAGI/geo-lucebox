// Both public entry points must expose the same budget. Run with `uma` on an
// integrated GPU to require physical Linux RAM rather than ROCr's virtual total.
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
int main(int argc,char **argv) {
    if(!ggml_backend_cuda_get_device_count())return 77;
    auto backend=ggml_backend_cuda_init(0);if(!backend)return 2;
    size_t a=0,b=0,ta=0,tb=0;
    ggml_backend_cuda_get_device_memory(0,&a,&ta);
    ggml_backend_dev_memory(ggml_backend_get_device(backend),&b,&tb);
    const size_t tolerance=size_t(256)<<20;
    std::printf("public_free=%zu backend_free=%zu public_total=%zu backend_total=%zu\n",a,b,ta,tb);
    if(!ta||ta!=tb||std::max(a,b)-std::min(a,b)>tolerance)return 3;
    if(argc==2&&std::string(argv[1])=="uma") {
        std::ifstream input("/proc/meminfo");std::string line;size_t total=0,available=0;
        while(std::getline(input,line)) {
            std::istringstream fields(line);std::string key,unit;size_t value;
            if(!(fields>>key>>value>>unit)||unit!="kB")continue;
            if(key=="MemTotal:")total=value*1024;
            if(key=="MemAvailable:")available=value*1024;
        }
        std::printf("physical_total=%zu physical_available=%zu\n",total,available);
        if(!total||ta!=total||a>total||std::max(a,available)-std::min(a,available)>tolerance)return 4;
    }
    ggml_backend_free(backend);return 0;
}
