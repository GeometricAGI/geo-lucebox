// Exercise actual GGML synchronous/asynchronous cross-device fallback paths.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cstdint>
#include <cstdio>
#include <vector>
int main() {
    if (ggml_backend_cuda_get_device_count() < 2) return 77;
    ggml_backend_t backends[2] = {ggml_backend_cuda_init(0), ggml_backend_cuda_init(1)};
    if (!backends[0] || !backends[1]) return 2;
    bool ok = true;
    for (int source = 0; source < 2; ++source) for (bool async : {false, true}) {
        const int destination = 1-source;
        constexpr int64_t n = 16384;
        ggml_init_params params{}; params.mem_size=4*ggml_tensor_overhead();params.no_alloc=true;
        ggml_context * ctx=ggml_init(params);
        ggml_tensor * src=ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_tensor * dst=ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_backend_buffer_t sb=ggml_backend_alloc_buffer(backends[source],n*4);
        ggml_backend_buffer_t db=ggml_backend_alloc_buffer(backends[destination],n*4);
        if (!sb || !db) return 3;
        ggml_backend_tensor_alloc(sb,src,ggml_backend_buffer_get_base(sb));
        ggml_backend_tensor_alloc(db,dst,ggml_backend_buffer_get_base(db));
        std::vector<uint32_t> expected(n), actual(n);
        for (int i=0;i<n;++i) expected[i]=0x12345678u+uint32_t(i)*17;
        ggml_backend_tensor_set(src,expected.data(),0,n*4);
        ggml_backend_tensor_set(dst,actual.data(),0,n*4);
        if (async) ggml_backend_tensor_copy_async(backends[source],backends[destination],src,dst);
        else ggml_backend_tensor_copy(src,dst);
        ggml_backend_synchronize(backends[source]);ggml_backend_synchronize(backends[destination]);
        ggml_backend_tensor_get(dst,actual.data(),0,n*4);
        size_t bad=0;for(int i=0;i<n;++i)bad+=actual[i]!=expected[i];
        std::printf("source=%d destination=%d async=%d mismatches=%zu\n",source,destination,int(async),bad);
        ok &= bad==0;
        ggml_backend_buffer_free(db);ggml_backend_buffer_free(sb);ggml_free(ctx);
    }
    ggml_backend_free(backends[1]);ggml_backend_free(backends[0]);return ok?0:1;
}
