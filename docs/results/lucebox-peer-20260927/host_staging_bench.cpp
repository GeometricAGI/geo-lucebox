#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>
#define H(call) do { auto e=(call); if(e!=hipSuccess) {fprintf(stderr,"%s: %s\n",#call,hipGetErrorString(e));return 2;} } while(0)
int main() {
 int count=0;H(hipGetDeviceCount(&count)); if(count!=2)return 3;
 for(int src=0;src<2;src++) for(bool pinned:{false,true}) for(size_t bytes:{size_t(20480),size_t(122880),size_t(655360),size_t(9953280)}) {
  int dst=1-src; std::vector<uint32_t> expected(bytes/4),actual(bytes/4),pageable(pinned?0:bytes/4);
  for(size_t i=0;i<expected.size();i++)expected[i]=0x12345678u+uint32_t(i)*17;
  void *source=nullptr,*target=nullptr,*staging=nullptr;
  H(hipSetDevice(src));H(hipMalloc(&source,bytes));H(hipMemcpy(source,expected.data(),bytes,hipMemcpyHostToDevice));
  if(pinned){H(hipHostMalloc(&staging,bytes,hipHostMallocPortable));}else staging=pageable.data();
  H(hipSetDevice(dst));H(hipMalloc(&target,bytes));
  std::vector<double> times;const int iterations=200;
  for(int i=-10;i<iterations;i++) {
   auto begin=std::chrono::steady_clock::now();
   H(hipSetDevice(src));H(hipMemcpy(staging,source,bytes,hipMemcpyDeviceToHost));H(hipDeviceSynchronize());
   H(hipSetDevice(dst));H(hipMemcpy(target,staging,bytes,hipMemcpyHostToDevice));H(hipDeviceSynchronize());
   auto end=std::chrono::steady_clock::now();if(i>=0)times.push_back(std::chrono::duration<double,std::micro>(end-begin).count());
  }
  H(hipMemcpy(actual.data(),target,bytes,hipMemcpyDeviceToHost));size_t bad=0;for(size_t i=0;i<actual.size();i++)bad+=actual[i]!=expected[i];
  std::sort(times.begin(),times.end());double sum=0;for(auto t:times)sum+=t;
  printf("{\"source\":%d,\"destination\":%d,\"pinned\":%s,\"bytes\":%zu,\"iterations\":%d,\"mismatches\":%zu,\"mean_us\":%.3f,\"median_us\":%.3f,\"p95_us\":%.3f}\n",src,dst,pinned?"true":"false",bytes,iterations,bad,sum/times.size(),times[times.size()/2],times[times.size()*95/100]);fflush(stdout);
  H(hipFree(target));H(hipSetDevice(src));H(hipFree(source));if(pinned)H(hipHostFree(staging));if(bad)return 4;
 }
}
