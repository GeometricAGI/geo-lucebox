#include "dsv41_native_fp4.h"
#include <stdexcept>
namespace luce::common {
std::vector<uint8_t> dsv41_native_fp4_to_mxfp4(const std::vector<uint8_t> & raw,
 const std::vector<uint8_t> & scales,uint64_t rows,uint64_t columns,uint64_t budget){
 if(!rows||!columns||columns%32||rows>UINT64_MAX/columns)throw std::runtime_error("Invalid native FP4 shape");
 uint64_t cells=rows*columns,blocks=cells/32;
 if(raw.size()!=cells/2||scales.size()!=blocks||blocks>UINT64_MAX/17||blocks*17>budget)
  throw std::runtime_error("Native FP4 size/budget mismatch");
 for(auto s:scales)if(s==255)throw std::runtime_error("Native FP4 NaN scale");
 std::vector<uint8_t> result(blocks*17);
 for(uint64_t b=0;b<blocks;++b){
  result[b*17]=scales[b];
  for(unsigned j=0;j<16;++j){
   unsigned lo=(raw[b*16+j/2]>>(4*(j%2)))&15;
   unsigned hi=(raw[b*16+(j+16)/2]>>(4*(j%2)))&15;
   result[b*17+1+j]=uint8_t(lo|(hi<<4));
  }
 }
 return result;
}
}
