#include "dsv41_nvfp4_payload.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace luce::common {
namespace {
using Json=nlohmann::json;
void require(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
uint64_t integer(const Json & j){require(j.is_number_unsigned()||(j.is_number_integer()&&j.get<int64_t>()>=0),"Invalid NVFP4 integer");return j.get<uint64_t>();}
uint64_t le(const uint8_t * p,unsigned n){uint64_t v=0;for(unsigned i=0;i<n;++i)v|=uint64_t(p[i])<<(8*i);return v;}
}
std::vector<uint8_t> dsv41_repack_nvfp4(const uint8_t * data,size_t bytes,uint64_t rows,uint64_t columns,uint64_t budget){
 require(rows&&columns&&columns%64==0&&rows<=uint64_t(INT64_MAX)/columns,"Invalid NVFP4 shape");
 uint64_t blocks=rows*(columns/64);
 require(blocks<=(UINT64_MAX-5)/36,"NVFP4 output overflow");uint64_t size=5+blocks*36;
 require(size<=budget&&size<=std::numeric_limits<size_t>::max(),"NVFP4 repack exceeds host budget");
 require(data&&bytes>=8,"Truncated NVFP4 envelope");uint64_t header_bytes=le(data,8);
 require(header_bytes<=bytes-8&&header_bytes<=(1u<<20),"NVFP4 metadata exceeds bound");
 auto header=Json::parse(data+8,data+8+header_bytes);
 require(header.is_object()&&header.size()==3&&header.contains("weight_packed")&&header.contains("weight_scale")&&header.contains("weight_global_scale"),"Unexpected NVFP4 fields");
 uint64_t base=8+header_bytes,payload=bytes-base;
 auto range=[&](const char * name,const char * dtype,std::initializer_list<uint64_t> dims,uint64_t element_bytes){
  const auto & t=header.at(name);const auto & shape=t.at("shape");const auto & off=t.at("data_offsets");
  require(t.at("dtype")==dtype&&shape.is_array()&&shape.size()==dims.size()&&off.is_array()&&off.size()==2,"Invalid NVFP4 tensor metadata");
  uint64_t count=1;unsigned i=0;for(auto dim:dims){require(integer(shape[i++])==dim&&count<=UINT64_MAX/dim,"Invalid NVFP4 tensor shape");count*=dim;}
  uint64_t lo=integer(off[0]),hi=integer(off[1]);require(count<=UINT64_MAX/element_bytes&&lo<=hi&&hi<=payload&&hi-lo==count*element_bytes,"Invalid NVFP4 tensor extent");
  return std::pair<uint64_t,uint64_t>{lo,hi};
 };
 auto w=range("weight_packed","U8",{rows,columns/2},1);
 auto s=range("weight_scale","F8_E4M3",{rows,columns/16},1);
 auto g=range("weight_global_scale","F32",{1},4);
 std::array<std::pair<uint64_t,uint64_t>,3> ranges={w,s,g};std::sort(ranges.begin(),ranges.end());uint64_t end=0;
 for(auto r:ranges){require(r.first==end,"Overlapping or unaccounted NVFP4 bytes");end=r.second;}require(end==payload,"Trailing NVFP4 data");
 uint32_t bits=uint32_t(le(data+base+g.first,4));float divisor;std::memcpy(&divisor,&bits,4);
 require(std::isfinite(divisor)&&divisor>0,"Invalid NVFP4 global divisor");
 for(uint64_t i=s.first;i<s.second;++i)require(data[base+i]<127,"Invalid NVFP4 E4M3 scale");
 std::vector<uint8_t> out(size_t(size),0);std::memcpy(out.data(),data+base+g.first,4);
 const auto * codes=data+base+w.first;const auto * scales=data+base+s.first;
 for(uint64_t b=0;b<blocks;++b){
  auto * dst=out.data()+5+b*36;std::memcpy(dst,scales+b*4,4);
  for(unsigned group=0;group<4;++group)for(unsigned j=0;j<8;++j){
   uint64_t first=b*64+group*16+j,second=first+8;
   unsigned lo=(codes[first/2]>>(4*(first%2)))&15,hi=(codes[second/2]>>(4*(second%2)))&15;
   dst[4+group*8+j]=uint8_t(lo|(hi<<4));
  }
 }
 return out;
}
}
