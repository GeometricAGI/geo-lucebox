#include "dsv41_int3_payload.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace luce::common {
namespace {
using Json=nlohmann::json;
void require(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
uint64_t integer(const Json & j){require(j.is_number_unsigned()||(j.is_number_integer()&&j.get<int64_t>()>=0),"Invalid INT3 integer");return j.get<uint64_t>();}
uint64_t le(const uint8_t * p,unsigned n){uint64_t v=0;for(unsigned i=0;i<n;++i)v|=uint64_t(p[i])<<(8*i);return v;}
}
std::vector<uint8_t> dsv41_repack_int3(const uint8_t * data,size_t bytes,uint64_t rows,uint64_t cols,unsigned group,uint64_t budget){
 require((group==64||group==128)&&rows&&cols&&cols%group==0&&rows<=uint64_t(INT64_MAX)/cols,"Invalid INT3 shape/group");
 uint64_t groups=cols/group,blocks=rows*groups,stride=4+group*3/8;
 require(blocks<=(UINT64_MAX-5)/stride,"INT3 output overflow");uint64_t size=5+blocks*stride;
 require(size<=budget&&size<=std::numeric_limits<size_t>::max(),"INT3 repack exceeds budget");
 require(data&&bytes>=8,"Truncated INT3 envelope");uint64_t hn=le(data,8);
 require(hn<=bytes-8&&hn<=(1u<<20),"INT3 header exceeds bound");
 auto h=Json::parse(data+8,data+8+hn);require(h.is_object()&&h.size()==2&&h.contains("weight_packed")&&h.contains("weight_scale"),"Unexpected INT3 fields");
 uint64_t base=8+hn,payload=bytes-base,words=(cols+9)/10;
 auto range=[&](const char * key,const char * dtype,uint64_t columns){
  const auto & t=h.at(key);const auto & shape=t.at("shape");const auto & off=t.at("data_offsets");
  require(t.at("dtype")==dtype&&shape.is_array()&&shape.size()==2&&integer(shape[0])==rows&&integer(shape[1])==columns&&off.is_array()&&off.size()==2,"Invalid INT3 tensor metadata");
  require(rows<=UINT64_MAX/columns&&rows*columns<=UINT64_MAX/4,"INT3 tensor overflow");
  uint64_t lo=integer(off[0]),hi=integer(off[1]);require(lo<=hi&&hi<=payload&&hi-lo==rows*columns*4,"Invalid INT3 tensor extent");return std::pair<uint64_t,uint64_t>{lo,hi};
 };
 auto w=range("weight_packed","I32",words),s=range("weight_scale","F32",groups);
 require((w.first==0&&w.second==s.first&&s.second==payload)||(s.first==0&&s.second==w.first&&w.second==payload),"Overlapping or unaccounted INT3 bytes");
 const auto * codes=data+base+w.first;const auto * scales=data+base+s.first;
 for(uint64_t b=0;b<blocks;++b){uint32_t bits=uint32_t(le(scales+4*b,4));float scale;std::memcpy(&scale,&bits,4);require(std::isfinite(scale)&&scale>=0,"Invalid INT3 scale");}
 for(uint64_t row=0;row<rows;++row){
  for(uint64_t word=0;word<words;++word)require((le(codes+4*(row*words+word),4)>>30)==0,"Noncanonical INT3 high bits");
  uint32_t last=uint32_t(le(codes+4*((row+1)*words-1),4));
  for(uint64_t col=cols;col<words*10;++col)require(((last>>(3*(col%10)))&7)==4,"Nonzero INT3 padded weights");
 }
 std::vector<uint8_t> out(size_t(size),0);out[2]=128;out[3]=63; // LE float32 1.0.
 for(uint64_t row=0;row<rows;++row)for(uint64_t g=0;g<groups;++g){
  auto * dst=out.data()+5+(row*groups+g)*stride;std::memcpy(dst,scales+4*(row*groups+g),4);
  for(unsigned j=0;j<group;++j){uint64_t col=g*group+j;unsigned code=unsigned((le(codes+4*(row*words+col/10),4)>>(3*(col%10)))&7);
   unsigned bit=j*3,shift=bit%8;dst[4+bit/8]|=uint8_t(code<<shift);if(shift+3>8)dst[5+bit/8]|=uint8_t(code>>(8-shift));
  }
 }
 return out;
}
}
