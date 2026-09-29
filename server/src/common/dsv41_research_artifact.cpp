#include "dsv41_research_artifact.h"
#include "dsv41_direct_io.h"
#include "gguf_inspect.h"
#include "dsv41_nvfp4_payload.h"
#include "dsv41_int3_payload.h"
#include "dsv41_cached_int3.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
namespace luce::common {
namespace {
using Json=nlohmann::json;
namespace fs=std::filesystem;
void require(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
uint64_t integer(const Json & value){
 require(value.is_number_unsigned()||(value.is_number_integer()&&value.get<int64_t>()>=0),"Expected nonnegative artifact integer");
 return value.get<uint64_t>();
}
bool digest(const std::string & value){
 return value.size()==64&&value.find_first_not_of("0123456789abcdef")==std::string::npos;
}
std::vector<uint8_t> read(const fs::path & path,uint64_t budget){
 std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"Cannot open research artifact file");
 auto end=f.tellg();require(end>=0&&uint64_t(end)<=budget&&uint64_t(end)<=uint64_t(std::numeric_limits<std::streamsize>::max()),"Research artifact file exceeds budget");
 std::vector<uint8_t> bytes(size_t(end),0);f.seekg(0);
 require(bool(f.read(reinterpret_cast<char *>(bytes.data()),std::streamsize(bytes.size()))),"Short research artifact read");
 require(f.peek()==std::char_traits<char>::eof()&&!f.bad(),"Research artifact changed size during read");return bytes;
}
struct Format{ggml_type type;bool awq,palette;};
Format format(std::string method){
 if(method=="gguf_q2_k"||method=="ggml_q2_k")return {GGML_TYPE_Q2_K,false,false};
 if(method=="gguf_iq2_xxs"||method=="ggml_iq2_xxs")return {GGML_TYPE_IQ2_XXS,false,false};
 if(method=="int3_g64"||method=="int3_g64_gptq")return {GGML_TYPE_DSV41_INT3_G64,false,false};
 if(method=="int3_g128"||method=="int3_g128_gptq")return {GGML_TYPE_DSV41_INT3_G128,false,false};
 if(method=="nvfp4_mse"||method=="nvfp4_gptq")return {GGML_TYPE_NVFP4,false,false};
 bool palette=false;const std::string suffix="_palette";
 if(method.size()>suffix.size()&&method.compare(method.size()-suffix.size(),suffix.size(),suffix)==0){method.resize(method.size()-suffix.size());palette=true;}
 for(auto item:{std::pair<const char *,ggml_type>{"gqh_t",GGML_TYPE_GQH_T},{"gqh_t_g32_r4",GGML_TYPE_GQH_T_G32_R4},{"gqh_t_g32_r3",GGML_TYPE_GQH_T_G32_R3},{"gqh2_h",GGML_TYPE_GQH2_H},{"gqh3",GGML_TYPE_GQH3},{"gqh4",GGML_TYPE_GQH4}}){
  std::string base=item.first;
  for(const char * fit:{"","_gptq","_ratio_gptq","_signround_gptq","_awq_gptq"})
   if(method==base+fit){require(base.rfind("gqh_t",0)==0||fit[0],"Unsupported bare scalar research method");require(!palette||base.rfind("gqh_t",0)==0,"Unsupported scalar GQH palette");return {item.second,std::string(fit)=="_awq_gptq",palette};}
 }
 throw std::runtime_error("Research artifact format has no qualified packed binding: "+method);
}
uint64_t le(const uint8_t * p,unsigned n){uint64_t v=0;for(unsigned i=0;i<n;++i)v|=uint64_t(p[i])<<(i*8);return v;}
}
struct Dsv41ResearchArtifact::Impl{fs::path root;Json replacements;std::shared_ptr<Dsv41PayloadCache> payload_cache;};
Dsv41ResearchArtifact::Dsv41ResearchArtifact(const std::string & directory,const std::string & source,const std::string & expected_manifest,std::shared_ptr<Dsv41PayloadCache> payload_cache):impl_(new Impl){
 impl_->payload_cache=std::move(payload_cache);
 require(digest(source),"Invalid expected source identity");
 impl_->root=fs::canonical(directory);
 auto bytes=read(impl_->root/"geoquant_artifact.json",64u<<20);
 require(expected_manifest.empty()||(digest(expected_manifest)&&sha256_bytes(bytes.data(),bytes.size())==expected_manifest),"Research artifact manifest identity mismatch");
 auto manifest=Json::parse(bytes);
 require(manifest.at("format")=="geoquant-v41-research-v1"&&manifest.at("status")=="complete"&&
         manifest.at("contract")=="native-fp8-activations-decoded-bf16-weight-gemm-v1","Unsupported research artifact contract");
 require(manifest.at("source_index_sha256")==source,"Research artifact source identity mismatch");
 impl_->replacements=manifest.at("replacements");require(impl_->replacements.is_object(),"Invalid replacement map");
 auto index_bytes=read(impl_->root/"model.safetensors.index.json",64u<<20);
 auto native=Json::parse(index_bytes).at("weight_map");require(native.is_object(),"Invalid native index");
 for(auto it=impl_->replacements.begin();it!=impl_->replacements.end();++it)
  require(!native.contains(it.key()+".weight")&&!native.contains(it.key()+".scale"),"Replacement also exists in native index");
}
uint64_t Dsv41ResearchArtifact::allocation_bytes(const std::string & name,uint64_t rows,uint64_t columns,ggml_backend_t backend) const {
 const auto & entry=impl_->replacements.at(name);
 const auto & shape=entry.at("shape");
 require(shape.is_array()&&shape.size()==2&&integer(shape[0])==rows&&integer(shape[1])==columns,"Research allocation shape mismatch");
 const auto info=format(entry.at("method").get<std::string>());
 return Dsv41PackedMatrix::allocation_bytes(backend,info.type,rows,columns,info.awq);
}

Dsv41ResearchArtifact::~Dsv41ResearchArtifact()=default;
std::unique_ptr<Dsv41PackedMatrix> Dsv41ResearchArtifact::load(const std::string & name,
 uint64_t rows,uint64_t columns,ggml_backend_t backend,uint64_t budget,uint64_t device_budget,bool cache_payload,ggml_backend_buffer_t arena,uint64_t arena_offset) const{
 auto * payload_cache=cache_payload?impl_->payload_cache.get():nullptr;
 const auto & entry=impl_->replacements.at(name);auto info=format(entry.at("method").get<std::string>());
 const auto & shape=entry.at("shape");require(shape.is_array()&&shape.size()==2&&integer(shape[0])==rows&&integer(shape[1])==columns,"Research replacement shape mismatch");
 auto hash=entry.at("sha256").get<std::string>();require(digest(hash),"Invalid replacement digest");
 auto file=fs::path(entry.at("file").get<std::string>());require(!file.is_absolute(),"Absolute research payload path");
 auto path=fs::canonical(impl_->root/file);auto relative=path.lexically_relative(impl_->root);
 require(!relative.empty()&&*relative.begin()!="..","Research payload escapes artifact root");
 uint64_t expected=integer(entry.at("bytes"));require(expected<=budget,"Research payload exceeds host budget");
 if(info.type==GGML_TYPE_DSV41_INT3_G64||info.type==GGML_TYPE_DSV41_INT3_G128){
  auto prepared=dsv41_read_cached_int3(payload_cache,path,expected,hash,
      rows,columns,info.type==GGML_TYPE_DSV41_INT3_G64?64:128,budget);
  return std::make_unique<Dsv41PackedMatrix>(backend,info.type,rows,columns,
      prepared.data(),prepared.size(),false,nullptr,0,budget-expected-prepared.size(),device_budget,arena,arena_offset);
 }
 auto blob=payload_cache?payload_cache->read(path,0,expected,true):dsv41_read_direct(path,0,expected,true);require(blob.size()==expected&&sha256_bytes(blob.data(),blob.size())==hash,"Research payload identity mismatch");
 budget-=blob.size();const uint8_t * wire=blob.data();size_t wire_bytes=blob.size();std::vector<float> awq;std::vector<uint8_t> repacked;
 if(info.type==GGML_TYPE_NVFP4){
  repacked=dsv41_repack_nvfp4(blob.data(),blob.size(),rows,columns,budget);budget-=repacked.size();wire=repacked.data();wire_bytes=repacked.size();
 }
 if(info.awq){
  require(blob.size()>=8,"Truncated AWQ safetensors header");uint64_t header_bytes=le(blob.data(),8);
  require(header_bytes<=blob.size()-8&&header_bytes<=(1u<<20),"AWQ metadata exceeds bound");
  auto header=Json::parse(blob.begin()+8,blob.begin()+8+header_bytes);
  require(header.is_object()&&header.size()==2&&header.contains("scaled_wire")&&header.contains("input_scale"),"Unexpected AWQ fields");
  uint64_t base=8+header_bytes,payload=blob.size()-base;
  auto range=[&](const char * key,const char * dtype,uint64_t elements,uint64_t width){
   const auto & t=header.at(key);const auto & sh=t.at("shape");const auto & off=t.at("data_offsets");
   require(t.at("dtype")==dtype&&sh.is_array()&&sh.size()==1&&integer(sh[0])==elements&&off.is_array()&&off.size()==2,"AWQ tensor metadata mismatch");
   uint64_t lo=integer(off[0]),hi=integer(off[1]);require(elements<=UINT64_MAX/width&&lo<=hi&&hi<=payload&&hi-lo==elements*width,"AWQ tensor range mismatch");return std::pair<uint64_t,uint64_t>{lo,hi};
  };
  const auto & ws=header.at("scaled_wire").at("shape");require(ws.is_array()&&ws.size()==1,"Invalid AWQ wire shape");
  auto w=range("scaled_wire","U8",integer(ws[0]),1),s=range("input_scale","F32",columns,4);
  require((w.first==0&&w.second==s.first&&s.second==payload)||(s.first==0&&s.second==w.first&&w.second==payload),"Overlapping or unaccounted AWQ bytes");
  require(columns<=budget/4,"AWQ extraction exceeds host budget");budget-=columns*4;
  awq.resize(size_t(columns));for(uint64_t i=0;i<columns;++i){uint32_t bits=uint32_t(le(blob.data()+base+s.first+i*4,4));std::memcpy(&awq[size_t(i)],&bits,4);}
  wire=blob.data()+base+w.first;wire_bytes=size_t(w.second-w.first);
 }
 return std::make_unique<Dsv41PackedMatrix>(backend,info.type,rows,columns,wire,wire_bytes,info.palette,awq.data(),awq.size(),budget,device_budget,arena,arena_offset);
}
}
