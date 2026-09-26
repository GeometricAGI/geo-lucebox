#include "dsv41_native_matrix.h"
#include "dsv41_direct_io.h"
#include "dsv41_native_fp4.h"
#include "gguf_inspect.h"
#include <nlohmann/json.hpp>
#include <climits>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <map>
#include <chrono>
#include <mutex>
#include <sys/stat.h>
namespace luce::common {
namespace {
using Json=nlohmann::json;
namespace fs=std::filesystem;
void require(bool v,const char * why){if(!v)throw std::runtime_error(why);}
uint64_t number(const Json & j){require(j.is_number_unsigned()||(j.is_number_integer()&&j.get<int64_t>()>=0),"Invalid native tensor integer");return j.get<uint64_t>();}
uint64_t align(uint64_t n,uint64_t a){require(a&&n<=UINT64_MAX-(a-1),"Native allocation overflow");return (n+a-1)/a*a;}
std::vector<uint8_t> read(std::ifstream & f,uint64_t offset,uint64_t size){
 require(offset<=uint64_t(std::numeric_limits<std::streamoff>::max())&&size<=uint64_t(std::numeric_limits<std::streamsize>::max())&&size<=SIZE_MAX,"Native read overflow");
 std::vector<uint8_t> out(size_t(size),0);f.seekg(std::streamoff(offset));require(bool(f)&&bool(f.read(reinterpret_cast<char *>(out.data()),std::streamsize(size))),"Short native tensor read");return out;
}
struct ShardIdentity {
 uint64_t device,inode,bytes;int64_t msec,mnano,csec,cnano;
 bool operator==(const ShardIdentity & b)const{return device==b.device&&inode==b.inode&&bytes==b.bytes&&msec==b.msec&&mnano==b.mnano&&csec==b.csec&&cnano==b.cnano;}
};
ShardIdentity shard_identity(const fs::path & path){
 struct stat info{};require(::stat(path.c_str(),&info)==0&&info.st_size>=8,"Cannot stat native shard");
#ifdef __APPLE__
 return {uint64_t(info.st_dev),uint64_t(info.st_ino),uint64_t(info.st_size),info.st_mtimespec.tv_sec,info.st_mtimespec.tv_nsec,info.st_ctimespec.tv_sec,info.st_ctimespec.tv_nsec};
#else
 return {uint64_t(info.st_dev),uint64_t(info.st_ino),uint64_t(info.st_size),info.st_mtim.tv_sec,info.st_mtim.tv_nsec,info.st_ctim.tv_sec,info.st_ctim.tv_nsec};
#endif
}
struct CachedHeader {ShardIdentity identity;Json tensors;uint64_t payload;};
struct HeaderCache {
 std::mutex mutex;std::map<fs::path,CachedHeader> shards;std::map<fs::path,fs::path> resolved_paths;uint64_t serialized_bytes=0;
 // JSON/container overhead is additional metadata covered by the runtime reserve.
 static constexpr uint64_t serialized_limit=64ull<<20;
};
struct Slice {fs::path path;Json metadata;uint64_t offset=0,bytes=0;};
Slice slice(const fs::path & root,const Json & mapping,HeaderCache & cache,const std::string & name){
 fs::path relative=mapping.at(name).get<std::string>();require(!relative.is_absolute(),"Absolute native shard path");
 auto path=fs::canonical(root/relative),confined=path.lexically_relative(root);
 require(!confined.empty()&&*confined.begin()!="..","Native shard escapes checkpoint");
 std::lock_guard<std::mutex> lock(cache.mutex);
 auto resolution=cache.resolved_paths.emplace(relative,path);
 require(resolution.second||resolution.first->second==path,"Native shard path changed after metadata was cached");
 const auto identity=shard_identity(path);auto found=cache.shards.find(path);
 if(found==cache.shards.end()){
  std::ifstream file(path,std::ios::binary);require(bool(file),"Cannot open native shard");
  auto prefix=read(file,0,8);uint64_t hn=0;for(unsigned i=0;i<8;++i)hn|=uint64_t(prefix[i])<<(i*8);
  require(hn<=HeaderCache::serialized_limit-cache.serialized_bytes&&hn<=identity.bytes-8,"Native header cache exceeds bound");
  auto header=Json::parse(read(file,8,hn));require(header.is_object(),"Invalid native header");
  require(shard_identity(path)==identity,"Native shard changed while reading metadata");
  found=cache.shards.emplace(path,CachedHeader{identity,std::move(header),8+hn}).first;
  cache.serialized_bytes+=hn;
 }else require(found->second.identity==identity,"Native shard changed after metadata was cached");
 const auto & header=found->second;Slice result;result.path=path;result.metadata=header.tensors.at(name);
 const auto & offsets=result.metadata.at("data_offsets");require(offsets.is_array()&&offsets.size()==2,"Invalid native offsets");
 uint64_t lo=number(offsets[0]),hi=number(offsets[1]);require(lo<=hi&&hi<=identity.bytes-header.payload,"Native extent outside shard");
 result.offset=header.payload+lo;result.bytes=hi-lo;return result;
}
void shape(const Slice & s,uint64_t rows,uint64_t columns,const char * dtype){
 const auto & v=s.metadata.at("shape");require(v.is_array()&&v.size()==2&&number(v[0])==rows&&number(v[1])==columns&&s.metadata.at("dtype")==dtype,"Native shape/dtype mismatch");
 require(rows&&columns&&rows<=UINT64_MAX/columns&&s.bytes==rows*columns,"Native byte extent mismatch");
}
}
struct Dsv41NativeMatrix::Impl {
 ggml_backend_t backend=nullptr;ggml_context * context=nullptr;ggml_backend_buffer_t buffer=nullptr;
 ggml_tensor * weight=nullptr,* scales=nullptr;
 ~Impl(){if(buffer){ggml_backend_synchronize(backend);ggml_backend_buffer_free(buffer);}if(context)ggml_free(context);}
};
Dsv41NativeMatrix::Dsv41NativeMatrix():impl_(new Impl){}
Dsv41NativeMatrix::~Dsv41NativeMatrix()=default;
ggml_tensor * Dsv41NativeMatrix::weight()const{return impl_->weight;}
ggml_tensor * Dsv41NativeMatrix::scales()const{return impl_->scales;}
uint64_t Dsv41NativeMatrix::device_bytes()const{return ggml_backend_buffer_get_size(impl_->buffer);}
struct Dsv41NativeStore::Impl {fs::path root;Json mapping;mutable HeaderCache headers;mutable std::mutex timing_mutex;mutable Dsv41NativeLoadStats timing;std::shared_ptr<Dsv41PayloadCache> payload_cache;};
Dsv41NativeStore::Dsv41NativeStore(const std::string & directory,const std::string & identity,std::shared_ptr<Dsv41PayloadCache> payload_cache):impl_(new Impl){
 impl_->payload_cache=std::move(payload_cache);
 require(identity.size()==64&&identity.find_first_not_of("0123456789abcdef")==std::string::npos,"Invalid native index identity");
 impl_->root=fs::canonical(directory);std::ifstream f(impl_->root/"model.safetensors.index.json",std::ios::binary|std::ios::ate);
 require(bool(f),"Missing native index");auto size=f.tellg();require(size>0&&size<=64*1024*1024,"Native index exceeds bound");
 auto bytes=read(f,0,uint64_t(size));require(sha256_bytes(bytes.data(),bytes.size())==identity,"Native index identity mismatch");
 impl_->mapping=Json::parse(bytes).at("weight_map");require(impl_->mapping.is_object(),"Invalid native mapping");
}
Dsv41NativeStore::~Dsv41NativeStore()=default;
Dsv41NativeLoadStats Dsv41NativeStore::load_stats()const{std::lock_guard<std::mutex> lock(impl_->timing_mutex);return impl_->timing;}
Dsv41NativeEngram Dsv41NativeStore::engram(const std::string & name, uint64_t rows) const {
    auto w = slice(impl_->root, impl_->mapping, impl_->headers, name + ".weight");
    auto s = slice(impl_->root, impl_->mapping, impl_->headers, name + ".scale");
    shape(w, rows, 256, "F8_E4M3");
    shape(s, rows, 8, "F8_E8M0");
    require(w.path == s.path, "Engram weight and scales must share a native shard");
    require(w.offset + w.bytes <= s.offset || s.offset + s.bytes <= w.offset, "Overlapping native engram ranges");
    return {w.path.string(), w.offset, s.offset, rows};
}

struct Dsv41NativeTensor::Impl {
 ggml_backend_t backend=nullptr;ggml_context * context=nullptr;ggml_backend_buffer_t buffer=nullptr;ggml_tensor * tensor=nullptr;
 ~Impl(){if(buffer){ggml_backend_synchronize(backend);ggml_backend_buffer_free(buffer);}if(context)ggml_free(context);}
};
Dsv41NativeTensor::Dsv41NativeTensor():impl_(new Impl){}
Dsv41NativeTensor::~Dsv41NativeTensor()=default;
ggml_tensor * Dsv41NativeTensor::tensor()const{return impl_->tensor;}
uint64_t Dsv41NativeTensor::device_bytes()const{return ggml_backend_buffer_get_size(impl_->buffer);}
std::unique_ptr<Dsv41NativeTensor> Dsv41NativeStore::load_tensor(const std::string & name,
 ggml_type type,const std::vector<uint64_t> & source_shape,ggml_backend_t backend,uint64_t host_budget,uint64_t device_budget,bool promote)const{
 require(backend&&source_shape.size()<=4&&(!promote||type==GGML_TYPE_BF16),"Invalid native dense tensor request");
 const char * dtype=type==GGML_TYPE_BF16?"BF16":type==GGML_TYPE_F32?"F32":type==GGML_TYPE_I32?"I32":type==GGML_TYPE_I64?"I64":nullptr;
 require(dtype,"Unsupported native dense dtype");uint64_t elements=1,width=type==GGML_TYPE_BF16?2:type==GGML_TYPE_I64?8:4;
 int64_t dims[4]={1,1,1,1};
 for(size_t i=0;i<source_shape.size();++i){uint64_t n=source_shape[i];require(n&&n<=uint64_t(INT64_MAX)/elements,"Native dense shape overflow");elements*=n;dims[source_shape.size()-1-i]=int64_t(n);}
 require(elements<=uint64_t(INT64_MAX)/width&&elements*width<=host_budget,"Native dense payload exceeds host budget");
 if(promote)require(elements*width<=host_budget/3,"Native dense promotion exceeds host budget");
 auto s=slice(impl_->root,impl_->mapping,impl_->headers,name);const auto & shape=s.metadata.at("shape");
 require(s.metadata.at("dtype")==dtype&&shape.is_array()&&shape.size()==source_shape.size()&&s.bytes==elements*width,"Native dense metadata mismatch");
 for(size_t i=0;i<source_shape.size();++i)require(number(shape[i])==source_shape[i],"Native dense shape mismatch");
 auto out=std::unique_ptr<Dsv41NativeTensor>(new Dsv41NativeTensor);auto & p=*out->impl_;p.backend=backend;
 p.context=ggml_init({4*ggml_tensor_overhead()+4096,nullptr,true});require(p.context,"Native dense context allocation failed");
 p.tensor=ggml_new_tensor(p.context,promote?GGML_TYPE_F32:type,int(source_shape.empty()?1:source_shape.size()),dims);
 auto buft=ggml_backend_get_default_buffer_type(backend);uint64_t allocation=align(ggml_backend_buft_get_alloc_size(buft,p.tensor),ggml_backend_buft_get_alignment(buft));
 require(allocation<=device_budget&&allocation<=SIZE_MAX,"Native dense device budget exceeded");
 auto raw=dsv41_read_direct(s.path,s.offset,s.bytes);
 if(type==GGML_TYPE_BF16)for(size_t i=0;i<raw.size();i+=2){unsigned bits=unsigned(raw[i])|(unsigned(raw[i+1])<<8);require((bits&0x7f80)!=0x7f80,"Nonfinite native BF16 weight");}
 if(type==GGML_TYPE_F32)for(size_t i=0;i<raw.size();i+=4){uint32_t bits=uint32_t(raw[i])|(uint32_t(raw[i+1])<<8)|(uint32_t(raw[i+2])<<16)|(uint32_t(raw[i+3])<<24);require((bits&0x7f800000)!=0x7f800000,"Nonfinite native F32 weight");}
 p.buffer=ggml_backend_alloc_buffer(backend,size_t(allocation));require(p.buffer&&ggml_backend_buffer_get_size(p.buffer)<=device_budget,"Native dense backend allocation failed or exceeded budget");
 require(ggml_backend_tensor_alloc(p.buffer,p.tensor,ggml_backend_buffer_get_base(p.buffer))==GGML_STATUS_SUCCESS,"Native dense tensor binding failed");
 if(promote){
  std::vector<float> expanded(size_t(elements),0.f);
  for(size_t i=0;i<expanded.size();++i){ggml_bf16_t value;value.bits=uint16_t(raw[2*i])|(uint16_t(raw[2*i+1])<<8);expanded[i]=ggml_bf16_to_fp32(value);}
  ggml_backend_tensor_set(p.tensor,expanded.data(),0,expanded.size()*sizeof(float));
 }else ggml_backend_tensor_set(p.tensor,raw.data(),0,raw.size());return out;
}
std::unique_ptr<Dsv41NativeTensor> Dsv41NativeStore::load_bf16_rows(const std::string & name,
 uint64_t rows,uint64_t columns,const std::vector<uint32_t> & ids,ggml_backend_t backend,uint64_t host_budget,uint64_t device_budget)const{
 require(backend&&rows&&columns&&rows<=uint64_t(INT64_MAX)/2/columns&&!ids.empty()&&ids.size()<=32,"Invalid native row request");
 require(columns<=uint64_t(INT64_MAX)/2/ids.size(),"Native row output shape overflow");
 require(columns*2<=host_budget,"Native row payload exceeds host budget");for(auto id:ids)require(id<rows,"Native row index out of range");
 auto s=slice(impl_->root,impl_->mapping,impl_->headers,name);const auto & dims=s.metadata.at("shape");
 require(s.metadata.at("dtype")=="BF16"&&dims.is_array()&&dims.size()==2&&number(dims[0])==rows&&number(dims[1])==columns&&s.bytes==rows*columns*2,"Native row metadata mismatch");
 auto out=std::unique_ptr<Dsv41NativeTensor>(new Dsv41NativeTensor);auto & p=*out->impl_;p.backend=backend;
 p.context=ggml_init({4*ggml_tensor_overhead()+4096,nullptr,true});require(p.context,"Native row context allocation failed");
 p.tensor=ggml_new_tensor_2d(p.context,GGML_TYPE_BF16,columns,ids.size());auto buft=ggml_backend_get_default_buffer_type(backend);
 uint64_t allocation=align(ggml_backend_buft_get_alloc_size(buft,p.tensor),ggml_backend_buft_get_alignment(buft));require(allocation<=device_budget&&allocation<=SIZE_MAX,"Native row device budget exceeded");
 p.buffer=ggml_backend_alloc_buffer(backend,size_t(allocation));require(p.buffer&&ggml_backend_buffer_get_size(p.buffer)<=device_budget,"Native row backend allocation failed or exceeded budget");
 require(ggml_backend_tensor_alloc(p.buffer,p.tensor,ggml_backend_buffer_get_base(p.buffer))==GGML_STATUS_SUCCESS,"Native row tensor binding failed");
 for(size_t i=0;i<ids.size();++i){auto raw=dsv41_read_direct(s.path,s.offset+uint64_t(ids[i])*columns*2,columns*2);
  for(size_t j=0;j<raw.size();j+=2){unsigned bits=unsigned(raw[j])|(unsigned(raw[j+1])<<8);require((bits&0x7f80)!=0x7f80,"Nonfinite native BF16 row");}
  ggml_backend_tensor_set(p.tensor,raw.data(),i*columns*2,raw.size());
 }
 return out;
}
std::unique_ptr<Dsv41NativeMatrix> Dsv41NativeStore::load(const std::string & name,uint64_t rows,uint64_t cols,
 ggml_backend_t backend,uint64_t host_budget,uint64_t device_budget,bool cache_payload)const{
 require(backend&&rows&&cols&&rows<=INT_MAX-31&&cols<=INT_MAX-31&&rows<=uint64_t(INT64_MAX)/cols,"Invalid native matrix dimensions");
 auto * payload_cache=cache_payload?impl_->payload_cache.get():nullptr;
 const auto metadata_begin=std::chrono::steady_clock::now();
 auto w=slice(impl_->root,impl_->mapping,impl_->headers,name+".weight"),s=slice(impl_->root,impl_->mapping,impl_->headers,name+".scale");
 require(w.path!=s.path||w.offset+w.bytes<=s.offset||s.offset+s.bytes<=w.offset,"Overlapping native matrix tensors");
 bool fp4=w.metadata.at("dtype")=="I8";
 if(fp4){require(cols%32==0,"Invalid native FP4 width");shape(w,rows,cols/2,"I8");shape(s,rows,cols/32,"F8_E8M0");}
 else{shape(w,rows,cols,"F8_E4M3");shape(s,(rows+31)/32,(cols+31)/32,"F8_E8M0");}
 require(w.bytes<=host_budget&&s.bytes<=host_budget-w.bytes,"Native input exceeds host budget");
 uint64_t left=host_budget-w.bytes-s.bytes;
 if(fp4)require(w.bytes+s.bytes<=left,"Native transcode exceeds host budget");
 auto out=std::unique_ptr<Dsv41NativeMatrix>(new Dsv41NativeMatrix);auto & p=*out->impl_;p.backend=backend;
 p.context=ggml_init({8*ggml_tensor_overhead()+4096,nullptr,true});require(p.context,"Native context allocation failed");
 p.weight=ggml_new_tensor_2d(p.context,fp4?GGML_TYPE_MXFP4:GGML_TYPE_I8,cols,rows);
 if(!fp4)p.scales=ggml_new_tensor_2d(p.context,GGML_TYPE_I8,(cols+31)/32,(rows+31)/32);
 auto buft=ggml_backend_get_default_buffer_type(backend);uint64_t alignment=ggml_backend_buft_get_alignment(buft);
 uint64_t offset=align(ggml_backend_buft_get_alloc_size(buft,p.weight),alignment);
 uint64_t sb=p.scales?align(ggml_backend_buft_get_alloc_size(buft,p.scales),alignment):0;
 require(offset<=device_budget&&sb<=device_budget-offset&&offset+sb<=SIZE_MAX,"Native device budget exceeded");
 const auto read_begin=std::chrono::steady_clock::now();
 const auto payload=[&](const Slice & slice){return payload_cache?payload_cache->read(slice.path,slice.offset,slice.bytes):dsv41_read_direct(slice.path,slice.offset,slice.bytes);};
 std::vector<uint8_t> weight,scales,packed;double preparation_seconds=0;
 if(fp4&&payload_cache){
  packed=payload_cache->read_prepared(
   {{w.path,w.offset,w.bytes,false},{s.path,s.offset,s.bytes,false}},
   "native-mxfp4-v1:"+std::to_string(rows)+":"+std::to_string(cols),w.bytes+s.bytes,
   [&](const std::vector<std::vector<uint8_t>> & inputs){
    const auto begin=std::chrono::steady_clock::now();
    auto value=dsv41_native_fp4_to_mxfp4(inputs[0],inputs[1],rows,cols,left);
    preparation_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();return value;
   });
 }else{weight=payload(w);scales=payload(s);}
 const auto validate_begin=std::chrono::steady_clock::now();
 if(!(fp4&&payload_cache)){
  for(auto code:scales)require(code!=255,"Native NaN scale");
  if(!fp4)for(auto code:weight)require((code&127)!=127,"Native NaN FP8 weight");
  if(fp4)packed=dsv41_native_fp4_to_mxfp4(weight,scales,rows,cols,left);
 }
 const auto allocation_begin=std::chrono::steady_clock::now();
 p.buffer=ggml_backend_alloc_buffer(backend,size_t(offset+sb));require(p.buffer&&ggml_backend_buffer_get_size(p.buffer)<=device_budget,"Native backend allocation failed or exceeded budget");
 auto * base=static_cast<uint8_t *>(ggml_backend_buffer_get_base(p.buffer));
 require(ggml_backend_tensor_alloc(p.buffer,p.weight,base)==GGML_STATUS_SUCCESS,"Native weight binding failed");
 const auto upload_begin=std::chrono::steady_clock::now();
 ggml_backend_tensor_set(p.weight,fp4?packed.data():weight.data(),0,ggml_nbytes(p.weight));
 if(p.scales){require(ggml_backend_tensor_alloc(p.buffer,p.scales,base+offset)==GGML_STATUS_SUCCESS,"Native scale binding failed");ggml_backend_tensor_set(p.scales,scales.data(),0,scales.size());}
 const auto done=std::chrono::steady_clock::now();
 const auto seconds=[](auto a,auto b){return std::chrono::duration<double>(b-a).count();};
 {std::lock_guard<std::mutex> lock(impl_->timing_mutex);auto & t=impl_->timing;
  ++t.matrices;t.payload_bytes+=w.bytes+s.bytes;
  t.metadata_seconds+=seconds(metadata_begin,read_begin);t.read_seconds+=seconds(read_begin,validate_begin)-preparation_seconds;
  t.validation_repack_seconds+=seconds(validate_begin,allocation_begin)+preparation_seconds;t.allocation_seconds+=seconds(allocation_begin,upload_begin);t.upload_seconds+=seconds(upload_begin,done);
 }
 return out;
}
}
