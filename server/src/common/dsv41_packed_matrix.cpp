#include "dsv41_packed_matrix.h"
#include "dsv41_scale_palette.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace luce::common {
namespace {
void require(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
uint64_t aligned(uint64_t bytes,uint64_t alignment){
 require(alignment&&bytes<=UINT64_MAX-(alignment-1),"Packed allocation overflow");
 return ((bytes+alignment-1)/alignment)*alignment;
}
}
struct Dsv41PackedMatrix::Impl {
 ggml_backend_t backend=nullptr;
 ggml_context * context=nullptr;
 ggml_backend_buffer_t buffer=nullptr;
 ggml_tensor * weight=nullptr;
 uint64_t metadata=0;
 bool registered=false, borrowed=false;
 uint64_t footprint=0;
 ~Impl(){
  if(buffer)ggml_backend_synchronize(backend);
  if(registered)ggml_gqh_unregister(weight->data);
  if(buffer&&!borrowed)ggml_backend_buffer_free(buffer);
  if(context)ggml_free(context);
 }
};
Dsv41PackedMatrix::Dsv41PackedMatrix(ggml_backend_t backend,ggml_type type,uint64_t rows,uint64_t columns,
 const uint8_t * payload,size_t bytes,bool palette,const float * awq,size_t count,
 uint64_t host_budget,uint64_t device_budget,ggml_backend_buffer_t arena,uint64_t arena_offset):impl_(new Impl){
 auto & p=*impl_;p.backend=backend;
 const bool nvfp4=type==GGML_TYPE_NVFP4;
 const unsigned int3=type==GGML_TYPE_DSV41_INT3_G64?64:type==GGML_TYPE_DSV41_INT3_G128?128:0;
 const uint64_t block_elements=int3?int3:nvfp4?64:256;
 require(backend&&rows&&columns&&columns%block_elements==0&&rows<=uint64_t(INT64_MAX)/columns,"Invalid packed matrix dimensions");
 uint32_t stride=type==GGML_TYPE_GQH_T?61:type==GGML_TYPE_GQH_T_G32_R4?57:type==GGML_TYPE_GQH_T_G32_R3?56:type==GGML_TYPE_GQH2_H?73:type==GGML_TYPE_GQH3?105:type==GGML_TYPE_GQH4?137:nvfp4?36:int3?4+int3*3/8:0;
 require(stride,"Unsupported packed matrix type");
 const bool scalar=stride>=73;
 require(!(nvfp4||int3)||(!palette&&!count),"Dense research weights cannot carry palette or AWQ column metadata");
 require(!palette||!scalar,"Scalar GQH palette has no canonical codec");
 require(count==0||(awq&&count==columns),"Invalid packed AWQ dimensions");
 require(columns<=UINT64_MAX/4,"AWQ size overflow");
 p.metadata=count*4;
 require(p.metadata<=host_budget,"Packed host metadata exceeds budget");
 for(size_t i=0;i<count;++i)require(std::isfinite(awq[i])&&awq[i]>0,"Invalid packed AWQ scale");
 std::vector<uint8_t> expanded;
 if(palette){expanded=dsv41_expand_scale_palette(payload,bytes,rows,columns,stride,host_budget-p.metadata);payload=expanded.data();bytes=expanded.size();}
 const uint64_t blocks=rows*(columns/block_elements);
 require(blocks<=(UINT64_MAX-5)/stride&&payload&&bytes==5+blocks*stride,"Invalid packed wire size");
 uint32_t bits=uint32_t(payload[0])|(uint32_t(payload[1])<<8)|(uint32_t(payload[2])<<16)|(uint32_t(payload[3])<<24);
 float global;std::memcpy(&global,&bits,4);
 require(std::isfinite(global)&&global>0&&(scalar?payload[4]<12:payload[4]==0),"Invalid GQH global header");
 require(!int3||global==1.f,"INT3 private header must be identity");
 for(uint64_t b=0;b<blocks;++b){
  const auto * block=payload+5+b*stride;
  if(int3){uint32_t bits=uint32_t(block[0])|(uint32_t(block[1])<<8)|(uint32_t(block[2])<<16)|(uint32_t(block[3])<<24);float scale;std::memcpy(&scale,&bits,4);require(std::isfinite(scale)&&scale>=0,"Invalid INT3 scale");}
  else if(nvfp4){for(unsigned i=0;i<4;++i)require(block[i]<127,"Invalid NVFP4 scale");}
  else require((block[0]&127)!=127,"Nonfinite GQH block scale");
  if(!scalar&&!nvfp4&&!int3){
  for(uint32_t i=stride-52;i<stride;++i)require(block[i]<243,"Invalid packed ternary digit");
  require(block[stride-1]<3,"Invalid packed ternary padding");
  }
 }
 p.context=ggml_init({8*ggml_tensor_overhead()+4096,nullptr,true});require(p.context,"Packed context allocation failed");
 p.weight=ggml_new_tensor_2d(p.context,type,int64_t(columns),int64_t(rows));
 auto * scales=count?ggml_new_tensor_1d(p.context,GGML_TYPE_F32,int64_t(columns)):nullptr;
 auto buft=ggml_backend_get_default_buffer_type(backend);auto alignment=ggml_backend_buft_get_alignment(buft);
 uint64_t offset=aligned(ggml_backend_buft_get_alloc_size(buft,p.weight),alignment);
 uint64_t scale_bytes=scales?aligned(ggml_backend_buft_get_alloc_size(buft,scales),alignment):0;
 require(offset<=device_budget&&scale_bytes<=device_budget-offset&&offset+scale_bytes<=std::numeric_limits<size_t>::max(),"Packed device allocation exceeds budget");
 p.footprint=offset+scale_bytes;
 p.borrowed=arena!=nullptr;
 if(arena) {
  require(ggml_backend_buffer_get_type(arena)==buft&&arena_offset%alignment==0,"Invalid packed arena backend/alignment");
  const uint64_t size=ggml_backend_buffer_get_size(arena);
  require(arena_offset<=size&&p.footprint<=size-arena_offset,"Packed matrix exceeds arena");
  p.buffer=arena;
 } else {
  require(arena_offset==0,"Offset supplied without packed arena");
  p.buffer=ggml_backend_alloc_buffer(backend,size_t(p.footprint));require(p.buffer,"Packed device allocation failed");
  p.footprint=ggml_backend_buffer_get_size(p.buffer);
  require(p.footprint<=device_budget,"Backend exceeded packed allocation budget");
 }
 auto * base=static_cast<uint8_t *>(ggml_backend_buffer_get_base(p.buffer))+arena_offset;
 require(ggml_backend_tensor_alloc(p.buffer,p.weight,base)==GGML_STATUS_SUCCESS,"Packed tensor binding failed");
 ggml_backend_tensor_set(p.weight,payload+5,0,bytes-5);
 if(scales){require(ggml_backend_tensor_alloc(p.buffer,scales,base+offset)==GGML_STATUS_SUCCESS,"Packed scale binding failed");ggml_backend_tensor_set(scales,awq,0,count*4);}
 ggml_gqh_register(p.weight->data,ggml_nbytes(p.weight),global,payload[4]);p.registered=true;
 if(scales)require((scalar?ggml_gqh_register_research_input_scale:ggml_gqh_register_ternary_input_scale)(p.weight,awq,int64_t(columns),scales->data),"Packed AWQ registration failed");
}
uint64_t Dsv41PackedMatrix::allocation_bytes(ggml_backend_t backend,ggml_type type,uint64_t rows,uint64_t columns,bool awq) {
 require(backend&&rows&&columns&&columns%ggml_blck_size(type)==0&&rows<=uint64_t(INT64_MAX)/columns,"Invalid packed allocation geometry");
 auto * ctx=ggml_init({4*ggml_tensor_overhead()+4096,nullptr,true});
 require(ctx,"Packed allocation metadata failed");
 const auto buft=ggml_backend_get_default_buffer_type(backend);
 const uint64_t alignment=ggml_backend_buft_get_alignment(buft);
 auto * w=ggml_new_tensor_2d(ctx,type,columns,rows);
 uint64_t bytes=aligned(ggml_backend_buft_get_alloc_size(buft,w),alignment);
 if(awq) {
  auto * s=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,columns);
  const uint64_t scales=aligned(ggml_backend_buft_get_alloc_size(buft,s),alignment);
  if(scales>UINT64_MAX-bytes){ggml_free(ctx);throw std::runtime_error("Packed allocation overflow");}
  bytes+=scales;
 }
 ggml_free(ctx);return bytes;
}

Dsv41PackedMatrix::~Dsv41PackedMatrix()=default;
ggml_tensor * Dsv41PackedMatrix::tensor() const{return impl_->weight;}
uint64_t Dsv41PackedMatrix::device_bytes() const{return impl_->footprint;}
uint64_t Dsv41PackedMatrix::metadata_bytes() const{return impl_->metadata;}
}
