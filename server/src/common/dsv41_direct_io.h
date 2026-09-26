#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
namespace luce::common {
inline constexpr size_t dsv41_io_scratch_bytes=1u<<20;
// Buffered mode allows the OS to retain weight pages on RAM-rich hosts. Engram
// row I/O is separate and remains direct. Resolve once for stable run provenance.
inline bool dsv41_weight_io_buffered(){
 static const bool buffered=[](){
  const char * mode=std::getenv("DSV41_WEIGHT_IO_MODE");
  if(!mode||std::strcmp(mode,"direct")==0)return false;
  if(std::strcmp(mode,"buffered")==0)return true;
  throw std::runtime_error("DSV41_WEIGHT_IO_MODE must be direct or buffered");
 }();
 return buffered;
}
inline const char * dsv41_weight_io_mode(){return dsv41_weight_io_buffered()?"buffered":"direct";}
namespace direct_io_detail {
inline void direct_require(bool ok,const char * message){if(!ok)throw std::runtime_error(message);}
// Payload reads bypass page cache by default; buffered mode is explicit.
// The fixed scratch allocation is a separate per-concurrent-load reserve.
inline std::vector<uint8_t> dsv41_read_direct(const std::filesystem::path & path,
 uint64_t offset,uint64_t bytes,bool whole_file=false){
 constexpr size_t alignment=4096,chunk=dsv41_io_scratch_bytes;
 direct_require(bytes<=SIZE_MAX&&offset<=uint64_t(std::numeric_limits<off_t>::max())&&
         bytes<=uint64_t(std::numeric_limits<off_t>::max())-offset,"Tensor payload read overflow");
 struct Fd {int value=-1;~Fd(){if(value>=0)close(value);}} fd;
 fd.value=open(path.c_str(),O_RDONLY|O_CLOEXEC|(dsv41_weight_io_buffered()?0:O_DIRECT));
 if(fd.value<0)throw std::runtime_error("Cannot open direct tensor payload: "+std::string(std::strerror(errno)));
 struct stat st{};
 direct_require(fstat(fd.value,&st)==0&&S_ISREG(st.st_mode)&&st.st_size>=0&&
         offset+bytes<=uint64_t(st.st_size),"Invalid direct tensor extent");
 direct_require(!whole_file||(offset==0&&bytes==uint64_t(st.st_size)),"Direct payload file size mismatch");
 void * ptr=nullptr;direct_require(posix_memalign(&ptr,alignment,chunk)==0,"Tensor I/O scratch allocation failed");
 std::unique_ptr<void,decltype(&std::free)> scratch(ptr,&std::free);
 std::vector<uint8_t> out(size_t(bytes),0);
 uint64_t done=0;
 while(done<bytes){
  const uint64_t position=offset+done,base=position/alignment*alignment;
  const size_t skip=size_t(position-base),take=size_t(std::min<uint64_t>(bytes-done,chunk-skip));
  const size_t count=size_t(((skip+take+alignment-1)/alignment*alignment));
  ssize_t got;
  do {got=pread(fd.value,ptr,count,off_t(base));}while(got<0&&errno==EINTR);
  if(got<0)throw std::runtime_error("Direct tensor payload read failed: "+std::string(std::strerror(errno)));
  // A final unaligned file extent can return fewer bytes than requested. It
  // must still contain the entire requested tensor slice; never pad payloads.
  direct_require(uint64_t(got)>=skip+take,"Short direct tensor payload read");
  std::memcpy(out.data()+size_t(done),static_cast<const uint8_t *>(ptr)+skip,take);
  done+=take;
 }
 struct stat after{};
 direct_require(fstat(fd.value,&after)==0&&after.st_size==st.st_size,"Direct payload changed size during read");
 return out;
}
}
using direct_io_detail::dsv41_read_direct;
}
