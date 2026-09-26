#pragma once
#include "dsv41_direct_io.h"
#include <list>
#include <map>
#include <mutex>
#include <tuple>
#include <functional>
namespace luce::common {
struct Dsv41PayloadCacheStats {
 uint64_t bytes=0,peak_bytes=0,hits=0,misses=0,evictions=0,disk_payload_bytes=0;
};
// Serialized bounded host LRU for exact payload slices. Returned vectors are
// caller-owned copies and remain charged to the caller's per-load host budget.
// Cache bytes are a separate explicit reservation; containers/allocator overhead
// require metadata reserve. Zero budget keeps the direct-I/O behavior.
class Dsv41PayloadCache {
public:
 using Source=std::tuple<std::filesystem::path,uint64_t,uint64_t,bool>;
 using Prepare=std::function<std::vector<uint8_t>(std::vector<std::vector<uint8_t>> &)>;
private:
 struct Identity {
  uint64_t device,inode,bytes;int64_t msec,mnano,csec,cnano;
  bool operator==(const Identity & b)const{return device==b.device&&inode==b.inode&&bytes==b.bytes&&msec==b.msec&&mnano==b.mnano&&csec==b.csec&&cnano==b.cnano;}
 };
 using Key=std::pair<std::string,std::vector<Source>>;
 struct Entry {std::vector<uint8_t> data;std::list<Key>::iterator position;};
 uint64_t budget_;mutable std::mutex mutex_;Dsv41PayloadCacheStats stats_;
 std::map<Key,Entry> entries_;std::list<Key> order_;
 std::map<std::filesystem::path,Identity> identities_;
 static Identity identity(const std::filesystem::path & path){
  struct stat s{};if(::stat(path.c_str(),&s)||!S_ISREG(s.st_mode)||s.st_size<0)throw std::runtime_error("Cannot stat cached payload");
#ifdef __APPLE__
  return {uint64_t(s.st_dev),uint64_t(s.st_ino),uint64_t(s.st_size),s.st_mtimespec.tv_sec,s.st_mtimespec.tv_nsec,s.st_ctimespec.tv_sec,s.st_ctimespec.tv_nsec};
#else
  return {uint64_t(s.st_dev),uint64_t(s.st_ino),uint64_t(s.st_size),s.st_mtim.tv_sec,s.st_mtim.tv_nsec,s.st_ctim.tv_sec,s.st_ctim.tv_nsec};
#endif
 }
public:
 explicit Dsv41PayloadCache(uint64_t budget):budget_(budget){}
 Dsv41PayloadCacheStats stats()const{std::lock_guard<std::mutex> lock(mutex_);return stats_;}
 // Encodings must uniquely identify the deterministic preparation and shape.
 // Prepared and raw entries share one byte budget; raw inputs are temporary and
 // are not also retained. The caller accounts for inputs plus output scratch.
 std::vector<uint8_t> read_prepared(const std::vector<Source> & sources,const std::string & encoding,
                                  uint64_t bytes,const Prepare & prepare){
  if(sources.empty()||encoding.empty())throw std::runtime_error("Invalid prepared payload key");
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<Source> canonical_sources;std::vector<Identity> versions;
  for(const auto & source:sources){
   auto canonical=std::filesystem::canonical(std::get<0>(source));auto version=identity(canonical);
   auto prior=identities_.find(canonical);
   if(prior!=identities_.end()&&!(prior->second==version))throw std::runtime_error("Payload changed after cache observation");
   canonical_sources.emplace_back(canonical,std::get<1>(source),std::get<2>(source),std::get<3>(source));versions.push_back(version);
  }
  Key key{encoding,canonical_sources};auto found=entries_.find(key);
  if(found!=entries_.end()){
   if(found->second.data.size()!=bytes)throw std::runtime_error("Prepared payload size changed");
   ++stats_.hits;order_.splice(order_.end(),order_,found->second.position);return found->second.data;
  }
  ++stats_.misses;std::vector<std::vector<uint8_t>> inputs;
  for(const auto & source:canonical_sources){
   inputs.push_back(dsv41_read_direct(std::get<0>(source),std::get<1>(source),std::get<2>(source),std::get<3>(source)));
   stats_.disk_payload_bytes+=inputs.back().size();
  }
  auto value=prepare(inputs);
  if(value.size()!=bytes)throw std::runtime_error("Prepared payload size mismatch");
  for(size_t i=0;i<canonical_sources.size();++i)
   if(!(identity(std::get<0>(canonical_sources[i]))==versions[i]))throw std::runtime_error("Payload changed during preparation");
  inputs.clear();
  if(bytes&&bytes<=budget_){
   while(bytes>budget_-stats_.bytes){auto old=entries_.find(order_.front());stats_.bytes-=old->second.data.size();entries_.erase(old);order_.pop_front();++stats_.evictions;}
   auto inserted=entries_.emplace(key,Entry{value,{}});
   try{order_.push_back(key);}catch(...){entries_.erase(inserted.first);throw;}
   inserted.first->second.position=std::prev(order_.end());
   stats_.bytes+=bytes;stats_.peak_bytes=std::max(stats_.peak_bytes,stats_.bytes);
   for(size_t i=0;i<canonical_sources.size();++i)identities_.emplace(std::get<0>(canonical_sources[i]),versions[i]);
  }
  return value;
 }
 std::vector<uint8_t> read(const std::filesystem::path & path,uint64_t offset,uint64_t bytes,bool whole_file=false){
  return read_prepared({Source{path,offset,bytes,whole_file}},"raw-v1",bytes,
   [](std::vector<std::vector<uint8_t>> & inputs){return std::move(inputs.front());});
 }

};
}
