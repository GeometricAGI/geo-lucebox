#pragma once
#include "dsv41_payload_cache.h"
#include "dsv41_int3_payload.h"
#include "gguf_inspect.h"
#include <limits>

namespace luce::common {
// Retain only authenticated, losslessly prepared bytes in the shared host LRU.
// Preserve the uncached loader's raw + prepared transient reservation, including
// on hits. The cache checks source identity before returning prepared bytes.
inline std::vector<uint8_t> dsv41_read_cached_int3(
    Dsv41PayloadCache * cache, const std::filesystem::path & path,
    uint64_t expected, const std::string & hash, uint64_t rows,
    uint64_t columns, unsigned group, uint64_t host_budget) {
    auto require=[](bool ok,const char * why){if(!ok)throw std::runtime_error(why);};
    require((group==64||group==128)&&rows&&columns&&columns%group==0&&
            rows<=uint64_t(INT64_MAX)/columns,"Invalid cached INT3 geometry");
    require(hash.size()==64&&hash.find_first_not_of("0123456789abcdef")==std::string::npos,
            "Invalid cached INT3 digest");
    const uint64_t blocks=rows*(columns/group),stride=4+group*3/8;
    require(blocks<=(UINT64_MAX-5)/stride,"Cached INT3 size overflow");
    const uint64_t prepared=5+blocks*stride;
    require(expected<=host_budget&&prepared<=host_budget-expected,
            "Cached INT3 exceeds transient host budget");
    auto prepare=[&](std::vector<std::vector<uint8_t>> & inputs){
        const auto & blob=inputs.at(0);
        require(blob.size()==expected&&sha256_bytes(blob.data(),blob.size())==hash,
                "Research payload identity mismatch");
        return dsv41_repack_int3(blob.data(),blob.size(),rows,columns,group,prepared);
    };
    if(cache){
        const auto encoding="research-int3-v1:"+hash+":"+std::to_string(rows)+":"+
            std::to_string(columns)+":"+std::to_string(group);
        return cache->read_prepared({{path,0,expected,true}},encoding,prepared,prepare);
    }
    std::vector<std::vector<uint8_t>> inputs;
    inputs.push_back(dsv41_read_direct(path,0,expected,true));
    return prepare(inputs);
}
}
