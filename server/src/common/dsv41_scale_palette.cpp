#include "dsv41_scale_palette.h"
#include <cstring>
#include <limits>
#include <stdexcept>
namespace luce::common {
namespace {
void require(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
uint64_t le(const uint8_t * p,unsigned bytes){uint64_t v=0;for(unsigned i=0;i<bytes;++i)v|=uint64_t(p[i])<<(8*i);return v;}
unsigned index_at(const uint8_t * indices,uint64_t block,unsigned bits){
    unsigned value=0;
    for(unsigned bit=0;bit<bits;++bit){auto pos=block*bits+bit;value|=unsigned((indices[pos/8]>>(pos%8))&1)<<bit;}
    return value;
}
}
Dsv41ScalePaletteInfo dsv41_inspect_scale_palette(const uint8_t * data,size_t bytes,
    uint64_t rows,uint64_t columns,uint32_t stride){
    require(data&&bytes>=21,"Truncated scale palette header");
    require(stride==56||stride==57||stride==61,"Unsupported scale palette block size");
    require(rows&&columns&&columns%256==0,"Invalid scale palette shape");
    require(rows<=UINT64_MAX/(columns/256),"Scale palette shape overflow");
    uint64_t blocks=rows*(columns/256);
    require(blocks<=(UINT64_MAX-5)/stride,"Scale palette output overflow");
    Dsv41ScalePaletteInfo out;
    out.blocks=blocks;out.wire_bytes=5+blocks*stride;out.stored_bytes=bytes;
    require(std::memcmp(data,"GQSP",4)==0&&le(data+8,8)==blocks,"Scale palette identity/shape mismatch");
    out.mode=data[4];out.index_bits=data[5];out.palette_size=uint16_t(le(data+6,2));
    if(out.mode==0){
        require(out.index_bits==0&&out.palette_size==0&&bytes-16==out.wire_bytes,"Invalid raw scale envelope");
        return out;
    }
    require(out.mode==1&&out.palette_size>=1&&out.palette_size<=256,"Invalid scale palette header");
    unsigned bits=1;while((1u<<bits)<out.palette_size)++bits;
    require(out.index_bits==bits,"Noncanonical scale palette bit width");
    // blocks*bits cannot overflow after the stricter stride check above.
    const uint64_t index_bytes=(blocks*bits+7)/8;
    const uint64_t body_bytes=blocks*(stride-1);
    require(index_bytes<=bytes && body_bytes<=bytes-index_bytes &&
            bytes-index_bytes-body_bytes==uint64_t(21+out.palette_size),"Invalid scale palette length");
    const auto * palette=data+21;
    for(unsigned i=1;i<out.palette_size;++i)require(palette[i]>palette[i-1],"Noncanonical scale palette ordering");
    const auto * indices=palette+out.palette_size;
    for(uint64_t i=0;i<blocks;++i)require(index_at(indices,i,bits)<out.palette_size,"Scale palette index out of range");
    const unsigned used=unsigned((blocks*bits)%8);
    if(used)require((indices[index_bytes-1]>>used)==0,"Nonzero scale palette padding");
    return out;
}
std::vector<uint8_t> dsv41_expand_scale_palette(const uint8_t * data,size_t bytes,
    uint64_t rows,uint64_t columns,uint32_t stride,uint64_t budget){
    auto info=dsv41_inspect_scale_palette(data,bytes,rows,columns,stride);
    require(info.wire_bytes<=budget&&info.wire_bytes<=std::numeric_limits<size_t>::max(),"Scale palette output exceeds budget");
    if(info.mode==0)return std::vector<uint8_t>(data+16,data+bytes);
    std::vector<uint8_t> out(size_t(info.wire_bytes));
    std::memcpy(out.data(),data+16,5);
    const auto * palette=data+21;
    const auto * indices=palette+info.palette_size;
    const auto * body=indices+(info.blocks*info.index_bits+7)/8;
    for(uint64_t i=0;i<info.blocks;++i){
        auto * dst=out.data()+5+i*stride;
        dst[0]=palette[index_at(indices,i,info.index_bits)];
        std::memcpy(dst+1,body+i*(stride-1),stride-1);
    }
    return out;
}
}
