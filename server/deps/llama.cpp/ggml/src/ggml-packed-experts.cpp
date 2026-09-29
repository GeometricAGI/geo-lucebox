#include "ggml-packed-experts-decode.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "gqh.h"
#include <climits>
#include <cmath>
#include <cstring>

bool ggml_packed_expert_init(ggml_packed_expert * out,const ggml_tensor * w) {
    if(!out||!w||!w->data||!ggml_is_contiguous(w)||w->ne[0]<=0||w->ne[0]>INT_MAX||w->ne[1]<=0||w->ne[1]>INT_MAX||w->ne[2]!=1||w->ne[3]!=1)return false;
    const bool header=w->type==GGML_TYPE_GQH_T||w->type==GGML_TYPE_GQH_T_G32_R4||w->type==GGML_TYPE_GQH_T_G32_R3||w->type==GGML_TYPE_GQH2_H||w->type==GGML_TYPE_GQH3||w->type==GGML_TYPE_GQH4||w->type==GGML_TYPE_NVFP4||w->type==GGML_TYPE_DSV41_INT3_G64||w->type==GGML_TYPE_DSV41_INT3_G128;
    if(!header&&w->type!=GGML_TYPE_Q2_K&&w->type!=GGML_TYPE_IQ2_XXS&&w->type!=GGML_TYPE_MXFP4&&w->type!=GGML_TYPE_BF16&&w->type!=GGML_TYPE_F32)return false;
    if(w->ne[0]%ggml_blck_size(w->type))return false;
    ggml_packed_expert d{};d.body=uintptr_t(w->data);d.type=w->type;d.rows=w->ne[1];d.columns=w->ne[0];d.global=1.f;
    int code=0;if(header&&(!ggml_gqh_lookup(w->data,&d.global,&code)||!std::isfinite(d.global)||d.global<=0||code<0||code>=GQH_GRID_CODES))return false;
    const float levels[]={0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};
    memcpy(d.levels,levels,sizeof(levels));memcpy(d.ratios,GQH_RATIO_Q,sizeof(d.ratios));
    if(w->type==GGML_TYPE_GQH2_H)memcpy(d.levels,GQH2H_GRID[code],4*sizeof(float));
    else if(w->type==GGML_TYPE_GQH3)memcpy(d.levels,GQH3_GRID[code],8*sizeof(float));
    else if(w->type==GGML_TYPE_GQH4)memcpy(d.levels,GQH4_GRID[code],16*sizeof(float));
    else if(header&&code!=0)return false;
    const float * host=nullptr;const void * backend=nullptr;int64_t columns=0;size_t offset=0;
    if(ggml_gqh_lookup_input_scale(w->data,&host,&backend,&columns,&offset)) {
        if(columns!=w->ne[0]||offset%ggml_row_size(w->type,w->ne[0]))return false;
        // For host tensors the validated host copy is the correct address.
        const bool host_weight = !w->buffer || ggml_backend_buffer_is_host(w->buffer);
        if (!host_weight && !backend) return false;
        d.input_scale=uintptr_t(host_weight ? host : backend);
        if (!d.input_scale) return false;
    }
    *out=d;return true;
}
static void packed_experts_cpu(ggml_tensor * dst,int ith,int nth,void * owner) {
    GGML_ASSERT(owner && ggml_backend_buffer_is_host(static_cast<ggml_backend_buffer_t>(owner)));
    const auto * desc=static_cast<const ggml_packed_expert *>(dst->src[0]->data);
    const auto * ids=static_cast<const int32_t *>(dst->src[2]->data);
    const auto * x=static_cast<const float *>(dst->src[1]->data);auto * y=static_cast<float *>(dst->data);
    const int64_t cols=dst->src[1]->ne[0],input_routes=dst->src[1]->ne[1],routes=dst->ne[1],rows=dst->ne[0];
    for(int64_t i=ith;i<ggml_nelements(dst);i+=nth) {
        int64_t row=i%rows,route=(i/rows)%routes,token=i/(rows*routes);int id=ids[token*routes+route];
        GGML_ASSERT(id>=0&&id<dst->src[0]->ne[1]);const auto & d=desc[id];GGML_ASSERT(d.rows==rows&&d.columns==cols);
        const float * input=x+(token*input_routes+route%input_routes)*cols;float sum=0;
        for(int c=0;c<cols;++c)sum+=packed_weight(d,row,c)*input[c];
        y[i]=sum;
    }
}
bool ggml_packed_experts_is_op(const ggml_tensor * op) {
    if(!op||op->op!=GGML_OP_CUSTOM)return false;
    ggml_custom_op_params p;memcpy(&p,op->op_params,sizeof(p));return p.fun==packed_experts_cpu;
}
bool ggml_packed_experts_supports_device(const ggml_tensor * op, ggml_backend_dev_t device) {
    if (!ggml_packed_experts_is_op(op) || !device) return false;
    ggml_custom_op_params p;
    memcpy(&p, op->op_params, sizeof(p));
    const auto owner=static_cast<ggml_backend_buffer_t>(p.userdata);
    if (!owner) return false;
    if (ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_CPU) return ggml_backend_buffer_is_host(owner);
    return !ggml_backend_buffer_is_host(owner) && ggml_backend_buft_get_device(ggml_backend_buffer_get_type(owner))==device;
}

ggml_tensor * ggml_packed_experts_mul_mat_id(ggml_context * ctx,ggml_tensor * d,ggml_tensor * x,ggml_tensor * ids,int64_t rows) {
    GGML_ASSERT(d&&d->buffer&&x&&ids&&rows>0&&rows<=INT_MAX&&d->type==GGML_TYPE_I8&&d->ne[0]==sizeof(ggml_packed_expert)&&d->ne[1]>0&&d->ne[2]==1&&d->ne[3]==1);
    GGML_ASSERT(x->type==GGML_TYPE_F32&&ids->type==GGML_TYPE_I32&&x->ne[0]>0&&x->ne[0]<=INT_MAX&&x->ne[3]==1&&ids->ne[2]==1&&ids->ne[3]==1);
    GGML_ASSERT(ids->ne[0]>0&&ids->ne[0]%x->ne[1]==0&&ids->ne[1]==x->ne[2]&&ids->ne[0]<=65535&&ids->ne[1]<=65535);
    GGML_ASSERT(ggml_is_contiguous(d)&&ggml_is_contiguous(x)&&ggml_is_contiguous(ids));
    ggml_tensor * args[]={d,x,ids};return ggml_custom_4d(ctx,GGML_TYPE_F32,rows,ids->ne[0],ids->ne[1],1,args,3,packed_experts_cpu,GGML_N_TASKS_MAX,d->buffer);
}
