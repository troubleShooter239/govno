#define GOVNO_OBJECTS_BINDING 1
#include "gpu-scene.slang"
struct CullInput { float4 minimum; float4 maximum; uint4 metadata; };
struct DrawCommand { uint indexCount; uint instanceCount; uint firstIndex; int vertexOffset; uint firstInstance; };
[[vk::binding(0,0)]] StructuredBuffer<CullInput> cullInput;
[[vk::binding(2,0)]] RWStructuredBuffer<DrawCommand> drawCommands;
[[vk::binding(3,0)]] RWStructuredBuffer<uint> drawCounts;
[[vk::binding(4,0)]] StructuredBuffer<float> hierarchy;
struct CullPush { column_major float4x4 viewProjection; uint4 options; uint4 screen; };
[[vk::push_constant]] ConstantBuffer<CullPush> params;
bool visible(CullInput input, out float3 minimum, out float3 maximum)
{
    minimum = float3(3.402823e38); maximum = -minimum;
    for (uint i = 0; i < 8; ++i) {
        float3 local = float3(((i&1) != 0) ? input.maximum.x : input.minimum.x,
                             ((i&2) != 0) ? input.maximum.y : input.minimum.y,
                             ((i&4) != 0) ? input.maximum.z : input.minimum.z);
        float3 p = mul(objects[input.metadata.x].model, float4(local,1)).xyz;
        if (!all(isfinite(p))) {minimum=float3(-1.0/0.0);maximum=-minimum;return true;}
        minimum = min(minimum,p); maximum = max(maximum,p);
    }
    if (params.options.y == 0) return true;
    float4x4 m = params.viewProjection;
    float4 planes[6] = {m[3]+m[0],m[3]-m[0],m[3]+m[1],m[3]-m[1],m[2],m[3]-m[2]};
    for (uint i = 0; i < 6; ++i) {
        float4 p = planes[i];
        float3 vertex = float3(p.x >= 0 ? maximum.x : minimum.x,
                               p.y >= 0 ? maximum.y : minimum.y,
                               p.z >= 0 ? maximum.z : minimum.z);
        float4 terms = p * float4(vertex,1);
        if (dot(terms,float4(1)) < -1e-7 * max(dot(abs(terms),float4(1)),1.0)) return false;
    }
    return true;
}
bool occluded(float3 minimum, float3 maximum)
{
    if (!all(isfinite(minimum)) || !all(isfinite(maximum))) return false;
    float2 low=float2(3.402823e38), high=-low;
    float nearest=0.0;
    for(uint i=0;i<8;++i) {
        float3 p=float3((i&1)!=0?maximum.x:minimum.x,(i&2)!=0?maximum.y:minimum.y,(i&4)!=0?maximum.z:minimum.z);
        float4 clip=mul(params.viewProjection,float4(p,1));
        if(!all(isfinite(clip)) || clip.w<=0 || clip.z<=0 || clip.z>=clip.w) return false;
        float3 ndc=clip.xyz/clip.w;
        low=min(low,ndc.xy);high=max(high,ndc.xy);nearest=max(nearest,ndc.z);
    }
    if(any(low<=-1.0)||any(high>=1.0)) return false;
    float2 size=float2(params.screen.xy);
    uint2 a=uint2(max(floor((low*0.5+0.5)*size-1.0),0.0));
    uint2 b=uint2(min(ceil((high*0.5+0.5)*size+1.0),size-1.0));
    uint offset=0;uint2 dimensions=params.screen.xy;uint level=0;
    while(any((b>>level)-(a>>level)>1) && any(dimensions>1)) {
        offset+=dimensions.x*dimensions.y;
        dimensions=(dimensions+1)/2; ++level;
    }
    a>>=level;b>>=level;
    float depth=1.0;
    for(uint y=a.y;y<=b.y;++y) for(uint x=a.x;x<=b.x;++x) depth=min(depth,hierarchy[offset+y*dimensions.x+x]);
    return nearest+max(1e-5,nearest*1e-4)<depth;
}
[shader("compute")]
[numthreads(64,1,1)]
void cullScene(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= params.options.x) return;
    CullInput input = cullInput[thread.x];
    float3 minimum,maximum;
    if (!visible(input,minimum,maximum)) return;
    if (params.options.z!=0 && occluded(minimum,maximum)) return;
    uint slot;
    InterlockedAdd(drawCounts[input.metadata.w],1,slot);
    DrawCommand command = {input.metadata.y,1,0,0,input.metadata.x};
    drawCommands[input.metadata.z+slot] = command;
}
