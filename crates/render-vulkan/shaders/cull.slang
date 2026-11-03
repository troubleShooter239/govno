#define GOVNO_OBJECTS_BINDING 1
#include "gpu-scene.slang"
struct CullInput { float4 minimum; float4 maximum; uint4 metadata; };
struct DrawCommand { uint indexCount; uint instanceCount; uint firstIndex; int vertexOffset; uint firstInstance; };
[[vk::binding(0,0)]] StructuredBuffer<CullInput> cullInput;
[[vk::binding(2,0)]] RWStructuredBuffer<DrawCommand> drawCommands;
[[vk::binding(3,0)]] RWStructuredBuffer<uint> drawCounts;
struct CullPush { column_major float4x4 viewProjection; uint4 options; };
[[vk::push_constant]] ConstantBuffer<CullPush> params;
bool visible(CullInput input)
{
    if (params.options.y == 0) return true;
    float3 minimum = float3(3.402823e38), maximum = -minimum;
    for (uint i = 0; i < 8; ++i) {
        float3 local = float3(((i&1) != 0) ? input.maximum.x : input.minimum.x,
                             ((i&2) != 0) ? input.maximum.y : input.minimum.y,
                             ((i&4) != 0) ? input.maximum.z : input.minimum.z);
        float3 p = mul(objects[input.metadata.x].model, float4(local,1)).xyz;
        if (!all(isfinite(p))) return true;
        minimum = min(minimum,p); maximum = max(maximum,p);
    }
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
[shader("compute")]
[numthreads(64,1,1)]
void cullScene(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= params.options.x) return;
    CullInput input = cullInput[thread.x];
    if (!visible(input)) return;
    uint slot;
    InterlockedAdd(drawCounts[input.metadata.w],1,slot);
    DrawCommand command = {input.metadata.y,1,0,0,input.metadata.x};
    drawCommands[input.metadata.z+slot] = command;
}
