#define GOVNO_LIGHT_COMPUTE 1
#include "lighting.slang"
struct LightPush { column_major float4x4 viewProjection; uint4 options; }; // width, height, light count, unused
[[vk::push_constant]] ConstantBuffer<LightPush> params;
groupshared float4 planes[6];
[shader("compute")]
[numthreads(32,1,1)]
void cullLights(uint3 tile : SV_GroupID, uint lane : SV_GroupIndex)
{
    if (lane < 6) {
        float2 lower = float2(tile.xy * 16) / float2(params.options.xy) * 2 - 1;
        float2 upper = float2(min((tile.xy+1)*16,params.options.xy)) / float2(params.options.xy) * 2 - 1;
        float4x4 m = params.viewProjection;
        float4 p;
        if (lane == 0) p=m[0]-lower.x*m[3];
        else if (lane == 1) p=upper.x*m[3]-m[0];
        else if (lane == 2) p=m[1]-lower.y*m[3];
        else if (lane == 3) p=upper.y*m[3]-m[1];
        else if (lane == 4) p=m[2];
        else p=m[3]-m[2];
        // Infinite reversed-Z may produce a constant far plane; retain it unnormalized.
        planes[lane]=p;
    }
    GroupMemoryBarrierWithGroupSync();
    uint mask=0;
    for (uint bit=0;bit<32;++bit) {
        uint id=lane*32+bit;
        if (id>=params.options.z) break;
        float4 light=lightVector(id,0);
        bool visible=true;
        for (uint i=0;i<6;++i) {
            float4 p=planes[i];
            float distance=dot(p,float4(light.xyz,1));
            float radius=light.w*length(p.xyz);
            float tolerance=1e-5*max(dot(abs(p),abs(float4(light.xyz,1)))+radius,1.0);
            if (distance < -radius-tolerance) visible=false;
        }
        if (visible) mask |= 1u<<bit;
    }
    uint index=tile.y*lighting[0]+tile.x;
    lighting[MASK_OFFSET+index*MASK_WORDS+lane]=mask;
}
