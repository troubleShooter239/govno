#include "gpu-scene.slang"
struct ShadowDraw { column_major float4x4 worldToLight; };
[[vk::push_constant]] ConstantBuffer<ShadowDraw> shadow;
[shader("vertex")]
float4 shadowVertex([[vk::location(0)]] float3 position : POSITION, uint objectId : SV_VulkanInstanceID) : SV_Position {
    return mul(shadow.worldToLight,mul(objects[objectId].model,float4(position,1.0)));
}
