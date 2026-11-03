struct DrawData
{
    column_major float4x4 model;
    float4 inverseScaleSquaredRoughness;
    float4 baseColorMetallic;
    float4 normalParams; // x scale, y enabled
    float4 emissiveOcclusion;
    uint4 textures0;
    uint4 textures1;
};
#ifndef GOVNO_OBJECTS_BINDING
#define GOVNO_OBJECTS_BINDING 10
#endif
[[vk::binding(GOVNO_OBJECTS_BINDING,0)]] StructuredBuffer<DrawData> objects;

