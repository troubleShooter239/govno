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
[[vk::binding(10,0)]] StructuredBuffer<DrawData> objects;

