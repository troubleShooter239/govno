[[vk::binding(0,0)]] Texture2D<float4> hdrImage;
struct OutputSettings { uint manualSRGB; float exposure; };
[[vk::push_constant]] ConstantBuffer<OutputSettings> settings;
[shader("vertex")]
float4 postVertex(uint vertex : SV_VulkanVertexID) : SV_Position {
    float2 positions[3] = { float2(-1,-1),float2(3,-1),float2(-1,3) };
    return float4(positions[vertex],0,1);
}
float encode(float c) { return c <= 0.0031308 ? c*12.92 : 1.055*pow(c,1.0/2.4)-0.055; }
[shader("fragment")]
float4 postFragment(float4 position : SV_Position) : SV_Target0 {
    float3 hdr = max(hdrImage.Load(int3(int2(position.xy),0)).rgb,0.0);
    hdr *= settings.exposure;
    float3 color = hdr / (1.0 + hdr);
    if (settings.manualSRGB != 0) color = float3(encode(color.r),encode(color.g),encode(color.b));
    return float4(color,1.0);
}
