[[vk::binding(0,0)]] Texture2D<float4> hdrImage;
[[vk::binding(1,0)]] Texture2D<float4> bloomImage;
[[vk::binding(2,0)]] StructuredBuffer<float> exposureState;
struct OutputSettings { uint manualSRGB; float bloomIntensity; };
[[vk::push_constant]] ConstantBuffer<OutputSettings> settings;
[shader("vertex")]
float4 postVertex(uint vertex : SV_VulkanVertexID) : SV_Position {
    float2 positions[3] = { float2(-1,-1),float2(3,-1),float2(-1,3) };
    return float4(positions[vertex],0,1);
}
float encode(float c) { return c <= 0.0031308 ? c*12.92 : 1.055*pow(c,1.0/2.4)-0.055; }
[shader("fragment")]
float4 postFragment(float4 position : SV_Position) : SV_Target0 {
    float3 hdr = clamp(hdrImage.Load(int3(int2(position.xy),0)).rgb,0.0,65504.0);
    hdr *= exposureState[0];
    uint bw,bh;bloomImage.GetDimensions(bw,bh);
    uint hw,hh;hdrImage.GetDimensions(hw,hh);
    float2 coordinate=position.xy/float2(hw,hh)*float2(bw,bh)-0.5;
    int2 pixel=int2(floor(coordinate));float2 f=frac(coordinate);int2 maximum=int2(bw-1,bh-1);
    float3 a=bloomImage.Load(int3(clamp(pixel,int2(0),maximum),0)).rgb;
    float3 b=bloomImage.Load(int3(clamp(pixel+int2(1,0),int2(0),maximum),0)).rgb;
    float3 c=bloomImage.Load(int3(clamp(pixel+int2(0,1),int2(0),maximum),0)).rgb;
    float3 d=bloomImage.Load(int3(clamp(pixel+int2(1,1),int2(0),maximum),0)).rgb;
    hdr += lerp(lerp(a,b,f.x),lerp(c,d,f.x),f.y)*settings.bloomIntensity;
    float3 color = hdr / (1.0 + hdr);
    if (settings.manualSRGB != 0) color = float3(encode(color.r),encode(color.g),encode(color.b));
    return float4(color,1.0);
}
