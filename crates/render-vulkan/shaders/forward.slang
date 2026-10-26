struct FrameData
{
    column_major float4x4 viewProjection;
    float4 cameraPosition;
    float4 lightDirection;
    float4 lightRadiance; // w = manually encode sRGB for UNORM swapchains
};
[[vk::binding(0, 0)]] ConstantBuffer<FrameData> frame;

[[vk::binding(1, 0)]] Texture2D<float4> baseColorTexture;
[[vk::binding(2, 0)]] SamplerState baseColorSampler;

[[vk::binding(1, 1)]] Texture2D<float4> normalTexture;
[[vk::binding(2, 1)]] SamplerState normalSampler;
[[vk::binding(1, 2)]] Texture2D<float4> metalRoughTexture;
[[vk::binding(2, 2)]] SamplerState metalRoughSampler;

[[vk::binding(1, 3)]] Texture2D<float4> emissiveTexture;
[[vk::binding(2, 3)]] SamplerState emissiveSampler;
[[vk::binding(3, 3)]] Texture2D<float4> occlusionTexture;
[[vk::binding(4, 3)]] SamplerState occlusionSampler;

struct DrawData
{
    column_major float4x4 model;
    float4 inverseScaleSquaredRoughness;
    float4 baseColorMetallic;
    float4 normalParams; // x scale, y enabled
    float4 emissiveOcclusion;
};
[[vk::push_constant]] ConstantBuffer<DrawData> draw;

struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
    [[vk::location(3)]] float4 tangent : TANGENT;
};
struct VertexOutput
{
    float4 position : SV_Position;
    float3 worldPosition : TEXCOORD0;
    float3 worldNormal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float4 worldTangent : TEXCOORD3;
};
[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;
    float4 worldPosition = mul(draw.model, float4(input.position, 1.0));
    output.position = mul(frame.viewProjection, worldPosition);
    output.worldPosition = worldPosition.xyz;
    // For positive TRS scales: model * (normal / scale^2) = inverse-transpose * normal.
    output.worldNormal = mul(draw.model, float4(input.normal * draw.inverseScaleSquaredRoughness.xyz, 0.0)).xyz;
    output.uv = input.uv;
    output.worldTangent = float4(mul(draw.model,float4(input.tangent.xyz,0.0)).xyz,input.tangent.w);
    return output;
}
float3 safeNormalize(float3 v) { return v * rsqrt(max(dot(v,v), 1e-12)); }
float3 encodeSRGB(float3 color)
{
    return float3(
        color.x <= 0.0031308 ? 12.92 * color.x : 1.055 * pow(color.x, 1.0/2.4) - 0.055,
        color.y <= 0.0031308 ? 12.92 * color.y : 1.055 * pow(color.y, 1.0/2.4) - 0.055,
        color.z <= 0.0031308 ? 12.92 * color.z : 1.055 * pow(color.z, 1.0/2.4) - 0.055);
}
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target0
{
    const float PI = 3.14159265359;
    float3 n = safeNormalize(input.worldNormal);
    if (draw.normalParams.y > 0.5) {
        float3 px = ddx(input.worldPosition), py = ddy(input.worldPosition);
        float2 ux = ddx(input.uv), uy = ddy(input.uv);
        float determinant = ux.x * uy.y - ux.y * uy.x;
        // Sample before fragment-varying frame validity branches for defined derivatives.
        float3 mapped = normalTexture.Sample(normalSampler,input.uv).xyz * 2.0 - 1.0;
        float3 tangent = float3(0.0);
        float3 bitangent = float3(0.0);
        bool validFrame = false;
        if (abs(input.worldTangent.w) > 0.5) {
            tangent = safeNormalize(input.worldTangent.xyz - n * dot(n,input.worldTangent.xyz));
            bitangent = cross(n,tangent) * (input.worldTangent.w < 0.0 ? -1.0 : 1.0);
            validFrame = dot(tangent,tangent) > 0.5;
        } else if (abs(determinant) > 1e-8) {
            tangent = (px * uy.y - py * ux.y) / determinant;
            bitangent = (py * ux.x - px * uy.x) / determinant;
            tangent = safeNormalize(tangent - n * dot(n,tangent));
            float handedness = dot(cross(n,tangent), bitangent) < 0.0 ? -1.0 : 1.0;
            bitangent = cross(n,tangent) * handedness;
            validFrame = dot(tangent,tangent) > 0.5;
        }
        if (validFrame) {

            mapped.xy *= draw.normalParams.x;
            n = safeNormalize(tangent * mapped.x + bitangent * mapped.y + n * mapped.z);
        }
    }
    float3 v = safeNormalize(frame.cameraPosition.xyz - input.worldPosition);
    float3 l = frame.lightDirection.xyz;
    float3 h = safeNormalize(v + l);
    float nv = max(dot(n,v), 0.0001);
    float nl = saturate(dot(n,l));
    float nh = saturate(dot(n,h));
    float vh = saturate(dot(v,h));
    float3 base = draw.baseColorMetallic.xyz * baseColorTexture.Sample(baseColorSampler, input.uv).rgb;
    float2 mr = metalRoughTexture.Sample(metalRoughSampler,input.uv).gb;
    float metal = saturate(draw.baseColorMetallic.w * mr.y);
    float roughness = clamp(draw.inverseScaleSquaredRoughness.w * mr.x,0.045,1.0);
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denominator = nh * nh * (alpha2 - 1.0) + 1.0;
    float distribution = alpha2 / max(PI * denominator * denominator, 1e-8);
    // Height-correlated Smith visibility for GGX.
    float visibility = 0.5 / max(nl * sqrt(nv*nv*(1.0-alpha2)+alpha2)
        + nv * sqrt(nl*nl*(1.0-alpha2)+alpha2), 1e-6);
    float3 f0 = lerp(float3(0.04), base, metal);
    float3 fresnel = f0 + (1.0-f0) * pow(1.0-vh, 5.0);
    float3 diffuse = (1.0-fresnel) * (1.0-metal) * base / PI;
    float3 specular = distribution * visibility * fresnel;
    float3 color = (diffuse + specular) * frame.lightRadiance.xyz * nl;
    // Temporary ambient fill; no IBL or shadows yet.
    float ao = lerp(1.0, occlusionTexture.Sample(occlusionSampler,input.uv).r, draw.emissiveOcclusion.w);
    color += base * 0.025 * (1.0-metal) * ao;
    color += draw.emissiveOcclusion.rgb * emissiveTexture.Sample(emissiveSampler,input.uv).rgb;
    color = color / (1.0 + color); // Reinhard display mapping
    if (frame.lightRadiance.w > 0.5) color = encodeSRGB(color);
    return float4(color, 1.0);
}
