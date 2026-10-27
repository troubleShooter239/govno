struct LocalProbeData{float4 positionSlot;float4 minimumFade;float4 maximumStrength;};
struct PointLightData { float4 positionRange; float4 intensity; };
struct FrameData
{
    column_major float4x4 viewProjection;
    float4 cameraPosition;
    float4 lightDirection;
    float4 lightRadiance; // rgb directional radiance; w point light count
    PointLightData pointLights[16];
    column_major float4x4 shadowViewProjection;
    float4 shadowParams; // x receiver bias, y enabled
    float4 diffuseSH[9]; // cosine-convolved irradiance/pi
    float4 environmentParams; // x specular enabled
    column_major float4x4 cascadeMatrices[4];
    float4 cascadeSplits;
    float4 cascadeParams;
    LocalProbeData localProbes[2];
};
[[vk::binding(0, 0)]] ConstantBuffer<FrameData> frame;

[[vk::binding(9,0)]] StructuredBuffer<float4> environmentSH;
[[vk::binding(6, 0)]] Texture2D<float4> specularAtlas;
[[vk::binding(7, 0)]] SamplerState environmentSampler;
[[vk::binding(8, 0)]] Texture2D<float4> brdfLut;
[[vk::binding(5, 0)]] Texture2D<float> shadowMap;
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
float cascadeVisibility(float3 position,uint cascade)
{
 float3 projected=mul(frame.cascadeMatrices[cascade],float4(position,1)).xyz;
 float2 uv=projected.xy*0.5+0.5;
 if(any(uv<0)||any(uv>=1)||projected.z<0||projected.z>1)return 1;
 int2 tile=int2(cascade%2,cascade/2)*2048;int2 pixel=int2(uv*2048)+tile;float visible=0;
 for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x){float depth=shadowMap.Load(int3(clamp(pixel+int2(x,y),tile,tile+2047),0));visible+=projected.z-frame.shadowParams.x<=depth?1:0;}
 return visible/9.0;
}
float directionalVisibility(float3 position)
{
 if(frame.shadowParams.y<0.5)return 1;
 // Forward view depth from clip w (infinite perspective, RH camera).
 float depth=mul(frame.viewProjection,float4(position,1)).w;
 if(depth>frame.cascadeSplits.w)return 1;
 uint cascade=depth<frame.cascadeSplits.x?0:depth<frame.cascadeSplits.y?1:depth<frame.cascadeSplits.z?2:3;
 float visible=cascadeVisibility(position,cascade);
 if(cascade<3){float begin=cascade==0?frame.cascadeParams.x:frame.cascadeSplits[cascade-1];float end=frame.cascadeSplits[cascade];float blend=max((end-begin)*frame.cascadeParams.y,0.00001);float t=saturate((depth-(end-blend))/blend);if(t>0)visible=lerp(visible,cascadeVisibility(position,cascade+1),t);}
 return visible;
}
float probeWeight(float3 point,uint i){LocalProbeData probe=frame.localProbes[i];float3 distance=min(point-probe.minimumFade.xyz,probe.maximumStrength.xyz-point);float t=saturate(min(distance.x,min(distance.y,distance.z))/probe.minimumFade.w);return t*t*(3-2*t)*probe.maximumStrength.w;}
float3 probeWeights(float3 point){float a=frame.environmentParams.z>0?probeWeight(point,0):0;float b=frame.environmentParams.z>1?probeWeight(point,1):0;float global=max(1-a-b,0);return float3(global,a,b)/max(global+a+b,0.00001);}
float3 boxProjection(float3 point,float3 ray,uint i){LocalProbeData probe=frame.localProbes[i];float3 safeRay=float3(abs(ray.x)<0.000001?(ray.x<0?-0.000001:0.000001):ray.x,abs(ray.y)<0.000001?(ray.y<0?-0.000001:0.000001):ray.y,abs(ray.z)<0.000001?(ray.z<0?-0.000001:0.000001):ray.z);float3 bounds=float3(ray.x>=0?probe.maximumStrength.x:probe.minimumFade.x,ray.y>=0?probe.maximumStrength.y:probe.minimumFade.y,ray.z>=0?probe.maximumStrength.z:probe.minimumFade.z);float3 distances=(bounds-point)/safeRay;float t=min(distances.x,min(distances.y,distances.z));return safeNormalize(point+ray*t-probe.positionSlot.xyz);}
float3 diffuseEnvironment(float3 n,float3 position)
{
    float basis[9] = { 0.2820948,0.48860252*n.y,0.48860252*n.z,0.48860252*n.x,
        1.0925485*n.x*n.y,1.0925485*n.y*n.z,0.31539157*(3.0*n.z*n.z-1.0),
        1.0925485*n.x*n.z,0.54627424*(n.x*n.x-n.y*n.y) };
    if(frame.environmentParams.y<0.5)return float3(0);
    float3 irradianceOverPi = float3(0.0);
    float3 weights=probeWeights(position);
    for(uint probe=0;probe<3;++probe){if(weights[probe]>0){uint slot=probe==0?0:uint(frame.localProbes[probe-1].positionSlot.w);for(uint i=0;i<9;++i)irradianceOverPi+=(probe==0&&frame.environmentParams.w<0.5?frame.diffuseSH[i].rgb:environmentSH[slot*9+i].rgb)*basis[i]*weights[probe];}}
    return max(irradianceOverPi,0.0);
}
float3 specularEnvironment(float3 n, float3 v, float roughness, float3 f0,float3 position)
{
    if (frame.environmentParams.x < 0.5) return float3(0.0);
    float3 r = reflect(-v,n);
    float3 weights=probeWeights(position);
    float3 totalRadiance=0;
    for(uint probe=0;probe<3;++probe){if(weights[probe]<=0)continue;uint slot=probe==0?0:uint(frame.localProbes[probe-1].positionSlot.w);float3 reflection=probe==0?r:boxProjection(position,r,probe-1);
    float u = frac(atan2(reflection.z,reflection.x) / 6.28318530718);
    float latitude = clamp(acos(clamp(reflection.y,-1.0,1.0))/3.14159265359,0.5/32.0,1.0-0.5/32.0);
    float level = roughness*5.0;
    float low = floor(level), high = min(low+1.0,5.0);
    float3 radiance = lerp(specularAtlas.SampleLevel(environmentSampler,float2(u,(slot*6+low+latitude)/18.0),0.0).rgb,
        specularAtlas.SampleLevel(environmentSampler,float2(u,(slot*6+high+latitude)/18.0),0.0).rgb,frac(level));
    totalRadiance+=radiance*weights[probe];}
    float2 lutUV = clamp(float2(saturate(dot(n,v)),roughness),float2(0.5/64.0),float2(1.0-0.5/64.0));
    float2 dfg = brdfLut.SampleLevel(environmentSampler,lutUV,0.0).rg;
    return totalRadiance*(f0*dfg.x+dfg.y);
}
float3 directLight(float3 n, float3 v, float3 l, float3 base, float metal, float roughness, float3 radiance)
{
    const float PI = 3.14159265359;
    float3 h = safeNormalize(v + l);
    float nv = max(dot(n,v), 0.0001);
    float nl = saturate(dot(n,l));
    float nh = saturate(dot(n,h));
    float vh = saturate(dot(v,h));
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denominator = nh * nh * (alpha2 - 1.0) + 1.0;
    float distribution = alpha2 / max(PI * denominator * denominator, 1e-8);
    float visibility = 0.5 / max(nl * sqrt(nv*nv*(1.0-alpha2)+alpha2)
        + nv * sqrt(nl*nl*(1.0-alpha2)+alpha2), 1e-6);
    float3 f0 = lerp(float3(0.04), base, metal);
    float3 fresnel = f0 + (1.0-f0) * pow(1.0-vh, 5.0);
    float3 diffuse = (1.0-fresnel) * (1.0-metal) * base / PI;
    float3 specular = distribution * visibility * fresnel;
    return (diffuse + specular) * radiance * nl;
}
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target0
{
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
    float3 base = draw.baseColorMetallic.xyz * baseColorTexture.Sample(baseColorSampler, input.uv).rgb;
    float2 mr = metalRoughTexture.Sample(metalRoughSampler,input.uv).gb;
    float metal = saturate(draw.baseColorMetallic.w * mr.y);
    float roughness = clamp(draw.inverseScaleSquaredRoughness.w * mr.x,0.045,1.0);
    float3 color = directLight(n,v,l,base,metal,roughness,frame.lightRadiance.xyz) * directionalVisibility(input.worldPosition);
    for (uint i = 0; i < uint(frame.lightRadiance.w); ++i) {
        PointLightData light = frame.pointLights[i];
        float3 delta = light.positionRange.xyz - input.worldPosition;
        float distanceSquared = dot(delta,delta);
        float distance = sqrt(distanceSquared);
        // Squared quartic window reaches zero smoothly at the finite range.
        float ratio = distance / light.positionRange.w;
        float window = saturate(1.0 - ratio*ratio*ratio*ratio);
        float attenuation = window * window / max(distanceSquared,0.01);
        color += directLight(n,v,safeNormalize(delta),base,metal,roughness,light.intensity.rgb * attenuation);
    }
    // Diffuse-only IBL. Directional shadows affect direct light only.
    float ao = lerp(1.0, occlusionTexture.Sample(occlusionSampler,input.uv).r, draw.emissiveOcclusion.w);
    float3 f0 = lerp(float3(0.04),base,metal);
    float3 ambientFresnel = f0 + (1.0-f0) * pow(1.0-saturate(dot(n,v)),5.0);
    color += specularEnvironment(n,v,roughness,f0,input.worldPosition);
    color += (1.0-ambientFresnel) * (1.0-metal) * base * diffuseEnvironment(n,input.worldPosition) * ao;
    color += draw.emissiveOcclusion.rgb * emissiveTexture.Sample(emissiveSampler,input.uv).rgb;


    if(frame.cascadeParams.z>0.5){float depth=mul(frame.viewProjection,float4(input.worldPosition,1)).w;uint i=depth<frame.cascadeSplits.x?0:depth<frame.cascadeSplits.y?1:depth<frame.cascadeSplits.z?2:3;float3 tint[4]={float3(1,0.25,0.25),float3(0.25,1,0.25),float3(0.25,0.4,1),float3(1,1,0.25)};color*=tint[i];}
    return float4(color, 1.0);
}
