// Shared storage ABI: header, 1024 pairs of float4 lights, then 32 mask words per tile.
static const uint MAX_LIGHTS = 1024;
static const uint MASK_WORDS = MAX_LIGHTS / 32;
static const uint MASK_OFFSET = 4 + MAX_LIGHTS * 8;
#ifdef GOVNO_LIGHT_COMPUTE
[[vk::binding(0,0)]] RWStructuredBuffer<uint> lighting;
#else
[[vk::binding(11,0)]] StructuredBuffer<uint> lighting;
#endif
float4 lightVector(uint light, uint component)
{
    uint base = 4 + light * 8 + component * 4;
    return asfloat(uint4(lighting[base],lighting[base+1],lighting[base+2],lighting[base+3]));
}
