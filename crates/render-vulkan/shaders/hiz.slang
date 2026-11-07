[[vk::binding(0,0)]] Texture2D<float> source;
[[vk::binding(2,0)]] RWStructuredBuffer<float> pyramid;
struct HiZPush { uint4 sourceInfo; uint4 targetInfo; }; // offsets, source dimensions; target dimensions, base level, unused
[[vk::push_constant]] ConstantBuffer<HiZPush> params;
[shader("compute")]
[numthreads(8,8,1)]
void reduceDepth(uint3 thread : SV_DispatchThreadID)
{
    uint2 p=thread.xy;
    if (any(p>=params.targetInfo.xy)) return;
    float value=1.0;
    if (params.targetInfo.z!=0) value=source.Load(int3(p,0));
    else {
        for (uint y=0;y<2;++y) for(uint x=0;x<2;++x) {
            uint2 q=p*2+uint2(x,y);
            if(all(q<params.sourceInfo.zw)) value=min(value,pyramid[params.sourceInfo.x+q.y*params.sourceInfo.z+q.x]);
        }
    }
    pyramid[params.sourceInfo.y+p.y*params.targetInfo.x+p.x]=value;
}
