[[vk::binding(0,0)]] Texture2D<float4> source;
[[vk::binding(1,0)]] RWStructuredBuffer<float> exposureState;
[[vk::binding(2,0)]][[vk::image_format("rgba16f")]] RWTexture2D<float4> target;
struct Parameters {float4 exposure;float4 bloom;float4 timing;};
[[vk::push_constant]] ConstantBuffer<Parameters> p;
groupshared float luminance[64];
[shader("compute")][numthreads(8,8,1)]
void meterExposure(uint3 id:SV_GroupThreadID,uint lane:SV_GroupIndex) {
 uint width,height;source.GetDimensions(width,height);
 float sum=0;
 for(uint y=id.y;y<64;y+=8) for(uint x=id.x;x<64;x+=8) {
  int2 pixel=min(int2((float2(x,y)+0.5)*float2(width,height)/64.0),int2(width-1,height-1));
  float3 c=clamp(source.Load(int3(pixel,0)).rgb,0.0,65504.0);
  sum+=log(max(dot(c,float3(0.2126,0.7152,0.0722)),0.00001));
 }
 luminance[lane]=sum;GroupMemoryBarrierWithGroupSync();
 for(uint stride=32;stride>0;stride/=2) {if(lane<stride)luminance[lane]+=luminance[lane+stride];GroupMemoryBarrierWithGroupSync();}
 if(lane==0) {
  float desired=p.bloom.w>0.5?clamp(p.exposure.y/exp(luminance[0]/4096.0),p.exposure.z,p.exposure.w):1.0;
  float prev=exposureState[2]>0.5?exposureState[1]:desired;
  float adapted=p.bloom.w>0.5?clamp(lerp(prev,desired,1.0-exp(-p.timing.x*p.timing.y)),p.exposure.z,p.exposure.w):1.0;
  exposureState[0]=adapted*p.exposure.x;exposureState[1]=adapted;exposureState[2]=1.0;exposureState[3]=exp(luminance[0]/4096.0);
 }
}
[shader("compute")][numthreads(8,8,1)]
void bloomExtract(uint3 id:SV_DispatchThreadID) {
 uint w,h;target.GetDimensions(w,h);if(any(id.xy>=uint2(w,h)))return;
 uint sw,sh;source.GetDimensions(sw,sh);
 float3 c=0;for(int y=0;y<2;++y)for(int x=0;x<2;++x)c+=clamp(source.Load(int3(min(id.xy*2+uint2(x,y),uint2(sw-1,sh-1)),0)).rgb,0.0,65504.0)*0.25;
 c=max(c*exposureState[0],0.0);float brightness=max(c.r,max(c.g,c.b));
 float knee=max(p.bloom.y,0.00001);float soft=clamp(brightness-p.bloom.x+knee,0.0,2.0*knee);soft=soft*soft/(4.0*knee);
 float contribution=max(brightness-p.bloom.x,soft)/max(brightness,0.00001);
 target[id.xy]=float4(p.bloom.z>0.0?min(c*contribution,65504.0):float3(0),1);
}
[shader("compute")][numthreads(8,8,1)]
void bloomBlur(uint3 id:SV_DispatchThreadID) {
 uint w,h;target.GetDimensions(w,h);if(any(id.xy>=uint2(w,h)))return;
 float weights[5]={0.227027,0.1945946,0.1216216,0.054054,0.016216};
 int2 direction=p.timing.w<0.5?int2(1,0):int2(0,1);int2 pixel=int2(id.xy);
 float3 color=source.Load(int3(pixel,0)).rgb*weights[0];
 for(int i=1;i<5;++i)for(int sign=-1;sign<=1;sign+=2)color+=source.Load(int3(clamp(pixel+direction*i*sign,int2(0),int2(w-1,h-1)),0)).rgb*weights[i];
 target[id.xy]=float4(color,1);
}
