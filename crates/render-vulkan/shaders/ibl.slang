[[vk::binding(0,0)]] Texture2D<float4> panorama;
[[vk::binding(1,0)]] SamplerState panoramaSampler;
[[vk::binding(2,0)]][[vk::image_format("rgba32f")]] RWTexture2D<float4> atlas;
[[vk::binding(3,0)]][[vk::image_format("rgba32f")]] RWTexture2D<float4> lut;
[[vk::binding(4,0)]] RWStructuredBuffer<float4> sh;
struct Params{uint probe;uint samples;uint pad0;uint pad1;};
[[vk::push_constant]] ConstantBuffer<Params> p;
float3 importanceGGX(uint i,float roughness){float x=float(i)/128.0;float y=float(reversebits(i))*2.3283064365386963e-10;float a2=pow(roughness,4.0);float c=sqrt((1-y)/(1+(a2-1)*y));float s=sqrt(max(1-c*c,0));return float3(cos(6.283185307*x)*s,sin(6.283185307*x)*s,c);}
float3 samplePanorama(float3 n){return panorama.SampleLevel(panoramaSampler,float2(frac(atan2(n.z,n.x)/6.283185307),acos(clamp(n.y,-1.0,1.0))/3.141592654),0).rgb;}
[shader("compute")][numthreads(8,8,1)]
void prefilterIBL(uint3 id:SV_DispatchThreadID){
 if(id.x>=64||id.y>=32||id.z>=6)return;
 float theta=3.141592654*(id.y+0.5)/32.0;float phi=6.283185307*(id.x+0.5)/64.0;
 float3 n=float3(sin(theta)*cos(phi),cos(theta),sin(theta)*sin(phi));float roughness=float(id.z)/5;
 float3 color=samplePanorama(n);
 if(id.z>0){float3 tangent=normalize(cross(abs(n.z)<0.99?float3(0,0,1):float3(1,0,0),n));float3 bitangent=cross(n,tangent);color=0;float total=0;
 for(uint i=0;i<128;++i){float3 h=importanceGGX(i,roughness);h=tangent*h.x+bitangent*h.y+n*h.z;float3 l=2*dot(n,h)*h-n;float weight=max(dot(n,l),0);if(weight>0){total+=weight;color=lerp(color,samplePanorama(l),weight/total);}}}
 atlas[int2(id.x,(p.probe*6+id.z)*32+id.y)]=float4(color,1);
}
[shader("compute")][numthreads(8,8,1)]
void integrateBRDF(uint3 id:SV_DispatchThreadID){
 if(any(id.xy>=64))return;float nv=(id.x+0.5)/64.0;float roughness=(id.y+0.5)/64.0;float3 v=float3(sqrt(1-nv*nv),0,nv);float a2=pow(roughness,4);float2 result=0;
 for(uint i=0;i<128;++i){float3 h=importanceGGX(i,roughness);float vh=max(dot(v,h),0);float3 l=2*vh*h-v;float nl=max(l.z,0);if(nl>0){float visibility=0.5/max(nl*sqrt(nv*nv*(1-a2)+a2)+nv*sqrt(nl*nl*(1-a2)+a2),1e-8);float weight=4*visibility*nl*vh/max(h.z,1e-6);float f=pow(1-vh,5);result+=float2(1-f,f)*weight;}}
 lut[id.xy]=float4(result/128.0,0,1);
}
[shader("compute")][numthreads(9,1,1)]
void projectSH(uint3 id:SV_DispatchThreadID){
 uint i=id.x;if(i>=9)return;float3 sum=0;
 for(uint y=0;y<32;++y){float t0=3.141592654*y/32.0;float t1=3.141592654*(y+1)/32.0;float theta=(t0+t1)*0.5;float weight=(cos(t0)-cos(t1))*6.283185307/64.0;
 for(uint x=0;x<64;++x){float phi=6.283185307*(x+0.5)/64.0;float3 n=float3(sin(theta)*cos(phi),cos(theta),sin(theta)*sin(phi));float basis[9]={0.2820948,0.48860252*n.y,0.48860252*n.z,0.48860252*n.x,1.0925485*n.x*n.y,1.0925485*n.y*n.z,0.31539157*(3*n.z*n.z-1),1.0925485*n.x*n.z,0.54627424*(n.x*n.x-n.y*n.y)};sum+=samplePanorama(n)*basis[i]*weight;}}
 sh[p.probe*9+i]=float4(sum*(i==0?1:i<4?2.0/3.0:0.25),0);
}
