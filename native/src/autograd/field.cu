// Device-pointer field primitives for the isolated full-chain reverse evaluator.
// No parameter differences or point/segment interaction tape are used.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr int threads = 128;
template<class T> struct Vec { T x,y,z; };
template<class T> __device__ Vec<T> add(Vec<T>a,Vec<T>b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
template<class T> __device__ Vec<T> mul(Vec<T>a,T b){return {a.x*b,a.y*b,a.z*b};}
template<class T> __device__ T dot(Vec<T>a,Vec<T>b){return a.x*b.x+a.y*b.y+a.z*b.z;}
template<class T> __device__ Vec<T> cross(Vec<T>a,Vec<T>b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
template<class T> __device__ Vec<T> get(const T*p,int i){return {p[3*i],p[3*i+1],p[3*i+2]};}
template<class T> __device__ void put(T*p,int i,Vec<T>a){p[3*i]=a.x;p[3*i+1]=a.y;p[3*i+2]=a.z;}
template<class T> __device__ T sum_block(T v){
    __shared__ T partial[4];
    for(int k=16;k;k>>=1)v+=__shfl_down_sync(0xffffffff,v,k);
    if((threadIdx.x&31)==0)partial[threadIdx.x>>5]=v;
    __syncthreads();
    T s=threadIdx.x<4?partial[threadIdx.x]:T(0);
    if(threadIdx.x<32)for(int k=16;k;k>>=1)s+=__shfl_down_sync(0xffffffff,s,k);
    __syncthreads();
    return s;
}
template<class T,bool Potential> __global__ void forward_kernel(
    const T* pos,const T* weights,const T* points,T* value,T* gradient,int ns,int np){
    const int p=blockIdx.x;if(p>=np)return;
    const Vec<T> q=get(points,p);T accum[12]={};
    for(int s=threadIdx.x;s<ns;s+=threads){
        const Vec<T> r=add(q,mul(get(pos,s),T(-1))),w=get(weights,s);
        const T r2=fmax(dot(r,r),T(1e-24));const T inv=T(1)/sqrt(r2),f3=inv*inv*inv;
        const Vec<T> v=Potential?mul(w,inv):mul(cross(w,r),f3);
        accum[0]+=v.x;accum[1]+=v.y;accum[2]+=v.z;
        if(!Potential&&gradient){
            const Vec<T> t=cross(w,r);const T rr[3]={r.x,r.y,r.z};
            const Vec<T> e[3]={{1,0,0},{0,1,0},{0,0,1}};
            for(int j=0;j<3;++j){
                const Vec<T> g=add(mul(cross(w,e[j]),f3),mul(t,-T(3)*rr[j]*f3/r2));
                accum[3+j]+=g.x;accum[6+j]+=g.y;accum[9+j]+=g.z;
            }
        }
    }
    const int count=(!Potential&&gradient)?12:3;
    for(int j=0;j<count;++j){const T v=sum_block(accum[j])*T(1e-7);
        if(threadIdx.x==0){if(j<3)value[3*p+j]=v;else gradient[9*p+j-3]=v;}}
}

template<class T,bool Potential> __device__ void pair_reverse(
    Vec<T> r,Vec<T>w,Vec<T>a,const T* h,Vec<T>&dw,Vec<T>&dr){
    const T r2=fmax(dot(r,r),T(1e-24)),inv=T(1)/sqrt(r2),f3=inv*inv*inv;
    const bool radial_active=dot(r,r)>T(1e-24);
    if(Potential){dw=mul(a,inv);dr=radial_active?mul(r,-dot(a,w)*f3):Vec<T>{0,0,0};}
    else {
        const Vec<T> t=cross(w,r);
        Vec<T> hr{0,0,0},ht{0,0,0},hw{0,0,0};T scalar=0;
        if(h){
            hr={h[0]*r.x+h[1]*r.y+h[2]*r.z,h[3]*r.x+h[4]*r.y+h[5]*r.z,h[6]*r.x+h[7]*r.y+h[8]*r.z};
            ht={h[0]*t.x+h[3]*t.y+h[6]*t.z,h[1]*t.x+h[4]*t.y+h[7]*t.z,h[2]*t.x+h[5]*t.y+h[8]*t.z};
            const Vec<T> e[3]={{1,0,0},{0,1,0},{0,0,1}};
            for(int j=0;j<3;++j){const Vec<T> col{h[j],h[3+j],h[6+j]};
                scalar+=dot(col,cross(w,e[j]));hw=add(hw,cross(e[j],col));}
        }
        const T f5=f3/r2;
        const Vec<T> at=add(mul(a,f3),mul(hr,-T(3)*f5));
        dw=add(cross(r,at),mul(hw,f3));dr=add(cross(at,w),mul(ht,-T(3)*f5));
        if(radial_active)dr=add(dr,mul(r,-T(3)*f5*(dot(a,t)+scalar)+T(15)*f5/r2*dot(t,hr)));
    }
    dw=mul(dw,T(1e-7));dr=mul(dr,T(1e-7));
}
template<class T,bool Potential,bool Segment> __global__ void reverse_kernel(
    const T* pos,const T* weights,const T* points,const T* av,const T* ag,
    T* dp,T* dw,T* dq,int ns,int np){
    const int outer=blockIdx.x;if(outer>=(Segment?ns:np))return;
    Vec<T> ar{0,0,0},aw{0,0,0};
    for(int inner=threadIdx.x;inner<(Segment?np:ns);inner+=threads){
        const int s=Segment?outer:inner,p=Segment?inner:outer;
        Vec<T> vr,vw;
        pair_reverse<T,Potential>(add(get(points,p),mul(get(pos,s),T(-1))),get(weights,s),
            av?get(av,p):Vec<T>{0,0,0},ag?ag+9*p:nullptr,vw,vr);
        ar=add(ar,vr);aw=add(aw,vw);
    }
    const T rx=sum_block(ar.x),ry=sum_block(ar.y),rz=sum_block(ar.z);
    if(Segment){
        const T wx=sum_block(aw.x),wy=sum_block(aw.y),wz=sum_block(aw.z);
        if(threadIdx.x==0){put(dp,outer,Vec<T>{-rx,-ry,-rz});put(dw,outer,Vec<T>{wx,wy,wz});}
    }else if(threadIdx.x==0)put(dq,outer,Vec<T>{rx,ry,rz});
}

template<class T,bool Potential> int forward(const void*p,const void*w,const void*q,void*v,void*g,int ns,int np,std::uint64_t stream){
    if(ns<=0||np<0)return int(cudaErrorInvalidValue);if(!np)return 0;
    forward_kernel<T,Potential><<<np,threads,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const T*>(p),static_cast<const T*>(w),static_cast<const T*>(q),static_cast<T*>(v),static_cast<T*>(g),ns,np);
    return int(cudaGetLastError());
}
template<class T,bool Potential> int backward(const void*p,const void*w,const void*q,const void*a,const void*h,void*dp,void*dw,void*dq,int ns,int np,std::uint64_t stream){
    if(ns<=0||np<=0)return int(cudaErrorInvalidValue);
    auto st=reinterpret_cast<cudaStream_t>(stream);
    if(dp&&dw)reverse_kernel<T,Potential,true><<<ns,threads,0,st>>>(static_cast<const T*>(p),static_cast<const T*>(w),static_cast<const T*>(q),static_cast<const T*>(a),static_cast<const T*>(h),static_cast<T*>(dp),static_cast<T*>(dw),static_cast<T*>(dq),ns,np);
    if(dq)reverse_kernel<T,Potential,false><<<np,threads,0,st>>>(static_cast<const T*>(p),static_cast<const T*>(w),static_cast<const T*>(q),static_cast<const T*>(a),static_cast<const T*>(h),static_cast<T*>(dp),static_cast<T*>(dw),static_cast<T*>(dq),ns,np);
    return int(cudaGetLastError());
}
}

#define FC_EXPORT(SUFFIX,TYPE) \
extern "C" int fc_field_##SUFFIX(const void*p,const void*w,const void*q,void*b,void*g,int ns,int np,std::uint64_t st){return forward<TYPE,false>(p,w,q,b,g,ns,np,st);} \
extern "C" int fc_potential_##SUFFIX(const void*p,const void*w,const void*q,void*a,int ns,int np,std::uint64_t st){return forward<TYPE,true>(p,w,q,a,nullptr,ns,np,st);} \
extern "C" int fc_field_vjp_##SUFFIX(const void*p,const void*w,const void*q,const void*a,const void*h,void*dp,void*dw,void*dq,int ns,int np,std::uint64_t st){return backward<TYPE,false>(p,w,q,a,h,dp,dw,dq,ns,np,st);} \
extern "C" int fc_potential_vjp_##SUFFIX(const void*p,const void*w,const void*q,const void*a,void*dp,void*dw,void*dq,int ns,int np,std::uint64_t st){return backward<TYPE,true>(p,w,q,a,nullptr,dp,dw,dq,ns,np,st);}
FC_EXPORT(f32,float)
FC_EXPORT(f64,double)
extern "C" const char* fc_error_string(int code){return cudaGetErrorString(static_cast<cudaError_t>(code));}
