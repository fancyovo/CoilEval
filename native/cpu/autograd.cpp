// CPU Biot-Savart primitives and discrete RK4 reverse. All buffers belong to
// the caller. OpenMP distributes independent points, segments, or trajectories;
// no CUDA runtime, interaction tape, or numerical differences are required.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include "autograd_simd_avx2.h"

namespace {
template<class T> struct Vec { T x, y, z; };
template<class T> Vec<T> add(Vec<T> a, Vec<T> b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
template<class T> Vec<T> mul(Vec<T> a, T b) { return {a.x*b,a.y*b,a.z*b}; }
template<class T> T dot(Vec<T> a, Vec<T> b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
template<class T> Vec<T> cross(Vec<T> a, Vec<T> b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
template<class T> Vec<T> get(const T* p, std::size_t i) { return {p[3*i],p[3*i+1],p[3*i+2]}; }
template<class T> void put(T* p, std::size_t i, Vec<T> v) { p[3*i]=v.x;p[3*i+1]=v.y;p[3*i+2]=v.z; }

template<class T, bool Potential> void point_field(
    const T* p, const T* w, Vec<T> q, int ns, T* value, T* gradient) {
    T sum[12]={};
    for(int s=0;s<ns;++s) {
        const Vec<T> r=add(q,mul(get(p,s),T(-1))), element=get(w,s);
        const T distance2=dot(r,r), r2=std::max(distance2,T(1e-24));
        const T inv=T(1)/std::sqrt(r2), inv3=inv*inv*inv;
        const Vec<T> t=cross(element,r), v=Potential?mul(element,inv):mul(t,inv3);
        sum[0]+=v.x;sum[1]+=v.y;sum[2]+=v.z;
        if(!Potential && gradient) {
            const T coords[3]={r.x,r.y,r.z};
            const Vec<T> axes[3]={{1,0,0},{0,1,0},{0,0,1}};
            for(int j=0;j<3;++j) {
                const T radial=distance2>T(1e-24)?-T(3)*coords[j]*inv3/r2:T(0);
                const Vec<T> g=add(mul(cross(element,axes[j]),inv3),mul(t,radial));
                sum[3+j]+=g.x;sum[6+j]+=g.y;sum[9+j]+=g.z;
            }
        }
    }
    for(int k=0;k<3;++k)value[k]=sum[k]*T(1e-7);
    if(!Potential && gradient)for(int k=0;k<9;++k)gradient[k]=sum[k+3]*T(1e-7);
}

template<class T, bool Potential> void pair_reverse(
    Vec<T> r, Vec<T> w, Vec<T> a, const T* h, Vec<T>& dw, Vec<T>& dr) {
    const T distance2=dot(r,r), r2=std::max(distance2,T(1e-24));
    const T inv=T(1)/std::sqrt(r2), f3=inv*inv*inv;
    const bool active=distance2>T(1e-24);
    if(Potential) {
        dw=mul(a,inv);dr=active?mul(r,-dot(a,w)*f3):Vec<T>{0,0,0};
    } else {
        const Vec<T> t=cross(w,r);
        Vec<T> hr{0,0,0},ht{0,0,0},hw{0,0,0};T scalar=0;
        if(h) {
            hr={h[0]*r.x+h[1]*r.y+h[2]*r.z,h[3]*r.x+h[4]*r.y+h[5]*r.z,h[6]*r.x+h[7]*r.y+h[8]*r.z};
            ht={h[0]*t.x+h[3]*t.y+h[6]*t.z,h[1]*t.x+h[4]*t.y+h[7]*t.z,h[2]*t.x+h[5]*t.y+h[8]*t.z};
            const Vec<T> axes[3]={{1,0,0},{0,1,0},{0,0,1}};
            for(int j=0;j<3;++j) {
                const Vec<T> col{h[j],h[3+j],h[6+j]};
                scalar+=dot(col,cross(w,axes[j]));hw=add(hw,cross(axes[j],col));
            }
        }
        const T f5=f3/r2;
        const Vec<T> at=active?add(mul(a,f3),mul(hr,-T(3)*f5)):mul(a,f3);
        dw=add(cross(r,at),mul(hw,f3));dr=cross(at,w);
        if(active) {
            dr=add(dr,mul(ht,-T(3)*f5));
            dr=add(dr,mul(r,-T(3)*f5*(dot(a,t)+scalar)+T(15)*f5/r2*dot(t,hr)));
        }
    }
    dw=mul(dw,T(1e-7));dr=mul(dr,T(1e-7));
}

template<class T, bool Potential> int field_forward(
    const void* pp,const void* ww,const void* qq,void* vv,void* gg,int ns,int np,int threads) {
    if(ns<=0||np<0||threads<1||!pp||!ww||(np&&(!qq||!vv)))return 1;
    const T *p=static_cast<const T*>(pp),*w=static_cast<const T*>(ww),*q=static_cast<const T*>(qq);
    T *v=static_cast<T*>(vv),*g=static_cast<T*>(gg);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int i=0;i<np;++i)point_field<T,Potential>(p,w,get(q,i),ns,v+3*i,g?g+9*i:nullptr);
    return 0;
}

template<class T, bool Potential> int field_reverse(
    const void* pp,const void* ww,const void* qq,const void* aa,const void* hh,
    void* dpp,void* dww,void* dqq,int ns,int np,int threads) {
    if(ns<=0||np<0||threads<1||!pp||!ww||(np&&!qq))return 1;
    const T *p=static_cast<const T*>(pp),*w=static_cast<const T*>(ww),*q=static_cast<const T*>(qq);
    const T *a=static_cast<const T*>(aa),*h=static_cast<const T*>(hh);
    T *dp=static_cast<T*>(dpp),*dw=static_cast<T*>(dww),*dq=static_cast<T*>(dqq);
    const coil_eval::cpu::VjpBuffers buffers{pp,ww,qq,aa,hh,dpp,dww,dqq,ns,np,threads};
    const auto simd=[&](bool segment_outputs){
        if constexpr(std::is_same_v<T,float>)return coil_eval::cpu::autograd_vjp_f32(buffers,Potential,segment_outputs);
        else return coil_eval::cpu::autograd_vjp_f64(buffers,Potential,segment_outputs);
    };
    if(dp||dw) {
        const int first=simd(true);
        #pragma omp parallel for num_threads(threads) schedule(static)
        for(int s=first;s<ns;++s) {
            Vec<T> ar{0,0,0},aw{0,0,0};const Vec<T> position=get(p,s),element=get(w,s);
            for(int i=0;i<np;++i) {
                Vec<T> vr,vw;
                pair_reverse<T,Potential>(add(get(q,i),mul(position,T(-1))),element,
                    a?get(a,i):Vec<T>{0,0,0},h?h+9*i:nullptr,vw,vr);
                ar=add(ar,vr);aw=add(aw,vw);
            }
            if(dp)put(dp,s,mul(ar,T(-1)));
            if(dw)put(dw,s,aw);
        }
    }
    if(dq) {
        const int first=simd(false);
        #pragma omp parallel for num_threads(threads) schedule(static)
        for(int i=first;i<np;++i) {
            Vec<T> ar{0,0,0};const Vec<T> point=get(q,i);
            for(int s=0;s<ns;++s) {
                Vec<T> vr,vw;
                pair_reverse<T,Potential>(add(point,mul(get(p,s),T(-1))),get(w,s),
                    a?get(a,i):Vec<T>{0,0,0},h?h+9*i:nullptr,vw,vr);
                ar=add(ar,vr);
            }
            put(dq,i,ar);
        }
    }
    return 0;
}

template<class T,class S> void rhs_stage(
    const T* p,const T* w,int ns,S R,S Z,S phi,T* xyz,S* J,S* M,S* rhs) {
    const S c=std::cos(phi),s=std::sin(phi);
    const Vec<T> point{T(R*c),T(R*s),T(Z)};T b[3],g[9];
    point_field<T,false>(p,w,point,ns,b,g);put(xyz,0,point);
    const S br=S(b[0])*c+S(b[1])*s,bp=-S(b[0])*s+S(b[1])*c,bz=S(b[2]);
    const bool active=std::abs(bp)>S(1e-14);
    const S den=active?bp:(bp<S(0)?-S(1e-14):S(1e-14));
    rhs[0]=R*br/den;rhs[1]=R*bz/den;
    const S dbp[3]={-s,c,0},dbr[3]={c,s,0};
    for(int j=0;j<3;++j) {
        M[j]=R*dbr[j]/den-(active?R*br*dbp[j]/(den*den):S(0));
        M[3+j]=(j==2?R/den:S(0))-(active?R*bz*dbp[j]/(den*den):S(0));
    }
    for(int i=0;i<2;++i) {
        S dR=(i==0?br:bz)/den,dZ=0;
        for(int j=0;j<3;++j) {
            dR+=M[3*i+j]*(S(g[3*j])*c+S(g[3*j+1])*s);
            dZ+=M[3*i+j]*S(g[3*j+2]);
        }
        J[2*i]=dR;J[2*i+1]=dZ;
    }
}

template<class T,class S> int trace_forward(
    const void* pp,const void* ww,const void* yy,const void* aa,const void* bb,
    void* ss,void* xx,void* jj,void* mm,int ns,int lines,int steps,int threads) {
    if(ns<=0||lines<=0||steps<=0||threads<1||!pp||!ww||!yy||!aa||!bb||!ss||!xx||!jj||!mm)return 1;
    const T *p=static_cast<const T*>(pp),*w=static_cast<const T*>(ww);
    const S *starts=static_cast<const S*>(yy),*phi0=static_cast<const S*>(aa),*phi1=static_cast<const S*>(bb);
    S *states=static_cast<S*>(ss),*J=static_cast<S*>(jj),*M=static_cast<S*>(mm);T* xyz=static_cast<T*>(xx);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int line=0;line<lines;++line) {
        const S begin=phi0[line],h=(phi1[line]-begin)/S(steps);
        S state[2]={starts[2*line],starts[2*line+1]};
        states[2*line]=state[0];states[2*line+1]=state[1];
        for(int step=0;step<steps;++step) {
            const S initial[2]={state[0],state[1]};S k[8];
            for(int stage=0;stage<4;++stage) {
                if(stage) {
                    const S scale=stage==3?h:h/S(2);
                    state[0]=initial[0]+scale*k[2*(stage-1)];
                    state[1]=initial[1]+scale*k[2*(stage-1)+1];
                }
                const std::size_t at=(static_cast<std::size_t>(step)*4+stage)*lines+line;
                const S phi=begin+h*S(step)+(stage==0?S(0):(stage==3?h:h/S(2)));
                rhs_stage<T,S>(p,w,ns,state[0],state[1],phi,xyz+3*at,J+4*at,M+6*at,k+2*stage);
            }
            state[0]=initial[0]+h*(k[0]+S(2)*k[2]+S(2)*k[4]+k[6])/S(6);
            state[1]=initial[1]+h*(k[1]+S(2)*k[3]+S(2)*k[5]+k[7])/S(6);
            const std::size_t at=(static_cast<std::size_t>(step)+1)*lines+line;
            states[2*at]=state[0];states[2*at+1]=state[1];
        }
    }
    return 0;
}

template<class T,class S> int trace_reverse(
    const void* jj,const void* mm,const void* aa,const void* bb,const void* vv,
    void* yy,void* ff,int lines,int steps,int all_states,int threads) {
    if(lines<=0||steps<=0||threads<1||!jj||!mm||!aa||!bb||!vv||!yy||!ff)return 1;
    const S *J=static_cast<const S*>(jj),*M=static_cast<const S*>(mm);
    const S *phi0=static_cast<const S*>(aa),*phi1=static_cast<const S*>(bb),*adj=static_cast<const S*>(vv);
    S* dy=static_cast<S*>(yy);T* dB=static_cast<T*>(ff);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int line=0;line<lines;++line) {
        const S h=(phi1[line]-phi0[line])/S(steps);
        const std::size_t last=all_states?static_cast<std::size_t>(steps)*lines+line:line;
        S carry[2]={adj[2*last],adj[2*last+1]};
        for(int step=steps-1;step>=0;--step) {
            S base[2]={carry[0],carry[1]},ak[8];
            for(int stage=0;stage<4;++stage) {
                const S weight=h/(stage==0||stage==3?S(6):S(3));
                ak[2*stage]=weight*carry[0];ak[2*stage+1]=weight*carry[1];
            }
            for(int stage=3;stage>=0;--stage) {
                const std::size_t at=(static_cast<std::size_t>(step)*4+stage)*lines+line;
                const S *j=J+4*at,*m=M+6*at,a0=ak[2*stage],a1=ak[2*stage+1];
                for(int k=0;k<3;++k)dB[3*at+k]=T(m[k]*a0+m[3+k]*a1);
                const S ay[2]={j[0]*a0+j[2]*a1,j[1]*a0+j[3]*a1};
                base[0]+=ay[0];base[1]+=ay[1];
                if(stage) {
                    const S scale=stage==3?h:h/S(2);
                    ak[2*(stage-1)]+=scale*ay[0];ak[2*(stage-1)+1]+=scale*ay[1];
                }
            }
            const std::size_t at=static_cast<std::size_t>(step)*lines+line;
            carry[0]=base[0]+(all_states?adj[2*at]:S(0));
            carry[1]=base[1]+(all_states?adj[2*at+1]:S(0));
        }
        dy[2*line]=carry[0];dy[2*line+1]=carry[1];
    }
    return 0;
}
} // namespace

#define CE_CPU_FIELD(SUFFIX,T) \
extern "C" int ce_cpu_autograd_field_##SUFFIX(const void*p,const void*w,const void*q,void*b,void*g,int ns,int np,int nt){return field_forward<T,false>(p,w,q,b,g,ns,np,nt);} \
extern "C" int ce_cpu_autograd_potential_##SUFFIX(const void*p,const void*w,const void*q,void*a,int ns,int np,int nt){return field_forward<T,true>(p,w,q,a,nullptr,ns,np,nt);} \
extern "C" int ce_cpu_autograd_field_vjp_##SUFFIX(const void*p,const void*w,const void*q,const void*a,const void*h,void*dp,void*dw,void*dq,int ns,int np,int nt){return field_reverse<T,false>(p,w,q,a,h,dp,dw,dq,ns,np,nt);} \
extern "C" int ce_cpu_autograd_potential_vjp_##SUFFIX(const void*p,const void*w,const void*q,const void*a,void*dp,void*dw,void*dq,int ns,int np,int nt){return field_reverse<T,true>(p,w,q,a,nullptr,dp,dw,dq,ns,np,nt);}
CE_CPU_FIELD(f32,float)
CE_CPU_FIELD(f64,double)

#define CE_CPU_TRACE(SUFFIX,T,S) \
extern "C" int ce_cpu_autograd_trace_forward_##SUFFIX(const void*p,const void*w,const void*y,const void*a,const void*b,void*s,void*x,void*j,void*m,int ns,int nl,int nstep,int nt){return trace_forward<T,S>(p,w,y,a,b,s,x,j,m,ns,nl,nstep,nt);} \
extern "C" int ce_cpu_autograd_trace_reverse_##SUFFIX(const void*j,const void*m,const void*a,const void*b,const void*v,void*y,void*f,int nl,int nstep,int all,int nt){return trace_reverse<T,S>(j,m,a,b,v,y,f,nl,nstep,all,nt);}
CE_CPU_TRACE(f32_f32,float,float)
CE_CPU_TRACE(f32_f64,float,double)
CE_CPU_TRACE(f64_f32,double,float)
CE_CPU_TRACE(f64_f64,double,double)
