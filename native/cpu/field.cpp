#include "pipeline/types.cuh"
#include "pipeline/axis.cuh"
#include "pipeline/coil_geometry.cuh"
#include "coil_field_internal.h"
#include "simd_avx2.h"
#include <exception>
#include <memory>
#include <type_traits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
thread_local std::string error;
struct Field {
    std::vector<double> p,w;
    std::vector<float> p32,w32;
    int threads=1;
};
template<class T> const T* positions(const Field& f) {
    if constexpr(std::is_same_v<T,float>)return f.p32.data();
    else return f.p.data();
}
template<class T> const T* elements(const Field& f) {
    if constexpr(std::is_same_v<T,float>)return f.w32.data();
    else return f.w.data();
}
template<class T> void point_field(const Field& f,const double* x,double* b,double* grad=nullptr,double* potential=nullptr) {
    // FP32 conversion is fixed when the immutable field is created. Preserve
    // the segment order and arithmetic of the original per-query conversion.
    const T* p=positions<T>(f);const T* w=elements<T>(f);
    T sum[3]={},jac[9]={},a[3]={};
    for(std::size_t k=0;k<f.p.size();k+=3) {
        const T rx=T(x[0])-p[k],ry=T(x[1])-p[k+1],rz=T(x[2])-p[k+2];
        const T wx=w[k],wy=w[k+1],wz=w[k+2];
        const T r2=rx*rx+ry*ry+rz*rz+T(sizeof(T)==4?1e-30:1e-300);
        const T q=T(1)/std::sqrt(r2),q3=q*q*q,q5=q3*q*q;
        const T cross[3]={wy*rz-wz*ry,wz*rx-wx*rz,wx*ry-wy*rx};
        for(int i=0;i<3;++i)sum[i]+=cross[i]*q3;
        if(potential){a[0]+=wx*q;a[1]+=wy*q;a[2]+=wz*q;}
        if(grad){
            const T r[3]={rx,ry,rz};const T skew[9]={0,-wz,wy,wz,0,-wx,-wy,wx,0};
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)jac[3*i+j]+=skew[3*i+j]*q3-T(3)*cross[i]*r[j]*q5;
        }
    }
    for(int i=0;i<3;++i){b[i]=double(sum[i]*T(1e-7));if(potential)potential[i]=double(a[i]*T(1e-7));}
    if(grad)for(int i=0;i<9;++i)grad[i]=double(jac[i]*T(1e-7));
}
void rhs(const Field& f,double R,double Z,double phi,bool mixed,double& r,double& z) {
    const double c=std::cos(phi),s=std::sin(phi),x[3]={R*c,R*s,Z};double b[3];
    if(mixed)point_field<float>(f,x,b);else point_field<double>(f,x,b);
    const double br=b[0]*c+b[1]*s,bp=-b[0]*s+b[1]*c;
    const double den=std::abs(bp)>1e-14?bp:std::copysign(1e-14,bp==0?1:bp);
    r=R*br/den;z=R*b[2]/den;
}
void step(const Field& f,double& r,double& z,double phi,double h,bool mixed) {
    double a,b,c,d,e,g,i,j;rhs(f,r,z,phi,mixed,a,b);
    rhs(f,r+.5*h*a,z+.5*h*b,phi+.5*h,mixed,c,d);
    rhs(f,r+.5*h*c,z+.5*h*d,phi+.5*h,mixed,e,g);
    rhs(f,r+h*e,z+h*g,phi+h,mixed,i,j);
    r+=h*(a+2*c+2*e+i)/6;z+=h*(b+2*d+2*g+j)/6;
}
void rhs8(const Field& f,const double* R,const double* Z,double phi,double* r,double* z) {
    const double c=std::cos(phi),s=std::sin(phi);
    double xyz[24],magnetic[24];
    for(int k=0;k<8;++k){xyz[3*k]=R[k]*c;xyz[3*k+1]=R[k]*s;xyz[3*k+2]=Z[k];}
    coil_eval::cpu::field8_f32(f.p32.data(),f.w32.data(),int(f.p32.size()/3),
                              xyz,magnetic,nullptr,nullptr);
    for(int k=0;k<8;++k){
        const double br=magnetic[3*k]*c+magnetic[3*k+1]*s;
        const double bp=-magnetic[3*k]*s+magnetic[3*k+1]*c;
        const double den=std::abs(bp)>1e-14?bp:std::copysign(1e-14,bp==0?1:bp);
        r[k]=R[k]*br/den;z[k]=R[k]*magnetic[3*k+2]/den;
    }
}
void step8(const Field& f,double* r,double* z,double phi,double h) {
    // Eight independent trajectories, each retaining the scalar FP64 RK4 order.
    double a[8],b[8],c[8],d[8],e[8],g[8],i[8],j[8],stage_r[8],stage_z[8];
    rhs8(f,r,z,phi,a,b);
    for(int k=0;k<8;++k){stage_r[k]=r[k]+.5*h*a[k];stage_z[k]=z[k]+.5*h*b[k];}
    rhs8(f,stage_r,stage_z,phi+.5*h,c,d);
    for(int k=0;k<8;++k){stage_r[k]=r[k]+.5*h*c[k];stage_z[k]=z[k]+.5*h*d[k];}
    rhs8(f,stage_r,stage_z,phi+.5*h,e,g);
    for(int k=0;k<8;++k){stage_r[k]=r[k]+h*e[k];stage_z[k]=z[k]+h*g[k];}
    rhs8(f,stage_r,stage_z,phi+h,i,j);
    for(int k=0;k<8;++k){r[k]+=h*(a[k]+2*c[k]+2*e[k]+i[k])/6;z[k]+=h*(b[k]+2*d[k]+2*g[k]+j[k])/6;}
}
int trace(void* h,const double* R,const double* Z,double* Re,double* Ze,int n,double p0,double p1,int steps,bool mixed){
    if(!h||steps<1||n<0)return 1;const auto& f=*static_cast<Field*>(h);const double dt=(p1-p0)/steps;
    if(mixed && coil_eval::cpu::avx2_available() && n>=8){
        const int groups=n/8;
        #pragma omp parallel for num_threads(f.threads) schedule(static)
        for(int group=0;group<groups;++group){
            double r[8],z[8];
            for(int k=0;k<8;++k){r[k]=R[8*group+k];z[k]=Z[8*group+k];}
            for(int j=0;j<steps;++j)step8(f,r,z,p0+j*dt,dt);
            for(int k=0;k<8;++k){Re[8*group+k]=r[k];Ze[8*group+k]=z[k];}
        }
        for(int k=8*groups;k<n;++k){double r=R[k],z=Z[k];for(int j=0;j<steps;++j)step(f,r,z,p0+j*dt,dt,true);Re[k]=r;Ze[k]=z;}
        return 0;
    }
    #pragma omp parallel for num_threads(f.threads) schedule(static)
    for(int k=0;k<n;++k){double r=R[k],z=Z[k];for(int j=0;j<steps;++j)step(f,r,z,p0+j*dt,dt,mixed);Re[k]=r;Ze[k]=z;}
    return 0;
}
}
namespace coil_eval::pipeline { thread_local sgpu_axis_coarse::Slot g_axis_coarse_slot; }
extern "C" {
int ce_cpu_openmp_enabled(){
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}
int ce_cpu_avx2_enabled(){return coil_eval::cpu::avx2_available()?1:0;}
const char* sgpu_last_error(){return error.c_str();}
void sgpu_internal_set_error(const char* s){error=s?s:"";}
int ce_cpu_open(const double* p,const double* w,int n,int threads,void** output){
    try {
        auto f=std::make_unique<Field>();
        f->p.assign(p,p+3*n);f->w.assign(w,w+3*n);
        f->p32.assign(p,p+3*n);f->w32.assign(w,w+3*n);
        f->threads=std::max(1,threads);*output=f.release();return 0;
    }
    catch(const std::exception& e){error=e.what();return 1;}
}
void ce_cpu_close(void* f){delete static_cast<Field*>(f);}
int ce_cpu_field(void* h,const double* xyz,int count,int mixed,double* B,double* gradient,double* A){
    if(!h||count<0)return 1;const auto& f=*static_cast<Field*>(h);
    if(mixed && coil_eval::cpu::avx2_available() && count>=8){
        const int groups=count/8;
        #pragma omp parallel for num_threads(f.threads) schedule(static)
        for(int group=0;group<groups;++group){
            const int first=8*group;
            coil_eval::cpu::field8_f32(f.p32.data(),f.w32.data(),int(f.p32.size()/3),
                xyz+3*first,B+3*first,gradient?gradient+9*first:nullptr,A?A+3*first:nullptr);
        }
        for(int k=8*groups;k<count;++k)point_field<float>(f,xyz+3*k,B+3*k,
            gradient?gradient+9*k:nullptr,A?A+3*k:nullptr);
        return 0;
    }
    #pragma omp parallel for num_threads(f.threads) schedule(static)
    for(int k=0;k<count;++k){double* g=gradient?gradient+9*k:nullptr;double* a=A?A+3*k:nullptr;
        if(mixed)point_field<float>(f,xyz+3*k,B+3*k,g,a);else point_field<double>(f,xyz+3*k,B+3*k,g,a);}
    return 0;
}
int sgpu_eval_B(void* h,const double* xyz,double* B,int n){return ce_cpu_field(h,xyz,n,0,B,nullptr,nullptr);}
int sgpu_trace_period_blockline(void* f,const double* R,const double* Z,double* Re,double* Ze,int n,int nfp,int steps,int){return trace(f,R,Z,Re,Ze,n,0,coil_eval::pipeline::TWOPI/nfp,steps,false);}
int sgpu_trace_period_blockline_mixed(void* f,const double* R,const double* Z,double* Re,double* Ze,int n,int nfp,int steps,int,int){return trace(f,R,Z,Re,Ze,n,0,coil_eval::pipeline::TWOPI/nfp,steps,true);}
int ce_cpu_trace(void* f,const double* R,const double* Z,double* Re,double* Ze,int n,double p0,double p1,int steps,int mixed){return trace(f,R,Z,Re,Ze,n,p0,p1,steps,mixed!=0);}
int sgpu_trace_axis_samples(void* h,double r,double z,int nfp,int integration_steps,int count,double* R,double* Z,double* Rp,double* Zp){
    if(!h||count<1)return 1;const auto& f=*static_cast<Field*>(h);int sub=(integration_steps+count-1)/count;const double dt=coil_eval::pipeline::TWOPI/(nfp*count*sub);
    for(int k=0;k<count;++k){R[k]=r;Z[k]=z;rhs(f,r,z,k*sub*dt,true,Rp[k],Zp[k]);for(int j=0;j<sub;++j)step(f,r,z,(k*sub+j)*dt,dt,true);}return 0;
}
int ce_cpu_axis(void* f,const double* x,const double* y,const double* z,int nb,int nc,int nfp,const SgpuScoreConfig* cfg,double* samples,double* info,double* timings){
    using namespace coil_eval::pipeline;AxisData axis;double time=0;
    if(!find_axis_native(f,x,y,z,nb,nc,nfp,*cfg,axis,time,timings))return 1;
    if(axis.R.empty())return axis.branch_lost?3:2;
    for(int k=0;k<cfg->axis_sample_count;++k){samples[4*k]=axis.R[k];samples[4*k+1]=axis.Z[k];samples[4*k+2]=axis.R_phi[k];samples[4*k+3]=axis.Z_phi[k];}
    info[0]=axis.selected.R;info[1]=axis.selected.Z;info[2]=axis.selected.residual;info[3]=axis.selected.topology_trace;info[4]=axis.selected.topology_det;info[5]=axis.selected.ellipse_aspect;info[6]=axis.candidate_count;info[7]=time;return 0;
}
int ce_cpu_coil_metrics(const double* x,const double* y,const double* z,const double* c,int nb,int nc,int nfp,double* values){
    auto m=coil_eval::pipeline::compute_coil_metrics(x,y,z,c,nb,nc,nfp);
    values[0]=m.length_mean;values[1]=m.curvature_p95;values[2]=m.curvature_max;values[3]=m.min_intercoil_distance;values[4]=m.min_axis_distance;values[5]=m.high_mode_fraction;values[6]=m.current_abs_max;return 0;
}
}
