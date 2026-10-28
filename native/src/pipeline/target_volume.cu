// Direct target-volume selection, center volume adjoint, and endpoint guards.
#include "pipeline/types.cuh"
#include "pipeline/device_helpers.cuh"
#include "pipeline/session_state.cuh"
#include "pipeline/axis.cuh"
#include "pipeline/common.cuh"
#include "pipeline/psi.cuh"
#include "pipeline/surface.cuh"
#include "pipeline/target_volume.cuh"
#include "coil_field_internal.h"
#include "host_profile.h"

namespace coil_eval::pipeline {

bool target_geometry(const AxisData& axis,const PsiData& psi,int nfp,
                     const SgpuScoreConfig& config,sgpu_target::Geometry& g,bool local=false) {
    CoilEvalHostProfile profile(local?"target.geometry.local":"target.geometry.formal");
    g.degree=config.psi_poly_degree;g.a=config.psi_a;
    if(local && g_target_volume_cache.radii.size()!=static_cast<size_t>(config.flux_phi_count)*config.flux_theta_count)return false;
    g.maximum_level=std::numeric_limits<double>::infinity();
    for(int ip=0;ip<config.flux_phi_count;++ip){
        const double phi=TWOPI*ip/(nfp*config.flux_phi_count);
        double R,Z,dR,dZ;
        periodic_hermite_host(phi,axis.R,axis.R_phi,nfp,R,dR);
        periodic_hermite_host(phi,axis.Z,axis.Z_phi,nfp,Z,dZ);
        const auto section=sgpu_target::collapse_section(psi.coeffs.data(),
            psi.modes.a.data(),psi.modes.b.data(),psi.modes.m.data(),
            psi.modes.kind.data(),psi.coeffs.size(),nfp*phi);
        for(int it=0;it<config.flux_theta_count;++it){
            sgpu_target::Ray ray;ray.R=R;ray.Z=Z;ray.phi=phi;ray.theta=TWOPI*it/config.flux_theta_count;
            const double c=std::cos(ray.theta),s=std::sin(ray.theta);
            ray.p=sgpu_target::radial_polynomial(section,g.degree,c,s);
            double lo=0,hi=config.surface_max_radius_scale;
            if(c<0)hi=std::min(hi,.999*R/(-c*g.a));
            if(!(ray.p[2]>1e-12)||!std::isfinite(ray.p[2]))return false;
            if(local)hi=std::min(hi,1.1*g_target_volume_cache.radii[g.rays.size()]/g.a);
            if(sgpu_target::nested(ray.p,g.degree,hi))lo=hi;
            else if(local)return false;
            else for(int j=0;j<32;++j){const double mid=.5*(lo+hi);if(sgpu_target::nested(ray.p,g.degree,mid))lo=mid;else hi=mid;}
            ray.umax=lo*(1-1e-6);
            const double bound=sgpu_target::value(ray.p,g.degree,ray.umax);
            if(!(bound>0&&std::isfinite(bound)))return false;
            g.maximum_level=std::min(g.maximum_level,bound);g.rays.push_back(ray);
        }
    }
    return !g.rays.empty();
}

bool screen_target_volume_native(void* field,const AxisData& axis,const PsiData& psi,
    int nfp,const SgpuScoreConfig& config,std::vector<SurfaceScreen>& screens,SgpuScoreResult& result) {
    CoilEvalHostProfile profile("target.total");
    result.target_volume_requested=config.target_volume;
    g_target_volume_cache=TargetVolumeCache{};
    sgpu_target::Geometry g;
    if(!target_geometry(axis,psi,nfp,config,g)){std::snprintf(result.error_message,sizeof(result.error_message),"no certified star-shaped nested psi branch on geometry grid");return true;}
    // Flux quadrature may use fewer toroidal sections than the fitted psi
    // representation. Certifying only those sections can miss a radial fold
    // between them. Keep the volume quadrature/adjoint unchanged, but limit
    // its admissible level using the psi fitting grid and at least two samples
    // per toroidal mode. This remains a sampled, not global, certificate.
    const int guard_phi=std::max(config.psi_n_phi,2*config.psi_m_tor+1);
    if(guard_phi>config.flux_phi_count){
        CoilEvalHostProfile guard_profile("target.capacity_guard");
        SgpuScoreConfig guard_config=config;
        guard_config.flux_phi_count=guard_phi;
        sgpu_target::Geometry guard;
        if(!target_geometry(axis,psi,nfp,guard_config,guard)){
            std::snprintf(result.error_message,sizeof(result.error_message),"no nested psi branch on toroidal capacity guard grid");return true;
        }
        g.maximum_level=std::min(g.maximum_level,guard.maximum_level);
    }
    double maximum,dummy;
    {
        CoilEvalHostProfile measure_profile("target.maximum_volume");
        if(!sgpu_target::measure(g,g.maximum_level,maximum,dummy))return true;
    }
    result.target_volume_maximum=maximum;
    double level,volume,dVds;int iterations=0;
    {
        CoilEvalHostProfile solve_profile("target.solve_volume");
        if(!sgpu_target::solve(g,config.target_volume,config.target_volume_relative_tolerance,
                              level,volume,dVds,iterations)){std::snprintf(result.error_message,sizeof(result.error_message),"target volume outside certified geometry domain or inverse solve failed");return true;}
    }
    result.target_volume_iterations=iterations;
    result.target_volume_dVds=dVds;
    result.target_volume_relative_error=volume/config.target_volume-1;
    SurfaceScreen screen;screen.level=level;
    std::vector<double> R(config.surface_theta_count),Z(R.size()),r(R.size()),Re,Ze;
    {
    CoilEvalHostProfile roots_profile("target.surface_roots");
    for(size_t j=0;j<R.size();++j){
        const double theta=TWOPI*j/R.size();std::array<double,25> p;
        ray_polynomial_phi0(psi,config,theta,p);r[j]=solve_ray_radius(p,level,config);
        if(std::abs(sgpu_target::value(p,config.psi_poly_degree,r[j]/config.psi_a)-level)>config.flux_boundary_tolerance)return true;
        R[j]=axis.R[0]+r[j]*std::cos(theta);Z[j]=axis.Z[0]+r[j]*std::sin(theta);
    }
    }
    {
        CoilEvalHostProfile trace_profile("target.surface_trace");
        if(!trace_map(field,R,Z,nfp,config.surface_trace_steps,false,Re,Ze))return false;
    }
    screen.radius_mean=std::accumulate(r.begin(),r.end(),0.0)/r.size();
    screen.radius_max=*std::max_element(r.begin(),r.end());std::vector<double> errors(r.size());
    for(size_t j=0;j<r.size();++j){
        double v,gR,gZ,gP;evaluate_psi_host(psi,axis,config,nfp,Re[j],Ze[j],TWOPI/nfp,v,gR,gZ,gP);
        errors[j]=std::abs(v-level)/std::max(1e-14,screen.radius_mean*std::sqrt(gR*gR+gZ*gZ+std::pow(gP/Re[j],2)));
        if(!std::isfinite(errors[j]))return true;
    }
    screen.relative_drift_p95=screen.one_period_relative_drift_p95=percentile(errors,.95);
    screen.drift_p95=screen.relative_drift_p95*screen.radius_mean;
    screen.stable=screen.strict=screen.verified=screen.long_verified=true;
    screen.long_trace_periods_completed=1;screens.push_back(screen);
    result.surface_effective_level=level;
    g_target_volume_cache.geometry=std::move(g);
    sgpu_target::measure(g_target_volume_cache.geometry,level,volume,dVds,&g_target_volume_cache.radii);
    g_target_volume_cache.level=level;g_target_volume_cache.volume=volume;g_target_volume_cache.dVds=dVds;
    g_target_volume_cache.coefficients=psi.coeffs;
    g_target_volume_cache.target=config.target_volume;g_target_volume_cache.id=++g_target_volume_sequence;
    result.target_volume_cache_id=g_target_volume_cache.id;
    return true;
}

bool predict_target_level(const sgpu_target::Geometry& query,const SgpuScoreConfig& config,
                          const SgpuScoreResult& center,double& level) {
    const auto& cache=g_target_volume_cache;
    if(cache.id==0||center.target_volume_cache_id!=cache.id||cache.target!=config.target_volume||
       cache.geometry.rays.size()!=query.rays.size()||cache.geometry.a!=query.a)return false;
    double delta=0;
    for(size_t j=0;j<query.rays.size();++j){
        const auto& c=cache.geometry.rays[j];const auto& q=query.rays[j];const double r=cache.radii[j];
        const double psir=sgpu_target::derivative(c.p,query.degree,r/query.a)/query.a;
        // Coordinates follow each axis: include the explicit axis-R volume derivative.
        const double dpsi=sgpu_target::value(q.p,query.degree,r/query.a)-cache.level;
        delta+=.5*r*r*(q.R-c.R)-(c.R+r*std::cos(c.theta))*r*dpsi/psir;
    }
    delta*=TWOPI*TWOPI/query.rays.size();level=cache.level-delta/cache.dVds;
    return std::isfinite(level)&&level>0&&level<=query.maximum_level;
}

void target_quality(const SgpuScoreConfig& config,SgpuScoreResult& result) {
    if(config.target_volume<=0)return;
    result.target_volume_requested=config.target_volume;
    result.target_volume_relative_error=result.surface_volume/config.target_volume-1;
    result.target_quality_error=result.surface_drift_relative_p95;
    result.target_quality_penalty=sgpu_target::penalty(result.target_quality_error,
        config.target_quality_threshold,config.target_quality_temperature,config.target_quality_weight);
}

// GPU contraction of the center's discrete volume adjoint. No neighbor geometry
// search, target-volume iteration, or per-neighbor nesting certificate.
__global__ void target_adjoint_weights_kernel(const double* rays, int nr,
    const int* ma,const int* mb,const int* mm,const int* mk,int nfp,double* weights) {
    const int k=blockIdx.x;
    double sum=0;
    for(int j=threadIdx.x;j<nr;j+=blockDim.x) {
        const double* r=rays+4*j;
        double basis=1;
        for(int i=0;i<ma[k];++i)basis*=r[0];
        for(int i=0;i<mb[k];++i)basis*=r[1];
        const double arg=mm[k]*nfp*r[2];
        sum+=r[3]*basis*(mk[k]?sin(arg):cos(arg));
    }
    __shared__ double values[256];values[threadIdx.x]=sum;__syncthreads();
    for(int step=128;step;step/=2){if(threadIdx.x<step)values[threadIdx.x]+=values[threadIdx.x+step];__syncthreads();}
    if(threadIdx.x==0)weights[k]=values[0];
}

__global__ void target_predict_levels_kernel(const double* coefficients,
    const double* center,const double* weights,const double* axis_delta,
    int nc,double residual,double level,double dVds,double* levels) {
    const int q=blockIdx.x;double sum=0;
    for(int k=threadIdx.x;k<nc;k+=blockDim.x)
        sum+=weights[k]*(coefficients[static_cast<size_t>(q)*nc+k]-center[k]);
    __shared__ double values[256];values[threadIdx.x]=sum;__syncthreads();
    for(int step=128;step;step/=2){if(threadIdx.x<step)values[threadIdx.x]+=values[threadIdx.x+step];__syncthreads();}
    if(threadIdx.x==0)levels[q]=level-(values[0]+axis_delta[q]+residual)/dVds;
}

bool predict_target_levels_gpu(const std::vector<AxisData>& axes,const PsiModes& modes,
    const double* coefficients,int nc,int nfp,const SgpuScoreConfig& config,
    const SgpuScoreResult& center,std::vector<double>& levels) {
    auto& cache=g_target_volume_cache;
    const int nr=config.flux_phi_count*config.flux_theta_count,nq=static_cast<int>(axes.size());
    if(!cache.id || center.target_volume_cache_id!=cache.id || cache.target!=config.target_volume ||
       cache.geometry.a!=config.psi_a || cache.geometry.rays.size()!=static_cast<size_t>(nr) ||
       cache.coefficients.size()!=static_cast<size_t>(nc) || !(cache.dVds>0)) {
        sgpu_internal_set_error("target-volume center adjoint cache mismatch");return false;
    }
    DeviceBuffer<double> dw;
    if(cache.adjoint_weights.empty()) {
        std::vector<double> rays(static_cast<size_t>(nr)*4);
        cache.axis_weights.assign(config.flux_phi_count,0);cache.adjoint_residual=0;
        const double w=TWOPI*TWOPI/nr;
        for(int j=0;j<nr;++j) {
            const auto& ray=cache.geometry.rays[j];const double r=cache.radii[j],u=r/config.psi_a;
            const double psir=sgpu_target::derivative(ray.p,config.psi_poly_degree,u)/config.psi_a;
            const double weight=-w*(ray.R+r*std::cos(ray.theta))*r/psir;
            rays[4*j]=u*std::cos(ray.theta);rays[4*j+1]=u*std::sin(ray.theta);
            rays[4*j+2]=ray.phi;rays[4*j+3]=weight;
            cache.axis_weights[j/config.flux_theta_count]+=.5*w*r*r;
            cache.adjoint_residual+=weight*(sgpu_target::value(ray.p,config.psi_poly_degree,u)-cache.level);
        }
        DeviceBuffer<double> dr;DeviceBuffer<int> da,db,dm,dk;
        if(!copy_to_device(dr,rays)||!copy_to_device(da,modes.a)||!copy_to_device(db,modes.b)||
           !copy_to_device(dm,modes.m)||!copy_to_device(dk,modes.kind)||!dw.allocate(nc))return false;
        target_adjoint_weights_kernel<<<nc,256>>>(dr.data(),nr,da.data(),db.data(),dm.data(),dk.data(),nfp,dw.data());
        std::vector<double> weights(nc);
        if(cudaGetLastError()!=cudaSuccess || cudaMemcpy(weights.data(),dw.data(),nc*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)return false;
        cache.adjoint_weights=std::move(weights);
    } else if(!copy_to_device(dw,cache.adjoint_weights))return false;
    std::vector<double> axis_delta(nq,0);
    for(int q=0;q<nq;++q)for(int ip=0;ip<config.flux_phi_count;++ip) {
        const auto& ray=cache.geometry.rays[ip*config.flux_theta_count];double R,dR;
        periodic_hermite_host(ray.phi,axes[q].R,axes[q].R_phi,nfp,R,dR);
        axis_delta[q]+=cache.axis_weights[ip]*(R-ray.R);
    }
    DeviceBuffer<double> dc(nq*static_cast<size_t>(nc)),dcenter,dd,dl(nq);
    if(!dc.data()||!dl.data()||!copy_to_device(dcenter,cache.coefficients)||!copy_to_device(dd,axis_delta)||
       cudaMemcpy(dc.data(),coefficients,nq*static_cast<size_t>(nc)*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess)return false;
    target_predict_levels_kernel<<<nq,256>>>(dc.data(),dcenter.data(),dw.data(),dd.data(),nc,
        cache.adjoint_residual,cache.level,cache.dVds,dl.data());
    if(cudaGetLastError()!=cudaSuccess || cudaMemcpy(levels.data(),dl.data(),nq*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)return false;
    for(double level:levels)if(!(std::isfinite(level)&&level>0)) {
        sgpu_internal_set_error("target-volume implicit level is invalid; gradient unavailable");return false;
    }
    return true;
}

// Reuse the flux boundary, rather than solving and measuring it again on CPU.
// This is a sampled numerical guard, not a global topology certificate.
__global__ void target_boundary_guard_kernel(const float* polys,const float* radii,
    const float* residuals,const float* axis_R,const float* axis_R_phi,
    int axis_count,int nr,int nt,int nl,int degree,int nfp,float a,float tolerance,
    double* derivatives) {
    const int q=blockIdx.x;double sum=0;bool valid=true;
    for(int j=threadIdx.x;j<nr;j+=blockDim.x) {
        const size_t at=(static_cast<size_t>(q)*nl+nl-1)*nr+j;
        const double r=radii[at],u=r/a;
        const float* p=polys+(static_cast<size_t>(q)*nr+j)*(degree+1);
        double psir=degree*p[degree];for(int k=degree-1;k>=1;--k)psir=psir*u+k*p[k];psir/=a;
        float R,dR;const float phi=static_cast<float>(TWOPI)*(j/nt)/(nfp*(nr/nt));
        periodic_hermite_device(phi,axis_R+static_cast<size_t>(q)*axis_count,
            axis_R_phi+static_cast<size_t>(q)*axis_count,axis_count,static_cast<float>(TWOPI)/nfp,R,dR);
        const double physical_R=R+r*cos(TWOPI*(j%nt)/nt);
        valid=valid&&isfinite(r)&&r>0&&isfinite(psir)&&psir>0&&physical_R>0&&
            isfinite(residuals[at])&&residuals[at]<=tolerance;
        sum+=physical_R*r/psir;
    }
    __shared__ double values[256];values[threadIdx.x]=valid?sum:nan("");__syncthreads();
    for(int step=128;step;step/=2){if(threadIdx.x<step)values[threadIdx.x]+=values[threadIdx.x+step];__syncthreads();}
    if(threadIdx.x==0)derivatives[q]=values[0]*TWOPI*TWOPI/nr;
}


}  // namespace coil_eval::pipeline
