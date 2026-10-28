#include "target_volume_geometry.h"
#include <cmath>
// Polynomial coefficients use normalized radius u=r/a, exactly as native.
extern "C" int ce_cpu_target(const double* polynomials,const double* axis_R,
    const double* axis_Z,int phi_count,int flux_phi_count,int theta_count,
    int degree,double a,double radius_limit,double target,double tolerance,
    double* output,double* radii) {
    sgpu_target::Geometry geometry;geometry.degree=degree;geometry.a=a;
    geometry.maximum_level=std::numeric_limits<double>::infinity();
    for(int ip=0;ip<phi_count;++ip)for(int it=0;it<theta_count;++it){
        sgpu_target::Ray ray;ray.R=axis_R[ip];ray.Z=axis_Z[ip];
        ray.theta=2*3.14159265358979323846*it/theta_count;
        const double* p=polynomials+(static_cast<std::size_t>(ip)*theta_count+it)*(degree+1);
        std::copy(p,p+degree+1,ray.p.begin());
        const double c=std::cos(ray.theta);double lo=0,hi=radius_limit;
        if(c<0)hi=std::min(hi,.999*ray.R/(-c*a));
        if(!(ray.p[2]>1e-12)||!std::isfinite(ray.p[2]))return 2;
        if(sgpu_target::nested(ray.p,degree,hi))lo=hi;
        else for(int j=0;j<32;++j){double mid=.5*(lo+hi);if(sgpu_target::nested(ray.p,degree,mid))lo=mid;else hi=mid;}
        ray.umax=lo*(1-1e-6);const double bound=sgpu_target::value(ray.p,degree,ray.umax);
        if(!(bound>0&&std::isfinite(bound)))return 2;
        geometry.maximum_level=std::min(geometry.maximum_level,bound);
        if(ip<flux_phi_count)geometry.rays.push_back(ray);
    }
    double maxv,dv;if(!sgpu_target::measure(geometry,geometry.maximum_level,maxv,dv))return 2;
    output[0]=maxv;output[1]=geometry.maximum_level;
    double level,volume;int iterations;
    if(!sgpu_target::solve(geometry,target,tolerance,level,volume,dv,iterations))return 2;
    output[2]=level;output[3]=volume;output[4]=dv;output[5]=iterations;
    std::vector<double> rr;sgpu_target::measure(geometry,level,volume,dv,&rr);
    std::copy(rr.begin(),rr.end(),radii);return 0;
}
