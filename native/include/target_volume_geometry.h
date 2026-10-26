// Host-side target geometry. Shared with analytic tests; no CUDA dependency.
#pragma once
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
namespace sgpu_target {
using Polynomial = std::array<double,25>;
using SectionCoefficients = std::array<Polynomial,25>;
// Collapse the toroidal series once per section. The subsequent ray expansion
// then visits each Cartesian monomial once, independently of toroidal order.
inline SectionCoefficients collapse_section(const double* coefficients,
    const int* mode_a,const int* mode_b,const int* mode_m,const int* mode_kind,
    std::size_t count,double nfp_phi) {
    SectionCoefficients section{};
    for(std::size_t k=0;k<count;++k) {
        const double arg=mode_m[k]*nfp_phi;
        const double trig=mode_kind[k]?std::sin(arg):std::cos(arg);
        section[mode_a[k]][mode_b[k]]+=coefficients[k]*trig;
    }
    return section;
}
inline Polynomial radial_polynomial(const SectionCoefficients& section,
                                    int degree,double c,double s) {
    Polynomial cp{},sp{},p{};cp[0]=sp[0]=1;
    for(int j=1;j<=degree;++j){cp[j]=cp[j-1]*c;sp[j]=sp[j-1]*s;}
    p[2]=c*c;
    for(int a=0;a<=degree;++a)for(int b=0;b<=degree-a;++b)
        p[a+b]+=section[a][b]*cp[a]*sp[b];
    return p;
}
inline double value(const Polynomial& p,int degree,double u) {
    double v=p[degree]; for(int j=degree-1;j>=0;--j) v=v*u+p[j]; return v;
}
inline double derivative(const Polynomial& p,int degree,double u) {
    double v=degree*p[degree]; for(int j=degree-1;j>=1;--j) v=v*u+j*p[j]; return v;
}
inline double choose(int n,int k) {
    double v=1;for(int j=1;j<=k;++j)v*=double(n-j+1)/j;return v;
}
inline bool positive_bernstein(std::vector<double> b,int depth=0) {
    const auto mm=std::minmax_element(b.begin(),b.end());
    if(*mm.first>1e-12) return true;
    if(*mm.second<=1e-12 || depth==10) return false;
    const int n=int(b.size())-1; std::vector<double> l(n+1),r(n+1);
    l[0]=b[0];r[n]=b[n];
    for(int k=1;k<=n;++k){for(int j=0;j<=n-k;++j)b[j]=.5*(b[j]+b[j+1]);l[k]=b[0];r[n-k]=b[n-k];}
    return positive_bernstein(l,depth+1)&&positive_bernstein(r,depth+1);
}
// Certify (d psi/du)/u > 0 over the whole radial interval via Bernstein bounds.
inline bool nested(const Polynomial& p,int degree,double maximum) {
    if(!(p[2]>1e-12) || !(maximum>0))return false;
    const int n=degree-2; std::vector<double>b(n+1);
    for(int k=0;k<=n;++k)for(int j=0;j<=k;++j)
        b[k]+=(j+2)*p[j+2]*std::pow(maximum,j)*choose(k,j)/choose(n,j);
    return positive_bernstein(b);
}
struct Ray { Polynomial p{}; double R=0,Z=0,theta=0,phi=0,umax=0; };
struct Geometry { std::vector<Ray> rays; int degree=0; double a=0,maximum_level=0; };
inline double radius(const Ray& ray,const Geometry& g,double level) {
    double lo=0,hi=ray.umax;
    for(int i=0;i<48;++i){double mid=.5*(lo+hi);if(value(ray.p,g.degree,mid)<level)lo=mid;else hi=mid;}
    return g.a*.5*(lo+hi);
}
inline bool measure(const Geometry& g,double level,double& volume,double& dVds,std::vector<double>* radii=nullptr) {
    volume=0;dVds=0;
    if(!(level>0&&level<=g.maximum_level)||g.rays.empty())return false;
    if(radii)radii->clear();
    for(const auto& ray:g.rays){
        const double r=radius(ray,g,level),R=ray.R+r*std::cos(ray.theta);
        const double psir=derivative(ray.p,g.degree,r/g.a)/g.a;
        if(!(psir>0&&R>0))return false;
        volume+=.5*ray.R*r*r+std::cos(ray.theta)*r*r*r/3;
        dVds+=R*r/psir;if(radii)radii->push_back(r);
    }
    const double w=4*3.14159265358979323846*3.14159265358979323846/g.rays.size();
    volume*=w;dVds*=w;return std::isfinite(volume)&&std::isfinite(dVds)&&dVds>0;
}
inline bool solve(const Geometry& g,double target,double tolerance,double& level,double& volume,double& dVds,int& iterations) {
    double maxv,deriv;if(!(target>0)||!measure(g,g.maximum_level,maxv,deriv)||target>maxv)return false;
    double lo=0,hi=g.maximum_level; level=hi*target/maxv;
    for(iterations=1;iterations<=60;++iterations){
        if(!measure(g,level,volume,dVds))return false;
        if(std::abs(volume/target-1)<=tolerance)return true;
        if(volume<target)lo=level;else hi=level;
        const double trial=level+(target-volume)/dVds;
        level=(trial>lo&&trial<hi)?trial:.5*(lo+hi);
    }return false;
}
inline double penalty(double error,double threshold,double temperature,double weight) {
    const double x=(error-threshold)/temperature;
    const double smooth=temperature*(std::max(x,0.0)+std::log1p(std::exp(-std::abs(x))));
    return weight*smooth*smooth;
}
}
