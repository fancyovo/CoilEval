// ABI-compatible optional geometry extension. 1024-point periodic quadrature.
#include "coil_field.h"
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
using V=std::array<double,3>;
static V cross3(const V&a,const V&b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
static double dot3(const V&a,const V&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
extern "C" int sgpu_coil_torsion_gradient(const double*x,const double*y,const double*z,int nc,int nk,int samples,double*value,double*gradient){
 if(!x||!y||!z||!value||nc<=0||nk<3||nk%2!=1||samples<32)return 1;
 const double*co[3]={x,y,z};int count=nc*nk;double numerator=0,length=0;
 std::vector<double> gn(3*count,0),gl(3*count,0),basis(3*nk);
 for(int s=0;s<samples;++s){
  double t=2*std::acos(-1.)*s/samples;std::fill(basis.begin(),basis.end(),0.);
  for(int k=1;k<=(nk-1)/2;++k)for(int d=1;d<=3;++d){double a=k*t+d*std::acos(-1.)/2;basis[(d-1)*nk+2*k-1]=std::pow(k,d)*std::sin(a);basis[(d-1)*nk+2*k]=std::pow(k,d)*std::cos(a);}
  for(int j=0;j<nc;++j){
   V a{},b{},c{};for(int h=0;h<3;++h)for(int k=0;k<nk;++k){double v=co[h][j*nk+k];a[h]+=v*basis[k];b[h]+=v*basis[nk+k];c[h]+=v*basis[2*nk+k];}
   V p=cross3(a,b);double speed=std::sqrt(dot3(a,a)),den=dot3(p,p),num=dot3(p,c);
   if(!std::isfinite(speed)||!std::isfinite(den)||!std::isfinite(num)||speed<=0||den<=1e-30)return 2;
   double tau=std::abs(num)/den,sign=(num>0)-(num<0);numerator+=tau*speed;length+=speed;
   if(gradient)for(int h=0;h<3;++h)for(int k=1;k<nk;++k){
    V da{},db{},dc{};da[h]=basis[k];db[h]=basis[nk+k];dc[h]=basis[2*nk+k];V p1=cross3(da,b),p2=cross3(a,db),dp{};for(int l=0;l<3;++l)dp[l]=p1[l]+p2[l];
    double dn=dot3(dp,c)+dot3(p,dc),dd=2*dot3(p,dp),ds=dot3(a,da)/speed;
    int index=h*count+j*nk+k;gn[index]+=sign*(dn/den-num*dd/(den*den))*speed+tau*ds;gl[index]+=ds;
   }
  }
 }
 *value=numerator/length;if(!std::isfinite(*value))return 2;
 if(gradient)for(int i=0;i<3*count;++i)gradient[i]=(gn[i]-*value*gl[i])/length;
 return 0;
}

// Fixed-grid weighted nearest-rank quantile; derivative follows selected point.
extern "C" int sgpu_coil_torsion_p95_gradient(const double*x,const double*y,const double*z,int nc,int nk,double*value,double*gradient){
 if(!x||!y||!z||!value||nc<=0||nk<3||nk%2!=1)return 1;
 struct Point {double tau,speed;int coil,sample;};std::vector<Point> points;double length=0;const double*co[3]={x,y,z};const int samples=256;
 auto evaluate=[&](int j,int s,V&a,V&b,V&c,std::vector<double>&basis){
  basis.assign(3*nk,0.);double t=2*std::acos(-1.)*s/samples;
  for(int k=1;k<=(nk-1)/2;++k)for(int d=1;d<=3;++d){double v=k*t+d*std::acos(-1.)/2;basis[(d-1)*nk+2*k-1]=std::pow(k,d)*std::sin(v);basis[(d-1)*nk+2*k]=std::pow(k,d)*std::cos(v);}
  for(int h=0;h<3;++h)for(int k=0;k<nk;++k){double v=co[h][j*nk+k];a[h]+=v*basis[k];b[h]+=v*basis[nk+k];c[h]+=v*basis[2*nk+k];}
 };
 for(int j=0;j<nc;++j)for(int s=0;s<samples;++s){V a{},b{},c{};std::vector<double> basis;evaluate(j,s,a,b,c,basis);V p=cross3(a,b);double speed=std::sqrt(dot3(a,a)),den=dot3(p,p),tau=std::abs(dot3(p,c))/std::max(den,1e-30);if(!std::isfinite(tau)||!std::isfinite(speed)||speed<=0)return 2;points.push_back({tau,speed,j,s});length+=speed;}
 std::stable_sort(points.begin(),points.end(),[](const Point&a,const Point&b){return a.tau<b.tau;});double cumulative=0;Point selected=points.back();for(const auto&p:points){cumulative+=p.speed;if(cumulative>=.95*length){selected=p;break;}}
 *value=selected.tau;
 if(gradient){std::fill(gradient,gradient+3*nc*nk,0.);V a{},b{},c{};std::vector<double> basis;evaluate(selected.coil,selected.sample,a,b,c,basis);V p=cross3(a,b);double den=dot3(p,p),num=dot3(p,c),sign=(num>0)-(num<0);if(den<=1e-30)return 2;
 for(int h=0;h<3;++h)for(int k=1;k<nk;++k){V da{},db{},dc{};da[h]=basis[k];db[h]=basis[nk+k];dc[h]=basis[2*nk+k];V u=cross3(da,b),v=cross3(a,db),dp{};for(int l=0;l<3;++l)dp[l]=u[l]+v[l];double dn=dot3(dp,c)+dot3(p,dc),dd=2*dot3(p,dp);gradient[h*nc*nk+selected.coil*nk+k]=sign*(dn/den-num*dd/(den*den));}}
 return 0;
}
