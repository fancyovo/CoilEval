#include "target_volume_geometry.h"
#include <cassert>
#include <iostream>

// Independent, uncollapsed expression used before the section optimization.
sgpu_target::Polynomial direct(const std::vector<double>& coefficients,
    const std::vector<int>& a,const std::vector<int>& b,const std::vector<int>& m,
    const std::vector<int>& kind,int degree,double phase,double theta) {
 sgpu_target::Polynomial cp{},sp{},p{};cp[0]=sp[0]=1;
 const double c=std::cos(theta),s=std::sin(theta);p[2]=c*c;
 for(int j=1;j<=degree;++j){cp[j]=cp[j-1]*c;sp[j]=sp[j-1]*s;}
 for(std::size_t k=0;k<coefficients.size();++k)
  p[a[k]+b[k]]+=coefficients[k]*cp[a[k]]*sp[b[k]]*
      (kind[k]?std::sin(m[k]*phase):std::cos(m[k]*phase));
 return p;
}

void section_equivalence() {
 // Include zero toroidal order and the validated maximum degree/order.
 for(int degree:{2,8,12,24})for(int toroidal:{0,12,32}) {
  std::vector<double> coefficients;
  std::vector<int>a,b,m,kind;
  for(int d=2;d<=degree;++d)for(int aa=0;aa<=d;++aa)
   for(int mm=0;mm<=toroidal;++mm)for(int kk=0;kk<(mm?2:1);++kk) {
    a.push_back(aa);b.push_back(d-aa);m.push_back(mm);kind.push_back(kk);
    coefficients.push_back(.03*std::sin(1.731*coefficients.size())/(d*d));
   }
  for(double phase:{0.,.137,1.57,3.1,6.0}) {
   const auto section=sgpu_target::collapse_section(coefficients.data(),a.data(),b.data(),
       m.data(),kind.data(),coefficients.size(),phase);
   for(double theta:{0.,.173,1.5707963267948966,2.9,4.7,6.}) {
    const auto expected=direct(coefficients,a,b,m,kind,degree,phase,theta);
    const auto actual=sgpu_target::radial_polynomial(section,degree,std::cos(theta),std::sin(theta));
    for(int d=0;d<=degree;++d)
     assert(std::abs(expected[d]-actual[d])<2e-13*(1.+std::abs(expected[d])));
    for(double u:{0.,.1,.9,1.}) {
     assert(std::abs(sgpu_target::value(expected,degree,u)-
                     sgpu_target::value(actual,degree,u))<2e-13);
     assert(std::abs(sgpu_target::derivative(expected,degree,u)-
                     sgpu_target::derivative(actual,degree,u))<2e-12);
    }
   }
  }
 }
}

double limit(int count,int mode=8,double phase=0.) {
 double level=1e100;
 for(int j=0;j<count;++j) {
  const double coefficients[]={-1.,std::cos(phase),-std::sin(phase)};
  const int a[]={4,4,4},b[]={0,0,0},m[]={0,mode,mode},kind[]={0,0,1};
  const auto section=sgpu_target::collapse_section(coefficients,a,b,m,kind,3,
      2.*3.141592653589793*j/count);
  const auto p=sgpu_target::radial_polynomial(section,4,1.,0.);
  double lo=0.,hi=1.;
  if(sgpu_target::nested(p,4,hi))lo=hi;
  else for(int k=0;k<32;++k){double m=(lo+hi)/2.;if(sgpu_target::nested(p,4,m))lo=m;else hi=m;}
  level=std::min(level,sgpu_target::value(p,4,lo*(1.-1e-6)));
 }return level;
}
int main(){section_equivalence();assert(limit(8)>.99);assert(limit(48)<.13);
 // The order-12 polynomial is resolved on 48 sections, yet its capacity
 // minimum lies halfway between them. A .135 level is falsely admitted.
 const int dense=sgpu_target::capacity_guard_phi_count(48,12);
 assert(dense==96);
 assert(limit(48,12,3.141592653589793/4.)>.135);
 assert(limit(dense,12,3.141592653589793/4.)<.135);
 std::cout<<"section polynomial equivalence and toroidal capacity aliasing passed\n";}
