#include "target_volume_geometry.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>

namespace original {
using sgpu_target::Polynomial;
bool positive(std::vector<double> b,int depth=0) {
    auto mm=std::minmax_element(b.begin(),b.end());
    if(*mm.first>1e-12)return true;
    if(*mm.second<=1e-12||depth==10)return false;
    int n=int(b.size())-1;std::vector<double>l(n+1),r(n+1);
    l[0]=b[0];r[n]=b[n];
    for(int k=1;k<=n;++k){for(int j=0;j<=n-k;++j)b[j]=.5*(b[j]+b[j+1]);l[k]=b[0];r[n-k]=b[n-k];}
    return positive(l,depth+1)&&positive(r,depth+1);
}
bool nested(const Polynomial& p,int degree,double maximum) {
    if(!(p[2]>1e-12)||!(maximum>0))return false;
    int n=degree-2;std::vector<double>b(n+1);
    for(int k=0;k<=n;++k)for(int j=0;j<=k;++j)
        b[k]+=(j+2)*p[j+2]*std::pow(maximum,j)*
            sgpu_target::choose(k,j)/sgpu_target::choose(n,j);
    return positive(b);
}
}

template<class Predicate> double boundary(const sgpu_target::Polynomial& p,int d,Predicate nested) {
    double lo=0,hi=1;
    if(nested(p,d,hi))return hi;
    for(int j=0;j<32;++j){double m=.5*(lo+hi);if(nested(p,d,m))lo=m;else hi=m;}
    return lo;
}

int main() {
    std::mt19937_64 rng(20261028);
    std::uniform_real_distribution<double> random(-1.,1.);
    std::vector<sgpu_target::Polynomial> cases;
    std::vector<int> degrees;
    for(int degree:{2,4,8,12,24})for(int sample=0;sample<200;++sample) {
        sgpu_target::Polynomial p{};
        for(int j=2;j<=degree;++j)p[j]=random(rng)/(j*j);
        p[2]=sample%9==0?1e-12*(1+1e-8*random(rng)):.05+std::abs(random(rng));
        for(double maximum:{0.,.001,.2,.9,1.,1.5,2.})
            assert(original::nested(p,degree,maximum)==sgpu_target::nested(p,degree,maximum));
        // Test the full selected boundary, not merely a Boolean at easy points.
        assert(boundary(p,degree,original::nested)==boundary(p,degree,sgpu_target::nested));
        cases.push_back(p);degrees.push_back(degree);
    }
    double sums[2]={},seconds[2]={};
    for(int arm=0;arm<2;++arm){
        auto start=std::chrono::steady_clock::now();
        for(int repeat=0;repeat<3;++repeat)for(std::size_t k=0;k<cases.size();++k)
            sums[arm]+=boundary(cases[k],degrees[k],arm?sgpu_target::nested:original::nested);
        seconds[arm]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    }
    assert(sums[0]==sums[1]);
    std::cout<<"1000 polynomial boundary branches identical; original_s="<<seconds[0]
             <<" optimized_s="<<seconds[1]<<" speedup="<<seconds[0]/seconds[1]<<"\n";
}
