#pragma once
#include <vector>

namespace coil_eval::pipeline {

// Evaluate the bounded Legendre basis without cancellation between large
// monomial coefficients near rho=1. Keep the stored coefficient format intact.
#ifdef __CUDACC__
__host__ __device__
#endif
inline float flux_legendre_radial_value(const float* coefficients,int degree,float rho) {
    int k=degree/2;
    while (k>0 && coefficients[2*k]==0.0f) --k;
    if (k==0) return 1.0f;
    const float x=2.0f*rho*rho-1.0f;
    float previous=1.0f,current=x;
    for (int j=2;j<=k;++j) {
        const float next=((2*j-1)*x*current-(j-1)*previous)/j;
        previous=current;
        current=next;
    }
    return current;
}

// Coefficients of P_k(2*rho^2-1), in ascending powers of rho.
// k=(l-m)/2 retains the existing angular/radial column count and ordering.
inline std::vector<float> flux_legendre_radial_coefficients(int l,int m,int width) {
    std::vector<float> result(width,0.0f);
    const int k=(l-m)/2;
    double choose_k_j=1.0,choose_kplusj_j=1.0;
    for (int j=0;j<=k;++j) {
        if (j>0) {
            choose_k_j*=static_cast<double>(k-j+1)/j;
            choose_kplusj_j*=static_cast<double>(k+j)/j;
        }
        result[2*j]=static_cast<float>(((k-j)&1 ? -1.0 : 1.0)*choose_k_j*choose_kplusj_j);
    }
    return result;
}

} // namespace coil_eval::pipeline
