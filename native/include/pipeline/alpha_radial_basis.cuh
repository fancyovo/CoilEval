#pragma once
#include <vector>

namespace coil_eval::pipeline {

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
