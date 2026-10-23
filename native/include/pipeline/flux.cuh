#pragma once
#include "pipeline/axis.cuh"
#include "pipeline/psi.cuh"

#include "pipeline/types.cuh"

namespace coil_eval::pipeline {

struct FluxCalibrationNative {
    double edge_level = 0.0;
    double edge_flux = 0.0;
    double fit_relative_rms = 0.0;
    double section_relative_std_edge = 0.0;
    double boundary_residual_max = 0.0;
    double derivative_min = 0.0;
    double derivative_max = 0.0;
    double volume = 0.0;
    double major_radius = 0.0;
    double effective_minor_radius = 0.0;
    std::vector<float> coefficients;
};

__global__ void build_ray_coefficients_kernel(
    const float* __restrict__ coeffs,
    const int* __restrict__ mode_a,
    const int* __restrict__ mode_b,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_kind,
    int n_coeff,
    int nfp,
    int degree,
    int m_tor,
    int n_phi,
    int n_theta,
    float phi_offset,
    float theta_offset,
    float theta_shear,
    float* __restrict__ ray_coefficients
);

__global__ void solve_boundary_radii_kernel(
    const float* __restrict__ ray_coefficients,
    int ray_count,
    int degree,
    float a_scale,
    const float* __restrict__ levels,
    int level_count,
    int iterations,
    float tolerance,
    float max_radius_scale,
    float* __restrict__ radii,
    float* __restrict__ residuals
);

void gauss_legendre(int count, std::vector<float>& nodes, std::vector<float>& weights);

bool solve_small_least_squares(
    const std::vector<double>& matrix,
    const std::vector<double>& rhs,
    int rows,
    int columns,
    std::vector<double>& solution
);

bool calibrate_flux_native(
    void* field,
    const DevicePsiData& device_psi,
    const AxisData& axis,
    const PsiData& psi,
    int nfp,
    double edge_level,
    const SgpuScoreConfig& config,
    FluxCalibrationNative& calibration,
    SgpuScoreResult& result
);


}  // namespace coil_eval::pipeline
