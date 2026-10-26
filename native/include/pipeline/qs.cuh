#pragma once
#include "pipeline/alpha.cuh"
#include "pipeline/volume.cuh"

#include "pipeline/types.cuh"

namespace coil_eval::pipeline {

__global__ void compute_qs_point_adjoint_kernel(
    const float* __restrict__ B,
    const float* __restrict__ grad_B,
    const float* __restrict__ grad_s,
    const float* __restrict__ flux_derivative,
    const float* __restrict__ rho,
    const float* __restrict__ volume_weight,
    int point_count,
    const float* __restrict__ iota_coefficients,
    int iota_degree,
    int helicity_M,
    int helicity_N,
    int nfp,
    double G,
    float edge_rho_threshold,
    double weight_sum,
    double edge_weight_sum,
    double target_rms,
    double target_edge_rms,
    double qa_rms,
    double qp_rms,
    double d_target_error,
    double d_target_edge_error,
    double d_qa_error,
    double d_qp_error,
    float* __restrict__ adj_B,
    float* __restrict__ adj_grad_B,
    double* __restrict__ adj_G
);

__global__ void compute_qs_iota_adjoint_kernel(
    const float* __restrict__ B,
    const float* __restrict__ grad_B,
    const float* __restrict__ grad_s,
    const float* __restrict__ flux_derivative,
    const float* __restrict__ rho,
    const float* __restrict__ volume_weight,
    int point_count,
    const float* __restrict__ iota_coefficients,
    int iota_degree,
    int helicity_M,
    int helicity_N,
    double G,
    float edge_rho_threshold,
    double weight_sum,
    double edge_weight_sum,
    double target_rms,
    double target_edge_rms,
    double qa_rms,
    double d_target_error,
    double d_target_edge_error,
    double d_qa_error,
    double* __restrict__ adj_iota
);

__global__ void alpha_residual_adjoint_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ rhs,
    int count,
    float residual_norm,
    float rhs_norm,
    float adj_relative,
    float* __restrict__ adj_residual,
    float* __restrict__ adj_rhs
);

__global__ void add_alpha_solution_adjoint_kernel(
    const float* __restrict__ adj_solution,
    const float* __restrict__ scales,
    int count,
    float* __restrict__ adj_scaled_solution
);

__global__ void alpha_column_dot_kernel(
    const float* __restrict__ scaled_matrix,
    int row_count,
    const float* __restrict__ adj_residual,
    const float* __restrict__ residual,
    const float* __restrict__ multiplier,
    const float* __restrict__ matrix_multiplier,
    const float* __restrict__ scaled_solution,
    float* __restrict__ dot_adjoint_matrix
);

__global__ void alpha_row_adjoint_kernel(
    const float* __restrict__ scaled_matrix,
    int row_count,
    int column_count,
    const float* __restrict__ adj_residual,
    const float* __restrict__ residual,
    const float* __restrict__ adj_rhs,
    const float* __restrict__ multiplier,
    const float* __restrict__ matrix_multiplier,
    const float* __restrict__ scaled_solution,
    const float* __restrict__ adj_solution,
    const float* __restrict__ scales,
    const float* __restrict__ dot_adjoint_matrix,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_n,
    const int* __restrict__ mode_kind,
    const float* __restrict__ radial_coefficients,
    int mode_count,
    int radial_order,
    int radial_basis,
    int iota_degree,
    int nfp,
    const float* __restrict__ rho,
    const float* __restrict__ theta,
    const float* __restrict__ phi,
    const float* __restrict__ b_theta,
    const float* __restrict__ b_phi,
    const float* __restrict__ weights,
    float* __restrict__ adj_weight,
    float* __restrict__ adj_b_theta,
    float* __restrict__ adj_b_phi
);

__global__ void alpha_B_adjoint_kernel(
    const float* __restrict__ B,
    const float* __restrict__ grad_s,
    const float* __restrict__ grad_theta,
    const float* __restrict__ grad_phi,
    int point_count,
    int fit_count,
    const float* __restrict__ weights,
    const float* __restrict__ adj_weight,
    const float* __restrict__ adj_b_theta,
    const float* __restrict__ adj_b_phi,
    float weight_scale,
    float adj_weight_dot_weight,
    float adj_normal_relative,
    float normal_relative,
    float normal_numerator,
    float field_denominator,
    float* __restrict__ adj_B
);

bool compute_qs_metric_native(
    const DeviceVolumePoints& points,
    const float* d_B,
    const float* d_grad_B,
    const AlphaFitNative& alpha,
    int nfp,
    double G,
    const SgpuScoreConfig& config,
    SgpuScoreResult& result,
    std::array<double, 9>* raw_sums
);


}  // namespace coil_eval::pipeline
