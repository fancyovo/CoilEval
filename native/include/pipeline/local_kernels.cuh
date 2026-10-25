#pragma once

#include "pipeline/types.cuh"

namespace coil_eval::pipeline {

__global__ void local_batch_build_ray_coefficients_kernel(
    const float* __restrict__ coefficients,
    const int* __restrict__ mode_a,
    const int* __restrict__ mode_b,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_kind,
    int coefficient_count,
    int query_count,
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

__global__ void local_batch_solve_boundary_radii_kernel(
    const float* __restrict__ ray_coefficients,
    int ray_count,
    int query_count,
    int degree,
    float radius_scale,
    const float* __restrict__ levels,
    int level_count,
    int iterations,
    float tolerance,
    float max_radius_scale,
    float* __restrict__ radii,
    float* __restrict__ residuals
);

__global__ void local_batch_generate_flux_points_kernel(
    const float* __restrict__ boundary_radii,
    const float* __restrict__ gauss_nodes,
    int query_count,
    int n_phi,
    int n_levels,
    int n_theta,
    int n_radial,
    int nfp,
    const float* __restrict__ axis_R,
    const float* __restrict__ axis_Z,
    const float* __restrict__ axis_R_phi,
    const float* __restrict__ axis_Z_phi,
    int axis_count,
    float* __restrict__ xyz
);

__global__ void local_batch_reduce_flux_sections_kernel(
    const float* __restrict__ B,
    const float* __restrict__ boundary_radii,
    const float* __restrict__ gauss_nodes,
    const float* __restrict__ gauss_weights,
    int query_count,
    int n_phi,
    int n_levels,
    int n_theta,
    int n_radial,
    int nfp,
    double* __restrict__ sections
);

__global__ void local_batch_generate_volume_points_kernel(
    const float* __restrict__ boundary_radii,
    const int* __restrict__ source_indices,
    int query_count,
    int n_phi,
    int n_theta,
    int n_radial,
    int output_count,
    int nfp,
    float rho_min,
    float edge_level,
    const float* __restrict__ flux_coefficients,
    int flux_degree,
    const float* __restrict__ edge_flux,
    const float* __restrict__ psi_coefficients,
    const int* __restrict__ mode_a,
    const int* __restrict__ mode_b,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_kind,
    int psi_coefficient_count,
    int psi_degree,
    int psi_m_tor,
    float psi_a,
    const float* __restrict__ axis_R,
    const float* __restrict__ axis_Z,
    const float* __restrict__ axis_R_phi,
    const float* __restrict__ axis_Z_phi,
    int axis_count,
    float* __restrict__ xyz,
    float* __restrict__ grad_s,
    float* __restrict__ grad_theta,
    float* __restrict__ grad_phi,
    float* __restrict__ rho_out,
    float* __restrict__ theta_out,
    float* __restrict__ phi_out,
    float* __restrict__ volume_weight,
    float* __restrict__ flux_derivative_out
);

__global__ void local_batch_alpha_histogram_kernel(
    const float* __restrict__ rho,
    int query_count,
    int point_count,
    int fit_count,
    int radial_bins,
    int* __restrict__ bin_counts
);

__global__ void local_batch_alpha_prepare_kernel(
    const float* __restrict__ B,
    const float* __restrict__ grad_s,
    const float* __restrict__ grad_theta,
    const float* __restrict__ grad_phi,
    const float* __restrict__ source_rho,
    const float* __restrict__ source_theta,
    const float* __restrict__ source_phi,
    int query_count,
    int point_count,
    int fit_count,
    int radial_bins,
    const int* __restrict__ bin_counts,
    float* __restrict__ b_theta,
    float* __restrict__ b_phi,
    float* __restrict__ weights,
    float* __restrict__ rho,
    float* __restrict__ theta,
    float* __restrict__ phi,
    double* __restrict__ diagnostics
);

__global__ void local_batch_normalize_alpha_kernel(
    float* __restrict__ weights,
    const float* __restrict__ b_theta,
    const double* __restrict__ diagnostics,
    int query_count,
    int fit_count,
    float* __restrict__ rhs
);

__global__ void local_batch_assemble_alpha_tile_kernel(
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_n,
    const int* __restrict__ mode_kind,
    const float* __restrict__ radial_coefficients,
    int mode_count,
    int radial_order,
    int iota_degree,
    int nfp,
    const float* __restrict__ rho,
    const float* __restrict__ theta,
    const float* __restrict__ phi,
    const float* __restrict__ b_theta,
    const float* __restrict__ b_phi,
    const float* __restrict__ weights,
    int fit_count,
    int query_begin,
    int tile_count,
    const float* __restrict__ center_scale,
    float* __restrict__ matrices
);

__global__ void local_batch_initialize_alpha_coefficients_kernel(
    const float* __restrict__ center,
    int column_count,
    int tile_count,
    float* __restrict__ coefficients
);

__global__ void local_batch_initialize_alpha_residual_kernel(
    const float* __restrict__ rhs,
    const float* __restrict__ prediction,
    int fit_count,
    int query_begin,
    int tile_count,
    float* __restrict__ residual
);

__global__ void local_batch_alpha_normal_ridge_kernel(
    float* __restrict__ normal,
    const float* __restrict__ coefficients,
    float ridge,
    size_t count
);

__global__ void local_batch_alpha_delta_kernel(
    const float* __restrict__ image,
    int fit_count,
    const float* __restrict__ direction,
    int column_count,
    int tile_count,
    float ridge,
    float* __restrict__ delta
);

__global__ void local_batch_column_norm_kernel(
    const float* __restrict__ values,
    int rows,
    int columns,
    float* __restrict__ norms
);

__global__ void local_batch_direction_update_kernel(
    float* __restrict__ direction,
    const float* __restrict__ normal,
    const float* __restrict__ gamma,
    const float* __restrict__ gamma_new,
    int rows,
    int columns
);

__global__ void local_batch_alpha_update_kernel(
    float* __restrict__ coefficients,
    float* __restrict__ residual,
    const float* __restrict__ direction,
    const float* __restrict__ image,
    const float* __restrict__ gamma,
    const float* __restrict__ delta,
    int column_count,
    int fit_count,
    int tile_count
);

__global__ void local_batch_extract_iota_kernel(
    const float* __restrict__ scaled_coefficients,
    const float* __restrict__ scale,
    int column_count,
    int mode_count,
    int iota_count,
    int query_begin,
    int tile_count,
    float* __restrict__ iota
);

__global__ void local_batch_qs_metric_kernel(
    const float* __restrict__ B,
    const float* __restrict__ grad_B,
    const float* __restrict__ grad_s,
    const float* __restrict__ flux_derivative,
    const float* __restrict__ rho,
    const float* __restrict__ volume_weight,
    int query_count,
    int point_count,
    const float* __restrict__ iota_coefficients,
    int iota_degree,
    int helicity_M,
    int helicity_N,
    int nfp,
    const double* __restrict__ G,
    float edge_rho_threshold,
    double* __restrict__ sums,
    unsigned int* __restrict__ psi_histogram,
    int histogram_bins,
    double* __restrict__ psi_sums
);


}  // namespace coil_eval::pipeline
