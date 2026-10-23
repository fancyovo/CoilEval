#pragma once

#include <cuda_runtime.h>

namespace coil_eval::field {

// Implemented once in psi_design.cu; launched by formal QR and batch PCGLS.
__global__ void build_psi_common_basis_kernel(
    const float* __restrict__ normalized_R,
    const float* __restrict__ normalized_Z,
    const float* __restrict__ phi,
    int point_count,
    const int* __restrict__ mode_a,
    const int* __restrict__ mode_b,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_kind,
    int coefficient_count,
    int nfp,
    float radius_scale,
    float* __restrict__ derivative_R,
    float* __restrict__ derivative_Z,
    float* __restrict__ derivative_phi
);

__global__ void build_psi_batch_features_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int segments_per_query,
    const float* __restrict__ normalized_R,
    const float* __restrict__ normalized_Z,
    const float* __restrict__ phi,
    int point_count,
    const float* __restrict__ axis_R,
    const float* __restrict__ axis_Z,
    const float* __restrict__ axis_R_phi,
    const float* __restrict__ axis_Z_phi,
    int axis_count,
    int nfp,
    float radius_scale,
    float* __restrict__ feature_R,
    float* __restrict__ feature_Z,
    float* __restrict__ feature_phi,
    float* __restrict__ rhs
);

__global__ void psi_fill_matrix_kernel_f64(
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ R,
    const double* __restrict__ Z,
    const double* __restrict__ phi,
    const double* __restrict__ axis_R,
    const double* __restrict__ axis_Z,
    const double* __restrict__ axis_R_phi,
    const double* __restrict__ axis_Z_phi,
    int n_axis,
    const int* __restrict__ mode_a,
    const int* __restrict__ mode_b,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_kind,
    int n_cols,
    int nfp,
    double a,
    int poly_degree,
    int m_tor,
    double* __restrict__ mat,
    double* __restrict__ rhs
);

__global__ void psi_fill_matrix_kernel_f32(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const float* __restrict__ R,
    const float* __restrict__ Z,
    const float* __restrict__ phi,
    const float* __restrict__ axis_R,
    const float* __restrict__ axis_Z,
    const float* __restrict__ axis_R_phi,
    const float* __restrict__ axis_Z_phi,
    int n_axis,
    const int* __restrict__ mode_a,
    const int* __restrict__ mode_b,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_kind,
    int n_cols,
    int nfp,
    float a,
    int poly_degree,
    int m_tor,
    float* __restrict__ mat,
    float* __restrict__ rhs
);

} // namespace coil_eval::field
