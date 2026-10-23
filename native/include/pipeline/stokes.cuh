#pragma once
#include "pipeline/types.cuh"
namespace coil_eval::pipeline {
extern thread_local bool g_stokes_flux;
__global__ void local_batch_stokes_points_kernel(
    const float* __restrict__ radii, int query_count, int n_phi,
    int n_levels, int n_theta, int nfp,
    const float* __restrict__ axis_R, const float* __restrict__ axis_Z,
    const float* __restrict__ axis_R_phi, const float* __restrict__ axis_Z_phi,
    int axis_count, const float* __restrict__ coefficients,
    const int* __restrict__ mode_a, const int* __restrict__ mode_b,
    const int* __restrict__ mode_m, const int* __restrict__ mode_kind,
    int coefficient_count, int psi_degree, int psi_m_tor, float psi_a,
    float* __restrict__ xyz, float* __restrict__ tangent,
    unsigned int* __restrict__ invalid_query
);
__global__ void local_batch_stokes_reduce_kernel(
    const float* __restrict__ potential, const float* __restrict__ tangent,
    int n_theta, double* __restrict__ sections
);
}
