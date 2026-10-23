#include "pipeline/device_helpers.cuh"
#include "pipeline/stokes.cuh"

namespace coil_eval::pipeline {
thread_local bool g_stokes_flux = false;

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
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int points_per_query = n_phi * n_levels * n_theta;
    if (flat >= static_cast<size_t>(query_count) * points_per_query) return;
    const int query = static_cast<int>(flat / points_per_query);
    int remainder = static_cast<int>(flat - static_cast<size_t>(query) * points_per_query);
    const int theta_index = remainder % n_theta;
    remainder /= n_theta;
    const int level_index = remainder % n_levels;
    const int phi_index = remainder / n_levels;
    const int ray_count = n_phi * n_theta;
    const float radius = radii[static_cast<size_t>(query) * n_levels * ray_count +
        level_index * ray_count + phi_index * n_theta + theta_index];
    const float phi = phi_index * static_cast<float>(TWOPI) / (nfp * n_phi);
    const float theta = theta_index * static_cast<float>(TWOPI) / n_theta;
    const size_t offset = static_cast<size_t>(query) * axis_count;
    float ar, arz, az, azz;
    periodic_hermite_device(phi, axis_R + offset, axis_R_phi + offset,
        axis_count, static_cast<float>(TWOPI) / nfp, ar, arz);
    periodic_hermite_device(phi, axis_Z + offset, axis_Z_phi + offset,
        axis_count, static_cast<float>(TWOPI) / nfp, az, azz);
    float sp, cp, st, ct;
    sincosf(phi, &sp, &cp);
    sincosf(theta, &st, &ct);
    const float R = ar + radius * ct, Z = az + radius * st;
    float value, gr, gz, gp;
    evaluate_psi_device(coefficients + static_cast<size_t>(query) * coefficient_count,
        mode_a, mode_b, mode_m, mode_kind, coefficient_count, nfp,
        psi_degree, psi_m_tor, psi_a, R, Z, phi, ar, az, arz, azz,
        value, gr, gz, gp);
    const float radial_derivative = gr * ct + gz * st;
    // The inherited target-V guard validates only the outer boundary. Stokes
    // additionally divides by psi_r at every inner level, so reject invalid
    // implicit tangents explicitly before evaluating the potential.
    unsigned int invalid = 0;
    if (!(isfinite(radius) && radius > 0.0f && isfinite(R) && R > 0.0f && isfinite(Z))) invalid |= 1u;
    if (!(isfinite(radial_derivative) && radial_derivative > 0.0f)) invalid |= 2u;
    if (!(isfinite(value) && isfinite(gr) && isfinite(gz))) invalid |= 4u;
    if (invalid) {
        atomicOr(invalid_query + query, invalid);
        return;
    }
    const float radius_theta = radius * (gr * st - gz * ct) / radial_derivative;
    const float R_theta = radius_theta * ct - radius * st;
    const float Z_theta = radius_theta * st + radius * ct;
    if (!(isfinite(radius_theta) && isfinite(R_theta) && isfinite(Z_theta))) {
        atomicOr(invalid_query + query, 8u);
        return;
    }
    xyz[3 * flat] = R * cp;
    xyz[3 * flat + 1] = R * sp;
    xyz[3 * flat + 2] = Z;
    tangent[3 * flat] = R_theta * cp;
    tangent[3 * flat + 1] = R_theta * sp;
    tangent[3 * flat + 2] = Z_theta;
}

__global__ void local_batch_stokes_reduce_kernel(
    const float* __restrict__ potential, const float* __restrict__ tangent,
    int n_theta, double* __restrict__ sections
) {
    extern __shared__ double shared[];
    double sum = 0.0;
    for (int theta_index = threadIdx.x; theta_index < n_theta; theta_index += blockDim.x) {
        const size_t offset = (static_cast<size_t>(blockIdx.x) * n_theta + theta_index) * 3;
        for (int axis = 0; axis < 3; ++axis)
            sum -= static_cast<double>(potential[offset + axis]) * tangent[offset + axis];
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) sections[blockIdx.x] = shared[0] / n_theta;
}
}

extern "C" int sgpu_set_flux_method(int method) {
    if (method != 0 && method != 1) return 1;
    coil_eval::pipeline::g_stokes_flux = method == 1;
    return 0;
}
