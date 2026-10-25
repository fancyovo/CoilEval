// Deterministic volume candidate generation, selection, and device point records.
#include "pipeline/types.cuh"
#include "pipeline/device_helpers.cuh"
#include "pipeline/common.cuh"
#include "pipeline/flux.cuh"
#include "pipeline/volume.cuh"
#include "pipeline/alpha.cuh"
#include "pipeline/session_state.cuh"
#include "coil_field_internal.h"
#include <cub/cub.cuh>
#include "pipeline/axis.cuh"
#include "pipeline/psi.cuh"

namespace coil_eval::pipeline {

__global__ void generate_volume_candidates_kernel(
    const float* __restrict__ boundary_radii,
    const float* __restrict__ boundary_residuals,
    int n_phi,
    int n_theta,
    int n_radial,
    int nfp,
    float rho_min,
    float edge_level,
    float boundary_tolerance,
    const float* __restrict__ flux_coefficients,
    int flux_degree,
    float edge_flux,
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
    float* __restrict__ s_out,
    float* __restrict__ rho_out,
    float* __restrict__ theta_out,
    float* __restrict__ phi_out,
    float* __restrict__ R_out,
    float* __restrict__ volume_weight,
    float* __restrict__ flux_derivative_out,
    unsigned char* __restrict__ valid,
    int* __restrict__ source_indices
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_phi * n_theta * n_radial;
    if (index >= total) return;
    int remainder = index;
    const int radial_index = remainder % n_radial;
    remainder /= n_radial;
    const int theta_index = remainder % n_theta;
    const int phi_index = remainder / n_theta;
    const int ray = phi_index * n_theta + theta_index;
    const float boundary = boundary_radii[ray];
    const float radial_fraction = sqrtf(
        rho_min * rho_min + (1.0f - rho_min * rho_min) *
        (radial_index + 0.371f) / n_radial
    );
    const float radius = boundary * radial_fraction;
    const float phi = (phi_index + 0.417f) * static_cast<float>(TWOPI) /
                      static_cast<float>(nfp * n_phi);
    float shifted_theta = theta_index + 0.613f + 0.217f * phi_index;
    shifted_theta -= floorf(shifted_theta / n_theta) * n_theta;
    const float geometric_theta = shifted_theta * static_cast<float>(TWOPI) / n_theta;
    float axis_radius, axis_radius_phi, axis_z, axis_z_phi;
    periodic_hermite_device(
        phi, axis_R, axis_R_phi, axis_count, static_cast<float>(TWOPI) / nfp,
        axis_radius, axis_radius_phi
    );
    periodic_hermite_device(
        phi, axis_Z, axis_Z_phi, axis_count, static_cast<float>(TWOPI) / nfp,
        axis_z, axis_z_phi
    );
    float sine_phi, cosine_phi, sine_theta, cosine_theta;
    sincosf(phi, &sine_phi, &cosine_phi);
    sincosf(geometric_theta, &sine_theta, &cosine_theta);
    const float R = axis_radius + radius * cosine_theta;
    const float Z = axis_z + radius * sine_theta;
    float s_value, grad_R, grad_Z, grad_phi_coordinate;
    evaluate_psi_device(
        psi_coefficients, mode_a, mode_b, mode_m, mode_kind, psi_coefficient_count,
        nfp, psi_degree, psi_m_tor, psi_a, R, Z, phi,
        axis_radius, axis_z, axis_radius_phi, axis_z_phi,
        s_value, grad_R, grad_Z, grad_phi_coordinate
    );
    float flux = 0.0f;
    float flux_derivative = 0.0f;
    float power = s_value;
    float derivative_power = 1.0f;
    for (int coefficient = 0; coefficient < flux_degree; ++coefficient) {
        flux += flux_coefficients[coefficient] * power;
        flux_derivative += (coefficient + 1) * flux_coefficients[coefficient] * derivative_power;
        power *= s_value;
        derivative_power *= s_value;
    }
    const float physical_rho = sqrtf(fmaxf(flux / edge_flux, 0.0f));
    const float physical_grad_phi = grad_phi_coordinate / R;
    xyz[3 * index] = R * cosine_phi;
    xyz[3 * index + 1] = R * sine_phi;
    xyz[3 * index + 2] = Z;
    grad_s[3 * index] = grad_R * cosine_phi - physical_grad_phi * sine_phi;
    grad_s[3 * index + 1] = grad_R * sine_phi + physical_grad_phi * cosine_phi;
    grad_s[3 * index + 2] = grad_Z;
    const float radius2 = fmaxf(radius * radius, 1.0e-30f);
    const float theta_R = radius * sine_theta / radius2;
    const float theta_Z = -radius * cosine_theta / radius2;
    const float theta_phi_coordinate =
        (radius * cosine_theta * axis_z_phi - radius * sine_theta * axis_radius_phi) / radius2;
    const float physical_theta_phi = theta_phi_coordinate / R;
    grad_theta[3 * index] = theta_R * cosine_phi - physical_theta_phi * sine_phi;
    grad_theta[3 * index + 1] = theta_R * sine_phi + physical_theta_phi * cosine_phi;
    grad_theta[3 * index + 2] = theta_Z;
    grad_phi[3 * index] = -sine_phi / R;
    grad_phi[3 * index + 1] = cosine_phi / R;
    grad_phi[3 * index + 2] = 0.0f;
    s_out[index] = s_value;
    rho_out[index] = physical_rho;
    theta_out[index] = -geometric_theta;
    phi_out[index] = phi;
    R_out[index] = R;
    volume_weight[index] = R * boundary * boundary;
    flux_derivative_out[index] = flux_derivative;
    const float lower = edge_level * rho_min * rho_min;
    valid[index] = static_cast<unsigned char>(
        isfinite(s_value) && isfinite(physical_rho) && R > 0.0f &&
        boundary_residuals[ray] <= boundary_tolerance &&
        s_value >= lower && s_value <= edge_level * (1.0f + 2.0e-5f) &&
        physical_rho <= 1.0001f
    );
    source_indices[index] = index;
}

__global__ void stratify_selected_indices_kernel(
    const int* __restrict__ available_indices,
    int available_count,
    int output_count,
    int* __restrict__ selected_indices
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_count) return;
    const int source = min(
        available_count - 1,
        static_cast<int>((static_cast<long long>(index) * available_count) / output_count)
    );
    selected_indices[index] = available_indices[source];
}

__global__ void gather_volume_points_kernel(
    const int* __restrict__ indices,
    int count,
    const float* __restrict__ source_xyz,
    const float* __restrict__ source_grad_s,
    const float* __restrict__ source_grad_theta,
    const float* __restrict__ source_grad_phi,
    const float* __restrict__ source_s,
    const float* __restrict__ source_rho,
    const float* __restrict__ source_theta,
    const float* __restrict__ source_phi,
    const float* __restrict__ source_R,
    const float* __restrict__ source_volume_weight,
    const float* __restrict__ source_flux_derivative,
    float* __restrict__ xyz,
    float* __restrict__ grad_s,
    float* __restrict__ grad_theta,
    float* __restrict__ grad_phi,
    float* __restrict__ s,
    float* __restrict__ rho,
    float* __restrict__ theta,
    float* __restrict__ phi,
    float* __restrict__ R,
    float* __restrict__ volume_weight,
    float* __restrict__ flux_derivative
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int source = indices[index];
    for (int component = 0; component < 3; ++component) {
        xyz[3 * index + component] = source_xyz[3 * source + component];
        grad_s[3 * index + component] = source_grad_s[3 * source + component];
        grad_theta[3 * index + component] = source_grad_theta[3 * source + component];
        grad_phi[3 * index + component] = source_grad_phi[3 * source + component];
    }
    s[index] = source_s[source];
    rho[index] = source_rho[source];
    theta[index] = source_theta[source];
    phi[index] = source_phi[source];
    R[index] = source_R[source];
    volume_weight[index] = source_volume_weight[source];
    flux_derivative[index] = source_flux_derivative[source];
}

bool capture_alpha_volume_sampling(const DeviceVolumePoints& points,
                                   const SgpuScoreConfig& config,
                                   std::uint64_t target_cache_id) {
    auto& warm = g_alpha_warm_preconditioner;
    warm.volume_sampling_ready = false;
    warm.volume_source_indices.clear();
    if (points.count != config.volume_point_count || points.count <= 0 ||
        points.source_indices.size() != static_cast<size_t>(points.count)) {
        sgpu_internal_set_error("formal volume sampling branch is unavailable");
        return false;
    }
    const int rays = config.volume_phi_count * config.volume_theta_count;
    const int nr = (static_cast<int>(std::ceil(config.volume_point_count * 1.25)) + rays - 1) / rays;
    warm.volume_source_indices.resize(points.count);
    if (cudaMemcpy(warm.volume_source_indices.data(), points.source_indices.data(),
                   static_cast<size_t>(points.count) * sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        warm.volume_source_indices.clear();
        sgpu_internal_set_error("formal volume sampling branch export failed");
        return false;
    }
    int previous = -1;
    for (int index : warm.volume_source_indices) {
        if (index <= previous || index >= rays * nr) {
            warm.volume_source_indices.clear();
            sgpu_internal_set_error("formal volume sampling branch indices are invalid");
            return false;
        }
        previous = index;
    }
    warm.volume_phi_count = config.volume_phi_count;
    warm.volume_theta_count = config.volume_theta_count;
    warm.volume_radial_count = nr;
    warm.volume_rho_min = config.volume_rho_min;
    warm.volume_device_id = config.device_id;
    warm.volume_target_cache_id = target_cache_id;
    warm.volume_sampling_ready = true;
    return true;
}

bool build_volume_points_native(
    const DevicePsiData& device_psi,
    const AxisData& axis,
    const PsiData& psi,
    const FluxCalibrationNative& flux,
    int nfp,
    const SgpuScoreConfig& config,
    DeviceVolumePoints& output,
    SgpuScoreResult& result
) {
    const int n_phi = config.volume_phi_count;
    const int n_theta = config.volume_theta_count;
    const int ray_count = n_phi * n_theta;
    const int candidate_target = static_cast<int>(std::ceil(config.volume_point_count * 1.25));
    const int n_radial = (candidate_target + ray_count - 1) / ray_count;
    const int candidate_count = ray_count * n_radial;
    result.volume_candidate_count = candidate_count;
    std::vector<float> edge_level{static_cast<float>(flux.edge_level)};
    DeviceBuffer<float> d_edge_level, d_flux_coefficients;
    DeviceBuffer<float> d_ray_coefficients(static_cast<size_t>(ray_count) * (config.psi_poly_degree + 1));
    DeviceBuffer<float> d_boundary_radii(ray_count), d_boundary_residuals(ray_count);
    DeviceVolumePoints candidates;
    DeviceBuffer<unsigned char> d_valid(candidate_count);
    DeviceBuffer<int> d_source_indices(candidate_count), d_available_indices(candidate_count), d_available_count(1);
    if (!copy_to_device(d_edge_level, edge_level) ||
        !copy_to_device(d_flux_coefficients, flux.coefficients) ||
        !d_ray_coefficients.data() || !d_boundary_radii.data() ||
        !d_boundary_residuals.data() || !candidates.allocate(candidate_count) ||
        !d_valid.data() || !d_source_indices.data() || !d_available_indices.data() ||
        !d_available_count.data()) {
        fail_result(&result, "volume point device allocation failed");
        return false;
    }
    constexpr int threads = 256;
    build_ray_coefficients_kernel<<<(ray_count + threads - 1) / threads, threads>>>(
        device_psi.coeffs.data(), device_psi.mode_a.data(), device_psi.mode_b.data(),
        device_psi.mode_m.data(), device_psi.mode_kind.data(), static_cast<int>(psi.coeffs.size()),
        nfp, config.psi_poly_degree, config.psi_m_tor, n_phi, n_theta, 0.417f, 0.613f, 0.217f,
        d_ray_coefficients.data()
    );
    solve_boundary_radii_kernel<<<(ray_count + threads - 1) / threads, threads>>>(
        d_ray_coefficients.data(), ray_count, config.psi_poly_degree,
        static_cast<float>(config.psi_a), d_edge_level.data(), 1,
        config.surface_newton_iters, static_cast<float>(config.surface_newton_tolerance),
        static_cast<float>(config.surface_max_radius_scale),
        d_boundary_radii.data(), d_boundary_residuals.data()
    );
    generate_volume_candidates_kernel<<<(candidate_count + threads - 1) / threads, threads>>>(
        d_boundary_radii.data(), d_boundary_residuals.data(), n_phi, n_theta, n_radial, nfp,
        static_cast<float>(config.volume_rho_min), static_cast<float>(flux.edge_level),
        static_cast<float>(config.flux_boundary_tolerance),
        d_flux_coefficients.data(), static_cast<int>(flux.coefficients.size()),
        static_cast<float>(flux.edge_flux), device_psi.coeffs.data(),
        device_psi.mode_a.data(), device_psi.mode_b.data(), device_psi.mode_m.data(),
        device_psi.mode_kind.data(), static_cast<int>(psi.coeffs.size()),
        config.psi_poly_degree, config.psi_m_tor, static_cast<float>(config.psi_a),
        device_psi.axis_R.data(), device_psi.axis_Z.data(),
        device_psi.axis_R_phi.data(), device_psi.axis_Z_phi.data(),
        static_cast<int>(axis.R.size()), candidates.xyz.data(), candidates.grad_s.data(),
        candidates.grad_theta.data(), candidates.grad_phi.data(), candidates.s.data(),
        candidates.rho.data(), candidates.theta.data(), candidates.phi.data(), candidates.R.data(),
        candidates.volume_weight.data(), candidates.flux_derivative.data(), d_valid.data(),
        d_source_indices.data()
    );
    if (!cuda_stage_ok(cudaGetLastError(), result, "volume candidate generation")) return false;
    size_t temporary_bytes = 0;
    cub::DeviceSelect::Flagged(
        nullptr, temporary_bytes, d_source_indices.data(), d_valid.data(),
        d_available_indices.data(), d_available_count.data(), candidate_count
    );
    DeviceBuffer<unsigned char> d_temporary(temporary_bytes);
    if (temporary_bytes > 0 && !d_temporary.data()) {
        fail_result(&result, "volume compaction workspace allocation failed");
        return false;
    }
    cub::DeviceSelect::Flagged(
        d_temporary.data(), temporary_bytes, d_source_indices.data(), d_valid.data(),
        d_available_indices.data(), d_available_count.data(), candidate_count
    );
    int available_count = 0;
    if (!cuda_stage_ok(cudaMemcpy(&available_count, d_available_count.data(), sizeof(int), cudaMemcpyDeviceToHost), result, "volume valid count")) {
        return false;
    }
    result.volume_available_count = available_count;
    result.volume_valid_fraction = static_cast<double>(available_count) /
        std::max(candidate_count, 1);
    const int minimum_count = std::max(
        std::max(config.alpha_fit_point_count, config.volume_point_count),
        static_cast<int>(std::ceil(0.95 * candidate_count))
    );
    if (available_count < minimum_count) {
        result.status = SGPU_SCORE_FLUX_REJECTED;
        std::snprintf(result.error_message, sizeof(result.error_message),
                      "volume sampler found %d valid points; minimum is %d", available_count, minimum_count);
        return false;
    }
    const int output_count = std::min(config.volume_point_count, available_count);
    DeviceBuffer<int> d_selected_indices(output_count);
    if (!d_selected_indices.data() || !output.allocate(output_count)) {
        fail_result(&result, "volume output allocation failed");
        return false;
    }
    stratify_selected_indices_kernel<<<(output_count + threads - 1) / threads, threads>>>(
        d_available_indices.data(), available_count, output_count, d_selected_indices.data()
    );
    gather_volume_points_kernel<<<(output_count + threads - 1) / threads, threads>>>(
        d_selected_indices.data(), output_count,
        candidates.xyz.data(), candidates.grad_s.data(), candidates.grad_theta.data(),
        candidates.grad_phi.data(), candidates.s.data(), candidates.rho.data(),
        candidates.theta.data(), candidates.phi.data(), candidates.R.data(),
        candidates.volume_weight.data(), candidates.flux_derivative.data(),
        output.xyz.data(), output.grad_s.data(), output.grad_theta.data(), output.grad_phi.data(),
        output.s.data(), output.rho.data(), output.theta.data(), output.phi.data(), output.R.data(),
        output.volume_weight.data(), output.flux_derivative.data()
    );
    if (!cuda_stage_ok(cudaDeviceSynchronize(), result, "volume point compaction")) return false;
    output.source_indices = std::move(d_selected_indices);
    result.volume_point_count = output_count;
    return true;
}


}  // namespace coil_eval::pipeline
