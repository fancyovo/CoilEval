// Batched endpoint boundary, volume, alpha PCGLS, and QS kernels.
#include "pipeline/types.cuh"
#include "pipeline/device_helpers.cuh"
#include "pipeline/local_kernels.cuh"
#include "radial_first_branch.h"

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
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int ray_count = n_phi * n_theta;
    const size_t total = static_cast<size_t>(query_count) * ray_count;
    if (flat >= total) return;
    const int query = static_cast<int>(flat / ray_count);
    const int ray = static_cast<int>(flat - static_cast<size_t>(query) * ray_count);
    const int phi_index = ray / n_theta;
    const int theta_index = ray - phi_index * n_theta;
    const float phi = (phi_index + phi_offset) * static_cast<float>(TWOPI) /
        static_cast<float>(nfp * n_phi);
    float shifted_theta = theta_index + theta_offset + theta_shear * phi_index;
    shifted_theta -= floorf(shifted_theta / n_theta) * n_theta;
    const float theta = shifted_theta * static_cast<float>(TWOPI) / n_theta;
    float sine, cosine;
    sincosf(theta, &sine, &cosine);
    float cpow[25], spow[25], cosv[33], sinv[33];
    cpow[0] = spow[0] = 1.0f;
    for (int power = 1; power <= degree; ++power) {
        cpow[power] = cpow[power - 1] * cosine;
        spow[power] = spow[power - 1] * sine;
    }
    cosv[0] = 1.0f;
    sinv[0] = 0.0f;
    for (int mode = 1; mode <= m_tor; ++mode) {
        sincosf(static_cast<float>(mode * nfp) * phi, &sinv[mode], &cosv[mode]);
    }
    float local[25];
    for (int power = 0; power <= degree; ++power) local[power] = 0.0f;
    local[2] = cosine * cosine;
    const float* query_coefficients = coefficients + static_cast<size_t>(query) * coefficient_count;
    for (int index = 0; index < coefficient_count; ++index) {
        const int mode = mode_m[index];
        const float trig = mode == 0 ? 1.0f :
            (mode_kind[index] == 0 ? cosv[mode] : sinv[mode]);
        local[mode_a[index] + mode_b[index]] += query_coefficients[index] *
            cpow[mode_a[index]] * spow[mode_b[index]] * trig;
    }
    float* output = ray_coefficients + flat * (degree + 1);
    for (int power = 0; power <= degree; ++power) output[power] = local[power];
}

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
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t per_query = static_cast<size_t>(ray_count) * level_count;
    if (flat >= static_cast<size_t>(query_count) * per_query) return;
    const int query = static_cast<int>(flat / per_query);
    const int local = static_cast<int>(flat - static_cast<size_t>(query) * per_query);
    const int level_index = local / ray_count;
    const int ray = local - level_index * ray_count;
    const float* polynomial = ray_coefficients +
        (static_cast<size_t>(query) * ray_count + ray) * (degree + 1);
    const float level = levels[static_cast<size_t>(query)*level_count+level_index];
    const float q = polynomial[2];
    float radius = q > 1.0e-10f
        ? radius_scale * sqrtf(fmaxf(level, 0.0f) / q)
        : radius_scale * sqrtf(fmaxf(level, 1.0e-16f));
    const float maximum = max_radius_scale * radius_scale;
    radius = fminf(maximum, fmaxf(1.0e-12f * radius_scale, radius));
    float final_residual = INFINITY;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const float u = radius / radius_scale;
        float value = 0.0f, derivative = 0.0f;
        float previous = u, power = u * u;
        for (int current_degree = 2; current_degree <= degree; ++current_degree) {
            if (current_degree > 2) {
                previous = power;
                power *= u;
            }
            value += polynomial[current_degree] * power;
            derivative += current_degree * polynomial[current_degree] * previous / radius_scale;
        }
        final_residual = value - level;
        if (fabsf(final_residual) <= tolerance) break;
        const float denominator = fabsf(derivative) > 1.0e-14f
            ? derivative : copysignf(1.0e-14f, derivative == 0.0f ? 1.0f : derivative);
        const float limit = 0.4f * fmaxf(radius, 1.0e-10f);
        radius = fminf(maximum, fmaxf(
            1.0e-12f * radius_scale,
            radius - fminf(limit, fmaxf(-limit, final_residual / denominator))
        ));
    }
    double connected = static_cast<double>(radius) / radius_scale;
    if (!coil_eval_radial::first_root(polynomial, degree, level,
            max_radius_scale, tolerance, connected)) {
        radii[flat] = 0.0f;
        residuals[flat] = INFINITY;
        return;
    }
    radius = static_cast<float>(connected * radius_scale);
    const float u = radius / radius_scale;
    float value = 0.0f, power = u * u;
    for (int current_degree = 2; current_degree <= degree; ++current_degree) {
        if (current_degree > 2) power *= u;
        value += polynomial[current_degree] * power;
    }
    radii[flat] = radius;
    residuals[flat] = fabsf(value - level);
}

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
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int points_per_query = n_phi * n_levels * n_theta * n_radial;
    if (flat >= static_cast<size_t>(query_count) * points_per_query) return;
    const int query = static_cast<int>(flat / points_per_query);
    int remainder = static_cast<int>(flat - static_cast<size_t>(query) * points_per_query);
    const int radial_index = remainder % n_radial;
    remainder /= n_radial;
    const int theta_index = remainder % n_theta;
    remainder /= n_theta;
    const int level_index = remainder % n_levels;
    const int phi_index = remainder / n_levels;
    const int ray_count = n_phi * n_theta;
    const int ray = phi_index * n_theta + theta_index;
    const float boundary = boundary_radii[
        static_cast<size_t>(query) * ray_count * n_levels + level_index * ray_count + ray
    ];
    const float radius = 0.5f * boundary * (gauss_nodes[radial_index] + 1.0f);
    const float phi = phi_index * static_cast<float>(TWOPI) / static_cast<float>(nfp * n_phi);
    const float theta = theta_index * static_cast<float>(TWOPI) / n_theta;
    float axis_radius, axis_radius_phi, axis_z, axis_z_phi;
    const size_t axis_offset = static_cast<size_t>(query) * axis_count;
    periodic_hermite_device(phi, axis_R + axis_offset, axis_R_phi + axis_offset,
        axis_count, static_cast<float>(TWOPI) / nfp, axis_radius, axis_radius_phi);
    periodic_hermite_device(phi, axis_Z + axis_offset, axis_Z_phi + axis_offset,
        axis_count, static_cast<float>(TWOPI) / nfp, axis_z, axis_z_phi);
    float sine_phi, cosine_phi, sine_theta, cosine_theta;
    sincosf(phi, &sine_phi, &cosine_phi);
    sincosf(theta, &sine_theta, &cosine_theta);
    const float R = axis_radius + radius * cosine_theta;
    xyz[3 * flat] = R * cosine_phi;
    xyz[3 * flat + 1] = R * sine_phi;
    xyz[3 * flat + 2] = axis_z + radius * sine_theta;
}

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
) {
    extern __shared__ double shared[];
    const int sections_per_query = n_phi * n_levels;
    const int flat_section = blockIdx.x;
    const int query = flat_section / sections_per_query;
    if (query >= query_count) return;
    const int section = flat_section - query * sections_per_query;
    const int level_index = section % n_levels;
    const int phi_index = section / n_levels;
    const int ray_count = n_phi * n_theta;
    const int points_per_query = n_phi * n_levels * n_theta * n_radial;
    const float phi = phi_index * static_cast<float>(TWOPI) / static_cast<float>(nfp * n_phi);
    float sine_phi, cosine_phi;
    sincosf(phi, &sine_phi, &cosine_phi);
    double sum = 0.0;
    for (int local = threadIdx.x; local < n_theta * n_radial; local += blockDim.x) {
        const int radial_index = local % n_radial;
        const int theta_index = local / n_radial;
        const int ray = phi_index * n_theta + theta_index;
        const float boundary = boundary_radii[
            static_cast<size_t>(query) * ray_count * n_levels + level_index * ray_count + ray
        ];
        const float radius = 0.5f * boundary * (gauss_nodes[radial_index] + 1.0f);
        const float radial_weight = 0.5f * boundary * gauss_weights[radial_index];
        const int point = ((phi_index * n_levels + level_index) * n_theta + theta_index) *
            n_radial + radial_index;
        const size_t point_offset = static_cast<size_t>(query) * points_per_query + point;
        const float b_phi = -B[3 * point_offset] * sine_phi + B[3 * point_offset + 1] * cosine_phi;
        sum += static_cast<double>(b_phi) * radius * radial_weight;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) sections[flat_section] = shared[0] / n_theta;
}

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
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= static_cast<size_t>(query_count) * output_count) return;
    const int query = static_cast<int>(flat / output_count);
    const int output_index = static_cast<int>(flat - static_cast<size_t>(query) * output_count);
    const int candidate_count = n_phi * n_theta * n_radial;
    int source = source_indices ? source_indices[output_index] :
        min(candidate_count - 1, static_cast<int>(
            static_cast<long long>(output_index) * candidate_count / output_count));
    const int radial_index = source % n_radial;
    source /= n_radial;
    const int theta_index = source % n_theta;
    const int phi_index = source / n_theta;
    const int ray_count = n_phi * n_theta;
    const int ray = phi_index * n_theta + theta_index;
    const float boundary = boundary_radii[static_cast<size_t>(query) * ray_count + ray];
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
    const size_t axis_offset = static_cast<size_t>(query) * axis_count;
    float axis_radius, axis_radius_phi, axis_z, axis_z_phi;
    periodic_hermite_device(phi, axis_R + axis_offset, axis_R_phi + axis_offset,
        axis_count, static_cast<float>(TWOPI) / nfp, axis_radius, axis_radius_phi);
    periodic_hermite_device(phi, axis_Z + axis_offset, axis_Z_phi + axis_offset,
        axis_count, static_cast<float>(TWOPI) / nfp, axis_z, axis_z_phi);
    float sine_phi, cosine_phi, sine_theta, cosine_theta;
    sincosf(phi, &sine_phi, &cosine_phi);
    sincosf(geometric_theta, &sine_theta, &cosine_theta);
    const float R = axis_radius + radius * cosine_theta;
    const float Z = axis_z + radius * sine_theta;
    float s_value, grad_R, grad_Z, grad_phi_coordinate;
    evaluate_psi_device(
        psi_coefficients + static_cast<size_t>(query) * psi_coefficient_count,
        mode_a, mode_b, mode_m, mode_kind, psi_coefficient_count,
        nfp, psi_degree, psi_m_tor, psi_a, R, Z, phi,
        axis_radius, axis_z, axis_radius_phi, axis_z_phi,
        s_value, grad_R, grad_Z, grad_phi_coordinate
    );
    const float* query_flux_coefficients =
        flux_coefficients + static_cast<size_t>(query) * flux_degree;
    float flux = 0.0f, flux_derivative = 0.0f;
    float power = s_value, derivative_power = 1.0f;
    for (int coefficient = 0; coefficient < flux_degree; ++coefficient) {
        flux += query_flux_coefficients[coefficient] * power;
        flux_derivative += (coefficient + 1) * query_flux_coefficients[coefficient] * derivative_power;
        power *= s_value;
        derivative_power *= s_value;
    }
    const float physical_rho = sqrtf(fmaxf(flux / edge_flux[query], 0.0f));
    const float physical_grad_phi = grad_phi_coordinate / R;
    xyz[3 * flat] = R * cosine_phi;
    xyz[3 * flat + 1] = R * sine_phi;
    xyz[3 * flat + 2] = Z;
    grad_s[3 * flat] = grad_R * cosine_phi - physical_grad_phi * sine_phi;
    grad_s[3 * flat + 1] = grad_R * sine_phi + physical_grad_phi * cosine_phi;
    grad_s[3 * flat + 2] = grad_Z;
    const float radius2 = fmaxf(radius * radius, 1.0e-30f);
    const float theta_R = radius * sine_theta / radius2;
    const float theta_Z = -radius * cosine_theta / radius2;
    const float theta_phi_coordinate =
        (radius * cosine_theta * axis_z_phi - radius * sine_theta * axis_radius_phi) / radius2;
    const float physical_theta_phi = theta_phi_coordinate / R;
    grad_theta[3 * flat] = theta_R * cosine_phi - physical_theta_phi * sine_phi;
    grad_theta[3 * flat + 1] = theta_R * sine_phi + physical_theta_phi * cosine_phi;
    grad_theta[3 * flat + 2] = theta_Z;
    grad_phi[3 * flat] = -sine_phi / R;
    grad_phi[3 * flat + 1] = cosine_phi / R;
    grad_phi[3 * flat + 2] = 0.0f;
    rho_out[flat] = physical_rho;
    theta_out[flat] = -geometric_theta;
    phi_out[flat] = phi;
    volume_weight[flat] = R * boundary * boundary;
    flux_derivative_out[flat] = flux_derivative;
}

__global__ void local_batch_alpha_histogram_kernel(
    const float* __restrict__ rho,
    int query_count,
    int point_count,
    int fit_count,
    int radial_bins,
    int* __restrict__ bin_counts
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= static_cast<size_t>(query_count) * fit_count) return;
    const int query = static_cast<int>(flat / fit_count);
    const int index = static_cast<int>(flat - static_cast<size_t>(query) * fit_count);
    const int point = min(point_count - 1, static_cast<int>(
        static_cast<long long>(index) * point_count / fit_count
    ));
    const float radial = rho[static_cast<size_t>(query) * point_count + point];
    const int bin = min(radial_bins - 1, max(0, static_cast<int>(radial * radial_bins)));
    atomicAdd(bin_counts + static_cast<size_t>(query) * radial_bins + bin, 1);
}

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
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= static_cast<size_t>(query_count) * fit_count) return;
    const int query = static_cast<int>(flat / fit_count);
    const int index = static_cast<int>(flat - static_cast<size_t>(query) * fit_count);
    const int point = min(point_count - 1, static_cast<int>(
        static_cast<long long>(index) * point_count / fit_count
    ));
    const size_t source = static_cast<size_t>(query) * point_count + point;
    float grad_norm2 = 0.0f, field_dot_grad = 0.0f, field_norm2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        const float field = B[3 * source + component];
        const float gradient = grad_s[3 * source + component];
        grad_norm2 += gradient * gradient;
        field_dot_grad += field * gradient;
        field_norm2 += field * field;
    }
    const float normal_coefficient = field_dot_grad / fmaxf(grad_norm2, 1.0e-30f);
    float along_theta = 0.0f, along_phi = 0.0f, normal_norm2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        const float normal = normal_coefficient * grad_s[3 * source + component];
        const float tangent = B[3 * source + component] - normal;
        along_theta += tangent * grad_theta[3 * source + component];
        along_phi += tangent * grad_phi[3 * source + component];
        normal_norm2 += normal * normal;
    }
    const float radial = source_rho[source];
    const int bin = min(radial_bins - 1, max(0, static_cast<int>(radial * radial_bins)));
    const float weight = rsqrtf(static_cast<float>(max(
        bin_counts[static_cast<size_t>(query) * radial_bins + bin], 1
    ))) / sqrtf(fmaxf(field_norm2, 1.0e-30f));
    b_theta[flat] = along_theta;
    b_phi[flat] = along_phi;
    weights[flat] = weight;
    rho[flat] = radial;
    theta[flat] = source_theta[source];
    phi[flat] = source_phi[source];
    atomic_add_double(diagnostics + static_cast<size_t>(query) * 3, static_cast<double>(weight) * weight);
    atomic_add_double(diagnostics + static_cast<size_t>(query) * 3 + 1, normal_norm2);
    atomic_add_double(diagnostics + static_cast<size_t>(query) * 3 + 2, field_norm2);
}

__global__ void local_batch_normalize_alpha_kernel(
    float* __restrict__ weights,
    const float* __restrict__ b_theta,
    const double* __restrict__ diagnostics,
    int query_count,
    int fit_count,
    float* __restrict__ rhs
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= static_cast<size_t>(query_count) * fit_count) return;
    const int query = static_cast<int>(flat / fit_count);
    const float scale = sqrtf(static_cast<float>(
        fit_count / fmax(diagnostics[static_cast<size_t>(query) * 3], 1.0e-300)
    ));
    weights[flat] *= scale;
    rhs[flat] = -weights[flat] * b_theta[flat];
}

__global__ void local_batch_assemble_alpha_tile_kernel(
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
    int fit_count,
    int query_begin,
    int tile_count,
    const float* __restrict__ center_scale,
    float* __restrict__ matrices
) {
    const int column = blockIdx.x;
    const int local_query = blockIdx.y;
    if (local_query >= tile_count) return;
    const int query = query_begin + local_query;
    const int column_count = mode_count + iota_degree + 1;
    float* matrix = matrices + static_cast<size_t>(local_query) * fit_count * column_count +
        static_cast<size_t>(column) * fit_count;
    const size_t query_offset = static_cast<size_t>(query) * fit_count;
    if (column < mode_count) {
        const int m = mode_m[column];
        const int n = mode_n[column];
        const int kind = mode_kind[column];
        const float* radial_coeff =
            radial_coefficients + static_cast<size_t>(column) * (radial_order + 1);
        for (int row = threadIdx.x; row < fit_count; row += blockDim.x) {
            const size_t index = query_offset + row;
            const float radial = evaluate_radial_polynomial(radial_coeff, radial_order, rho[index], radial_basis);
            const float argument = m * theta[index] - n * nfp * phi[index];
            float sine, cosine;
            sincosf(argument, &sine, &cosine);
            const float derivative_theta = kind == 0 ? -m * radial * sine : m * radial * cosine;
            const float derivative_phi = kind == 0 ? n * nfp * radial * sine : -n * nfp * radial * cosine;
            matrix[row] = weights[index] *
                (derivative_theta * b_theta[index] + derivative_phi * b_phi[index]) /
                center_scale[column];
        }
    } else {
        const int power_index = column - mode_count;
        if (power_index > iota_degree) return;
        for (int row = threadIdx.x; row < fit_count; row += blockDim.x) {
            const size_t index = query_offset + row;
            float radial_power = 1.0f;
            const float u = rho[index] * rho[index];
            for (int power = 0; power < power_index; ++power) radial_power *= u;
            matrix[row] = -weights[index] * radial_power * b_phi[index] /
                center_scale[column];
        }
    }
}

__global__ void local_batch_initialize_alpha_coefficients_kernel(
    const float* __restrict__ center,
    int column_count,
    int tile_count,
    float* __restrict__ coefficients
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat < static_cast<size_t>(column_count) * tile_count) {
        coefficients[flat] = center[flat % column_count];
    }
}

__global__ void local_batch_initialize_alpha_residual_kernel(
    const float* __restrict__ rhs,
    const float* __restrict__ prediction,
    int fit_count,
    int query_begin,
    int tile_count,
    float* __restrict__ residual
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= static_cast<size_t>(fit_count) * tile_count) return;
    const int local_query = static_cast<int>(flat / fit_count);
    const int row = static_cast<int>(flat - static_cast<size_t>(local_query) * fit_count);
    residual[flat] = rhs[static_cast<size_t>(query_begin + local_query) * fit_count + row] -
        prediction[flat];
}

__global__ void local_batch_alpha_normal_ridge_kernel(
    float* __restrict__ normal,
    const float* __restrict__ coefficients,
    float ridge,
    size_t count
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat < count) normal[flat] -= ridge * coefficients[flat];
}

__global__ void local_batch_alpha_delta_kernel(
    const float* __restrict__ image,
    int fit_count,
    const float* __restrict__ direction,
    int column_count,
    int tile_count,
    float ridge,
    float* __restrict__ delta
) {
    const int query = blockIdx.x;
    if (query >= tile_count) return;
    float sum = 0.0f;
    for (int row = threadIdx.x; row < fit_count; row += blockDim.x) {
        const float value = image[static_cast<size_t>(query) * fit_count + row];
        sum += value * value;
    }
    for (int column = threadIdx.x; column < column_count; column += blockDim.x) {
        const float value = direction[static_cast<size_t>(query) * column_count + column];
        sum += ridge * value * value;
    }
    extern __shared__ float local_shared_float[];
    local_shared_float[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            local_shared_float[threadIdx.x] += local_shared_float[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) delta[query] = local_shared_float[0];
}

__global__ void local_batch_column_norm_kernel(
    const float* __restrict__ values,
    int rows,
    int columns,
    float* __restrict__ norms
) {
    const int column = blockIdx.x;
    if (column >= columns) return;
    float sum = 0.0f;
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        const float value = values[static_cast<size_t>(column) * rows + row];
        sum += value * value;
    }
    extern __shared__ float local_shared_float[];
    local_shared_float[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            local_shared_float[threadIdx.x] += local_shared_float[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) norms[column] = local_shared_float[0];
}

__global__ void local_batch_direction_update_kernel(
    float* __restrict__ direction,
    const float* __restrict__ normal,
    const float* __restrict__ gamma,
    const float* __restrict__ gamma_new,
    int rows,
    int columns
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= static_cast<size_t>(rows) * columns) return;
    const int column = static_cast<int>(flat / rows);
    direction[flat] = normal[flat] +
        gamma_new[column] / fmaxf(gamma[column], 1.0e-30f) * direction[flat];
}

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
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t point_total = static_cast<size_t>(fit_count) * tile_count;
    const size_t coefficient_total = static_cast<size_t>(column_count) * tile_count;
    if (flat < point_total) {
        const int query = static_cast<int>(flat / fit_count);
        residual[flat] -= gamma[query] / fmaxf(delta[query], 1.0e-30f) * image[flat];
    }
    if (flat < coefficient_total) {
        const int query = static_cast<int>(flat / column_count);
        coefficients[flat] += gamma[query] / fmaxf(delta[query], 1.0e-30f) * direction[flat];
    }
}

__global__ void local_batch_extract_iota_kernel(
    const float* __restrict__ scaled_coefficients,
    const float* __restrict__ scale,
    int column_count,
    int mode_count,
    int iota_count,
    int query_begin,
    int tile_count,
    float* __restrict__ iota
) {
    const int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= tile_count * iota_count) return;
    const int local_query = flat / iota_count;
    const int degree = flat - local_query * iota_count;
    iota[static_cast<size_t>(query_begin + local_query) * iota_count + degree] =
        scaled_coefficients[static_cast<size_t>(local_query) * column_count + mode_count + degree] /
        scale[mode_count + degree];
}

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
) {
    const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= static_cast<size_t>(query_count) * point_count) return;
    const int query = static_cast<int>(flat / point_count);
    float magnitude2 = 0.0f, grad_s_norm2 = 0.0f, field_dot_grad_s = 0.0f;
    for (int component = 0; component < 3; ++component) {
        const float field = B[3 * flat + component];
        const float gradient = grad_s[3 * flat + component];
        magnitude2 += field * field;
        grad_s_norm2 += gradient * gradient;
        field_dot_grad_s += field * gradient;
    }
    const float magnitude = sqrtf(fmaxf(magnitude2, 1.0e-30f));
    float grad_magnitude[3];
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        float sum = 0.0f;
        for (int component = 0; component < 3; ++component) {
            sum += grad_B[9 * flat + 3 * component + coordinate] * B[3 * flat + component];
        }
        grad_magnitude[coordinate] = sum / magnitude;
    }
    float grad_psi[3];
    for (int component = 0; component < 3; ++component) {
        grad_psi[component] = flux_derivative[flat] * grad_s[3 * flat + component];
    }
    const float cross_x = B[3 * flat + 1] * grad_psi[2] - B[3 * flat + 2] * grad_psi[1];
    const float cross_y = B[3 * flat + 2] * grad_psi[0] - B[3 * flat] * grad_psi[2];
    const float cross_z = B[3 * flat] * grad_psi[1] - B[3 * flat + 1] * grad_psi[0];
    const float A = cross_x * grad_magnitude[0] + cross_y * grad_magnitude[1] +
        cross_z * grad_magnitude[2];
    const float C = B[3 * flat] * grad_magnitude[0] + B[3 * flat + 1] * grad_magnitude[1] +
        B[3 * flat + 2] * grad_magnitude[2];
    const float u = rho[flat] * rho[flat];
    const float* query_iota = iota_coefficients + static_cast<size_t>(query) * (iota_degree + 1);
    float iota = 0.0f, power = 1.0f;
    for (int degree = 0; degree <= iota_degree; ++degree) {
        iota += query_iota[degree] * power;
        power *= u;
    }
    const double f_c = (helicity_M * static_cast<double>(iota) - helicity_N) * A -
        helicity_M * G[query] * C;
    const double f_c_qa = static_cast<double>(iota) * A - G[query] * C;
    const double f_c_qp = -static_cast<double>(nfp) * A;
    const double denominator = fmax(static_cast<double>(magnitude) * magnitude * magnitude, 1.0e-30);
    const double normalized = f_c / denominator;
    const double normalized_qa = f_c_qa / denominator;
    const double normalized_qp = f_c_qp / denominator;
    const double weight = volume_weight[flat];
    double* query_sums = sums + static_cast<size_t>(query) * 9;
    atomic_add_double(query_sums, weight);
    atomic_add_double(query_sums + 1, weight * normalized * normalized);
    atomic_add_double(query_sums + 4, weight * weight);
    atomic_add_double(query_sums + 7, weight * normalized_qa * normalized_qa);
    atomic_add_double(query_sums + 8, weight * normalized_qp * normalized_qp);
    if (rho[flat] >= edge_rho_threshold) {
        atomic_add_double(query_sums + 2, weight);
        atomic_add_double(query_sums + 3, weight * normalized * normalized);
        atomic_add_double(query_sums + 5, weight * weight);
        atomic_add_double(query_sums + 6, 1.0);
    }
    const double psi_scale2 = static_cast<double>(magnitude2) * grad_s_norm2;
    atomic_add_double(psi_sums + static_cast<size_t>(query) * 2,
        static_cast<double>(field_dot_grad_s) * field_dot_grad_s);
    atomic_add_double(psi_sums + static_cast<size_t>(query) * 2 + 1, psi_scale2);
    const float angle = fabsf(field_dot_grad_s) / sqrtf(fmaxf(magnitude2 * grad_s_norm2, 1.0e-30f));
    const float log_angle = log10f(fmaxf(angle, 1.0e-10f));
    const int bin = min(histogram_bins - 1, max(0, static_cast<int>(
        (log_angle + 10.0f) * histogram_bins / 8.0f
    )));
    atomicAdd(psi_histogram + static_cast<size_t>(query) * histogram_bins + bin, 1u);
}


}  // namespace coil_eval::pipeline
