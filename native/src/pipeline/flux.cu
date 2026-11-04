// Flux quadrature, polynomial calibration, and shared boundary-radius kernels.
#include "pipeline/types.cuh"
#include "pipeline/device_helpers.cuh"
#include "pipeline/common.cuh"
#include "pipeline/flux.cuh"
#include "pipeline/psi.cuh"
#include "pipeline/axis.cuh"
#include "coil_field_internal.h"
#include "radial_first_branch.h"

namespace coil_eval::pipeline {

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
) {
    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    const int ray_count = n_phi * n_theta;
    if (ray >= ray_count) return;
    const int phi_index = ray / n_theta;
    const int theta_index = ray - phi_index * n_theta;
    const float phi = (phi_index + phi_offset) * static_cast<float>(TWOPI) /
                      static_cast<float>(nfp * n_phi);
    float shifted_theta = theta_index + theta_offset + theta_shear * phi_index;
    shifted_theta -= floorf(shifted_theta / n_theta) * n_theta;
    const float theta = shifted_theta * static_cast<float>(TWOPI) / n_theta;
    float cosine, sine;
    sincosf(theta, &sine, &cosine);
    float cpow[25];
    float spow[25];
    float cosv[33];
    float sinv[33];
    cpow[0] = 1.0f;
    spow[0] = 1.0f;
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
    for (int index = 0; index < n_coeff; ++index) {
        const int mode = mode_m[index];
        const float trig = mode == 0 ? 1.0f : (mode_kind[index] == 0 ? cosv[mode] : sinv[mode]);
        const int total = mode_a[index] + mode_b[index];
        local[total] += coeffs[index] * cpow[mode_a[index]] * spow[mode_b[index]] * trig;
    }
    for (int power = 0; power <= degree; ++power) {
        ray_coefficients[static_cast<size_t>(ray) * (degree + 1) + power] = local[power];
    }
}

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
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ray_count * level_count;
    if (index >= total) return;
    const int level_index = index / ray_count;
    const int ray = index - level_index * ray_count;
    const float* polynomial = ray_coefficients + static_cast<size_t>(ray) * (degree + 1);
    const float level = levels[level_index];
    const float q = polynomial[2];
    float radius = q > 1.0e-10f
        ? a_scale * sqrtf(fmaxf(level, 0.0f) / q)
        : a_scale * sqrtf(fmaxf(level, 1.0e-16f));
    const float maximum = max_radius_scale * a_scale;
    radius = fminf(maximum, fmaxf(1.0e-12f * a_scale, radius));
    float final_residual = INFINITY;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const float u = radius / a_scale;
        float value = 0.0f;
        float derivative = 0.0f;
        float previous = u;
        float power = u * u;
        for (int degree_now = 2; degree_now <= degree; ++degree_now) {
            if (degree_now > 2) {
                previous = power;
                power *= u;
            }
            value += polynomial[degree_now] * power;
            derivative += degree_now * polynomial[degree_now] * previous / a_scale;
        }
        final_residual = value - level;
        if (fabsf(final_residual) <= tolerance) break;
        const float denominator = fabsf(derivative) > 1.0e-14f
            ? derivative : copysignf(1.0e-14f, derivative == 0.0f ? 1.0f : derivative);
        const float limit = 0.4f * fmaxf(radius, 1.0e-10f);
        const float step = fminf(limit, fmaxf(-limit, final_residual / denominator));
        radius = fminf(maximum, fmaxf(1.0e-12f * a_scale, radius - step));
    }
    // Newton can converge exactly to a later branch, even with positive slope.
    double connected = static_cast<double>(radius) / a_scale;
    if (!coil_eval_radial::first_root(polynomial, degree, level,
            max_radius_scale, tolerance, connected)) {
        radii[index] = 0.0f;
        residuals[index] = INFINITY;
        return;
    }
    radius = static_cast<float>(connected * a_scale);
    {
        const float u = radius / a_scale;
        float value = 0.0f;
        float power = u * u;
        for (int degree_now = 2; degree_now <= degree; ++degree_now) {
            if (degree_now > 2) power *= u;
            value += polynomial[degree_now] * power;
        }
        final_residual = value - level;
    }
    radii[index] = radius;
    residuals[index] = fabsf(final_residual);
}

void gauss_legendre(int count, std::vector<float>& nodes, std::vector<float>& weights) {
    nodes.resize(count);
    weights.resize(count);
    const int half = (count + 1) / 2;
    for (int index = 0; index < half; ++index) {
        double root = std::cos(PI * (index + 0.75) / (count + 0.5));
        double derivative = 0.0;
        for (int iteration = 0; iteration < 32; ++iteration) {
            double p0 = 1.0;
            double p1 = root;
            for (int degree = 2; degree <= count; ++degree) {
                const double next = ((2.0 * degree - 1.0) * root * p1 - (degree - 1.0) * p0) / degree;
                p0 = p1;
                p1 = next;
            }
            derivative = count * (root * p1 - p0) / (root * root - 1.0);
            const double step = p1 / derivative;
            root -= step;
            if (std::abs(step) < 1.0e-15) break;
        }
        const double weight = 2.0 / ((1.0 - root * root) * derivative * derivative);
        nodes[index] = static_cast<float>(-root);
        nodes[count - 1 - index] = static_cast<float>(root);
        weights[index] = static_cast<float>(weight);
        weights[count - 1 - index] = static_cast<float>(weight);
    }
}

__global__ void generate_flux_points_kernel(
    const float* __restrict__ boundary_radii,
    const float* __restrict__ gauss_nodes,
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
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_phi * n_levels * n_theta * n_radial;
    if (index >= total) return;
    int remainder = index;
    const int radial_index = remainder % n_radial;
    remainder /= n_radial;
    const int theta_index = remainder % n_theta;
    remainder /= n_theta;
    const int level_index = remainder % n_levels;
    const int phi_index = remainder / n_levels;
    const int ray = phi_index * n_theta + theta_index;
    const float boundary = boundary_radii[level_index * (n_phi * n_theta) + ray];
    const float radius = 0.5f * boundary * (gauss_nodes[radial_index] + 1.0f);
    const float phi = phi_index * static_cast<float>(TWOPI) / static_cast<float>(nfp * n_phi);
    const float theta = theta_index * static_cast<float>(TWOPI) / n_theta;
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
    sincosf(theta, &sine_theta, &cosine_theta);
    const float R = axis_radius + radius * cosine_theta;
    xyz[3 * index] = R * cosine_phi;
    xyz[3 * index + 1] = R * sine_phi;
    xyz[3 * index + 2] = axis_z + radius * sine_theta;
}

__global__ void reduce_flux_sections_kernel(
    const float* __restrict__ B,
    const float* __restrict__ boundary_radii,
    const float* __restrict__ gauss_nodes,
    const float* __restrict__ gauss_weights,
    int n_phi,
    int n_levels,
    int n_theta,
    int n_radial,
    int nfp,
    double* __restrict__ sections
) {
    extern __shared__ double shared[];
    const int section = blockIdx.x;
    const int level_index = section % n_levels;
    const int phi_index = section / n_levels;
    const float phi = phi_index * static_cast<float>(TWOPI) / static_cast<float>(nfp * n_phi);
    float sine_phi, cosine_phi;
    sincosf(phi, &sine_phi, &cosine_phi);
    double sum = 0.0;
    const int per_section = n_theta * n_radial;
    for (int local = threadIdx.x; local < per_section; local += blockDim.x) {
        const int radial_index = local % n_radial;
        const int theta_index = local / n_radial;
        const int ray = phi_index * n_theta + theta_index;
        const float boundary = boundary_radii[level_index * (n_phi * n_theta) + ray];
        const float radius = 0.5f * boundary * (gauss_nodes[radial_index] + 1.0f);
        const float radial_weight = 0.5f * boundary * gauss_weights[radial_index];
        const int point = ((phi_index * n_levels + level_index) * n_theta + theta_index) * n_radial + radial_index;
        const float B_phi = -B[3 * point] * sine_phi + B[3 * point + 1] * cosine_phi;
        sum += static_cast<double>(B_phi) * radius * radial_weight;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) sections[section] = shared[0] / n_theta;
}

bool solve_small_least_squares(
    const std::vector<double>& matrix,
    const std::vector<double>& rhs,
    int rows,
    int columns,
    std::vector<double>& solution
) {
    std::vector<double> q(matrix);
    std::vector<double> r(static_cast<size_t>(columns) * columns, 0.0);
    for (int column = 0; column < columns; ++column) {
        for (int previous = 0; previous < column; ++previous) {
            double dot = 0.0;
            for (int row = 0; row < rows; ++row) {
                dot += q[static_cast<size_t>(row) * columns + previous] *
                       q[static_cast<size_t>(row) * columns + column];
            }
            r[static_cast<size_t>(previous) * columns + column] = dot;
            for (int row = 0; row < rows; ++row) {
                q[static_cast<size_t>(row) * columns + column] -=
                    dot * q[static_cast<size_t>(row) * columns + previous];
            }
        }
        double norm2 = 0.0;
        for (int row = 0; row < rows; ++row) {
            const double value = q[static_cast<size_t>(row) * columns + column];
            norm2 += value * value;
        }
        const double norm = std::sqrt(norm2);
        if (!(norm > 1.0e-20)) return false;
        r[static_cast<size_t>(column) * columns + column] = norm;
        for (int row = 0; row < rows; ++row) q[static_cast<size_t>(row) * columns + column] /= norm;
    }
    std::vector<double> qtb(columns, 0.0);
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < rows; ++row) {
            qtb[column] += q[static_cast<size_t>(row) * columns + column] * rhs[row];
        }
    }
    solution.assign(columns, 0.0);
    for (int row = columns - 1; row >= 0; --row) {
        double value = qtb[row];
        for (int column = row + 1; column < columns; ++column) {
            value -= r[static_cast<size_t>(row) * columns + column] * solution[column];
        }
        solution[row] = value / r[static_cast<size_t>(row) * columns + row];
    }
    return true;
}

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
) {
    const int n_phi = config.flux_phi_count;
    const int n_levels = config.flux_level_count;
    const int n_theta = config.flux_theta_count;
    const int n_radial = config.flux_radial_quadrature;
    const int ray_count = n_phi * n_theta;
    const int field_points = n_phi * n_levels * n_theta * n_radial;
    std::vector<float> levels(n_levels);
    for (int index = 0; index < n_levels; ++index) {
        const double fraction = static_cast<double>(index + 1) / n_levels;
        levels[index] = static_cast<float>(edge_level * fraction * fraction);
    }
    std::vector<float> gauss_nodes, gauss_weights;
    gauss_legendre(n_radial, gauss_nodes, gauss_weights);
    DeviceBuffer<float> d_levels, d_gauss_nodes, d_gauss_weights;
    DeviceBuffer<float> d_ray_coefficients(static_cast<size_t>(ray_count) * (config.psi_poly_degree + 1));
    DeviceBuffer<float> d_radii(static_cast<size_t>(ray_count) * n_levels);
    DeviceBuffer<float> d_residuals(d_radii.size());
    DeviceBuffer<float> d_xyz(static_cast<size_t>(field_points) * 3);
    DeviceBuffer<float> d_B(d_xyz.size());
    DeviceBuffer<double> d_sections(static_cast<size_t>(n_phi) * n_levels);
    if (!copy_to_device(d_levels, levels) || !copy_to_device(d_gauss_nodes, gauss_nodes) ||
        !copy_to_device(d_gauss_weights, gauss_weights) || !d_ray_coefficients.data() ||
        !d_radii.data() || !d_residuals.data() || !d_xyz.data() || !d_B.data() || !d_sections.data()) {
        fail_result(&result, "flux device allocation failed");
        return false;
    }
    constexpr int threads = 256;
    build_ray_coefficients_kernel<<<(ray_count + threads - 1) / threads, threads>>>(
        device_psi.coeffs.data(), device_psi.mode_a.data(), device_psi.mode_b.data(),
        device_psi.mode_m.data(), device_psi.mode_kind.data(), static_cast<int>(psi.coeffs.size()),
        nfp, config.psi_poly_degree, config.psi_m_tor, n_phi, n_theta, 0.0f, 0.0f, config.target_volume>0 ? 0.0f : 0.217f,
        d_ray_coefficients.data()
    );
    const int radius_count = ray_count * n_levels;
    solve_boundary_radii_kernel<<<(radius_count + threads - 1) / threads, threads>>>(
        d_ray_coefficients.data(), ray_count, config.psi_poly_degree,
        static_cast<float>(config.psi_a), d_levels.data(), n_levels,
        config.surface_newton_iters, static_cast<float>(config.surface_newton_tolerance),
        static_cast<float>(config.surface_max_radius_scale), d_radii.data(), d_residuals.data()
    );
    generate_flux_points_kernel<<<(field_points + threads - 1) / threads, threads>>>(
        d_radii.data(), d_gauss_nodes.data(), n_phi, n_levels, n_theta, n_radial, nfp,
        device_psi.axis_R.data(), device_psi.axis_Z.data(),
        device_psi.axis_R_phi.data(), device_psi.axis_Z_phi.data(),
        static_cast<int>(axis.R.size()), d_xyz.data()
    );
    if (!cuda_stage_ok(cudaGetLastError(), result, "flux geometry kernels") ||
        sgpu_internal_eval_B_f32_device(field, d_xyz.data(), d_B.data(), field_points)) {
        if (result.status != SGPU_SCORE_INTERNAL_ERROR) fail_from_backend(&result, "flux field evaluation");
        return false;
    }
    reduce_flux_sections_kernel<<<n_phi * n_levels, threads, threads * sizeof(double)>>>(
        d_B.data(), d_radii.data(), d_gauss_nodes.data(), d_gauss_weights.data(),
        n_phi, n_levels, n_theta, n_radial, nfp, d_sections.data()
    );
    if (!cuda_stage_ok(cudaDeviceSynchronize(), result, "flux reduction")) return false;

    std::vector<float> radii(d_radii.size()), residuals(d_residuals.size());
    std::vector<double> sections(d_sections.size());
    if (!cuda_stage_ok(cudaMemcpy(radii.data(), d_radii.data(), radii.size() * sizeof(float), cudaMemcpyDeviceToHost), result, "flux copy radii") ||
        !cuda_stage_ok(cudaMemcpy(residuals.data(), d_residuals.data(), residuals.size() * sizeof(float), cudaMemcpyDeviceToHost), result, "flux copy residuals") ||
        !cuda_stage_ok(cudaMemcpy(sections.data(), d_sections.data(), sections.size() * sizeof(double), cudaMemcpyDeviceToHost), result, "flux copy sections")) {
        return false;
    }
    calibration.boundary_residual_max = *std::max_element(residuals.begin(), residuals.end());
    std::vector<double> mean(n_levels, 0.0), section_std(n_levels, 0.0);
    for (int level = 0; level < n_levels; ++level) {
        for (int phi_index = 0; phi_index < n_phi; ++phi_index) {
            mean[level] += sections[static_cast<size_t>(phi_index) * n_levels + level] / n_phi;
        }
        for (int phi_index = 0; phi_index < n_phi; ++phi_index) {
            const double delta = sections[static_cast<size_t>(phi_index) * n_levels + level] - mean[level];
            section_std[level] += delta * delta / n_phi;
        }
        section_std[level] = std::sqrt(section_std[level]);
    }
    calibration.section_relative_std_edge = section_std.back() / std::max(std::abs(mean.back()), 1.0e-30);
    const int polynomial_degree = config.flux_polynomial_degree;
    std::vector<double> matrix(static_cast<size_t>(n_levels) * polynomial_degree);
    for (int row = 0; row < n_levels; ++row) {
        double power = levels[row];
        for (int column = 0; column < polynomial_degree; ++column) {
            matrix[static_cast<size_t>(row) * polynomial_degree + column] = power;
            power *= levels[row];
        }
    }
    std::vector<double> coefficients;
    if (!solve_small_least_squares(matrix, mean, n_levels, polynomial_degree, coefficients)) {
        fail_result(&result, "flux calibration linear least squares failed");
        return false;
    }
    calibration.coefficients.assign(coefficients.begin(), coefficients.end());
    double fit_error2 = 0.0;
    calibration.derivative_min = std::numeric_limits<double>::infinity();
    calibration.derivative_max = -std::numeric_limits<double>::infinity();
    for (int row = 0; row < n_levels; ++row) {
        double fitted = 0.0;
        double power = levels[row];
        for (int column = 0; column < polynomial_degree; ++column) {
            fitted += coefficients[column] * power;
            power *= levels[row];
        }
        const double delta = fitted - mean[row];
        fit_error2 += delta * delta / n_levels;
    }
    for (int index = 0; index <= 200; ++index) {
        const double level = edge_level * index / 200.0;
        double derivative = 0.0;
        double power = 1.0;
        for (int column = 0; column < polynomial_degree; ++column) {
            derivative += (column + 1) * coefficients[column] * power;
            power *= level;
        }
        calibration.derivative_min = std::min(calibration.derivative_min, derivative);
        calibration.derivative_max = std::max(calibration.derivative_max, derivative);
    }
    calibration.edge_level = edge_level;
    calibration.edge_flux = 0.0;
    double edge_power = edge_level;
    for (double coefficient : coefficients) {
        calibration.edge_flux += coefficient * edge_power;
        edge_power *= edge_level;
    }
    calibration.fit_relative_rms = std::sqrt(fit_error2) / std::max(std::abs(mean.back()), 1.0e-30);
    double volume_integrand_sum = 0.0;
    double major_radius_sum = 0.0;
    for (int phi_index = 0; phi_index < n_phi; ++phi_index) {
        const double phi = TWOPI * phi_index / static_cast<double>(nfp * n_phi);
        double axis_R, axis_R_phi, axis_Z, axis_Z_phi;
        periodic_hermite_host(phi, axis.R, axis.R_phi, nfp, axis_R, axis_R_phi);
        periodic_hermite_host(phi, axis.Z, axis.Z_phi, nfp, axis_Z, axis_Z_phi);
        major_radius_sum += axis_R / n_phi;
        for (int theta_index = 0; theta_index < n_theta; ++theta_index) {
            const size_t ray = static_cast<size_t>(phi_index) * n_theta + theta_index;
            const double radius = radii[static_cast<size_t>(n_levels - 1) * ray_count + ray];
            const double theta = TWOPI * theta_index / n_theta;
            volume_integrand_sum += (0.5 * axis_R * radius * radius +
                                     std::cos(theta) * radius * radius * radius / 3.0) /
                                    static_cast<double>(n_phi * n_theta);
        }
    }
    calibration.volume = TWOPI * TWOPI * volume_integrand_sum;
    calibration.major_radius = major_radius_sum;
    calibration.effective_minor_radius = std::sqrt(
        std::max(calibration.volume, 0.0) /
        std::max(2.0 * PI * PI * calibration.major_radius, 1.0e-30)
    );
    const bool monotone = calibration.edge_flux >= 0.0
        ? calibration.derivative_min > 0.0
        : calibration.derivative_max < 0.0;
    return calibration.boundary_residual_max <= config.flux_boundary_tolerance &&
           (config.target_volume>0 || calibration.section_relative_std_edge <= config.flux_section_relative_std_tolerance) &&
           monotone;
}


}  // namespace coil_eval::pipeline
