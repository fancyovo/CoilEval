// QS values and the point/field adjoint kernels used by the retained derivative ABI.
#include "pipeline/types.cuh"
#include "pipeline/device_helpers.cuh"
#include "pipeline/common.cuh"
#include "pipeline/qs.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include "pipeline/alpha.cuh"
#include "pipeline/volume.cuh"

namespace coil_eval::pipeline {

__global__ void compute_qs_metric_kernel(
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
    float* __restrict__ absolute_normalized,
    double* __restrict__ sums
) {
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= point_count) return;
    float magnitude2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        const float value = B[3 * point + component];
        magnitude2 += value * value;
    }
    const float magnitude = sqrtf(fmaxf(magnitude2, 1.0e-30f));
    float grad_magnitude[3];
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        float sum = 0.0f;
        for (int component = 0; component < 3; ++component) {
            sum += grad_B[9 * point + 3 * component + coordinate] * B[3 * point + component];
        }
        grad_magnitude[coordinate] = sum / magnitude;
    }
    float grad_psi[3];
    for (int component = 0; component < 3; ++component) {
        grad_psi[component] = flux_derivative[point] * grad_s[3 * point + component];
    }
    const float cross_x = B[3 * point + 1] * grad_psi[2] - B[3 * point + 2] * grad_psi[1];
    const float cross_y = B[3 * point + 2] * grad_psi[0] - B[3 * point] * grad_psi[2];
    const float cross_z = B[3 * point] * grad_psi[1] - B[3 * point + 1] * grad_psi[0];
    const float A = cross_x * grad_magnitude[0] + cross_y * grad_magnitude[1] +
                    cross_z * grad_magnitude[2];
    const float C = B[3 * point] * grad_magnitude[0] + B[3 * point + 1] * grad_magnitude[1] +
                    B[3 * point + 2] * grad_magnitude[2];
    const float u = rho[point] * rho[point];
    float iota = 0.0f;
    float power = 1.0f;
    for (int degree = 0; degree <= iota_degree; ++degree) {
        iota += iota_coefficients[degree] * power;
        power *= u;
    }
    const double f_c = (helicity_M * static_cast<double>(iota) - helicity_N) * A -
                       helicity_M * G * C;
    const double f_c_qa = static_cast<double>(iota) * A - G * C;
    const double f_c_qp = -static_cast<double>(nfp) * A;
    const double normalized = f_c /
        fmax(static_cast<double>(magnitude) * magnitude * magnitude, 1.0e-30);
    const double normalized_qa = f_c_qa /
        fmax(static_cast<double>(magnitude) * magnitude * magnitude, 1.0e-30);
    const double normalized_qp = f_c_qp /
        fmax(static_cast<double>(magnitude) * magnitude * magnitude, 1.0e-30);
    const double weight = volume_weight[point];
    absolute_normalized[point] = static_cast<float>(fabs(normalized));
    atomic_add_double(sums, weight);
    atomic_add_double(sums + 1, weight * normalized * normalized);
    atomic_add_double(sums + 4, weight * weight);
    atomic_add_double(sums + 7, weight * normalized_qa * normalized_qa);
    atomic_add_double(sums + 8, weight * normalized_qp * normalized_qp);
    if (rho[point] >= edge_rho_threshold) {
        atomic_add_double(sums + 2, weight);
        atomic_add_double(sums + 3, weight * normalized * normalized);
        atomic_add_double(sums + 5, weight * weight);
        atomic_add_double(sums + 6, 1.0);
    }
}

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
) {
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= point_count) return;
    float b[3];
    float magnitude2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        b[component] = B[3 * point + component];
        magnitude2 += b[component] * b[component];
    }
    const float magnitude = sqrtf(fmaxf(magnitude2, 1.0e-30f));
    float q[3] = {};
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        for (int component = 0; component < 3; ++component) {
            q[coordinate] += grad_B[9 * point + 3 * component + coordinate] * b[component];
        }
        q[coordinate] /= magnitude;
    }
    float p[3];
    for (int component = 0; component < 3; ++component) {
        p[component] = flux_derivative[point] * grad_s[3 * point + component];
    }
    const float cross[3] = {
        b[1] * p[2] - b[2] * p[1],
        b[2] * p[0] - b[0] * p[2],
        b[0] * p[1] - b[1] * p[0],
    };
    const float A = cross[0] * q[0] + cross[1] * q[1] + cross[2] * q[2];
    const float C = b[0] * q[0] + b[1] * q[1] + b[2] * q[2];
    const float u = rho[point] * rho[point];
    float iota = 0.0f;
    float power = 1.0f;
    for (int degree = 0; degree <= iota_degree; ++degree) {
        iota += iota_coefficients[degree] * power;
        power *= u;
    }
    const double k_target = helicity_M * static_cast<double>(iota) - helicity_N;
    const double f_target = k_target * A - helicity_M * G * C;
    const double f_qa = static_cast<double>(iota) * A - G * C;
    const double f_qp = -static_cast<double>(nfp) * A;
    const double magnitude3 = fmax(static_cast<double>(magnitude) * magnitude * magnitude, 1.0e-30);
    const double r_target = f_target / magnitude3;
    const double r_qa = f_qa / magnitude3;
    const double r_qp = f_qp / magnitude3;
    const double weight = volume_weight[point];
    const double target_norm = fmax(
        hypot(static_cast<double>(helicity_M), static_cast<double>(helicity_N)), 1.0
    );
    const double qp_norm = nfp > 1 ? static_cast<double>(nfp) : 1.0;
    double lambda_target = 0.0;
    if (target_rms > 1.0e-30 && weight_sum > 0.0) {
        lambda_target += d_target_error * weight * r_target /
            (weight_sum * target_rms * target_norm);
    }
    if (rho[point] >= edge_rho_threshold && target_edge_rms > 1.0e-30 && edge_weight_sum > 0.0) {
        lambda_target += d_target_edge_error * weight * r_target /
            (edge_weight_sum * target_edge_rms * target_norm);
    }
    const double lambda_qa = qa_rms > 1.0e-30 && weight_sum > 0.0
        ? d_qa_error * weight * r_qa / (weight_sum * qa_rms)
        : 0.0;
    const double lambda_qp = qp_rms > 1.0e-30 && weight_sum > 0.0
        ? d_qp_error * weight * r_qp / (weight_sum * qp_rms * qp_norm)
        : 0.0;

    const double adj_A =
        (lambda_target * k_target + lambda_qa * iota - lambda_qp * nfp) / magnitude3;
    const double adj_C = (-lambda_target * helicity_M * G - lambda_qa * G) / magnitude3;
    double adj_magnitude = -3.0 * (
        lambda_target * f_target + lambda_qa * f_qa + lambda_qp * f_qp
    ) / (magnitude3 * magnitude);
    const double point_adj_G =
        (-lambda_target * helicity_M * C - lambda_qa * C) / magnitude3;
    double adj_b[3] = {
        adj_A * (p[1] * q[2] - p[2] * q[1]) + adj_C * q[0],
        adj_A * (p[2] * q[0] - p[0] * q[2]) + adj_C * q[1],
        adj_A * (p[0] * q[1] - p[1] * q[0]) + adj_C * q[2],
    };
    const double adj_q[3] = {
        adj_A * cross[0] + adj_C * b[0],
        adj_A * cross[1] + adj_C * b[1],
        adj_A * cross[2] + adj_C * b[2],
    };
    double adj_y[3];
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        adj_y[coordinate] = adj_q[coordinate] / magnitude;
        adj_magnitude -= adj_q[coordinate] * q[coordinate] / magnitude;
    }
    for (int component = 0; component < 3; ++component) {
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            adj_b[component] +=
                grad_B[9 * point + 3 * component + coordinate] * adj_y[coordinate];
            adj_grad_B[9 * point + 3 * component + coordinate] =
                static_cast<float>(b[component] * adj_y[coordinate]);
        }
        adj_b[component] += adj_magnitude * b[component] / magnitude;
        adj_B[3 * point + component] = static_cast<float>(adj_b[component]);
    }
    atomic_add_double(adj_G, point_adj_G);
}

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
) {
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= point_count) return;
    float b[3];
    float magnitude2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        b[component] = B[3 * point + component];
        magnitude2 += b[component] * b[component];
    }
    const float magnitude = sqrtf(fmaxf(magnitude2, 1.0e-30f));
    float q[3] = {};
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        for (int component = 0; component < 3; ++component) {
            q[coordinate] += grad_B[9 * point + 3 * component + coordinate] * b[component];
        }
        q[coordinate] /= magnitude;
    }
    float grad_psi[3];
    for (int component = 0; component < 3; ++component) {
        grad_psi[component] = flux_derivative[point] * grad_s[3 * point + component];
    }
    const float cross[3] = {
        b[1] * grad_psi[2] - b[2] * grad_psi[1],
        b[2] * grad_psi[0] - b[0] * grad_psi[2],
        b[0] * grad_psi[1] - b[1] * grad_psi[0],
    };
    const double A = cross[0] * q[0] + cross[1] * q[1] + cross[2] * q[2];
    const double C = b[0] * q[0] + b[1] * q[1] + b[2] * q[2];
    const float u = rho[point] * rho[point];
    float iota = 0.0f;
    float power = 1.0f;
    for (int degree = 0; degree <= iota_degree; ++degree) {
        iota += iota_coefficients[degree] * power;
        power *= u;
    }
    const double magnitude3 = fmax(static_cast<double>(magnitude) * magnitude * magnitude, 1.0e-30);
    const double target_residual =
        ((helicity_M * static_cast<double>(iota) - helicity_N) * A - helicity_M * G * C) /
        magnitude3;
    const double qa_residual = (static_cast<double>(iota) * A - G * C) / magnitude3;
    const double weight = volume_weight[point];
    const double target_norm = fmax(
        hypot(static_cast<double>(helicity_M), static_cast<double>(helicity_N)), 1.0
    );
    double lambda_target = 0.0;
    if (target_rms > 1.0e-30 && weight_sum > 0.0) {
        lambda_target += d_target_error * weight * target_residual /
            (weight_sum * target_rms * target_norm);
    }
    if (rho[point] >= edge_rho_threshold && target_edge_rms > 1.0e-30 && edge_weight_sum > 0.0) {
        lambda_target += d_target_edge_error * weight * target_residual /
            (edge_weight_sum * target_edge_rms * target_norm);
    }
    const double lambda_qa = qa_rms > 1.0e-30 && weight_sum > 0.0
        ? d_qa_error * weight * qa_residual / (weight_sum * qa_rms)
        : 0.0;
    const double adj_iota_value = A / magnitude3 *
        (helicity_M * lambda_target + lambda_qa);
    double radial_power = 1.0;
    for (int degree = 0; degree <= iota_degree; ++degree) {
        atomic_add_double(adj_iota + degree, adj_iota_value * radial_power);
        radial_power *= u;
    }
}

__global__ void alpha_residual_adjoint_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ rhs,
    int count,
    float residual_norm,
    float rhs_norm,
    float adj_relative,
    float* __restrict__ adj_residual,
    float* __restrict__ adj_rhs
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    const float safe_residual = fmaxf(residual_norm, 1.0e-20f);
    const float safe_rhs = fmaxf(rhs_norm, 1.0e-20f);
    const float value = adj_relative * residual[row] / (safe_residual * safe_rhs);
    adj_residual[row] = value;
    adj_rhs[row] = -value - adj_relative * residual_norm * rhs[row] /
        (safe_rhs * safe_rhs * safe_rhs);
}

__global__ void add_alpha_solution_adjoint_kernel(
    const float* __restrict__ adj_solution,
    const float* __restrict__ scales,
    int count,
    float* __restrict__ adj_scaled_solution
) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column < count) {
        adj_scaled_solution[column] += adj_solution[column] / scales[column];
    }
}

__global__ void alpha_column_dot_kernel(
    const float* __restrict__ scaled_matrix,
    int row_count,
    const float* __restrict__ adj_residual,
    const float* __restrict__ residual,
    const float* __restrict__ multiplier,
    const float* __restrict__ matrix_multiplier,
    const float* __restrict__ scaled_solution,
    float* __restrict__ dot_adjoint_matrix
) {
    extern __shared__ double shared[];
    const int column = blockIdx.x;
    double sum = 0.0;
    const float lambda = multiplier[column];
    const float solution = scaled_solution[column];
    for (int row = threadIdx.x; row < row_count; row += blockDim.x) {
        const float matrix_value = scaled_matrix[static_cast<size_t>(column) * row_count + row];
        const float adjoint = adj_residual[row] * solution - residual[row] * lambda -
            matrix_multiplier[row] * solution;
        sum += static_cast<double>(adjoint) * matrix_value;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) dot_adjoint_matrix[column] = static_cast<float>(shared[0]);
}

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
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    const float bt = b_theta[row];
    const float bp = b_phi[row];
    const float weight = weights[row];
    float weight_adjoint = -adj_rhs[row] * bt;
    float theta_adjoint = -adj_rhs[row] * weight;
    float phi_adjoint = 0.0f;
    for (int column = 0; column < column_count; ++column) {
        const float scale = scales[column];
        const float matrix_value = scaled_matrix[static_cast<size_t>(column) * row_count + row];
        const float matrix_adjoint =
            adj_residual[row] * scaled_solution[column] -
            residual[row] * multiplier[column] -
            matrix_multiplier[row] * scaled_solution[column];
        const float scale_adjoint =
            -adj_solution[column] * scaled_solution[column] / (scale * scale) -
            dot_adjoint_matrix[column] / scale;
        const float design_adjoint = matrix_adjoint / scale + matrix_value * scale_adjoint;
        if (column < mode_count) {
            const int m = mode_m[column];
            const int n = mode_n[column];
            const float radial = evaluate_radial_polynomial(
                radial_coefficients + static_cast<size_t>(column) * (radial_order + 1),
                radial_order,
                rho[row], radial_basis
            );
            float sine, cosine;
            sincosf(m * theta[row] - n * nfp * phi[row], &sine, &cosine);
            const float derivative_theta = mode_kind[column] == 0
                ? -m * radial * sine : m * radial * cosine;
            const float derivative_phi = mode_kind[column] == 0
                ? n * nfp * radial * sine : -n * nfp * radial * cosine;
            weight_adjoint += design_adjoint * (derivative_theta * bt + derivative_phi * bp);
            theta_adjoint += design_adjoint * weight * derivative_theta;
            phi_adjoint += design_adjoint * weight * derivative_phi;
        } else {
            const int power_index = column - mode_count;
            if (power_index <= iota_degree) {
                float radial_power = 1.0f;
                const float u = rho[row] * rho[row];
                for (int degree = 0; degree < power_index; ++degree) radial_power *= u;
                weight_adjoint -= design_adjoint * radial_power * bp;
                phi_adjoint -= design_adjoint * weight * radial_power;
            }
        }
    }
    adj_weight[row] = weight_adjoint;
    adj_b_theta[row] = theta_adjoint;
    adj_b_phi[row] = phi_adjoint;
}

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
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= fit_count) return;
    const int point = min(
        point_count - 1,
        static_cast<int>((static_cast<long long>(row) * point_count) / fit_count)
    );
    float b[3];
    float gs[3];
    float gt[3];
    float gp[3];
    float magnitude2 = 0.0f;
    float gs2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        b[component] = B[3 * point + component];
        gs[component] = grad_s[3 * point + component];
        gt[component] = grad_theta[3 * point + component];
        gp[component] = grad_phi[3 * point + component];
        magnitude2 += b[component] * b[component];
        gs2 += gs[component] * gs[component];
    }
    float projection[3];
    float projection_dot_gs = 0.0f;
    for (int component = 0; component < 3; ++component) {
        projection[component] =
            adj_b_theta[row] * gt[component] + adj_b_phi[row] * gp[component];
        projection_dot_gs += projection[component] * gs[component];
    }
    const float inverse_gs2 = 1.0f / fmaxf(gs2, 1.0e-30f);
    const float normalized_weight = weights[row];
    const float base_weight = normalized_weight / weight_scale;
    const float base_weight_adjoint = weight_scale * (
        adj_weight[row] - normalized_weight * adj_weight_dot_weight / fit_count
    );
    const float normal_coefficient =
        (b[0] * gs[0] + b[1] * gs[1] + b[2] * gs[2]) * inverse_gs2;
    for (int component = 0; component < 3; ++component) {
        float value = projection[component] - projection_dot_gs * inverse_gs2 * gs[component];
        value -= base_weight_adjoint * base_weight * b[component] /
            fmaxf(magnitude2, 1.0e-30f);
        if (adj_normal_relative != 0.0f && normal_relative > 1.0e-20f &&
            field_denominator > 1.0e-20f && normal_numerator >= 0.0f) {
            value += adj_normal_relative * (
                normal_coefficient * gs[component] /
                    (normal_relative * field_denominator) -
                normal_relative * b[component] / field_denominator
            );
        }
        adj_B[3 * point + component] = value;
    }
}

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
) {
    DeviceBuffer<float> d_iota;
    DeviceBuffer<float> d_absolute(points.count);
    DeviceBuffer<double> d_sums(9);
    if (!copy_to_device(d_iota, alpha.iota_coefficients) || !d_absolute.data() || !d_sums.data() ||
        !cuda_stage_ok(cudaMemset(d_sums.data(), 0, d_sums.size() * sizeof(double)), result, "QS reduction clear")) {
        if (result.status != SGPU_SCORE_INTERNAL_ERROR) fail_result(&result, "QS allocation failed");
        return false;
    }
    if (!std::isfinite(G)) {
        fail_result(&result, "QS metric requires finite vacuum G");
        return false;
    }
    const float edge_threshold = static_cast<float>(
        config.volume_rho_min + (1.0 - config.volume_rho_min) *
        (config.radial_bin_count - 1.0) / config.radial_bin_count
    );
    constexpr int threads = 256;
    compute_qs_metric_kernel<<<(points.count + threads - 1) / threads, threads>>>(
        d_B, d_grad_B, points.grad_s.data(), points.flux_derivative.data(),
        points.rho.data(), points.volume_weight.data(), points.count,
        d_iota.data(), config.iota_degree, config.target_M, config.target_N,
        nfp, G, edge_threshold, d_absolute.data(), d_sums.data()
    );
    if (!cuda_stage_ok(cudaDeviceSynchronize(), result, "QS metric kernel")) return false;
    thrust::device_ptr<float> begin(d_absolute.data());
    thrust::sort(begin, begin + points.count);
    const int percentile_index = std::min(points.count - 1, static_cast<int>(std::floor(0.95 * (points.count - 1))));
    float p95 = 0.0f;
    std::array<double, 9> sums{};
    if (!cuda_stage_ok(cudaMemcpy(&p95, d_absolute.data() + percentile_index, sizeof(float), cudaMemcpyDeviceToHost), result, "QS p95 copy") ||
        !cuda_stage_ok(cudaMemcpy(sums.data(), d_sums.data(), sizeof(sums), cudaMemcpyDeviceToHost), result, "QS sums copy")) {
        return false;
    }
    result.qs_global_error = std::sqrt(sums[1] / std::max(sums[0], 1.0e-300));
    result.qs_edge_error = std::sqrt(sums[3] / std::max(sums[2], 1.0e-300));
    result.qs_qa_global_error = std::sqrt(sums[7] / std::max(sums[0], 1.0e-300));
    result.qs_qp_global_error_raw = std::sqrt(sums[8] / std::max(sums[0], 1.0e-300));
    const double target_helicity_norm = std::max(
        std::hypot(static_cast<double>(config.target_M), static_cast<double>(config.target_N)),
        1.0
    );
    const double qp_helicity_norm = std::max(std::abs(nfp), 1);
    result.qs_vacuum_G = G;
    result.qs_target_global_error_per_helicity = result.qs_global_error / target_helicity_norm;
    result.qs_target_edge_error_per_helicity = result.qs_edge_error / target_helicity_norm;
    result.qs_qa_global_error_per_helicity = result.qs_qa_global_error;
    result.qs_qp_global_error_per_helicity = result.qs_qp_global_error_raw / qp_helicity_norm;
    // Preserve the ABI-8 field semantics while exposing both explicit forms above.
    result.qs_qp_global_error = result.qs_qp_global_error_per_helicity;
    result.volume_weight_effective_fraction =
        sums[0] * sums[0] / std::max(points.count * sums[4], 1.0e-300);
    result.edge_weight_effective_fraction =
        sums[2] * sums[2] / std::max(sums[6] * sums[5], 1.0e-300);
    result.qs_abs_p95 = p95;
    result.qs_abs_p95_per_helicity = p95 / target_helicity_norm;
    if (raw_sums) *raw_sums = sums;
    return true;
}


}  // namespace coil_eval::pipeline
