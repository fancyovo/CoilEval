#include "field/state.h"
#include "field/psi_design.cuh"
#include "coil_field.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace coil_eval::field {

__global__ void combine_psi_feature_kernel(
    float* __restrict__ output,
    const float* __restrict__ feature,
    const float* __restrict__ image,
    size_t count,
    int initialize
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = feature[index] * image[index];
        output[index] = initialize ? value : output[index] + value;
    }
}

__global__ void weight_psi_residual_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ feature,
    float* __restrict__ weighted,
    size_t count
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) weighted[index] = residual[index] * feature[index];
}

__global__ void initialize_psi_batch_coefficients_kernel(
    const float* __restrict__ center_coefficients,
    const float* __restrict__ scale,
    int coefficient_count,
    int query_count,
    float* __restrict__ scaled_coefficients
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(coefficient_count) * query_count;
    if (index < count) {
        const int coefficient = static_cast<int>(index % coefficient_count);
        scaled_coefficients[index] = center_coefficients[coefficient] * scale[coefficient];
    }
}

__global__ void unscale_psi_batch_coefficients_kernel(
    const float* __restrict__ scaled,
    const float* __restrict__ scale,
    int coefficient_count,
    int query_count,
    float* __restrict__ unscaled
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(coefficient_count) * query_count;
    if (index < count) {
        const int coefficient = static_cast<int>(index % coefficient_count);
        unscaled[index] = scaled[index] / scale[coefficient];
    }
}

__global__ void initialize_psi_residual_kernel(
    const float* __restrict__ rhs,
    const float* __restrict__ prediction,
    size_t count,
    float* __restrict__ residual
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) residual[index] = rhs[index] - prediction[index];
}

__global__ void initialize_psi_ridge_residual_kernel(
    const float* __restrict__ scaled_coefficients,
    float ridge_lambda,
    size_t count,
    float* __restrict__ ridge_residual
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) ridge_residual[index] = -ridge_lambda * scaled_coefficients[index];
}

__global__ void scale_psi_batch_normal_kernel(
    float* __restrict__ normal,
    const float* __restrict__ scale,
    const float* __restrict__ ridge_residual,
    float ridge_lambda,
    int coefficient_count,
    int query_count
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(coefficient_count) * query_count;
    if (index < count) {
        const int coefficient = static_cast<int>(index % coefficient_count);
        normal[index] = normal[index] / scale[coefficient] +
            ridge_lambda * ridge_residual[index];
    }
}

__global__ void psi_batch_column_norm_kernel(
    const float* __restrict__ values,
    int rows,
    int columns,
    float* __restrict__ norms
) {
    const int column = blockIdx.x;
    float sum = 0.0f;
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        const float value = values[static_cast<size_t>(column) * rows + row];
        sum += value * value;
    }
    extern __shared__ float reduction[];
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) reduction[threadIdx.x] += reduction[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0 && column < columns) norms[column] = reduction[0];
}

__global__ void psi_batch_delta_kernel(
    const float* __restrict__ image,
    int point_count,
    const float* __restrict__ ridge_image,
    int coefficient_count,
    int query_count,
    float* __restrict__ delta
) {
    const int query = blockIdx.x;
    float sum = 0.0f;
    for (int row = threadIdx.x; row < point_count; row += blockDim.x) {
        const float value = image[static_cast<size_t>(query) * point_count + row];
        sum += value * value;
    }
    for (int coefficient = threadIdx.x; coefficient < coefficient_count; coefficient += blockDim.x) {
        const float value = ridge_image[static_cast<size_t>(query) * coefficient_count + coefficient];
        sum += value * value;
    }
    extern __shared__ float reduction[];
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) reduction[threadIdx.x] += reduction[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0 && query < query_count) delta[query] = reduction[0];
}

__global__ void psi_batch_update_kernel(
    float* __restrict__ scaled_coefficients,
    float* __restrict__ residual,
    float* __restrict__ ridge_residual,
    const float* __restrict__ transformed_direction,
    const float* __restrict__ image,
    const float* __restrict__ ridge_image,
    const float* __restrict__ gamma,
    const float* __restrict__ delta,
    int coefficient_count,
    int point_count,
    int query_count
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t point_total = static_cast<size_t>(point_count) * query_count;
    const size_t coefficient_total = static_cast<size_t>(coefficient_count) * query_count;
    if (index < point_total) {
        const int query = static_cast<int>(index / point_count);
        const float alpha = gamma[query] / fmaxf(delta[query], 1.0e-30f);
        residual[index] -= alpha * image[index];
    }
    if (index < coefficient_total) {
        const int query = static_cast<int>(index / coefficient_count);
        const float alpha = gamma[query] / fmaxf(delta[query], 1.0e-30f);
        scaled_coefficients[index] += alpha * transformed_direction[index];
        ridge_residual[index] -= alpha * ridge_image[index];
    }
}

__global__ void psi_batch_direction_update_kernel(
    float* __restrict__ direction,
    const float* __restrict__ normal,
    const float* __restrict__ gamma_old,
    const float* __restrict__ gamma_new,
    int coefficient_count,
    int query_count
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(coefficient_count) * query_count;
    if (index < count) {
        const int query = static_cast<int>(index / coefficient_count);
        const float beta = gamma_new[query] / fmaxf(gamma_old[query], 1.0e-30f);
        direction[index] = normal[index] + beta * direction[index];
    }
}

__global__ void psi_batch_ridge_image_kernel(
    const float* __restrict__ direction,
    float ridge_lambda,
    size_t count,
    float* __restrict__ image
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) image[index] = ridge_lambda * direction[index];
}

} // namespace coil_eval::field

// The public ABI stays global and unmangled; implementation state is named.
using namespace coil_eval::field;

extern "C" {

int sgpu_fit_psi_batch_pcgls_f32(
    void* batch_field_handle,
    const double* axis_R_host,
    const double* axis_Z_host,
    const double* axis_R_phi_host,
    const double* axis_Z_phi_host,
    int axis_count,
    const int* mode_a_host,
    const int* mode_b_host,
    const int* mode_m_host,
    const int* mode_kind_host,
    int coefficient_count,
    int nfp,
    double radius_scale,
    int radial_grid,
    int vertical_grid,
    int phi_grid,
    double rho_min,
    double ridge,
    int iterations,
    const double* center_coefficients_host,
    double* coefficients_host,
    double* train_rms_host,
    double* stats_out,
    int stats_len
) {
    using clock = std::chrono::steady_clock;
    const auto total_started = clock::now();
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(batch_field_handle);
    if (!field || !axis_R_host || !axis_Z_host || !axis_R_phi_host || !axis_Z_phi_host ||
        axis_count <= 1 || !mode_a_host || !mode_b_host || !mode_m_host || !mode_kind_host ||
        coefficient_count <= 0 || nfp <= 0 || !(radius_scale > 0.0) || radial_grid <= 1 ||
        vertical_grid <= 1 || phi_grid <= 0 || rho_min < 0.0 || ridge < 0.0 ||
        iterations < 0 || !center_coefficients_host || !coefficients_host || !train_rms_host) {
        set_error("invalid batch psi PCGLS arguments");
        return 1;
    }
    if (!sgpu_has_psi_warm_preconditioner(coefficient_count)) {
        set_error("batch psi PCGLS requires a captured center QR preconditioner");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "batch psi cudaSetDevice")) return 1;
    const int query_count = field->query_count;
    std::vector<float> normalized_R;
    std::vector<float> normalized_Z;
    std::vector<float> phi;
    normalized_R.reserve(static_cast<size_t>(radial_grid) * vertical_grid * phi_grid);
    normalized_Z.reserve(normalized_R.capacity());
    phi.reserve(normalized_R.capacity());
    for (int radial = 0; radial < radial_grid; ++radial) {
        const double dR = -radius_scale + 2.0 * radius_scale * radial /
            static_cast<double>(radial_grid - 1);
        for (int vertical = 0; vertical < vertical_grid; ++vertical) {
            const double dZ = -radius_scale + 2.0 * radius_scale * vertical /
                static_cast<double>(vertical_grid - 1);
            const double radius = std::hypot(dR, dZ);
            if (radius < rho_min || radius > radius_scale) continue;
            for (int toroidal = 0; toroidal < phi_grid; ++toroidal) {
                normalized_R.push_back(static_cast<float>(dR / radius_scale));
                normalized_Z.push_back(static_cast<float>(dZ / radius_scale));
                phi.push_back(static_cast<float>(
                    TWOPI * toroidal / static_cast<double>(nfp * phi_grid)
                ));
            }
        }
    }
    const int point_count = static_cast<int>(normalized_R.size());
    if (point_count <= 0) {
        set_error("batch psi grid contains no points");
        return 1;
    }
    const size_t point_query_count = static_cast<size_t>(point_count) * query_count;
    const size_t coefficient_query_count = static_cast<size_t>(coefficient_count) * query_count;
    const size_t basis_count = static_cast<size_t>(point_count) * coefficient_count;
    const size_t axis_query_count = static_cast<size_t>(axis_count) * query_count;
    std::vector<float> axis_R(axis_query_count), axis_Z(axis_query_count);
    std::vector<float> axis_R_phi(axis_query_count), axis_Z_phi(axis_query_count);
    for (size_t index = 0; index < axis_query_count; ++index) {
        axis_R[index] = static_cast<float>(axis_R_host[index]);
        axis_Z[index] = static_cast<float>(axis_Z_host[index]);
        axis_R_phi[index] = static_cast<float>(axis_R_phi_host[index]);
        axis_Z_phi[index] = static_cast<float>(axis_Z_phi_host[index]);
    }
    std::vector<float> center_coefficients(coefficient_count);
    std::vector<float> center_scale(coefficient_count);
    for (int coefficient = 0; coefficient < coefficient_count; ++coefficient) {
        center_coefficients[coefficient] = static_cast<float>(center_coefficients_host[coefficient]);
        center_scale[coefficient] = static_cast<float>(
            g_psi_warm_preconditioner.center_scale[static_cast<size_t>(coefficient)]
        );
    }

    float *d_normalized_R = nullptr, *d_normalized_Z = nullptr, *d_phi = nullptr;
    float *d_axis_R = nullptr, *d_axis_Z = nullptr, *d_axis_R_phi = nullptr, *d_axis_Z_phi = nullptr;
    int *d_mode_a = nullptr, *d_mode_b = nullptr, *d_mode_m = nullptr, *d_mode_kind = nullptr;
    float *d_basis_R = nullptr, *d_basis_Z = nullptr, *d_basis_phi = nullptr;
    float *d_feature_R = nullptr, *d_feature_Z = nullptr, *d_feature_phi = nullptr, *d_rhs = nullptr;
    float *d_center_coefficients = nullptr, *d_scale = nullptr, *d_center_factor = nullptr;
    float *d_scaled_coefficients = nullptr, *d_unscaled = nullptr, *d_residual = nullptr;
    float *d_ridge_residual = nullptr, *d_normal = nullptr, *d_direction = nullptr;
    float *d_transformed_direction = nullptr, *d_image = nullptr, *d_temporary = nullptr;
    float *d_weighted = nullptr, *d_ridge_image = nullptr, *d_gamma = nullptr;
    float *d_gamma_new = nullptr, *d_delta = nullptr;
    cublasHandle_t blas = nullptr;
    auto cleanup = [&] {
        if (blas) cublasDestroy(blas);
        cudaFree(d_normalized_R); cudaFree(d_normalized_Z); cudaFree(d_phi);
        cudaFree(d_axis_R); cudaFree(d_axis_Z); cudaFree(d_axis_R_phi); cudaFree(d_axis_Z_phi);
        cudaFree(d_mode_a); cudaFree(d_mode_b); cudaFree(d_mode_m); cudaFree(d_mode_kind);
        cudaFree(d_basis_R); cudaFree(d_basis_Z); cudaFree(d_basis_phi);
        cudaFree(d_feature_R); cudaFree(d_feature_Z); cudaFree(d_feature_phi); cudaFree(d_rhs);
        cudaFree(d_center_coefficients); cudaFree(d_scale); cudaFree(d_center_factor);
        cudaFree(d_scaled_coefficients); cudaFree(d_unscaled); cudaFree(d_residual);
        cudaFree(d_ridge_residual); cudaFree(d_normal); cudaFree(d_direction);
        cudaFree(d_transformed_direction); cudaFree(d_image); cudaFree(d_temporary);
        cudaFree(d_weighted); cudaFree(d_ridge_image); cudaFree(d_gamma);
        cudaFree(d_gamma_new); cudaFree(d_delta);
    };
    auto allocate = [&](void** pointer, size_t bytes, const char* label) {
        if (cuda_check(cudaMalloc(pointer, bytes), label)) return false;
        return true;
    };
    const size_t common_point_bytes = static_cast<size_t>(point_count) * sizeof(float);
    const size_t axis_bytes = axis_query_count * sizeof(float);
    const size_t mode_bytes = static_cast<size_t>(coefficient_count) * sizeof(int);
    const size_t basis_bytes = basis_count * sizeof(float);
    const size_t point_query_bytes = point_query_count * sizeof(float);
    const size_t coefficient_query_bytes = coefficient_query_count * sizeof(float);
    const size_t coefficient_bytes = static_cast<size_t>(coefficient_count) * sizeof(float);
    const size_t factor_bytes = static_cast<size_t>(coefficient_count) * coefficient_count * sizeof(float);
    if (!allocate(reinterpret_cast<void**>(&d_normalized_R), common_point_bytes, "batch psi normalized R") ||
        !allocate(reinterpret_cast<void**>(&d_normalized_Z), common_point_bytes, "batch psi normalized Z") ||
        !allocate(reinterpret_cast<void**>(&d_phi), common_point_bytes, "batch psi phi") ||
        !allocate(reinterpret_cast<void**>(&d_axis_R), axis_bytes, "batch psi axis R") ||
        !allocate(reinterpret_cast<void**>(&d_axis_Z), axis_bytes, "batch psi axis Z") ||
        !allocate(reinterpret_cast<void**>(&d_axis_R_phi), axis_bytes, "batch psi axis R phi") ||
        !allocate(reinterpret_cast<void**>(&d_axis_Z_phi), axis_bytes, "batch psi axis Z phi") ||
        !allocate(reinterpret_cast<void**>(&d_mode_a), mode_bytes, "batch psi mode a") ||
        !allocate(reinterpret_cast<void**>(&d_mode_b), mode_bytes, "batch psi mode b") ||
        !allocate(reinterpret_cast<void**>(&d_mode_m), mode_bytes, "batch psi mode m") ||
        !allocate(reinterpret_cast<void**>(&d_mode_kind), mode_bytes, "batch psi mode kind") ||
        !allocate(reinterpret_cast<void**>(&d_basis_R), basis_bytes, "batch psi basis R") ||
        !allocate(reinterpret_cast<void**>(&d_basis_Z), basis_bytes, "batch psi basis Z") ||
        !allocate(reinterpret_cast<void**>(&d_basis_phi), basis_bytes, "batch psi basis phi") ||
        !allocate(reinterpret_cast<void**>(&d_feature_R), point_query_bytes, "batch psi feature R") ||
        !allocate(reinterpret_cast<void**>(&d_feature_Z), point_query_bytes, "batch psi feature Z") ||
        !allocate(reinterpret_cast<void**>(&d_feature_phi), point_query_bytes, "batch psi feature phi") ||
        !allocate(reinterpret_cast<void**>(&d_rhs), point_query_bytes, "batch psi rhs") ||
        !allocate(reinterpret_cast<void**>(&d_center_coefficients), coefficient_bytes, "batch psi center coeff") ||
        !allocate(reinterpret_cast<void**>(&d_scale), coefficient_bytes, "batch psi scale") ||
        !allocate(reinterpret_cast<void**>(&d_center_factor), factor_bytes, "batch psi center factor") ||
        !allocate(reinterpret_cast<void**>(&d_scaled_coefficients), coefficient_query_bytes, "batch psi coefficients") ||
        !allocate(reinterpret_cast<void**>(&d_unscaled), coefficient_query_bytes, "batch psi unscaled") ||
        !allocate(reinterpret_cast<void**>(&d_residual), point_query_bytes, "batch psi residual") ||
        !allocate(reinterpret_cast<void**>(&d_ridge_residual), coefficient_query_bytes, "batch psi ridge residual") ||
        !allocate(reinterpret_cast<void**>(&d_normal), coefficient_query_bytes, "batch psi normal") ||
        !allocate(reinterpret_cast<void**>(&d_direction), coefficient_query_bytes, "batch psi direction") ||
        !allocate(reinterpret_cast<void**>(&d_transformed_direction), coefficient_query_bytes, "batch psi transformed direction") ||
        !allocate(reinterpret_cast<void**>(&d_image), point_query_bytes, "batch psi image") ||
        !allocate(reinterpret_cast<void**>(&d_temporary), point_query_bytes, "batch psi temporary") ||
        !allocate(reinterpret_cast<void**>(&d_weighted), point_query_bytes, "batch psi weighted") ||
        !allocate(reinterpret_cast<void**>(&d_ridge_image), coefficient_query_bytes, "batch psi ridge image") ||
        !allocate(reinterpret_cast<void**>(&d_gamma), static_cast<size_t>(query_count) * sizeof(float), "batch psi gamma") ||
        !allocate(reinterpret_cast<void**>(&d_gamma_new), static_cast<size_t>(query_count) * sizeof(float), "batch psi gamma new") ||
        !allocate(reinterpret_cast<void**>(&d_delta), static_cast<size_t>(query_count) * sizeof(float), "batch psi delta")) {
        cleanup();
        return 1;
    }
    if (cublas_check(cublasCreate(&blas), "batch psi cublasCreate")) {
        cleanup();
        return 1;
    }
    cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH);
    auto copy_host = [&](void* target, const void* source, size_t bytes, const char* label) {
        return cuda_check(cudaMemcpy(target, source, bytes, cudaMemcpyHostToDevice), label) == 0;
    };
    if (!copy_host(d_normalized_R, normalized_R.data(), common_point_bytes, "batch psi copy normalized R") ||
        !copy_host(d_normalized_Z, normalized_Z.data(), common_point_bytes, "batch psi copy normalized Z") ||
        !copy_host(d_phi, phi.data(), common_point_bytes, "batch psi copy phi") ||
        !copy_host(d_axis_R, axis_R.data(), axis_bytes, "batch psi copy axis R") ||
        !copy_host(d_axis_Z, axis_Z.data(), axis_bytes, "batch psi copy axis Z") ||
        !copy_host(d_axis_R_phi, axis_R_phi.data(), axis_bytes, "batch psi copy axis R phi") ||
        !copy_host(d_axis_Z_phi, axis_Z_phi.data(), axis_bytes, "batch psi copy axis Z phi") ||
        !copy_host(d_mode_a, mode_a_host, mode_bytes, "batch psi copy mode a") ||
        !copy_host(d_mode_b, mode_b_host, mode_bytes, "batch psi copy mode b") ||
        !copy_host(d_mode_m, mode_m_host, mode_bytes, "batch psi copy mode m") ||
        !copy_host(d_mode_kind, mode_kind_host, mode_bytes, "batch psi copy mode kind") ||
        !copy_host(d_center_coefficients, center_coefficients.data(), coefficient_bytes, "batch psi copy center coeff") ||
        !copy_host(d_scale, center_scale.data(), coefficient_bytes, "batch psi copy scale") ||
        !copy_host(d_center_factor, g_psi_warm_preconditioner.upper_factor.data(), factor_bytes, "batch psi copy factor")) {
        cleanup();
        return 1;
    }

    constexpr int threads = 256;
    const auto basis_started = clock::now();
    const int basis_blocks = static_cast<int>((basis_count + threads - 1) / threads);
    build_psi_common_basis_kernel<<<basis_blocks, threads>>>(
        d_normalized_R, d_normalized_Z, d_phi, point_count,
        d_mode_a, d_mode_b, d_mode_m, d_mode_kind, coefficient_count,
        nfp, static_cast<float>(radius_scale), d_basis_R, d_basis_Z, d_basis_phi
    );
    if (cuda_check(cudaGetLastError(), "batch psi basis kernel") ||
        cuda_check(cudaDeviceSynchronize(), "batch psi basis sync")) {
        cleanup();
        return 1;
    }
    const double basis_seconds = std::chrono::duration<double>(clock::now() - basis_started).count();

    const auto feature_started = clock::now();
    const size_t feature_shared = static_cast<size_t>(SEG_TILE * 6 + threads * 3 + 4) * sizeof(float);
    build_psi_batch_features_kernel<<<static_cast<int>(point_query_count), threads, feature_shared>>>(
        field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz,
        field->n_segments, d_normalized_R, d_normalized_Z, d_phi, point_count,
        d_axis_R, d_axis_Z, d_axis_R_phi, d_axis_Z_phi, axis_count, nfp,
        static_cast<float>(radius_scale), d_feature_R, d_feature_Z, d_feature_phi, d_rhs
    );
    if (cuda_check(cudaGetLastError(), "batch psi feature kernel") ||
        cuda_check(cudaDeviceSynchronize(), "batch psi feature sync")) {
        cleanup();
        return 1;
    }
    const double feature_seconds = std::chrono::duration<double>(clock::now() - feature_started).count();

    const float one = 1.0f;
    const float zero = 0.0f;
    auto apply_operator = [&](const float* coefficients, float* output) {
        const float* bases[] = {d_basis_R, d_basis_Z, d_basis_phi};
        const float* features[] = {d_feature_R, d_feature_Z, d_feature_phi};
        for (int component = 0; component < 3; ++component) {
            if (cublas_check(cublasSgemm(
                    blas, CUBLAS_OP_N, CUBLAS_OP_N,
                    point_count, query_count, coefficient_count,
                    &one, bases[component], point_count, coefficients, coefficient_count,
                    &zero, d_temporary, point_count), "batch psi forward GEMM")) {
                return false;
            }
            combine_psi_feature_kernel<<<
                static_cast<int>((point_query_count + threads - 1) / threads), threads
            >>>(output, features[component], d_temporary, point_query_count, component == 0);
            if (cuda_check(cudaGetLastError(), "batch psi forward combine")) return false;
        }
        return true;
    };
    auto compute_normal = [&](float* output) {
        const float* bases[] = {d_basis_R, d_basis_Z, d_basis_phi};
        const float* features[] = {d_feature_R, d_feature_Z, d_feature_phi};
        for (int component = 0; component < 3; ++component) {
            weight_psi_residual_kernel<<<
                static_cast<int>((point_query_count + threads - 1) / threads), threads
            >>>(d_residual, features[component], d_weighted, point_query_count);
            if (cuda_check(cudaGetLastError(), "batch psi residual weighting")) return false;
            const float beta = component == 0 ? 0.0f : 1.0f;
            if (cublas_check(cublasSgemm(
                    blas, CUBLAS_OP_T, CUBLAS_OP_N,
                    coefficient_count, query_count, point_count,
                    &one, bases[component], point_count, d_weighted, point_count,
                    &beta, output, coefficient_count), "batch psi transpose GEMM")) {
                return false;
            }
        }
        scale_psi_batch_normal_kernel<<<
            static_cast<int>((coefficient_query_count + threads - 1) / threads), threads
        >>>(output, d_scale, d_ridge_residual, static_cast<float>(std::sqrt(ridge)),
            coefficient_count, query_count);
        return cuda_check(cudaGetLastError(), "batch psi normal scaling") == 0;
    };

    const auto solve_started = clock::now();
    initialize_psi_batch_coefficients_kernel<<<
        static_cast<int>((coefficient_query_count + threads - 1) / threads), threads
    >>>(d_center_coefficients, d_scale, coefficient_count, query_count, d_scaled_coefficients);
    unscale_psi_batch_coefficients_kernel<<<
        static_cast<int>((coefficient_query_count + threads - 1) / threads), threads
    >>>(d_scaled_coefficients, d_scale, coefficient_count, query_count, d_unscaled);
    if (cuda_check(cudaGetLastError(), "batch psi coefficient initialization") ||
        !apply_operator(d_unscaled, d_image)) {
        cleanup();
        return 1;
    }
    initialize_psi_residual_kernel<<<
        static_cast<int>((point_query_count + threads - 1) / threads), threads
    >>>(d_rhs, d_image, point_query_count, d_residual);
    const float ridge_lambda = static_cast<float>(std::sqrt(ridge));
    initialize_psi_ridge_residual_kernel<<<
        static_cast<int>((coefficient_query_count + threads - 1) / threads), threads
    >>>(d_scaled_coefficients, ridge_lambda, coefficient_query_count, d_ridge_residual);
    if (cuda_check(cudaGetLastError(), "batch psi residual initialization") ||
        !compute_normal(d_normal) ||
        cublas_check(cublasStrsm(
            blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
            CUBLAS_DIAG_NON_UNIT, coefficient_count, query_count, &one,
            d_center_factor, coefficient_count, d_normal, coefficient_count
        ), "batch psi center R^-T") ||
        cublas_check(cublasScopy(
            blas, static_cast<int>(coefficient_query_count), d_normal, 1, d_direction, 1
        ), "batch psi initial direction")) {
        cleanup();
        return 1;
    }
    psi_batch_column_norm_kernel<<<query_count, threads, threads * sizeof(float)>>>(
        d_normal, coefficient_count, query_count, d_gamma
    );
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (cublas_check(cublasScopy(
                blas, static_cast<int>(coefficient_query_count), d_direction, 1,
                d_transformed_direction, 1), "batch psi copy direction") ||
            cublas_check(cublasStrsm(
                blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, coefficient_count, query_count, &one,
                d_center_factor, coefficient_count, d_transformed_direction, coefficient_count
            ), "batch psi center R^-1")) {
            cleanup();
            return 1;
        }
        unscale_psi_batch_coefficients_kernel<<<
            static_cast<int>((coefficient_query_count + threads - 1) / threads), threads
        >>>(d_transformed_direction, d_scale, coefficient_count, query_count, d_unscaled);
        if (cuda_check(cudaGetLastError(), "batch psi direction unscale") ||
            !apply_operator(d_unscaled, d_image)) {
            cleanup();
            return 1;
        }
        psi_batch_ridge_image_kernel<<<
            static_cast<int>((coefficient_query_count + threads - 1) / threads), threads
        >>>(d_transformed_direction, ridge_lambda, coefficient_query_count, d_ridge_image);
        psi_batch_delta_kernel<<<query_count, threads, threads * sizeof(float)>>>(
            d_image, point_count, d_ridge_image, coefficient_count, query_count, d_delta
        );
        psi_batch_update_kernel<<<
            static_cast<int>((std::max(point_query_count, coefficient_query_count) + threads - 1) / threads),
            threads
        >>>(d_scaled_coefficients, d_residual, d_ridge_residual,
            d_transformed_direction, d_image, d_ridge_image, d_gamma, d_delta,
            coefficient_count, point_count, query_count);
        if (cuda_check(cudaGetLastError(), "batch psi PCGLS update") ||
            !compute_normal(d_normal) ||
            cublas_check(cublasStrsm(
                blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT, coefficient_count, query_count, &one,
                d_center_factor, coefficient_count, d_normal, coefficient_count
            ), "batch psi next center R^-T")) {
            cleanup();
            return 1;
        }
        psi_batch_column_norm_kernel<<<query_count, threads, threads * sizeof(float)>>>(
            d_normal, coefficient_count, query_count, d_gamma_new
        );
        psi_batch_direction_update_kernel<<<
            static_cast<int>((coefficient_query_count + threads - 1) / threads), threads
        >>>(d_direction, d_normal, d_gamma, d_gamma_new, coefficient_count, query_count);
        if (cuda_check(cudaGetLastError(), "batch psi direction update") ||
            cuda_check(cudaMemcpy(
                d_gamma, d_gamma_new, static_cast<size_t>(query_count) * sizeof(float),
                cudaMemcpyDeviceToDevice), "batch psi gamma update")) {
            cleanup();
            return 1;
        }
    }
    unscale_psi_batch_coefficients_kernel<<<
        static_cast<int>((coefficient_query_count + threads - 1) / threads), threads
    >>>(d_scaled_coefficients, d_scale, coefficient_count, query_count, d_unscaled);
    psi_batch_column_norm_kernel<<<query_count, threads, threads * sizeof(float)>>>(
        d_residual, point_count, query_count, d_gamma
    );
    if (cuda_check(cudaGetLastError(), "batch psi final kernels") ||
        cuda_check(cudaDeviceSynchronize(), "batch psi solve sync")) {
        cleanup();
        return 1;
    }
    const double solve_seconds = std::chrono::duration<double>(clock::now() - solve_started).count();
    std::vector<float> coefficient_output(coefficient_query_count);
    std::vector<float> residual_norm2(query_count);
    if (cuda_check(cudaMemcpy(
            coefficient_output.data(), d_unscaled, coefficient_query_bytes,
            cudaMemcpyDeviceToHost), "batch psi copy coefficients") ||
        cuda_check(cudaMemcpy(
            residual_norm2.data(), d_gamma, static_cast<size_t>(query_count) * sizeof(float),
            cudaMemcpyDeviceToHost), "batch psi copy residuals")) {
        cleanup();
        return 1;
    }
    for (size_t index = 0; index < coefficient_query_count; ++index) {
        coefficients_host[index] = static_cast<double>(coefficient_output[index]);
    }
    for (int query = 0; query < query_count; ++query) {
        train_rms_host[query] = std::sqrt(
            std::max(static_cast<double>(residual_norm2[query]), 0.0) / point_count
        );
    }
    const double total_seconds = std::chrono::duration<double>(clock::now() - total_started).count();
    if (stats_out && stats_len >= 6) {
        stats_out[0] = point_count;
        stats_out[1] = basis_seconds;
        stats_out[2] = feature_seconds;
        stats_out[3] = solve_seconds;
        stats_out[4] = total_seconds;
        stats_out[5] = static_cast<double>(basis_bytes) * 3.0 / (1024.0 * 1024.0 * 1024.0);
    }
    cleanup();
    set_error("");
    return 0;
}

} // extern "C"
