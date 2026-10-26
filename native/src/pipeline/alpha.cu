// Clebsch basis construction and center alpha/iota weighted least-squares fit.
#include "pipeline/types.cuh"
#include "pipeline/device_helpers.cuh"
#include "pipeline/session_state.cuh"
#include "pipeline/alpha.cuh"
#include "pipeline/alpha_radial_basis.cuh"
#include "pipeline/common.cuh"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "pipeline/volume.cuh"

namespace coil_eval::pipeline {

double factorial_int(int value) {
    double result = 1.0;
    for (int index = 2; index <= value; ++index) result *= index;
    return result;
}

ClebschModesNative build_clebsch_modes_native(
    int radial_order,
    int poloidal_order,
    int toroidal_order,
    int radial_basis
) {
    ClebschModesNative modes;
    modes.radial_order = radial_order;
    modes.radial_basis = radial_basis;
    const int m_max = std::min(radial_order, poloidal_order);
    for (int m = 0; m <= m_max; ++m) {
        const int n_min = m == 0 ? 1 : -toroidal_order;
        const int n_max = toroidal_order;
        for (int n = n_min; n <= n_max; ++n) {
            for (int l = m; l <= radial_order; l += 2) {
                for (int kind = 0; kind < 2; ++kind) {
                    modes.l.push_back(l);
                    modes.m.push_back(m);
                    modes.n.push_back(n);
                    modes.kind.push_back(kind);
                    const size_t old_size = modes.radial_coefficients.size();
                    modes.radial_coefficients.resize(old_size + radial_order + 1, 0.0f);
                    if (radial_basis == 1) {
                        const auto coefficients=flux_legendre_radial_coefficients(l,m,radial_order+1);
                        std::copy(coefficients.begin(),coefficients.end(),modes.radial_coefficients.begin()+old_size);
                        continue;
                    }
                    const int half_plus = (l + m) / 2;
                    const int half_minus = (l - m) / 2;
                    for (int k = 0; k <= half_minus; ++k) {
                        double coefficient = ((k & 1) ? -1.0 : 1.0) * factorial_int(l - k);
                        coefficient /= factorial_int(k) * factorial_int(half_plus - k) *
                                       factorial_int(half_minus - k);
                        modes.radial_coefficients[old_size + l - 2 * k] = static_cast<float>(coefficient);
                    }
                }
            }
        }
    }
    return modes;
}

__global__ void alpha_radial_histogram_kernel(
    const float* __restrict__ rho,
    int point_count,
    int fit_count,
    int radial_bins,
    int* __restrict__ bin_counts
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= fit_count) return;
    const int point = min(
        point_count - 1,
        static_cast<int>((static_cast<long long>(index) * point_count) / fit_count)
    );
    const int bin = min(radial_bins - 1, max(0, static_cast<int>(rho[point] * radial_bins)));
    atomicAdd(bin_counts + bin, 1);
}

__global__ void alpha_prepare_kernel(
    const float* __restrict__ B,
    const float* __restrict__ grad_s,
    const float* __restrict__ grad_theta,
    const float* __restrict__ grad_phi,
    const float* __restrict__ source_rho,
    const float* __restrict__ source_theta,
    const float* __restrict__ source_phi,
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
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= fit_count) return;
    const int point = min(
        point_count - 1,
        static_cast<int>((static_cast<long long>(index) * point_count) / fit_count)
    );
    float grad_norm2 = 0.0f;
    float field_dot_grad = 0.0f;
    float field_norm2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        const float field = B[3 * point + component];
        const float gradient = grad_s[3 * point + component];
        grad_norm2 += gradient * gradient;
        field_dot_grad += field * gradient;
        field_norm2 += field * field;
    }
    const float normal_coefficient = field_dot_grad / fmaxf(grad_norm2, 1.0e-30f);
    float along_theta = 0.0f;
    float along_phi = 0.0f;
    float normal_norm2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        const float normal = normal_coefficient * grad_s[3 * point + component];
        const float tangent = B[3 * point + component] - normal;
        along_theta += tangent * grad_theta[3 * point + component];
        along_phi += tangent * grad_phi[3 * point + component];
        normal_norm2 += normal * normal;
    }
    const float radial = source_rho[point];
    const int bin = min(radial_bins - 1, max(0, static_cast<int>(radial * radial_bins)));
    const float field_norm = sqrtf(fmaxf(field_norm2, 1.0e-30f));
    const float weight = rsqrtf(static_cast<float>(max(bin_counts[bin], 1))) / field_norm;
    b_theta[index] = along_theta;
    b_phi[index] = along_phi;
    weights[index] = weight;
    rho[index] = radial;
    theta[index] = source_theta[point];
    phi[index] = source_phi[point];
    atomic_add_double(diagnostics, static_cast<double>(weight) * weight);
    atomic_add_double(diagnostics + 1, static_cast<double>(normal_norm2));
    atomic_add_double(diagnostics + 2, static_cast<double>(field_norm2));
}

// Fixed-shape FP64 partial reductions avoid histogram edges and atomic-order noise.
__global__ void alpha_soft_count_partials_kernel(
    const float* rho,int point_count,int fit_count,int radial_bins,double sigma,
    int partial_count,double* partials
) {
    extern __shared__ double sums[];
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    const int bin=blockIdx.y;
    double value=0.;
    if (index<fit_count) {
        const int point=min(point_count-1,static_cast<int>(
            (static_cast<long long>(index)*point_count)/fit_count));
        const double distance=(static_cast<double>(rho[point])-(bin+0.5)/radial_bins)/sigma;
        value=exp(-0.5*distance*distance);
    }
    sums[threadIdx.x]=value;__syncthreads();
    for (int stride=blockDim.x/2;stride>0;stride>>=1) {
        if (threadIdx.x<stride) sums[threadIdx.x]+=sums[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x==0) partials[static_cast<size_t>(bin)*partial_count+blockIdx.x]=sums[0];
}

__global__ void alpha_soft_count_reduce_kernel(
    const double* partials,int partial_count,double* counts
) {
    extern __shared__ double sums[];
    double value=0.;
    for (int index=threadIdx.x;index<partial_count;index+=blockDim.x)
        value+=partials[static_cast<size_t>(blockIdx.x)*partial_count+index];
    sums[threadIdx.x]=value;__syncthreads();
    for (int stride=blockDim.x/2;stride>0;stride>>=1) {
        if (threadIdx.x<stride) sums[threadIdx.x]+=sums[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x==0) counts[blockIdx.x]=sums[0];
}

__global__ void alpha_prepare_smooth_kernel(
    const float* __restrict__ B,
    const float* __restrict__ grad_s,
    const float* __restrict__ grad_theta,
    const float* __restrict__ grad_phi,
    const float* __restrict__ source_rho,
    const float* __restrict__ source_theta,
    const float* __restrict__ source_phi,
    int point_count,
    int fit_count,
    int radial_bins,
    const double* __restrict__ soft_counts,
    double sigma,
    double count_floor,
    float* __restrict__ b_theta,
    float* __restrict__ b_phi,
    float* __restrict__ weights,
    float* __restrict__ rho,
    float* __restrict__ theta,
    float* __restrict__ phi,
    double* __restrict__ diagnostics
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= fit_count) return;
    const int point = min(
        point_count - 1,
        static_cast<int>((static_cast<long long>(index) * point_count) / fit_count)
    );
    float grad_norm2 = 0.0f;
    float field_dot_grad = 0.0f;
    float field_norm2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        const float field = B[3 * point + component];
        const float gradient = grad_s[3 * point + component];
        grad_norm2 += gradient * gradient;
        field_dot_grad += field * gradient;
        field_norm2 += field * field;
    }
    const float normal_coefficient = field_dot_grad / fmaxf(grad_norm2, 1.0e-30f);
    float along_theta = 0.0f;
    float along_phi = 0.0f;
    float normal_norm2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        const float normal = normal_coefficient * grad_s[3 * point + component];
        const float tangent = B[3 * point + component] - normal;
        along_theta += tangent * grad_theta[3 * point + component];
        along_phi += tangent * grad_phi[3 * point + component];
        normal_norm2 += normal * normal;
    }
    const float radial = source_rho[point];
    const float field_norm = sqrtf(fmaxf(field_norm2, 1.0e-30f));
    double balance=0.;
    for (int radial_bin=0;radial_bin<radial_bins;++radial_bin) {
        const double center=(radial_bin+0.5)/radial_bins;
        const double distance=(static_cast<double>(radial)-center)/sigma;
        balance+=exp(-0.5*distance*distance)/(soft_counts[radial_bin]+count_floor);
    }
    const float weight=static_cast<float>(sqrt(balance)/static_cast<double>(field_norm));
    b_theta[index] = along_theta;
    b_phi[index] = along_phi;
    weights[index] = weight;
    rho[index] = radial;
    theta[index] = source_theta[point];
    phi[index] = source_phi[point];
    atomic_add_double(diagnostics, static_cast<double>(weight) * weight);
    atomic_add_double(diagnostics + 1, static_cast<double>(normal_norm2));
    atomic_add_double(diagnostics + 2, static_cast<double>(field_norm2));
}

__global__ void normalize_alpha_weights_kernel(
    float* __restrict__ weights,
    int count,
    double weight_norm2,
    const float* __restrict__ b_theta,
    float* __restrict__ rhs,
    int rhs_rows
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float scale = sqrtf(static_cast<float>(count / fmax(weight_norm2, 1.0e-300)));
        weights[index] *= scale;
        rhs[index] = -weights[index] * b_theta[index];
    }
    if (index >= count && index < rhs_rows) rhs[index] = 0.0f;
}

__global__ void assemble_alpha_design_kernel(
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
    int row_count,
    int leading_dimension,
    float* __restrict__ matrix
) {
    const int column = blockIdx.x;
    if (column < mode_count) {
        const int m = mode_m[column];
        const int n = mode_n[column];
        const int kind = mode_kind[column];
        const float* radial_coeff = radial_coefficients + static_cast<size_t>(column) * (radial_order + 1);
        for (int row = threadIdx.x; row < row_count; row += blockDim.x) {
            const float radial = evaluate_radial_polynomial(radial_coeff, radial_order, rho[row], radial_basis);
            const float argument = m * theta[row] - n * nfp * phi[row];
            float sine, cosine;
            sincosf(argument, &sine, &cosine);
            const float derivative_theta = kind == 0 ? -m * radial * sine : m * radial * cosine;
            const float derivative_phi = kind == 0 ? n * nfp * radial * sine : -n * nfp * radial * cosine;
            matrix[static_cast<size_t>(column) * leading_dimension + row] =
                weights[row] * (derivative_theta * b_theta[row] + derivative_phi * b_phi[row]);
        }
    } else {
        const int power = column - mode_count;
        if (power > iota_degree) return;
        for (int row = threadIdx.x; row < row_count; row += blockDim.x) {
            float radial_power = 1.0f;
            const float u = rho[row] * rho[row];
            for (int index = 0; index < power; ++index) radial_power *= u;
            matrix[static_cast<size_t>(column) * leading_dimension + row] =
                -weights[row] * radial_power * b_phi[row];
        }
    }
}

__global__ void scale_and_augment_alpha_columns_kernel(
    float* __restrict__ matrix,
    int data_rows,
    int leading_dimension,
    int column_count,
    float ridge_sqrt,
    float* __restrict__ scales
) {
    extern __shared__ double shared[];
    const int column = blockIdx.x;
    double sum = 0.0;
    for (int row = threadIdx.x; row < data_rows; row += blockDim.x) {
        const float value = matrix[static_cast<size_t>(column) * leading_dimension + row];
        sum += static_cast<double>(value) * value;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = fmaxf(static_cast<float>(sqrt(shared[0])), 1.0e-20f);
    if (threadIdx.x == 0) scales[column] = scale;
    __syncthreads();
    for (int row = threadIdx.x; row < data_rows; row += blockDim.x) {
        matrix[static_cast<size_t>(column) * leading_dimension + row] /= scale;
    }
    for (int row = data_rows + threadIdx.x; row < leading_dimension; row += blockDim.x) {
        matrix[static_cast<size_t>(column) * leading_dimension + row] =
            row == data_rows + column ? ridge_sqrt : 0.0f;
    }
}

__global__ void unscale_alpha_solution_kernel(
    const float* __restrict__ scaled_solution,
    const float* __restrict__ scales,
    int count,
    float* __restrict__ solution
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) solution[index] = scaled_solution[index] / scales[index];
}

bool fit_alpha_native(
    const DeviceVolumePoints& points,
    const float* d_B,
    int nfp,
    const SgpuScoreConfig& config,
    AlphaFitNative& fit,
    SgpuScoreResult& result
) {
    const auto assembly_started = Clock::now();
    const auto weighting=g_alpha_radial_weighting;
    const bool smooth=weighting.mode==1;
    fit.radial_weighting_mode=weighting.mode;
    fit.radial_weight_sigma=weighting.sigma;
    fit.radial_count_floor=weighting.count_floor;
    fit.radial_soft_counts.clear();

    const int fit_count = std::min(config.alpha_fit_point_count, points.count);
    const ClebschModesNative modes = build_clebsch_modes_native(
        config.alpha_radial_order,
        config.alpha_poloidal_order,
        config.alpha_toroidal_order,
        g_alpha_radial_basis
    );
    const int mode_count = static_cast<int>(modes.m.size());
    const int column_count = mode_count + config.iota_degree + 1;
    const int qr_rows = fit_count + column_count;
    fit.column_count = column_count;
    result.alpha_column_count = column_count;
    DeviceBuffer<int> d_mode_m, d_mode_n, d_mode_kind, d_bin_counts(config.radial_bin_count);
    DeviceBuffer<float> d_radial_coefficients;
    DeviceBuffer<float> d_b_theta(fit_count), d_b_phi(fit_count), d_weights(fit_count);
    DeviceBuffer<float> d_rho(fit_count), d_theta(fit_count), d_phi(fit_count);
    DeviceBuffer<double> d_diagnostics(3);
    DeviceBuffer<float> d_matrix(static_cast<size_t>(qr_rows) * column_count);
    DeviceBuffer<float> d_matrix_reference(static_cast<size_t>(fit_count) * column_count);
    DeviceBuffer<float> d_rhs(qr_rows), d_rhs_reference(fit_count), d_scales(column_count), d_solution(column_count);
    if (!copy_to_device(d_mode_m, modes.m) || !copy_to_device(d_mode_n, modes.n) ||
        !copy_to_device(d_mode_kind, modes.kind) ||
        !copy_to_device(d_radial_coefficients, modes.radial_coefficients) ||
        !d_bin_counts.data() || !d_b_theta.data() || !d_b_phi.data() || !d_weights.data() ||
        !d_rho.data() || !d_theta.data() || !d_phi.data() || !d_diagnostics.data() ||
        !d_matrix.data() || !d_matrix_reference.data() || !d_rhs.data() || !d_rhs_reference.data() ||
        !d_scales.data() || !d_solution.data()) {
        fail_result(&result, "alpha device allocation failed");
        return false;
    }
    cudaMemset(d_bin_counts.data(), 0, d_bin_counts.size() * sizeof(int));
    cudaMemset(d_diagnostics.data(), 0, d_diagnostics.size() * sizeof(double));
    constexpr int threads = 256;
    DeviceBuffer<double> d_soft_counts, d_soft_partials;
    if (smooth) {
        const int partial_count=(fit_count+threads-1)/threads;
        if (!d_soft_counts.allocate(config.radial_bin_count) ||
            !d_soft_partials.allocate(static_cast<size_t>(config.radial_bin_count)*partial_count)) {
            fail_result(&result,"alpha smooth density allocation failed"); return false;
        }
        alpha_soft_count_partials_kernel<<<dim3(partial_count,config.radial_bin_count),threads,threads*sizeof(double)>>>(
            points.rho.data(),points.count,fit_count,config.radial_bin_count,weighting.sigma,
            partial_count,d_soft_partials.data());
        alpha_soft_count_reduce_kernel<<<config.radial_bin_count,threads,threads*sizeof(double)>>>(
            d_soft_partials.data(),partial_count,d_soft_counts.data());
        alpha_prepare_smooth_kernel<<<(fit_count+threads-1)/threads,threads>>>(
            d_B,points.grad_s.data(),points.grad_theta.data(),points.grad_phi.data(),
            points.rho.data(),points.theta.data(),points.phi.data(),points.count,fit_count,
            config.radial_bin_count,d_soft_counts.data(),weighting.sigma,weighting.count_floor,
            d_b_theta.data(),d_b_phi.data(),d_weights.data(),d_rho.data(),d_theta.data(),d_phi.data(),d_diagnostics.data());
        fit.radial_soft_counts.resize(config.radial_bin_count);
        if (!cuda_stage_ok(cudaMemcpy(fit.radial_soft_counts.data(),d_soft_counts.data(),
                config.radial_bin_count*sizeof(double),cudaMemcpyDeviceToHost),result,"alpha soft counts")) return false;
    } else {
    alpha_radial_histogram_kernel<<<(fit_count + threads - 1) / threads, threads>>>(
        points.rho.data(), points.count, fit_count, config.radial_bin_count, d_bin_counts.data()
    );
    alpha_prepare_kernel<<<(fit_count + threads - 1) / threads, threads>>>(
        d_B, points.grad_s.data(), points.grad_theta.data(), points.grad_phi.data(),
        points.rho.data(), points.theta.data(), points.phi.data(), points.count, fit_count,
        config.radial_bin_count, d_bin_counts.data(), d_b_theta.data(), d_b_phi.data(),
        d_weights.data(), d_rho.data(), d_theta.data(), d_phi.data(), d_diagnostics.data()
    );
    }
    std::array<double, 3> diagnostics{};
    if (!cuda_stage_ok(cudaMemcpy(diagnostics.data(), d_diagnostics.data(), sizeof(diagnostics), cudaMemcpyDeviceToHost), result, "alpha diagnostics") ||
        !cuda_stage_ok(cudaMemset(d_matrix.data(), 0, d_matrix.size() * sizeof(float)), result, "alpha matrix clear")) {
        return false;
    }
    normalize_alpha_weights_kernel<<<(qr_rows + threads - 1) / threads, threads>>>(
        d_weights.data(), fit_count, diagnostics[0], d_b_theta.data(), d_rhs.data(), qr_rows
    );
    assemble_alpha_design_kernel<<<column_count, threads>>>(
        d_mode_m.data(), d_mode_n.data(), d_mode_kind.data(), d_radial_coefficients.data(),
        mode_count, config.alpha_radial_order, modes.radial_basis, config.iota_degree, nfp,
        d_rho.data(), d_theta.data(), d_phi.data(), d_b_theta.data(), d_b_phi.data(),
        d_weights.data(), fit_count, qr_rows, d_matrix.data()
    );
    scale_and_augment_alpha_columns_kernel<<<column_count, threads, threads * sizeof(double)>>>(
        d_matrix.data(), fit_count, qr_rows, column_count,
        static_cast<float>(std::sqrt(std::max(config.alpha_ridge, 0.0))), d_scales.data()
    );
    if (!cuda_stage_ok(cudaMemcpy(d_rhs_reference.data(), d_rhs.data(), fit_count * sizeof(float), cudaMemcpyDeviceToDevice), result, "alpha rhs reference") ||
        !cuda_stage_ok(cudaMemcpy2D(
            d_matrix_reference.data(), static_cast<size_t>(fit_count) * sizeof(float),
            d_matrix.data(), static_cast<size_t>(qr_rows) * sizeof(float),
            static_cast<size_t>(fit_count) * sizeof(float), column_count,
            cudaMemcpyDeviceToDevice
        ), result, "alpha matrix reference") ||
        !cuda_stage_ok(cudaDeviceSynchronize(), result, "alpha assembly")) {
        return false;
    }
    result.timings[SGPU_SCORE_TIME_ALPHA_ASSEMBLE] = seconds_since(assembly_started);
    fit.normal_B_relative_l2 = std::sqrt(diagnostics[1] / std::max(diagnostics[2], 1.0e-300));

    const auto solve_started = Clock::now();
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    DeviceBuffer<float> d_tau(column_count);
    DeviceBuffer<int> d_info(1);
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS ||
        cusolverDnCreate(&solver) != CUSOLVER_STATUS_SUCCESS ||
        !d_tau.data() || !d_info.data()) {
        if (blas) cublasDestroy(blas);
        if (solver) cusolverDnDestroy(solver);
        fail_result(&result, "alpha cuBLAS/cuSOLVER initialization failed");
        return false;
    }
    int info = 0;
    cublasStatus_t blas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t solver_status = CUSOLVER_STATUS_SUCCESS;
    float* d_scaled_solution = nullptr;
    DeviceBuffer<float> d_normal_matrix;
    DeviceBuffer<float> d_normal_rhs;
    DeviceBuffer<float> d_normal_work;
    if (config.alpha_solver_mode == 1) {
        if (!d_normal_matrix.allocate(static_cast<size_t>(column_count) * column_count) ||
            !d_normal_rhs.allocate(column_count)) {
            cublasDestroy(blas);
            cusolverDnDestroy(solver);
            fail_result(&result, "alpha normal-equation allocation failed");
            return false;
        }
        const float one = 1.0f;
        const float zero = 0.0f;
        blas_status = cublasSgemm(
            blas, CUBLAS_OP_T, CUBLAS_OP_N, column_count, column_count, qr_rows,
            &one, d_matrix.data(), qr_rows, d_matrix.data(), qr_rows,
            &zero, d_normal_matrix.data(), column_count
        );
        if (blas_status == CUBLAS_STATUS_SUCCESS) {
            blas_status = cublasSgemv(
                blas, CUBLAS_OP_T, qr_rows, column_count, &one,
                d_matrix.data(), qr_rows, d_rhs.data(), 1,
                &zero, d_normal_rhs.data(), 1
            );
        }
        int work_size = 0;
        if (blas_status == CUBLAS_STATUS_SUCCESS) {
            solver_status = cusolverDnSpotrf_bufferSize(
                solver, CUBLAS_FILL_MODE_LOWER, column_count,
                d_normal_matrix.data(), column_count, &work_size
            );
        }
        if (solver_status == CUSOLVER_STATUS_SUCCESS && !d_normal_work.allocate(work_size)) {
            solver_status = CUSOLVER_STATUS_ALLOC_FAILED;
        }
        if (solver_status == CUSOLVER_STATUS_SUCCESS && blas_status == CUBLAS_STATUS_SUCCESS) {
            solver_status = cusolverDnSpotrf(
                solver, CUBLAS_FILL_MODE_LOWER, column_count,
                d_normal_matrix.data(), column_count, d_normal_work.data(), work_size,
                d_info.data()
            );
            cudaMemcpy(&info, d_info.data(), sizeof(int), cudaMemcpyDeviceToHost);
        }
        if (solver_status == CUSOLVER_STATUS_SUCCESS && info == 0) {
            solver_status = cusolverDnSpotrs(
                solver, CUBLAS_FILL_MODE_LOWER, column_count, 1,
                d_normal_matrix.data(), column_count, d_normal_rhs.data(), column_count,
                d_info.data()
            );
            cudaMemcpy(&info, d_info.data(), sizeof(int), cudaMemcpyDeviceToHost);
        }
        if (solver_status != CUSOLVER_STATUS_SUCCESS || info != 0 ||
            blas_status != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(blas);
            cusolverDnDestroy(solver);
            fail_result(&result, "alpha FP32 Cholesky normal-equation solve failed");
            return false;
        }
        d_scaled_solution = d_normal_rhs.data();
    } else {
        int geqrf_work = 0;
        int ormqr_work = 0;
        solver_status = cusolverDnSgeqrf_bufferSize(
            solver, qr_rows, column_count, d_matrix.data(), qr_rows, &geqrf_work
        );
        if (solver_status == CUSOLVER_STATUS_SUCCESS) {
            solver_status = cusolverDnSormqr_bufferSize(
                solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, column_count,
                d_matrix.data(), qr_rows, d_tau.data(), d_rhs.data(), qr_rows, &ormqr_work
            );
        }
        DeviceBuffer<float> d_work(std::max(geqrf_work, ormqr_work));
        if (solver_status != CUSOLVER_STATUS_SUCCESS || !d_work.data()) {
            cublasDestroy(blas);
            cusolverDnDestroy(solver);
            fail_result(&result, "alpha QR workspace query failed");
            return false;
        }
        solver_status = cusolverDnSgeqrf(
            solver, qr_rows, column_count, d_matrix.data(), qr_rows,
            d_tau.data(), d_work.data(), static_cast<int>(d_work.size()), d_info.data()
        );
        cudaMemcpy(&info, d_info.data(), sizeof(int), cudaMemcpyDeviceToHost);
        if (solver_status == CUSOLVER_STATUS_SUCCESS && info == 0) {
            solver_status = cusolverDnSormqr(
                solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, column_count,
                d_matrix.data(), qr_rows, d_tau.data(), d_rhs.data(), qr_rows,
                d_work.data(), static_cast<int>(d_work.size()), d_info.data()
            );
            cudaMemcpy(&info, d_info.data(), sizeof(int), cudaMemcpyDeviceToHost);
        }
        if (solver_status == CUSOLVER_STATUS_SUCCESS && info == 0) {
            blas_status = cublasStrsv(
                blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                column_count, d_matrix.data(), qr_rows, d_rhs.data(), 1
            );
        }
        if (solver_status != CUSOLVER_STATUS_SUCCESS || info != 0 ||
            blas_status != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(blas);
            cusolverDnDestroy(solver);
            fail_result(&result, "alpha FP32 QR solve failed");
            return false;
        }
        d_scaled_solution = d_rhs.data();
    }
    unscale_alpha_solution_kernel<<<(column_count + threads - 1) / threads, threads>>>(
        d_scaled_solution, d_scales.data(), column_count, d_solution.data()
    );
    DeviceBuffer<float> d_prediction(fit_count);
    if (!d_prediction.data()) {
        cublasDestroy(blas);
        cusolverDnDestroy(solver);
        fail_result(&result, "alpha residual allocation failed");
        return false;
    }
    const float one = 1.0f;
    const float zero = 0.0f;
    const float minus_one = -1.0f;
    blas_status = cublasSgemv(
        blas, CUBLAS_OP_N, fit_count, column_count, &one,
        d_matrix_reference.data(), fit_count, d_scaled_solution, 1, &zero, d_prediction.data(), 1
    );
    if (blas_status == CUBLAS_STATUS_SUCCESS) {
        blas_status = cublasSaxpy(
            blas, fit_count, &minus_one, d_rhs_reference.data(), 1, d_prediction.data(), 1
        );
    }
    float residual_norm = 0.0f;
    float rhs_norm = 0.0f;
    if (blas_status == CUBLAS_STATUS_SUCCESS) {
        blas_status = cublasSnrm2(blas, fit_count, d_prediction.data(), 1, &residual_norm);
    }
    if (blas_status == CUBLAS_STATUS_SUCCESS) {
        blas_status = cublasSnrm2(blas, fit_count, d_rhs_reference.data(), 1, &rhs_norm);
    }
    if (blas_status != CUBLAS_STATUS_SUCCESS ||
        !cuda_stage_ok(cudaDeviceSynchronize(), result, "alpha QR completion")) {
        cublasDestroy(blas);
        cusolverDnDestroy(solver);
        if (result.status != SGPU_SCORE_INTERNAL_ERROR) fail_result(&result, "alpha residual evaluation failed");
        return false;
    }
    fit.relative_l2 = residual_norm / std::max(rhs_norm, 1.0e-20f);
    fit.residual_norm = residual_norm;
    fit.rhs_norm = rhs_norm;
    fit.iota_coefficients.resize(config.iota_degree + 1);
    if (!cuda_stage_ok(cudaMemcpy(
            fit.iota_coefficients.data(), d_solution.data() + mode_count,
            fit.iota_coefficients.size() * sizeof(float), cudaMemcpyDeviceToHost
        ), result, "alpha iota copy")) {
        cublasDestroy(blas);
        cusolverDnDestroy(solver);
        return false;
    }
    if (g_active_gradient_group >= 3) {
        DeviceBuffer<float> d_saved_scaled_solution(column_count);
        if (!d_saved_scaled_solution.data() ||
            !cuda_stage_ok(cudaMemcpy(
                d_saved_scaled_solution.data(), d_scaled_solution,
                column_count * sizeof(float), cudaMemcpyDeviceToDevice
            ), result, "alpha scaled solution cache")) {
            cublasDestroy(blas);
            cusolverDnDestroy(solver);
            return false;
        }
        fit.mode_count = mode_count;
        fit.fit_count = fit_count;
        fit.qr_rows = qr_rows;
        fit.weight_norm2 = diagnostics[0];
        fit.normal_numerator = diagnostics[1];
        fit.field_denominator = diagnostics[2];
        fit.modes = modes;
        fit.qr_matrix = std::move(d_matrix);
        fit.matrix_reference = std::move(d_matrix_reference);
        fit.rhs_reference = std::move(d_rhs_reference);
        fit.scales = std::move(d_scales);
        fit.scaled_solution = std::move(d_saved_scaled_solution);
        fit.residual = std::move(d_prediction);
        fit.b_theta = std::move(d_b_theta);
        fit.b_phi = std::move(d_b_phi);
        fit.weights = std::move(d_weights);
        fit.rho = std::move(d_rho);
        fit.theta = std::move(d_theta);
        fit.phi = std::move(d_phi);
        fit.adjoint_ready = true;
    }
    cublasDestroy(blas);
    cusolverDnDestroy(solver);
    result.timings[SGPU_SCORE_TIME_ALPHA_QR] = seconds_since(solve_started);
    return true;
}


}  // namespace coil_eval::pipeline
