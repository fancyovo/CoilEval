#include "field/state.h"
#include "field/psi_design.cuh"
#include "coil_field.h"
#include "nvtx_profile.h"
#include "psi_qr_snapshot.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <type_traits>

namespace coil_eval::field {

struct GpuFitStats {
    double copy_in_s = 0.0;
    double assemble_s = 0.0;
    double linear_prep_s = 0.0;
    double solve_s = 0.0;
    double residual_s = 0.0;
    double copy_out_s = 0.0;
    double total_s = 0.0;
    double qr_transpose_s = 0.0;
    double qr_scale_s = 0.0;
    double qr_factor_s = 0.0;
    double qr_apply_qtb_s = 0.0;
    double qr_tri_s = 0.0;
};

int psi_mode_count(int poly_degree, int m_tor) {
    int modes = 0;
    for (int deg = 2; deg <= poly_degree; ++deg) {
        for (int ax = deg; ax >= 0; --ax) {
            int bz = deg - ax;
            for (int m = 0; m <= m_tor; ++m) {
                if (ax == 2 && bz == 0 && m == 0) continue;
                modes += (m == 0) ? 1 : 2;
            }
        }
    }
    return modes;
}

bool write_device_bytes(std::ofstream& stream, const void* device_data, std::size_t bytes) {
    constexpr std::size_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
    std::vector<unsigned char> host_buffer(std::min(bytes, chunk_bytes));
    const auto* source = static_cast<const unsigned char*>(device_data);
    for (std::size_t offset = 0; offset < bytes; offset += host_buffer.size()) {
        const std::size_t count = std::min(host_buffer.size(), bytes - offset);
        if (cuda_check(cudaMemcpy(host_buffer.data(), source + offset, count, cudaMemcpyDeviceToHost),
                       "psi QR snapshot copy to host")) {
            return false;
        }
        stream.write(reinterpret_cast<const char*>(host_buffer.data()), static_cast<std::streamsize>(count));
        if (!stream) {
            set_error("failed to write psi QR snapshot payload");
            return false;
        }
    }
    return true;
}

bool maybe_write_psi_qr_snapshot(
    const float* matrix,
    const float* rhs,
    const double* scale,
    int rows,
    int cols,
    int data_rows,
    double ridge
) {
    const char* path = std::getenv("SGPU_PSI_QR_SNAPSHOT");
    if (!path || path[0] == '\0') return true;

    sgpu::PsiQrSnapshotHeader header{};
    std::memcpy(header.magic, sgpu::kPsiQrSnapshotMagic, sizeof(header.magic));
    header.version = sgpu::kPsiQrSnapshotVersion;
    header.scalar_type = sgpu::kPsiQrScalarFloat32;
    header.layout = sgpu::kPsiQrLayoutColumnMajor;
    header.rows = static_cast<std::uint64_t>(rows);
    header.cols = static_cast<std::uint64_t>(cols);
    header.data_rows = static_cast<std::uint64_t>(data_rows);
    header.ridge = ridge;
    header.matrix_bytes = static_cast<std::uint64_t>(rows) * cols * sizeof(float);
    header.rhs_bytes = static_cast<std::uint64_t>(rows) * sizeof(float);
    header.scale_bytes = static_cast<std::uint64_t>(cols) * sizeof(double);

    const std::uint64_t expected_bytes = sizeof(header) + header.matrix_bytes +
        header.rhs_bytes + header.scale_bytes;
    std::ifstream existing(path, std::ios::binary | std::ios::ate);
    if (existing.good() && static_cast<std::uint64_t>(existing.tellg()) == expected_bytes) return true;
    existing.close();

    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        set_error(std::string("failed to open psi QR snapshot: ") + path);
        return false;
    }
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!stream ||
        !write_device_bytes(stream, matrix, static_cast<std::size_t>(header.matrix_bytes)) ||
        !write_device_bytes(stream, rhs, static_cast<std::size_t>(header.rhs_bytes)) ||
        !write_device_bytes(stream, scale, static_cast<std::size_t>(header.scale_bytes))) {
        return false;
    }
    stream.close();
    if (!stream) {
        set_error("failed to finalize psi QR snapshot");
        return false;
    }
    return true;
}

__global__ void convert_double_to_float_kernel(const double* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<float>(src[i]);
}

__global__ void convert_float_to_double_kernel(const float* src, double* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<double>(src[i]);
}

__global__ void scale_rhs_kernel(double* atb, const double* scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atb[i] /= scale[i];
}

__global__ void extract_scale_kernel(const double* ata, double* scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double d = ata[static_cast<size_t>(i) * n + i];
        scale[i] = sqrt(fmax(d, 1.0e-30));
    }
}

__global__ void scale_system_kernel(double* ata, const double* scale, int n, double ridge) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        double v = ata[static_cast<size_t>(i) * n + j] / (scale[i] * scale[j]);
        if (i == j) v += ridge;
        ata[static_cast<size_t>(i) * n + j] = v;
    }
}

__global__ void unscale_coeff_kernel(double* coeff, const double* scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) coeff[i] /= scale[i];
}

__global__ void unscale_coeff_kernel_f32(float* coeff, const double* scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) coeff[i] /= static_cast<float>(scale[i]);
}

__global__ void scale_coeff_kernel_f32(float* coeff, const double* scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) coeff[i] *= static_cast<float>(scale[i]);
}

__global__ void scale_coeff_float_kernel(float* coeff, const float* scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) coeff[i] *= scale[i];
}

template <typename T>
__global__ void axpy_negative_kernel(const T* x, const T* y, T* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] - y[i];
}

template <typename T>
__global__ void transpose_rowmajor_to_colmajor_kernel(const T* src, T* dst, int n_rows, int n_cols, int dst_lda) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n_rows && col < n_cols) {
        dst[static_cast<size_t>(col) * dst_lda + row] = src[static_cast<size_t>(row) * n_cols + col];
    }
}

__global__ void column_norms_kernel_f64(const double* mat, double* scale, int n_rows, int lda, int n_cols) {
    int col = blockIdx.x;
    if (col >= n_cols) return;
    __shared__ double sh[THREADS_PER_BLOCK];
    double sum = 0.0;
    const double* col_ptr = mat + static_cast<size_t>(col) * lda;
    for (int row = threadIdx.x; row < n_rows; row += blockDim.x) {
        double v = col_ptr[row];
        sum += v * v;
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) scale[col] = sqrt(fmax(sh[0], 1.0e-30));
}

__global__ void column_norms_kernel_f32(const float* mat, double* scale, int n_rows, int lda, int n_cols) {
    int col = blockIdx.x;
    if (col >= n_cols) return;
    __shared__ double sh[THREADS_PER_BLOCK];
    double sum = 0.0;
    const float* col_ptr = mat + static_cast<size_t>(col) * lda;
    for (int row = threadIdx.x; row < n_rows; row += blockDim.x) {
        double v = static_cast<double>(col_ptr[row]);
        sum += v * v;
    }
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) scale[col] = sqrt(fmax(sh[0], 1.0e-30));
}

__global__ void scale_columns_kernel_f64(double* mat, const double* scale, int n_rows, int lda, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n_rows && col < n_cols) {
        mat[static_cast<size_t>(col) * lda + row] /= scale[col];
    }
}

__global__ void scale_columns_kernel_f32(float* mat, const double* scale, int n_rows, int lda, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n_rows && col < n_cols) {
        mat[static_cast<size_t>(col) * lda + row] /= static_cast<float>(scale[col]);
    }
}

template <typename T>
__global__ void set_ridge_tail_kernel(T* mat, int n_rows_data, int lda, int n_cols, T lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_cols * n_cols;
    if (idx < total) {
        int col = idx / n_cols;
        int tail_row = idx - col * n_cols;
        mat[static_cast<size_t>(col) * lda + (n_rows_data + tail_row)] = (tail_row == col) ? lambda : static_cast<T>(0);
    }
}

} // namespace coil_eval::field

// The public ABI stays global and unmangled; implementation state is named.
using namespace coil_eval::field;

extern "C" {

int sgpu_normal_eq(
    void* handle,
    const double* mat_host,
    const double* rhs_host,
    double* ata_host,
    double* atb_host,
    int n_rows,
    int n_cols
) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !mat_host || !rhs_host || !ata_host || !atb_host || n_rows < 0 || n_cols <= 0) {
        set_error("invalid normal_eq arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    double *d_mat = nullptr, *d_rhs = nullptr, *d_ata = nullptr, *d_atb = nullptr;
    size_t mat_bytes = static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols) * sizeof(double);
    size_t rhs_bytes = static_cast<size_t>(n_rows) * sizeof(double);
    size_t ata_bytes = static_cast<size_t>(n_cols) * static_cast<size_t>(n_cols) * sizeof(double);
    size_t atb_bytes = static_cast<size_t>(n_cols) * sizeof(double);
    cublasHandle_t blas = nullptr;
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mat), mat_bytes), "normal_eq d_mat") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rhs), rhs_bytes), "normal_eq d_rhs") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ata), ata_bytes), "normal_eq d_ata") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_atb), atb_bytes), "normal_eq d_atb") ||
        cuda_check(cudaMemcpy(d_mat, mat_host, mat_bytes, cudaMemcpyHostToDevice), "normal_eq copy mat") ||
        cuda_check(cudaMemcpy(d_rhs, rhs_host, rhs_bytes, cudaMemcpyHostToDevice), "normal_eq copy rhs") ||
        cuda_check(cudaMemset(d_ata, 0, ata_bytes), "normal_eq memset ata") ||
        cuda_check(cudaMemset(d_atb, 0, atb_bytes), "normal_eq memset atb") ||
        cublas_check(cublasCreate(&blas), "cublasCreate")) {
        cudaFree(d_mat); cudaFree(d_rhs); cudaFree(d_ata); cudaFree(d_atb);
        if (blas) cublasDestroy(blas);
        return 1;
    }
    const double one = 1.0;
    const double zero = 0.0;
    // mat_host is C-order A[n_rows, n_cols]. The same memory interpreted as
    // column-major is B[n_cols, n_rows] = A^T, so B * B^T = A^T A.
    if (cublas_check(
            cublasDgemm(
                blas,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                n_cols,
                n_cols,
                n_rows,
                &one,
                d_mat,
                n_cols,
                d_mat,
                n_cols,
                &zero,
                d_ata,
                n_cols),
            "cublasDgemm normal_eq") ||
        cublas_check(
            cublasDgemv(
                blas,
                CUBLAS_OP_N,
                n_cols,
                n_rows,
                &one,
                d_mat,
                n_cols,
                d_rhs,
                1,
                &zero,
                d_atb,
                1),
            "cublasDgemv normal_eq") ||
        cuda_check(cudaMemcpy(ata_host, d_ata, ata_bytes, cudaMemcpyDeviceToHost), "normal_eq copy ata") ||
        cuda_check(cudaMemcpy(atb_host, d_atb, atb_bytes, cudaMemcpyDeviceToHost), "normal_eq copy atb")) {
        cublasDestroy(blas);
        cudaFree(d_mat); cudaFree(d_rhs); cudaFree(d_ata); cudaFree(d_atb);
        return 1;
    }
    cublasDestroy(blas);
    cudaFree(d_mat);
    cudaFree(d_rhs);
    cudaFree(d_ata);
    cudaFree(d_atb);
    set_error("");
    return 0;
}

int sgpu_normal_eq_f32(
    void* handle,
    const float* mat_host,
    const float* rhs_host,
    float* ata_host,
    float* atb_host,
    int n_rows,
    int n_cols
) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !mat_host || !rhs_host || !ata_host || !atb_host || n_rows < 0 || n_cols <= 0) {
        set_error("invalid normal_eq_f32 arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    float *d_mat = nullptr, *d_rhs = nullptr, *d_ata = nullptr, *d_atb = nullptr;
    size_t mat_bytes = static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols) * sizeof(float);
    size_t rhs_bytes = static_cast<size_t>(n_rows) * sizeof(float);
    size_t ata_bytes = static_cast<size_t>(n_cols) * static_cast<size_t>(n_cols) * sizeof(float);
    size_t atb_bytes = static_cast<size_t>(n_cols) * sizeof(float);
    cublasHandle_t blas = nullptr;
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mat), mat_bytes), "normal_eq_f32 d_mat") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rhs), rhs_bytes), "normal_eq_f32 d_rhs") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ata), ata_bytes), "normal_eq_f32 d_ata") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_atb), atb_bytes), "normal_eq_f32 d_atb") ||
        cuda_check(cudaMemcpy(d_mat, mat_host, mat_bytes, cudaMemcpyHostToDevice), "normal_eq_f32 copy mat") ||
        cuda_check(cudaMemcpy(d_rhs, rhs_host, rhs_bytes, cudaMemcpyHostToDevice), "normal_eq_f32 copy rhs") ||
        cuda_check(cudaMemset(d_ata, 0, ata_bytes), "normal_eq_f32 memset ata") ||
        cuda_check(cudaMemset(d_atb, 0, atb_bytes), "normal_eq_f32 memset atb") ||
        cublas_check(cublasCreate(&blas), "cublasCreate f32")) {
        cudaFree(d_mat); cudaFree(d_rhs); cudaFree(d_ata); cudaFree(d_atb);
        if (blas) cublasDestroy(blas);
        return 1;
    }
    const float one = 1.0f;
    const float zero = 0.0f;
    if (cublas_check(
            cublasSgemm(
                blas,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                n_cols,
                n_cols,
                n_rows,
                &one,
                d_mat,
                n_cols,
                d_mat,
                n_cols,
                &zero,
                d_ata,
                n_cols),
            "cublasSgemm normal_eq_f32") ||
        cublas_check(
            cublasSgemv(
                blas,
                CUBLAS_OP_N,
                n_cols,
                n_rows,
                &one,
                d_mat,
                n_cols,
                d_rhs,
                1,
                &zero,
                d_atb,
                1),
            "cublasSgemv normal_eq_f32") ||
        cuda_check(cudaMemcpy(ata_host, d_ata, ata_bytes, cudaMemcpyDeviceToHost), "normal_eq_f32 copy ata") ||
        cuda_check(cudaMemcpy(atb_host, d_atb, atb_bytes, cudaMemcpyDeviceToHost), "normal_eq_f32 copy atb")) {
        cublasDestroy(blas);
        cudaFree(d_mat); cudaFree(d_rhs); cudaFree(d_ata); cudaFree(d_atb);
        return 1;
    }
    cublasDestroy(blas);
    cudaFree(d_mat);
    cudaFree(d_rhs);
    cudaFree(d_ata);
    cudaFree(d_atb);
    set_error("");
    return 0;
}

int sgpu_fit_psi_fullgpu(
    void* handle,
    const double* R_host,
    const double* Z_host,
    const double* phi_host,
    int n_points,
    const double* axis_R_host,
    const double* axis_Z_host,
    const double* axis_R_phi_host,
    const double* axis_Z_phi_host,
    int n_axis,
    const int* mode_a_host,
    const int* mode_b_host,
    const int* mode_m_host,
    const int* mode_kind_host,
    int n_coeff,
    int nfp,
    double a,
    int poly_degree,
    int m_tor,
    double ridge,
    int solver_mode,
    int precision_mode,
    double* coeff_host,
    double* train_rms_out,
    double* stats_out,
    int stats_len
) {
    SGPU_NVTX_RANGE("psi.fullgpu.total");
    using clock = std::chrono::steady_clock;

    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !R_host || !Z_host || !phi_host || !axis_R_host || !axis_Z_host || !axis_R_phi_host || !axis_Z_phi_host ||
        !mode_a_host || !mode_b_host || !mode_m_host || !mode_kind_host || !coeff_host || !train_rms_out ||
        n_points <= 0 || n_axis <= 1 || nfp <= 0 || poly_degree < 2 || m_tor < 0 || n_coeff <= 0) {
        set_error("invalid fit_psi_fullgpu arguments");
        return 1;
    }
    if (poly_degree > MAX_PSI_DEGREE || m_tor > MAX_PSI_MTOR) {
        set_error("psi degree exceeds compiled GPU limits");
        return 1;
    }
    int expected = psi_mode_count(poly_degree, m_tor);
    if (n_coeff != expected) {
        set_error("n_coeff does not match poly_degree/m_tor");
        return 1;
    }
    const bool warm_preconditioned = solver_mode >= 2000;
    const bool warm_solver = solver_mode >= 1000;
    const int warm_iterations = warm_solver
        ? solver_mode - (warm_preconditioned ? 2000 : 1000)
        : 0;
    if (solver_mode != 1 && solver_mode != 2 && !warm_solver) {
        set_error("solver_mode must be 1 (normal_eq), 2 (qr), or an experimental warm mode");
        return 1;
    }
    if (precision_mode != 1 && precision_mode != 2) {
        set_error("precision_mode must be 1 (fp64) or 2 (fp32)");
        return 1;
    }
    if (warm_solver && precision_mode != 2) {
        set_error("experimental warm psi solve requires fp32");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;

    GpuFitStats stats;
    auto t_total = clock::now();

    double *d_R = nullptr, *d_Z = nullptr, *d_phi = nullptr;
    double *d_axis_R = nullptr, *d_axis_Z = nullptr, *d_axis_R_phi = nullptr, *d_axis_Z_phi = nullptr;
    int *d_mode_a = nullptr, *d_mode_b = nullptr, *d_mode_m = nullptr, *d_mode_kind = nullptr;
    double *d_scale = nullptr, *d_info_rhs = nullptr, *d_ata_d = nullptr, *d_atb_d = nullptr;
    int *d_info = nullptr, *d_ipiv = nullptr;
    double train_rms = 0.0;
    int h_info = 0;

    float *d_R_f = nullptr, *d_Z_f = nullptr, *d_phi_f = nullptr;
    float *d_axis_R_f = nullptr, *d_axis_Z_f = nullptr, *d_axis_R_phi_f = nullptr, *d_axis_Z_phi_f = nullptr;
    float *d_mat_f = nullptr, *d_rhs_f = nullptr, *d_ata_f = nullptr, *d_atb_f = nullptr, *d_coeff_f = nullptr, *d_pred_f = nullptr;
    double *d_mat_d = nullptr, *d_rhs_d = nullptr, *d_pred_d = nullptr;
    double *d_qr_mat_d = nullptr, *d_qr_rhs_d = nullptr, *d_tau_d = nullptr, *d_work_d = nullptr;
    float *d_qr_mat_f = nullptr, *d_qr_rhs_f = nullptr, *d_tau_f = nullptr, *d_work_f = nullptr;
    float *d_warm_residual_f = nullptr, *d_warm_normal_f = nullptr;
    float *d_warm_direction_f = nullptr, *d_warm_image_f = nullptr;
    float *d_warm_transformed_direction_f = nullptr, *d_warm_scale_ratio_f = nullptr;
    float *d_warm_center_factor_f = nullptr;
    std::vector<float> coeff_host_f;
    double* d_coeff_src_d = nullptr;
    float* d_coeff_src_f = nullptr;

    auto cleanup = [&]() {
        SGPU_NVTX_RANGE("psi.fullgpu.cleanup");
        cudaFree(d_R); cudaFree(d_Z); cudaFree(d_phi);
        cudaFree(d_axis_R); cudaFree(d_axis_Z); cudaFree(d_axis_R_phi); cudaFree(d_axis_Z_phi);
        cudaFree(d_mode_a); cudaFree(d_mode_b); cudaFree(d_mode_m); cudaFree(d_mode_kind);
        cudaFree(d_scale); cudaFree(d_info_rhs); cudaFree(d_ata_d); cudaFree(d_atb_d); cudaFree(d_info); cudaFree(d_ipiv);
        cudaFree(d_R_f); cudaFree(d_Z_f); cudaFree(d_phi_f);
        cudaFree(d_axis_R_f); cudaFree(d_axis_Z_f); cudaFree(d_axis_R_phi_f); cudaFree(d_axis_Z_phi_f);
        cudaFree(d_mat_f); cudaFree(d_rhs_f); cudaFree(d_ata_f); cudaFree(d_atb_f); cudaFree(d_coeff_f); cudaFree(d_pred_f);
        cudaFree(d_mat_d); cudaFree(d_rhs_d); cudaFree(d_pred_d);
        cudaFree(d_qr_mat_d); cudaFree(d_qr_rhs_d); cudaFree(d_tau_d); cudaFree(d_work_d);
        cudaFree(d_qr_mat_f); cudaFree(d_qr_rhs_f); cudaFree(d_tau_f); cudaFree(d_work_f);
        cudaFree(d_warm_residual_f); cudaFree(d_warm_normal_f);
        cudaFree(d_warm_direction_f); cudaFree(d_warm_image_f);
        cudaFree(d_warm_transformed_direction_f); cudaFree(d_warm_scale_ratio_f);
        cudaFree(d_warm_center_factor_f);
    };

    auto fail = [&](const char* msg) {
        set_error(msg);
        cleanup();
        return 1;
    };

    size_t point_bytes = static_cast<size_t>(n_points) * sizeof(double);
    size_t axis_bytes = static_cast<size_t>(n_axis) * sizeof(double);
    size_t mode_bytes = static_cast<size_t>(n_coeff) * sizeof(int);
    auto t = clock::now();
    {
        SGPU_NVTX_RANGE("psi.fullgpu.copy_in");
        if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R), point_bytes), "fit d_R") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z), point_bytes), "fit d_Z") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_phi), point_bytes), "fit d_phi") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_R), axis_bytes), "fit d_axis_R") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_Z), axis_bytes), "fit d_axis_Z") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_R_phi), axis_bytes), "fit d_axis_R_phi") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_Z_phi), axis_bytes), "fit d_axis_Z_phi") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mode_a), mode_bytes), "fit d_mode_a") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mode_b), mode_bytes), "fit d_mode_b") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mode_m), mode_bytes), "fit d_mode_m") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mode_kind), mode_bytes), "fit d_mode_kind") ||
            cuda_check(cudaMemcpy(d_R, R_host, point_bytes, cudaMemcpyHostToDevice), "fit copy R") ||
            cuda_check(cudaMemcpy(d_Z, Z_host, point_bytes, cudaMemcpyHostToDevice), "fit copy Z") ||
            cuda_check(cudaMemcpy(d_phi, phi_host, point_bytes, cudaMemcpyHostToDevice), "fit copy phi") ||
            cuda_check(cudaMemcpy(d_axis_R, axis_R_host, axis_bytes, cudaMemcpyHostToDevice), "fit copy axis_R") ||
            cuda_check(cudaMemcpy(d_axis_Z, axis_Z_host, axis_bytes, cudaMemcpyHostToDevice), "fit copy axis_Z") ||
            cuda_check(cudaMemcpy(d_axis_R_phi, axis_R_phi_host, axis_bytes, cudaMemcpyHostToDevice), "fit copy axis_R_phi") ||
            cuda_check(cudaMemcpy(d_axis_Z_phi, axis_Z_phi_host, axis_bytes, cudaMemcpyHostToDevice), "fit copy axis_Z_phi") ||
            cuda_check(cudaMemcpy(d_mode_a, mode_a_host, mode_bytes, cudaMemcpyHostToDevice), "fit copy mode_a") ||
            cuda_check(cudaMemcpy(d_mode_b, mode_b_host, mode_bytes, cudaMemcpyHostToDevice), "fit copy mode_b") ||
            cuda_check(cudaMemcpy(d_mode_m, mode_m_host, mode_bytes, cudaMemcpyHostToDevice), "fit copy mode_m") ||
            cuda_check(cudaMemcpy(d_mode_kind, mode_kind_host, mode_bytes, cudaMemcpyHostToDevice), "fit copy mode_kind")) {
            cleanup();
            return 1;
        }
    }
    stats.copy_in_s = std::chrono::duration<double>(clock::now() - t).count();

    double rhs_norm2 = 0.0;
    size_t mat_elems = static_cast<size_t>(n_points) * static_cast<size_t>(n_coeff);
    size_t mat_bytes_d = mat_elems * sizeof(double);
    size_t mat_bytes_f = mat_elems * sizeof(float);
    int threads = THREADS_PER_BLOCK;
    int blocks1d_points = (n_points + threads - 1) / threads;
    int blocks1d_coeff = (n_coeff + threads - 1) / threads;
    int shared_d = static_cast<int>((SEG_TILE * 6 + threads * 3 + 2 * (MAX_PSI_DEGREE + 1) + 2 * (MAX_PSI_MTOR + 1) + 4) * sizeof(double));
    int shared_f = static_cast<int>((SEG_TILE * 6 + threads * 3 + 2 * (MAX_PSI_DEGREE + 1) + 2 * (MAX_PSI_MTOR + 1) + 4) * sizeof(float));

    t = clock::now();
    {
    SGPU_NVTX_RANGE("psi.fullgpu.assemble");
    if (precision_mode == 1) {
        if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mat_d), mat_bytes_d), "fit d_mat_d") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rhs_d), point_bytes), "fit d_rhs_d")) {
            cleanup();
            return 1;
        }
        psi_fill_matrix_kernel_f64<<<n_points, threads, shared_d>>>(
            f->d_x, f->d_y, f->d_z, f->d_wx, f->d_wy, f->d_wz, f->n_segments,
            d_R, d_Z, d_phi, d_axis_R, d_axis_Z, d_axis_R_phi, d_axis_Z_phi, n_axis,
            d_mode_a, d_mode_b, d_mode_m, d_mode_kind, n_coeff, nfp, a, poly_degree, m_tor, d_mat_d, d_rhs_d
        );
        if (cuda_check(cudaGetLastError(), "fit psi_fill_matrix f64") ||
            cublas_check(cublasDdot(f->blas, n_points, d_rhs_d, 1, d_rhs_d, 1, &rhs_norm2), "fit rhs_norm2 d64") ||
            cuda_check(cudaDeviceSynchronize(), "fit assemble sync f64")) {
            cleanup();
            return 1;
        }
    } else {
        if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R_f), static_cast<size_t>(n_points) * sizeof(float)), "fit d_R_f") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z_f), static_cast<size_t>(n_points) * sizeof(float)), "fit d_Z_f") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_phi_f), static_cast<size_t>(n_points) * sizeof(float)), "fit d_phi_f") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_R_f), static_cast<size_t>(n_axis) * sizeof(float)), "fit d_axis_R_f") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_Z_f), static_cast<size_t>(n_axis) * sizeof(float)), "fit d_axis_Z_f") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_R_phi_f), static_cast<size_t>(n_axis) * sizeof(float)), "fit d_axis_R_phi_f") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_axis_Z_phi_f), static_cast<size_t>(n_axis) * sizeof(float)), "fit d_axis_Z_phi_f") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mat_f), mat_bytes_f), "fit d_mat_f") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rhs_f), static_cast<size_t>(n_points) * sizeof(float)), "fit d_rhs_f")) {
            cleanup();
            return 1;
        }
        convert_double_to_float_kernel<<<blocks1d_points, threads>>>(d_R, d_R_f, n_points);
        convert_double_to_float_kernel<<<blocks1d_points, threads>>>(d_Z, d_Z_f, n_points);
        convert_double_to_float_kernel<<<blocks1d_points, threads>>>(d_phi, d_phi_f, n_points);
        convert_double_to_float_kernel<<<(n_axis + threads - 1) / threads, threads>>>(d_axis_R, d_axis_R_f, n_axis);
        convert_double_to_float_kernel<<<(n_axis + threads - 1) / threads, threads>>>(d_axis_Z, d_axis_Z_f, n_axis);
        convert_double_to_float_kernel<<<(n_axis + threads - 1) / threads, threads>>>(d_axis_R_phi, d_axis_R_phi_f, n_axis);
        convert_double_to_float_kernel<<<(n_axis + threads - 1) / threads, threads>>>(d_axis_Z_phi, d_axis_Z_phi_f, n_axis);
        psi_fill_matrix_kernel_f32<<<n_points, threads, shared_f>>>(
            f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f, f->n_segments,
            d_R_f, d_Z_f, d_phi_f, d_axis_R_f, d_axis_Z_f, d_axis_R_phi_f, d_axis_Z_phi_f, n_axis,
            d_mode_a, d_mode_b, d_mode_m, d_mode_kind, n_coeff, nfp, static_cast<float>(a), poly_degree, m_tor, d_mat_f, d_rhs_f
        );
        float rhs_norm2_f = 0.0f;
        if (cuda_check(cudaGetLastError(), "fit psi_fill_matrix f32") ||
            cublas_check(cublasSdot(f->blas, n_points, d_rhs_f, 1, d_rhs_f, 1, &rhs_norm2_f), "fit rhs_norm2 s32") ||
            cuda_check(cudaDeviceSynchronize(), "fit assemble sync f32")) {
            cleanup();
            return 1;
        }
        rhs_norm2 = static_cast<double>(rhs_norm2_f);
    }
    stats.assemble_s = std::chrono::duration<double>(clock::now() - t).count();
    }

    if (solver_mode == 1) {
        SgpuNvtxRange solver_range(precision_mode == 1
            ? "psi.fullgpu.normal_eq.fp64" : "psi.fullgpu.normal_eq.fp32");
        t = clock::now();
        if (precision_mode == 1) {
            size_t ata_bytes = static_cast<size_t>(n_coeff) * static_cast<size_t>(n_coeff) * sizeof(double);
            size_t atb_bytes = static_cast<size_t>(n_coeff) * sizeof(double);
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ata_d), ata_bytes), "fit d_ata_d") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_atb_d), atb_bytes), "fit d_atb_d")) {
                cleanup();
                return 1;
            }
            const double one = 1.0, zero = 0.0;
            if (cublas_check(cublasDgemm(f->blas, CUBLAS_OP_N, CUBLAS_OP_T, n_coeff, n_coeff, n_points, &one, d_mat_d, n_coeff, d_mat_d, n_coeff, &zero, d_ata_d, n_coeff), "fit dgemm") ||
                cublas_check(cublasDgemv(f->blas, CUBLAS_OP_N, n_coeff, n_points, &one, d_mat_d, n_coeff, d_rhs_d, 1, &zero, d_atb_d, 1), "fit dgemv") ||
                cuda_check(cudaDeviceSynchronize(), "fit normal_eq sync d64")) {
                cleanup();
                return 1;
            }
        } else {
            size_t ata_bytes_f = static_cast<size_t>(n_coeff) * static_cast<size_t>(n_coeff) * sizeof(float);
            size_t atb_bytes_f = static_cast<size_t>(n_coeff) * sizeof(float);
            size_t ata_bytes_d = static_cast<size_t>(n_coeff) * static_cast<size_t>(n_coeff) * sizeof(double);
            size_t atb_bytes_d = static_cast<size_t>(n_coeff) * sizeof(double);
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ata_f), ata_bytes_f), "fit d_ata_f") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_atb_f), atb_bytes_f), "fit d_atb_f")) {
                cleanup();
                return 1;
            }
            const float one = 1.0f, zero = 0.0f;
            if (cublas_check(cublasSgemm(f->blas, CUBLAS_OP_N, CUBLAS_OP_T, n_coeff, n_coeff, n_points, &one, d_mat_f, n_coeff, d_mat_f, n_coeff, &zero, d_ata_f, n_coeff), "fit sgemm") ||
                cublas_check(cublasSgemv(f->blas, CUBLAS_OP_N, n_coeff, n_points, &one, d_mat_f, n_coeff, d_rhs_f, 1, &zero, d_atb_f, 1), "fit sgemv")) {
                cleanup();
                return 1;
            }
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ata_d), ata_bytes_d), "fit d_ata_d from f") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_atb_d), atb_bytes_d), "fit d_atb_d from f")) {
                cleanup();
                return 1;
            }
            convert_float_to_double_kernel<<<(n_coeff * n_coeff + threads - 1) / threads, threads>>>(d_ata_f, d_ata_d, n_coeff * n_coeff);
            convert_float_to_double_kernel<<<blocks1d_coeff, threads>>>(d_atb_f, d_atb_d, n_coeff);
            if (cuda_check(cudaGetLastError(), "fit convert ata/atb f32->f64") ||
                cuda_check(cudaDeviceSynchronize(), "fit normal_eq sync f32")) {
                cleanup();
                return 1;
            }
        }
        stats.linear_prep_s = std::chrono::duration<double>(clock::now() - t).count();

        t = clock::now();
        if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_scale), static_cast<size_t>(n_coeff) * sizeof(double)), "fit d_scale") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)), "fit d_info")) {
            cleanup();
            return 1;
        }
        extract_scale_kernel<<<blocks1d_coeff, threads>>>(d_ata_d, d_scale, n_coeff);
        dim3 block2(16, 16);
        dim3 grid2((n_coeff + block2.x - 1) / block2.x, (n_coeff + block2.y - 1) / block2.y);
        scale_system_kernel<<<grid2, block2>>>(d_ata_d, d_scale, n_coeff, ridge);
        scale_rhs_kernel<<<blocks1d_coeff, threads>>>(d_atb_d, d_scale, n_coeff);
        if (cuda_check(cudaGetLastError(), "fit scale system") || cuda_check(cudaDeviceSynchronize(), "fit scale sync")) {
            cleanup();
            return 1;
        }
        int lwork = 0;
        if (cusolver_check(cusolverDnDgetrf_bufferSize(f->solver, n_coeff, n_coeff, d_ata_d, n_coeff, &lwork), "fit getrf buffer") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_info_rhs), static_cast<size_t>(lwork) * sizeof(double)), "fit getrf work") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ipiv), static_cast<size_t>(n_coeff) * sizeof(int)), "fit getrf piv")) {
            cleanup();
            return 1;
        }
        if (cusolver_check(cusolverDnDgetrf(f->solver, n_coeff, n_coeff, d_ata_d, n_coeff, d_info_rhs, d_ipiv, d_info), "fit getrf") ||
            cuda_check(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "fit getrf info copy")) {
            cleanup();
            return 1;
        }
        if (h_info != 0) return fail("fit getrf failed");
        if (cusolver_check(cusolverDnDgetrs(f->solver, CUBLAS_OP_N, n_coeff, 1, d_ata_d, n_coeff, d_ipiv, d_atb_d, n_coeff, d_info), "fit getrs") ||
            cuda_check(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "fit getrs info copy")) {
            cleanup();
            return 1;
        }
        if (h_info != 0) return fail("fit getrs failed");
        unscale_coeff_kernel<<<blocks1d_coeff, threads>>>(d_atb_d, d_scale, n_coeff);
        if (cuda_check(cudaGetLastError(), "fit unscale coeff") || cuda_check(cudaDeviceSynchronize(), "fit solve sync")) {
            cleanup();
            return 1;
        }
        stats.solve_s = std::chrono::duration<double>(clock::now() - t).count();
        d_coeff_src_d = d_atb_d;
    } else {
        SgpuNvtxRange solver_range(precision_mode == 1
            ? "psi.fullgpu.qr.fp64" : "psi.fullgpu.qr.fp32");
        if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_scale), static_cast<size_t>(n_coeff) * sizeof(double)), "fit qr d_scale") ||
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)), "fit qr d_info")) {
            cleanup();
            return 1;
        }
        int qr_rows = n_points + n_coeff;
        size_t rhs_qr_bytes_d = static_cast<size_t>(qr_rows) * sizeof(double);
        size_t rhs_qr_bytes_f = static_cast<size_t>(qr_rows) * sizeof(float);
        dim3 block_t(16, 16);
        dim3 grid_t((n_coeff + block_t.x - 1) / block_t.x, (n_points + block_t.y - 1) / block_t.y);
        dim3 block_s(16, 16);
        dim3 grid_s((n_coeff + block_s.x - 1) / block_s.x, (n_points + block_s.y - 1) / block_s.y);
        double ridge_lambda = sqrt(fmax(ridge, 0.0));

        auto t_qr = clock::now();
        if (precision_mode == 1) {
            size_t mat_qr_bytes_d = static_cast<size_t>(qr_rows) * static_cast<size_t>(n_coeff) * sizeof(double);
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_qr_mat_d), mat_qr_bytes_d), "fit qr d_qr_mat_d") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_qr_rhs_d), rhs_qr_bytes_d), "fit qr d_qr_rhs_d")) {
                cleanup();
                return 1;
            }
            transpose_rowmajor_to_colmajor_kernel<<<grid_t, block_t>>>(d_mat_d, d_qr_mat_d, n_points, n_coeff, qr_rows);
            if (cuda_check(cudaGetLastError(), "fit qr transpose d64") ||
                cuda_check(cudaMemcpy(d_qr_rhs_d, d_rhs_d, point_bytes, cudaMemcpyDeviceToDevice), "fit qr copy rhs d64") ||
                cuda_check(cudaMemset(d_qr_rhs_d + n_points, 0, static_cast<size_t>(n_coeff) * sizeof(double)), "fit qr zero rhs tail d64") ||
                cuda_check(cudaDeviceSynchronize(), "fit qr transpose sync d64")) {
                cleanup();
                return 1;
            }
        } else {
            size_t mat_qr_bytes_f = static_cast<size_t>(qr_rows) * static_cast<size_t>(n_coeff) * sizeof(float);
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_qr_mat_f), mat_qr_bytes_f), "fit qr d_qr_mat_f") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_qr_rhs_f), rhs_qr_bytes_f), "fit qr d_qr_rhs_f")) {
                cleanup();
                return 1;
            }
            transpose_rowmajor_to_colmajor_kernel<<<grid_t, block_t>>>(d_mat_f, d_qr_mat_f, n_points, n_coeff, qr_rows);
            if (cuda_check(cudaGetLastError(), "fit qr transpose f32") ||
                cuda_check(cudaMemcpy(d_qr_rhs_f, d_rhs_f, static_cast<size_t>(n_points) * sizeof(float), cudaMemcpyDeviceToDevice), "fit qr copy rhs f32") ||
                cuda_check(cudaMemset(d_qr_rhs_f + n_points, 0, static_cast<size_t>(n_coeff) * sizeof(float)), "fit qr zero rhs tail f32") ||
                cuda_check(cudaDeviceSynchronize(), "fit qr transpose sync f32")) {
                cleanup();
                return 1;
            }
        }
        stats.qr_transpose_s = std::chrono::duration<double>(clock::now() - t_qr).count();

        t_qr = clock::now();
        if (warm_solver) {
            column_norms_kernel_f32<<<n_coeff, threads>>>(d_qr_mat_f, d_scale, n_points, qr_rows, n_coeff);
            scale_columns_kernel_f32<<<grid_s, block_s>>>(d_qr_mat_f, d_scale, n_points, qr_rows, n_coeff);
            set_ridge_tail_kernel<float><<<(n_coeff * n_coeff + threads - 1) / threads, threads>>>(
                d_qr_mat_f, n_points, qr_rows, n_coeff, static_cast<float>(ridge_lambda));
            if (cuda_check(cudaGetLastError(), "fit warm scale/augment") ||
                cuda_check(cudaDeviceSynchronize(), "fit warm scale/augment sync")) {
                cleanup();
                return 1;
            }
            stats.qr_scale_s = std::chrono::duration<double>(clock::now() - t_qr).count();
            stats.linear_prep_s = stats.qr_transpose_s + stats.qr_scale_s;
            t_qr = clock::now();
            coeff_host_f.resize(static_cast<size_t>(n_coeff));
            for (int index = 0; index < n_coeff; ++index) {
                coeff_host_f[static_cast<size_t>(index)] = static_cast<float>(coeff_host[index]);
            }
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_coeff_f), static_cast<size_t>(n_coeff) * sizeof(float)), "fit warm d_coeff_f") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_warm_residual_f), rhs_qr_bytes_f), "fit warm residual") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_warm_normal_f), static_cast<size_t>(n_coeff) * sizeof(float)), "fit warm normal") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_warm_direction_f), static_cast<size_t>(n_coeff) * sizeof(float)), "fit warm direction") ||
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_warm_image_f), rhs_qr_bytes_f), "fit warm image") ||
                cuda_check(cudaMemcpy(d_coeff_f, coeff_host_f.data(), static_cast<size_t>(n_coeff) * sizeof(float), cudaMemcpyHostToDevice), "fit warm copy initial")) {
                cleanup();
                return 1;
            }
            if (warm_preconditioned) {
                if (!sgpu_has_psi_warm_preconditioner(n_coeff)) {
                    return fail("fit warm center QR preconditioner is unavailable");
                }
                std::vector<double> endpoint_scale(static_cast<size_t>(n_coeff));
                std::vector<float> scale_ratio(static_cast<size_t>(n_coeff));
                if (cuda_check(cudaMemcpy(
                        endpoint_scale.data(), d_scale,
                        static_cast<size_t>(n_coeff) * sizeof(double), cudaMemcpyDeviceToHost
                    ), "fit warm copy endpoint scale")) {
                    cleanup();
                    return 1;
                }
                for (int index = 0; index < n_coeff; ++index) {
                    scale_ratio[static_cast<size_t>(index)] = static_cast<float>(
                        endpoint_scale[static_cast<size_t>(index)] /
                        g_psi_warm_preconditioner.center_scale[static_cast<size_t>(index)]
                    );
                }
                const size_t factor_bytes = static_cast<size_t>(n_coeff) * n_coeff * sizeof(float);
                if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_warm_transformed_direction_f), static_cast<size_t>(n_coeff) * sizeof(float)), "fit warm transformed direction") ||
                    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_warm_scale_ratio_f), static_cast<size_t>(n_coeff) * sizeof(float)), "fit warm scale ratio") ||
                    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_warm_center_factor_f), factor_bytes), "fit warm center factor") ||
                    cuda_check(cudaMemcpy(d_warm_scale_ratio_f, scale_ratio.data(), static_cast<size_t>(n_coeff) * sizeof(float), cudaMemcpyHostToDevice), "fit warm upload scale ratio") ||
                    cuda_check(cudaMemcpy(d_warm_center_factor_f, g_psi_warm_preconditioner.upper_factor.data(), factor_bytes, cudaMemcpyHostToDevice), "fit warm upload center factor")) {
                    cleanup();
                    return 1;
                }
            }
            scale_coeff_kernel_f32<<<blocks1d_coeff, threads>>>(d_coeff_f, d_scale, n_coeff);
            const float one = 1.0f, minus_one = -1.0f, zero = 0.0f;
            if (cuda_check(cudaGetLastError(), "fit warm scale initial") ||
                cuda_check(cudaMemcpy(d_warm_residual_f, d_qr_rhs_f, rhs_qr_bytes_f, cudaMemcpyDeviceToDevice), "fit warm copy rhs") ||
                cublas_check(cublasSgemv(f->blas, CUBLAS_OP_N, qr_rows, n_coeff, &minus_one,
                    d_qr_mat_f, qr_rows, d_coeff_f, 1, &one, d_warm_residual_f, 1), "fit warm initial residual") ||
                cublas_check(cublasSgemv(f->blas, CUBLAS_OP_T, qr_rows, n_coeff, &one,
                    d_qr_mat_f, qr_rows, d_warm_residual_f, 1, &zero, d_warm_normal_f, 1), "fit warm initial normal")) {
                cleanup();
                return 1;
            }
            if (warm_preconditioned) {
                scale_coeff_float_kernel<<<blocks1d_coeff, threads>>>(
                    d_warm_normal_f, d_warm_scale_ratio_f, n_coeff
                );
                if (cuda_check(cudaGetLastError(), "fit warm precondition normal scale") ||
                    cublas_check(cublasStrsv(
                        f->blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                        CUBLAS_DIAG_NON_UNIT, n_coeff, d_warm_center_factor_f,
                        n_coeff, d_warm_normal_f, 1
                    ), "fit warm center R^-T")) {
                    cleanup();
                    return 1;
                }
            }
            if (cublas_check(cublasScopy(
                    f->blas, n_coeff, d_warm_normal_f, 1,
                    d_warm_direction_f, 1
                ), "fit warm initial direction")) {
                cleanup();
                return 1;
            }
            float gamma = 0.0f;
            if (cublas_check(cublasSdot(f->blas, n_coeff, d_warm_normal_f, 1, d_warm_normal_f, 1, &gamma), "fit warm initial gamma")) {
                cleanup();
                return 1;
            }
            for (int iteration = 0; iteration < warm_iterations; ++iteration) {
                if (!(gamma > 0.0f) || !std::isfinite(gamma)) break;
                float* search_direction = d_warm_direction_f;
                if (warm_preconditioned) {
                    if (cublas_check(cublasScopy(
                            f->blas, n_coeff, d_warm_direction_f, 1,
                            d_warm_transformed_direction_f, 1
                        ), "fit warm copy preconditioned direction") ||
                        cublas_check(cublasStrsv(
                            f->blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                            CUBLAS_DIAG_NON_UNIT, n_coeff, d_warm_center_factor_f,
                            n_coeff, d_warm_transformed_direction_f, 1
                        ), "fit warm center R^-1")) {
                        cleanup();
                        return 1;
                    }
                    scale_coeff_float_kernel<<<blocks1d_coeff, threads>>>(
                        d_warm_transformed_direction_f, d_warm_scale_ratio_f, n_coeff
                    );
                    if (cuda_check(cudaGetLastError(), "fit warm precondition direction scale")) {
                        cleanup();
                        return 1;
                    }
                    search_direction = d_warm_transformed_direction_f;
                }
                if (cublas_check(cublasSgemv(f->blas, CUBLAS_OP_N, qr_rows, n_coeff, &one,
                        d_qr_mat_f, qr_rows, search_direction, 1, &zero, d_warm_image_f, 1), "fit warm A p")) {
                    cleanup();
                    return 1;
                }
                float delta = 0.0f;
                if (cublas_check(cublasSdot(f->blas, qr_rows, d_warm_image_f, 1, d_warm_image_f, 1, &delta), "fit warm delta")) {
                    cleanup();
                    return 1;
                }
                if (!(delta > 0.0f) || !std::isfinite(delta)) break;
                const float alpha = gamma / delta;
                const float minus_alpha = -alpha;
                if (cublas_check(cublasSaxpy(f->blas, n_coeff, &alpha, search_direction, 1, d_coeff_f, 1), "fit warm solution update") ||
                    cublas_check(cublasSaxpy(f->blas, qr_rows, &minus_alpha, d_warm_image_f, 1, d_warm_residual_f, 1), "fit warm residual update") ||
                    cublas_check(cublasSgemv(f->blas, CUBLAS_OP_T, qr_rows, n_coeff, &one,
                        d_qr_mat_f, qr_rows, d_warm_residual_f, 1, &zero, d_warm_normal_f, 1), "fit warm A^T r")) {
                    cleanup();
                    return 1;
                }
                if (warm_preconditioned) {
                    scale_coeff_float_kernel<<<blocks1d_coeff, threads>>>(
                        d_warm_normal_f, d_warm_scale_ratio_f, n_coeff
                    );
                    if (cuda_check(cudaGetLastError(), "fit warm precondition next normal scale") ||
                        cublas_check(cublasStrsv(
                            f->blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                            CUBLAS_DIAG_NON_UNIT, n_coeff, d_warm_center_factor_f,
                            n_coeff, d_warm_normal_f, 1
                        ), "fit warm next center R^-T")) {
                        cleanup();
                        return 1;
                    }
                }
                float gamma_new = 0.0f;
                if (cublas_check(cublasSdot(f->blas, n_coeff, d_warm_normal_f, 1, d_warm_normal_f, 1, &gamma_new), "fit warm gamma")) {
                    cleanup();
                    return 1;
                }
                if (!(gamma_new > 0.0f) || !std::isfinite(gamma_new)) break;
                const float beta = gamma_new / gamma;
                if (cublas_check(cublasSscal(f->blas, n_coeff, &beta, d_warm_direction_f, 1), "fit warm direction scale") ||
                    cublas_check(cublasSaxpy(f->blas, n_coeff, &one, d_warm_normal_f, 1, d_warm_direction_f, 1), "fit warm direction update")) {
                    cleanup();
                    return 1;
                }
                gamma = gamma_new;
            }
            unscale_coeff_kernel_f32<<<blocks1d_coeff, threads>>>(d_coeff_f, d_scale, n_coeff);
            if (cuda_check(cudaGetLastError(), "fit warm unscale") ||
                cuda_check(cudaDeviceSynchronize(), "fit warm solve sync")) {
                cleanup();
                return 1;
            }
            stats.qr_factor_s = 0.0;
            stats.qr_apply_qtb_s = std::chrono::duration<double>(clock::now() - t_qr).count();
            stats.qr_tri_s = 0.0;
            stats.solve_s = stats.qr_apply_qtb_s;
            d_coeff_src_f = d_coeff_f;
        } else if (precision_mode == 1) {
            column_norms_kernel_f64<<<n_coeff, threads>>>(d_qr_mat_d, d_scale, n_points, qr_rows, n_coeff);
            scale_columns_kernel_f64<<<grid_s, block_s>>>(d_qr_mat_d, d_scale, n_points, qr_rows, n_coeff);
            set_ridge_tail_kernel<double><<<(n_coeff * n_coeff + threads - 1) / threads, threads>>>(d_qr_mat_d, n_points, qr_rows, n_coeff, ridge_lambda);
        } else {
            column_norms_kernel_f32<<<n_coeff, threads>>>(d_qr_mat_f, d_scale, n_points, qr_rows, n_coeff);
            scale_columns_kernel_f32<<<grid_s, block_s>>>(d_qr_mat_f, d_scale, n_points, qr_rows, n_coeff);
            set_ridge_tail_kernel<float><<<(n_coeff * n_coeff + threads - 1) / threads, threads>>>(d_qr_mat_f, n_points, qr_rows, n_coeff, static_cast<float>(ridge_lambda));
        }
        if (!warm_solver) {
            if (cuda_check(cudaGetLastError(), "fit qr scale/augment") ||
                cuda_check(cudaDeviceSynchronize(), "fit qr scale/augment sync")) {
                cleanup();
                return 1;
            }
            stats.qr_scale_s = std::chrono::duration<double>(clock::now() - t_qr).count();
            stats.linear_prep_s = stats.qr_transpose_s + stats.qr_scale_s;
        }

        if (precision_mode == 2 &&
            !maybe_write_psi_qr_snapshot(
                d_qr_mat_f, d_qr_rhs_f, d_scale, qr_rows, n_coeff, n_points, ridge
            )) {
            cleanup();
            return 1;
        }

        t_qr = clock::now();
        if (!warm_solver && precision_mode == 1) {
            int lwork_geqrf = 0, lwork_ormqr = 0;
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_tau_d), static_cast<size_t>(n_coeff) * sizeof(double)), "fit qr d_tau_d") ||
                cusolver_check(cusolverDnDgeqrf_bufferSize(f->solver, qr_rows, n_coeff, d_qr_mat_d, qr_rows, &lwork_geqrf), "fit qr geqrf buffer d64") ||
                cusolver_check(cusolverDnDormqr_bufferSize(f->solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, n_coeff, d_qr_mat_d, qr_rows, d_tau_d, d_qr_rhs_d, qr_rows, &lwork_ormqr), "fit qr ormqr buffer d64")) {
                cleanup();
                return 1;
            }
            int lwork = (lwork_geqrf > lwork_ormqr) ? lwork_geqrf : lwork_ormqr;
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_work_d), static_cast<size_t>(lwork) * sizeof(double)), "fit qr d_work_d")) {
                cleanup();
                return 1;
            }
            if (cusolver_check(cusolverDnDgeqrf(f->solver, qr_rows, n_coeff, d_qr_mat_d, qr_rows, d_tau_d, d_work_d, lwork, d_info), "fit qr geqrf d64") ||
                cuda_check(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "fit qr geqrf info d64")) {
                cleanup();
                return 1;
            }
            if (h_info != 0) return fail("fit qr geqrf failed");
        } else if (!warm_solver) {
            int lwork_geqrf = 0, lwork_ormqr = 0;
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_tau_f), static_cast<size_t>(n_coeff) * sizeof(float)), "fit qr d_tau_f") ||
                cusolver_check(cusolverDnSgeqrf_bufferSize(f->solver, qr_rows, n_coeff, d_qr_mat_f, qr_rows, &lwork_geqrf), "fit qr geqrf buffer f32") ||
                cusolver_check(cusolverDnSormqr_bufferSize(f->solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, n_coeff, d_qr_mat_f, qr_rows, d_tau_f, d_qr_rhs_f, qr_rows, &lwork_ormqr), "fit qr ormqr buffer f32")) {
                cleanup();
                return 1;
            }
            int lwork = (lwork_geqrf > lwork_ormqr) ? lwork_geqrf : lwork_ormqr;
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_work_f), static_cast<size_t>(lwork) * sizeof(float)), "fit qr d_work_f")) {
                cleanup();
                return 1;
            }
            if (cusolver_check(cusolverDnSgeqrf(f->solver, qr_rows, n_coeff, d_qr_mat_f, qr_rows, d_tau_f, d_work_f, lwork, d_info), "fit qr geqrf f32") ||
                cuda_check(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "fit qr geqrf info f32")) {
                cleanup();
                return 1;
            }
            if (h_info != 0) return fail("fit qr geqrf failed");
        }
        if (!warm_solver) {
            if (cuda_check(cudaDeviceSynchronize(), "fit qr geqrf sync")) {
                cleanup();
                return 1;
            }
            stats.qr_factor_s = std::chrono::duration<double>(clock::now() - t_qr).count();
            if (precision_mode == 2 && solver_mode == 2 &&
                g_psi_warm_preconditioner.capture_enabled) {
                g_psi_warm_preconditioner.coefficient_count = n_coeff;
                g_psi_warm_preconditioner.upper_factor.resize(
                    static_cast<size_t>(n_coeff) * n_coeff
                );
                g_psi_warm_preconditioner.center_scale.resize(static_cast<size_t>(n_coeff));
                if (cuda_check(cudaMemcpy2D(
                        g_psi_warm_preconditioner.upper_factor.data(),
                        static_cast<size_t>(n_coeff) * sizeof(float),
                        d_qr_mat_f, static_cast<size_t>(qr_rows) * sizeof(float),
                        static_cast<size_t>(n_coeff) * sizeof(float), n_coeff,
                        cudaMemcpyDeviceToHost
                    ), "fit capture center R") ||
                    cuda_check(cudaMemcpy(
                        g_psi_warm_preconditioner.center_scale.data(), d_scale,
                        static_cast<size_t>(n_coeff) * sizeof(double), cudaMemcpyDeviceToHost
                    ), "fit capture center scale")) {
                    cleanup();
                    return 1;
                }
            }
        }

        t_qr = clock::now();
        if (!warm_solver && precision_mode == 1) {
            int lwork_ormqr = 0;
            if (cusolver_check(cusolverDnDormqr_bufferSize(f->solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, n_coeff, d_qr_mat_d, qr_rows, d_tau_d, d_qr_rhs_d, qr_rows, &lwork_ormqr), "fit qr ormqr buffer2 d64")) {
                cleanup();
                return 1;
            }
            cudaFree(d_work_d);
            d_work_d = nullptr;
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_work_d), static_cast<size_t>(lwork_ormqr) * sizeof(double)), "fit qr d_work_d ormqr")) {
                cleanup();
                return 1;
            }
            if (cusolver_check(cusolverDnDormqr(f->solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, n_coeff, d_qr_mat_d, qr_rows, d_tau_d, d_qr_rhs_d, qr_rows, d_work_d, lwork_ormqr, d_info), "fit qr ormqr d64") ||
                cuda_check(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "fit qr ormqr info d64")) {
                cleanup();
                return 1;
            }
            if (h_info != 0) return fail("fit qr ormqr failed");
        } else if (!warm_solver) {
            int lwork_ormqr = 0;
            if (cusolver_check(cusolverDnSormqr_bufferSize(f->solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, n_coeff, d_qr_mat_f, qr_rows, d_tau_f, d_qr_rhs_f, qr_rows, &lwork_ormqr), "fit qr ormqr buffer2 f32")) {
                cleanup();
                return 1;
            }
            cudaFree(d_work_f);
            d_work_f = nullptr;
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_work_f), static_cast<size_t>(lwork_ormqr) * sizeof(float)), "fit qr d_work_f ormqr")) {
                cleanup();
                return 1;
            }
            if (cusolver_check(cusolverDnSormqr(f->solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, n_coeff, d_qr_mat_f, qr_rows, d_tau_f, d_qr_rhs_f, qr_rows, d_work_f, lwork_ormqr, d_info), "fit qr ormqr f32") ||
                cuda_check(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "fit qr ormqr info f32")) {
                cleanup();
                return 1;
            }
            if (h_info != 0) return fail("fit qr ormqr failed");
        }
        if (!warm_solver) {
            if (cuda_check(cudaDeviceSynchronize(), "fit qr ormqr sync")) {
                cleanup();
                return 1;
            }
            stats.qr_apply_qtb_s = std::chrono::duration<double>(clock::now() - t_qr).count();
        }

        t_qr = clock::now();
        if (!warm_solver && precision_mode == 1) {
            if (cublas_check(cublasDtrsv(f->blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n_coeff, d_qr_mat_d, qr_rows, d_qr_rhs_d, 1), "fit qr trsv d64")) {
                cleanup();
                return 1;
            }
            unscale_coeff_kernel<<<blocks1d_coeff, threads>>>(d_qr_rhs_d, d_scale, n_coeff);
        } else if (!warm_solver) {
            if (cublas_check(cublasStrsv(f->blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n_coeff, d_qr_mat_f, qr_rows, d_qr_rhs_f, 1), "fit qr trsv f32")) {
                cleanup();
                return 1;
            }
            unscale_coeff_kernel_f32<<<blocks1d_coeff, threads>>>(d_qr_rhs_f, d_scale, n_coeff);
        }
        if (!warm_solver) {
            if (cuda_check(cudaGetLastError(), "fit qr unscale coeff") ||
                cuda_check(cudaDeviceSynchronize(), "fit qr trsv sync")) {
                cleanup();
                return 1;
            }
            stats.qr_tri_s = std::chrono::duration<double>(clock::now() - t_qr).count();
            stats.solve_s = stats.qr_factor_s + stats.qr_apply_qtb_s + stats.qr_tri_s;
        }
        if (!warm_solver) {
            if (precision_mode == 1) d_coeff_src_d = d_qr_rhs_d;
            else d_coeff_src_f = d_qr_rhs_f;
        }
    }

    t = clock::now();
    {
    SGPU_NVTX_RANGE("psi.fullgpu.residual");
    if (precision_mode == 1) {
        if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pred_d), point_bytes), "fit d_pred_d")) {
            cleanup();
            return 1;
        }
        const double one = 1.0, zero = 0.0;
        if (cublas_check(cublasDgemv(f->blas, CUBLAS_OP_T, n_coeff, n_points, &one, d_mat_d, n_coeff, d_coeff_src_d, 1, &zero, d_pred_d, 1), "fit residual gemv d64")) {
            cleanup();
            return 1;
        }
        axpy_negative_kernel<<<blocks1d_points, threads>>>(d_pred_d, d_rhs_d, d_pred_d, n_points);
        double resid2 = 0.0;
        if (cuda_check(cudaGetLastError(), "fit residual axpy d64") ||
            cublas_check(cublasDdot(f->blas, n_points, d_pred_d, 1, d_pred_d, 1, &resid2), "fit residual dot d64") ||
            cuda_check(cudaDeviceSynchronize(), "fit residual sync d64")) {
            cleanup();
            return 1;
        }
        train_rms = sqrt(fmax(resid2, 0.0) / static_cast<double>(n_points));
    } else {
        if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pred_f), static_cast<size_t>(n_points) * sizeof(float)), "fit d_pred_f")) {
            cleanup();
            return 1;
        }
        if (solver_mode == 1) {
            if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_coeff_f), static_cast<size_t>(n_coeff) * sizeof(float)), "fit d_coeff_f")) {
                cleanup();
                return 1;
            }
            convert_double_to_float_kernel<<<blocks1d_coeff, threads>>>(d_atb_d, d_coeff_f, n_coeff);
            d_coeff_src_f = d_coeff_f;
        }
        const float one = 1.0f, zero = 0.0f;
        if (cublas_check(cublasSgemv(f->blas, CUBLAS_OP_T, n_coeff, n_points, &one, d_mat_f, n_coeff, d_coeff_src_f, 1, &zero, d_pred_f, 1), "fit residual gemv f32")) {
            cleanup();
            return 1;
        }
        axpy_negative_kernel<<<blocks1d_points, threads>>>(d_pred_f, d_rhs_f, d_pred_f, n_points);
        float resid2_f = 0.0f;
        if (cuda_check(cudaGetLastError(), "fit residual axpy f32") ||
            cublas_check(cublasSdot(f->blas, n_points, d_pred_f, 1, d_pred_f, 1, &resid2_f), "fit residual dot f32") ||
            cuda_check(cudaDeviceSynchronize(), "fit residual sync f32")) {
            cleanup();
            return 1;
        }
        train_rms = sqrt(fmax(static_cast<double>(resid2_f), 0.0) / static_cast<double>(n_points));
    }
    stats.residual_s = std::chrono::duration<double>(clock::now() - t).count();
    }

    t = clock::now();
    {
    SGPU_NVTX_RANGE("psi.fullgpu.copy_out");
    if (precision_mode == 1) {
        if (cuda_check(cudaMemcpy(coeff_host, d_coeff_src_d, static_cast<size_t>(n_coeff) * sizeof(double), cudaMemcpyDeviceToHost), "fit copy coeff d64")) {
            cleanup();
            return 1;
        }
    } else {
        coeff_host_f.resize(static_cast<size_t>(n_coeff));
        if (cuda_check(cudaMemcpy(coeff_host_f.data(), d_coeff_src_f, static_cast<size_t>(n_coeff) * sizeof(float), cudaMemcpyDeviceToHost), "fit copy coeff f32")) {
            cleanup();
            return 1;
        }
        for (int i = 0; i < n_coeff; ++i) coeff_host[i] = static_cast<double>(coeff_host_f[static_cast<size_t>(i)]);
    }
    *train_rms_out = train_rms;
    stats.copy_out_s = std::chrono::duration<double>(clock::now() - t).count();
    }
    stats.total_s = std::chrono::duration<double>(clock::now() - t_total).count();
    if (stats_out && stats_len >= 12) {
        stats_out[0] = stats.copy_in_s;
        stats_out[1] = stats.assemble_s;
        stats_out[2] = stats.linear_prep_s;
        stats_out[3] = stats.solve_s;
        stats_out[4] = stats.residual_s;
        stats_out[5] = stats.copy_out_s;
        stats_out[6] = stats.total_s;
        stats_out[7] = stats.qr_transpose_s;
        stats_out[8] = stats.qr_scale_s;
        stats_out[9] = stats.qr_factor_s;
        stats_out[10] = stats.qr_apply_qtb_s;
        stats_out[11] = stats.qr_tri_s;
    }
    cleanup();
    set_error("");
    return 0;
}

} // extern "C"
