#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace coil_eval::field {

constexpr double MU0_OVER_4PI = 1.0e-7;
constexpr double TWOPI = 6.283185307179586476925286766559;
constexpr int WARP_SIZE = 32;
#ifndef SGPU_WARPS_PER_BLOCK
#define SGPU_WARPS_PER_BLOCK 8
#endif
#ifndef SGPU_SEG_TILE
#define SGPU_SEG_TILE 256
#endif
constexpr int WARPS_PER_BLOCK = SGPU_WARPS_PER_BLOCK;
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;
constexpr int SEG_TILE = SGPU_SEG_TILE;
constexpr int MAX_PSI_DEGREE = 24;
constexpr int MAX_PSI_MTOR = 32;
constexpr int SURFACE_THREADS = 64;

struct PsiWarmPreconditionerCache {
    bool capture_enabled = false;
    int coefficient_count = 0;
    std::vector<float> upper_factor;
    std::vector<double> center_scale;
};

struct CoilField {
    int device_id = 0;
    int n_segments = 0;
    double* d_x = nullptr;
    double* d_y = nullptr;
    double* d_z = nullptr;
    double* d_wx = nullptr;
    double* d_wy = nullptr;
    double* d_wz = nullptr;
    float* d_x_f = nullptr;
    float* d_y_f = nullptr;
    float* d_z_f = nullptr;
    float* d_wx_f = nullptr;
    float* d_wy_f = nullptr;
    float* d_wz_f = nullptr;
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
};

struct BatchCoilField {
    int device_id = 0;
    int query_count = 0;
    int n_segments = 0;
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_z = nullptr;
    float* d_wx = nullptr;
    float* d_wy = nullptr;
    float* d_wz = nullptr;
};

// Owned by state.cpp. Every host thread has exactly one error and one center
// QR preconditioner, shared by the formal-fit and nearby-PCGLS translation units.
extern thread_local std::string g_last_error;
extern thread_local PsiWarmPreconditionerCache g_psi_warm_preconditioner;

void set_error(const char* msg);
void set_error(const std::string& msg);
int cuda_check(cudaError_t err, const char* where);
int cublas_check(cublasStatus_t status, const char* where);
int cusolver_check(cusolverStatus_t status, const char* where);

} // namespace coil_eval::field
