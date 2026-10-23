#include "field/device_field.cuh"
#include "coil_field.h"
#include "coil_field_internal.h"
#include "nvtx_profile.h"


namespace coil_eval::field {

__global__ void eval_B_kernel(
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ xyz,
    double* __restrict__ B,
    int npoints
) {
    extern __shared__ double sh[];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & 31;
    int point = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    double bx = 0.0, by = 0.0, bz = 0.0;
    if (point < npoints) {
        double px = xyz[3 * point + 0];
        double py = xyz[3 * point + 1];
        double pz = xyz[3 * point + 2];
        eval_B_warp(px, py, pz, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, bx, by, bz);
        if (lane == 0) {
            B[3 * point + 0] = bx;
            B[3 * point + 1] = by;
            B[3 * point + 2] = bz;
        }
    } else {
        // Still participate in shared-memory synchronizations inside eval_B_warp.
        eval_B_warp(0.0, 0.0, 0.0, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, bx, by, bz);
    }
}

template <typename T>
__global__ void eval_B_only_kernel(
    const T* __restrict__ seg_x,
    const T* __restrict__ seg_y,
    const T* __restrict__ seg_z,
    const T* __restrict__ seg_wx,
    const T* __restrict__ seg_wy,
    const T* __restrict__ seg_wz,
    int nseg,
    const T* __restrict__ xyz,
    T* __restrict__ B,
    int npoints
) {
    extern __shared__ unsigned char shared_raw[];
    T* sx = reinterpret_cast<T*>(shared_raw);
    T* sy = sx + SEG_TILE;
    T* sz = sy + SEG_TILE;
    T* swx = sz + SEG_TILE;
    T* swy = swx + SEG_TILE;
    T* swz = swy + SEG_TILE;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int point = blockIdx.x * WARPS_PER_BLOCK + tid / WARP_SIZE;
    bool valid = point < npoints;
    T px = valid ? xyz[3 * point + 0] : T(0);
    T py = valid ? xyz[3 * point + 1] : T(0);
    T pz = valid ? xyz[3 * point + 2] : T(0);
    T accum[3] = {};

    for (int base = 0; base < nseg; base += SEG_TILE) {
        int count = min(SEG_TILE, nseg - base);
        for (int j = tid; j < count; j += blockDim.x) {
            int index = base + j;
            sx[j] = seg_x[index];
            sy[j] = seg_y[index];
            sz[j] = seg_z[index];
            swx[j] = seg_wx[index];
            swy[j] = seg_wy[index];
            swz[j] = seg_wz[index];
        }
        __syncthreads();
        if (valid) {
            for (int j = lane; j < count; j += WARP_SIZE) {
                T rx = px - sx[j];
                T ry = py - sy[j];
                T rz = pz - sz[j];
                T invr = T(1) / sqrt(rx * rx + ry * ry + rz * rz + T(1.0e-30));
                T invr3 = invr * invr * invr;
                T wx = swx[j];
                T wy = swy[j];
                T wz = swz[j];
                accum[0] += (wy * rz - wz * ry) * invr3;
                accum[1] += (wz * rx - wx * rz) * invr3;
                accum[2] += (wx * ry - wy * rx) * invr3;
            }
        }
        __syncthreads();
    }
    for (int component = 0; component < 3; ++component) {
        accum[component] = warp_sum_t(accum[component]) * T(MU0_OVER_4PI);
    }
    if (valid && lane == 0) {
        B[3 * point + 0] = accum[0];
        B[3 * point + 1] = accum[1];
        B[3 * point + 2] = accum[2];
    }
}

template <typename T>
__global__ void eval_B_grad_kernel(
    const T* __restrict__ seg_x,
    const T* __restrict__ seg_y,
    const T* __restrict__ seg_z,
    const T* __restrict__ seg_wx,
    const T* __restrict__ seg_wy,
    const T* __restrict__ seg_wz,
    int nseg,
    const T* __restrict__ xyz,
    T* __restrict__ B,
    T* __restrict__ grad_B,
    int npoints
) {
    extern __shared__ unsigned char shared_raw[];
    T* sx = reinterpret_cast<T*>(shared_raw);
    T* sy = sx + SEG_TILE;
    T* sz = sy + SEG_TILE;
    T* swx = sz + SEG_TILE;
    T* swy = swx + SEG_TILE;
    T* swz = swy + SEG_TILE;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid / WARP_SIZE;
    int point = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    bool valid = point < npoints;
    T px = valid ? xyz[3 * point + 0] : T(0);
    T py = valid ? xyz[3 * point + 1] : T(0);
    T pz = valid ? xyz[3 * point + 2] : T(0);

    T accum[12] = {};
    for (int base = 0; base < nseg; base += SEG_TILE) {
        int count = min(SEG_TILE, nseg - base);
        for (int j = tid; j < count; j += blockDim.x) {
            int index = base + j;
            sx[j] = seg_x[index];
            sy[j] = seg_y[index];
            sz[j] = seg_z[index];
            swx[j] = seg_wx[index];
            swy[j] = seg_wy[index];
            swz[j] = seg_wz[index];
        }
        __syncthreads();

        if (valid) {
            for (int j = lane; j < count; j += WARP_SIZE) {
                T rx = px - sx[j];
                T ry = py - sy[j];
                T rz = pz - sz[j];
                T r2 = rx * rx + ry * ry + rz * rz + T(1.0e-30);
                T invr = T(1) / sqrt(r2);
                T invr2 = invr * invr;
                T invr3 = invr * invr2;
                T invr5 = invr3 * invr2;
                T wx = swx[j];
                T wy = swy[j];
                T wz = swz[j];
                T ux = wy * rz - wz * ry;
                T uy = wz * rx - wx * rz;
                T uz = wx * ry - wy * rx;

                accum[0] += ux * invr3;
                accum[1] += uy * invr3;
                accum[2] += uz * invr3;

                T common_x = T(-3) * rx * invr5;
                T common_y = T(-3) * ry * invr5;
                T common_z = T(-3) * rz * invr5;
                accum[3] += ux * common_x;
                accum[4] += -wz * invr3 + ux * common_y;
                accum[5] += wy * invr3 + ux * common_z;
                accum[6] += wz * invr3 + uy * common_x;
                accum[7] += uy * common_y;
                accum[8] += -wx * invr3 + uy * common_z;
                accum[9] += -wy * invr3 + uz * common_x;
                accum[10] += wx * invr3 + uz * common_y;
                accum[11] += uz * common_z;
            }
        }
        __syncthreads();
    }

    for (int component = 0; component < 12; ++component) {
        accum[component] = warp_sum_t(accum[component]) * T(MU0_OVER_4PI);
    }
    if (valid && lane == 0) {
        B[3 * point + 0] = accum[0];
        B[3 * point + 1] = accum[1];
        B[3 * point + 2] = accum[2];
        for (int component = 0; component < 9; ++component) {
            grad_B[9 * point + component] = accum[3 + component];
        }
    }
}

template <bool WITH_GRADIENT>
__global__ void eval_B_batch_f32_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const float* __restrict__ xyz,
    float* __restrict__ B,
    float* __restrict__ grad_B,
    int points_per_query,
    int blocks_per_query
) {
    extern __shared__ float shared[];
    float* sx = shared;
    float* sy = sx + SEG_TILE;
    float* sz = sy + SEG_TILE;
    float* swx = sz + SEG_TILE;
    float* swy = swx + SEG_TILE;
    float* swz = swy + SEG_TILE;

    const int query = blockIdx.x / blocks_per_query;
    const int query_block = blockIdx.x - query * blocks_per_query;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int point = query_block * WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE;
    const bool valid = point < points_per_query;
    const size_t point_index = static_cast<size_t>(query) * points_per_query + point;
    const size_t segment_offset = static_cast<size_t>(query) * nseg;
    const float px = valid ? xyz[3 * point_index] : 0.0f;
    const float py = valid ? xyz[3 * point_index + 1] : 0.0f;
    const float pz = valid ? xyz[3 * point_index + 2] : 0.0f;
    float accum[12] = {};

    for (int base = 0; base < nseg; base += SEG_TILE) {
        const int count = min(SEG_TILE, nseg - base);
        for (int item = threadIdx.x; item < count; item += blockDim.x) {
            const size_t source = segment_offset + base + item;
            sx[item] = seg_x[source];
            sy[item] = seg_y[source];
            sz[item] = seg_z[source];
            swx[item] = seg_wx[source];
            swy[item] = seg_wy[source];
            swz[item] = seg_wz[source];
        }
        __syncthreads();
        if (valid) {
            for (int item = lane; item < count; item += WARP_SIZE) {
                const float rx = px - sx[item];
                const float ry = py - sy[item];
                const float rz = pz - sz[item];
                const float r2 = rx * rx + ry * ry + rz * rz + 1.0e-30f;
                const float invr = rsqrtf(r2);
                const float invr2 = invr * invr;
                const float invr3 = invr * invr2;
                const float wx_value = swx[item];
                const float wy_value = swy[item];
                const float wz_value = swz[item];
                const float ux = wy_value * rz - wz_value * ry;
                const float uy = wz_value * rx - wx_value * rz;
                const float uz = wx_value * ry - wy_value * rx;
                accum[0] += ux * invr3;
                accum[1] += uy * invr3;
                accum[2] += uz * invr3;
                if constexpr (WITH_GRADIENT) {
                    const float invr5 = invr3 * invr2;
                    const float common_x = -3.0f * rx * invr5;
                    const float common_y = -3.0f * ry * invr5;
                    const float common_z = -3.0f * rz * invr5;
                    accum[3] += ux * common_x;
                    accum[4] += -wz_value * invr3 + ux * common_y;
                    accum[5] += wy_value * invr3 + ux * common_z;
                    accum[6] += wz_value * invr3 + uy * common_x;
                    accum[7] += uy * common_y;
                    accum[8] += -wx_value * invr3 + uy * common_z;
                    accum[9] += -wy_value * invr3 + uz * common_x;
                    accum[10] += wx_value * invr3 + uz * common_y;
                    accum[11] += uz * common_z;
                }
            }
        }
        __syncthreads();
    }
    const int component_count = WITH_GRADIENT ? 12 : 3;
    for (int component = 0; component < component_count; ++component) {
        accum[component] = warp_sum_t(accum[component]) *
            static_cast<float>(MU0_OVER_4PI);
    }
    if (valid && lane == 0) {
        B[3 * point_index] = accum[0];
        B[3 * point_index + 1] = accum[1];
        B[3 * point_index + 2] = accum[2];
        if constexpr (WITH_GRADIENT) {
            for (int component = 0; component < 9; ++component) {
                grad_B[9 * point_index + component] = accum[3 + component];
            }
        }
    }
}

} // namespace coil_eval::field

// The public ABI stays global and unmangled; implementation state is named.
using namespace coil_eval::field;

extern "C" {

int sgpu_batch_eval_B_f32(
    void* handle,
    const float* xyz_host,
    float* B_host,
    int points_per_query
) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field || !xyz_host || !B_host || points_per_query <= 0) {
        set_error("invalid batch eval_B arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "batch eval_B cudaSetDevice")) return 1;
    const size_t point_count = static_cast<size_t>(field->query_count) * points_per_query;
    const size_t vector_bytes = point_count * 3 * sizeof(float);
    float* d_xyz = nullptr;
    float* d_B = nullptr;
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_xyz), vector_bytes), "batch eval_B xyz allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_B), vector_bytes), "batch eval_B output allocation")) {
        cudaFree(d_xyz);
        cudaFree(d_B);
        return 1;
    }
    const int blocks_per_query = (points_per_query + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t shared_bytes = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    const bool failed =
        cuda_check(cudaMemcpy(d_xyz, xyz_host, vector_bytes, cudaMemcpyHostToDevice), "batch eval_B input copy") ||
        (eval_B_batch_f32_kernel<false><<<field->query_count * blocks_per_query, THREADS_PER_BLOCK, shared_bytes>>>(
            field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz,
            field->n_segments, d_xyz, d_B, nullptr, points_per_query, blocks_per_query
        ), false) ||
        cuda_check(cudaGetLastError(), "batch eval_B kernel") ||
        cuda_check(cudaDeviceSynchronize(), "batch eval_B sync") ||
        cuda_check(cudaMemcpy(B_host, d_B, vector_bytes, cudaMemcpyDeviceToHost), "batch eval_B output copy");
    cudaFree(d_xyz);
    cudaFree(d_B);
    if (failed) return 1;
    set_error("");
    return 0;
}

int sgpu_internal_batch_eval_B_f32_device(
    void* handle,
    const float* xyz_device,
    float* B_device,
    int points_per_query
) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field || !xyz_device || !B_device || points_per_query <= 0) {
        set_error("invalid device batch eval_B arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "device batch eval_B cudaSetDevice")) return 1;
    const int blocks_per_query = (points_per_query + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t shared_bytes = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    eval_B_batch_f32_kernel<false><<<
        field->query_count * blocks_per_query, THREADS_PER_BLOCK, shared_bytes
    >>>(
        field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz,
        field->n_segments, xyz_device, B_device, nullptr,
        points_per_query, blocks_per_query
    );
    return cuda_check(cudaGetLastError(), "device batch eval_B kernel");
}

int sgpu_internal_batch_eval_B_grad_f32_device(
    void* handle,
    const float* xyz_device,
    float* B_device,
    float* grad_B_device,
    int points_per_query
) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field || !xyz_device || !B_device || !grad_B_device || points_per_query <= 0) {
        set_error("invalid device batch eval_B_grad arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "device batch eval_B_grad cudaSetDevice")) return 1;
    const int blocks_per_query = (points_per_query + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t shared_bytes = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    eval_B_batch_f32_kernel<true><<<
        field->query_count * blocks_per_query, THREADS_PER_BLOCK, shared_bytes
    >>>(
        field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz,
        field->n_segments, xyz_device, B_device, grad_B_device,
        points_per_query, blocks_per_query
    );
    return cuda_check(cudaGetLastError(), "device batch eval_B_grad kernel");
}

int sgpu_batch_eval_B_grad_f32(
    void* handle,
    const float* xyz_host,
    float* B_host,
    float* grad_B_host,
    int points_per_query
) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field || !xyz_host || !B_host || !grad_B_host || points_per_query <= 0) {
        set_error("invalid batch eval_B_grad arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "batch eval_B_grad cudaSetDevice")) return 1;
    const size_t point_count = static_cast<size_t>(field->query_count) * points_per_query;
    const size_t vector_bytes = point_count * 3 * sizeof(float);
    const size_t gradient_bytes = point_count * 9 * sizeof(float);
    float* d_xyz = nullptr;
    float* d_B = nullptr;
    float* d_grad_B = nullptr;
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_xyz), vector_bytes), "batch eval_B_grad xyz allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_B), vector_bytes), "batch eval_B_grad B allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_grad_B), gradient_bytes), "batch eval_B_grad gradient allocation")) {
        cudaFree(d_xyz);
        cudaFree(d_B);
        cudaFree(d_grad_B);
        return 1;
    }
    const int blocks_per_query = (points_per_query + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t shared_bytes = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    const bool failed =
        cuda_check(cudaMemcpy(d_xyz, xyz_host, vector_bytes, cudaMemcpyHostToDevice), "batch eval_B_grad input copy") ||
        (eval_B_batch_f32_kernel<true><<<field->query_count * blocks_per_query, THREADS_PER_BLOCK, shared_bytes>>>(
            field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz,
            field->n_segments, d_xyz, d_B, d_grad_B, points_per_query, blocks_per_query
        ), false) ||
        cuda_check(cudaGetLastError(), "batch eval_B_grad kernel") ||
        cuda_check(cudaDeviceSynchronize(), "batch eval_B_grad sync") ||
        cuda_check(cudaMemcpy(B_host, d_B, vector_bytes, cudaMemcpyDeviceToHost), "batch eval_B_grad B copy") ||
        cuda_check(cudaMemcpy(grad_B_host, d_grad_B, gradient_bytes, cudaMemcpyDeviceToHost), "batch eval_B_grad gradient copy");
    cudaFree(d_xyz);
    cudaFree(d_B);
    cudaFree(d_grad_B);
    if (failed) return 1;
    set_error("");
    return 0;
}

int sgpu_eval_B(void* handle, const double* xyz_host, double* B_host, int n_points) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !xyz_host || !B_host || n_points < 0) {
        set_error("invalid eval_B arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    double *d_xyz = nullptr, *d_B = nullptr;
    size_t xyz_bytes = static_cast<size_t>(n_points) * 3 * sizeof(double);
    if (cuda_check(cudaMalloc(&d_xyz, xyz_bytes), "cudaMalloc d_xyz")) return 1;
    if (cuda_check(cudaMalloc(&d_B, xyz_bytes), "cudaMalloc d_B")) { cudaFree(d_xyz); return 1; }
    if (cuda_check(cudaMemcpy(d_xyz, xyz_host, xyz_bytes, cudaMemcpyHostToDevice), "copy xyz")) { cudaFree(d_xyz); cudaFree(d_B); return 1; }
    int blocks = (n_points + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t shmem = static_cast<size_t>(SEG_TILE) * 6 * sizeof(double);
    eval_B_kernel<<<blocks, THREADS_PER_BLOCK, shmem>>>(f->d_x, f->d_y, f->d_z, f->d_wx, f->d_wy, f->d_wz, f->n_segments, d_xyz, d_B, n_points);
    if (cuda_check(cudaGetLastError(), "eval_B kernel")) { cudaFree(d_xyz); cudaFree(d_B); return 1; }
    if (cuda_check(cudaDeviceSynchronize(), "eval_B sync")) { cudaFree(d_xyz); cudaFree(d_B); return 1; }
    if (cuda_check(cudaMemcpy(B_host, d_B, xyz_bytes, cudaMemcpyDeviceToHost), "copy B")) { cudaFree(d_xyz); cudaFree(d_B); return 1; }
    cudaFree(d_xyz);
    cudaFree(d_B);
    set_error("");
    return 0;
}

int sgpu_eval_B_f32(void* handle, const float* xyz_host, float* B_host, int n_points) {
    SGPU_NVTX_RANGE("field.eval_B.fp32");
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !xyz_host || !B_host || n_points < 0) {
        set_error("invalid eval_B_f32 arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    float *d_xyz = nullptr, *d_B = nullptr;
    size_t xyz_bytes = static_cast<size_t>(n_points) * 3 * sizeof(float);
    if (cuda_check(cudaMalloc(&d_xyz, xyz_bytes), "eval_B_f32 d_xyz") ||
        cuda_check(cudaMalloc(&d_B, xyz_bytes), "eval_B_f32 d_B")) {
        cudaFree(d_xyz); cudaFree(d_B);
        return 1;
    }
    if (cuda_check(cudaMemcpy(d_xyz, xyz_host, xyz_bytes, cudaMemcpyHostToDevice), "eval_B_f32 copy xyz")) {
        cudaFree(d_xyz); cudaFree(d_B);
        return 1;
    }
    int blocks = (n_points + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t shmem = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    eval_B_only_kernel<float><<<blocks, THREADS_PER_BLOCK, shmem>>>(
        f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f,
        f->n_segments, d_xyz, d_B, n_points
    );
    if (cuda_check(cudaGetLastError(), "eval_B_f32 kernel") ||
        cuda_check(cudaDeviceSynchronize(), "eval_B_f32 sync") ||
        cuda_check(cudaMemcpy(B_host, d_B, xyz_bytes, cudaMemcpyDeviceToHost), "eval_B_f32 copy B")) {
        cudaFree(d_xyz); cudaFree(d_B);
        return 1;
    }
    cudaFree(d_xyz); cudaFree(d_B);
    set_error("");
    return 0;
}

int sgpu_eval_B_grad(void* handle, const double* xyz_host, double* B_host, double* grad_B_host, int n_points) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !xyz_host || !B_host || !grad_B_host || n_points < 0) {
        set_error("invalid eval_B_grad arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    double *d_xyz = nullptr, *d_B = nullptr, *d_grad_B = nullptr;
    size_t xyz_bytes = static_cast<size_t>(n_points) * 3 * sizeof(double);
    size_t grad_bytes = static_cast<size_t>(n_points) * 9 * sizeof(double);
    if (cuda_check(cudaMalloc(&d_xyz, xyz_bytes), "eval_B_grad d_xyz") ||
        cuda_check(cudaMalloc(&d_B, xyz_bytes), "eval_B_grad d_B") ||
        cuda_check(cudaMalloc(&d_grad_B, grad_bytes), "eval_B_grad d_grad_B")) {
        cudaFree(d_xyz); cudaFree(d_B); cudaFree(d_grad_B);
        return 1;
    }
    if (cuda_check(cudaMemcpy(d_xyz, xyz_host, xyz_bytes, cudaMemcpyHostToDevice), "eval_B_grad copy xyz")) {
        cudaFree(d_xyz); cudaFree(d_B); cudaFree(d_grad_B);
        return 1;
    }
    int blocks = (n_points + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t shmem = static_cast<size_t>(SEG_TILE) * 6 * sizeof(double);
    eval_B_grad_kernel<double><<<blocks, THREADS_PER_BLOCK, shmem>>>(
        f->d_x, f->d_y, f->d_z, f->d_wx, f->d_wy, f->d_wz,
        f->n_segments, d_xyz, d_B, d_grad_B, n_points
    );
    if (cuda_check(cudaGetLastError(), "eval_B_grad kernel") ||
        cuda_check(cudaDeviceSynchronize(), "eval_B_grad sync") ||
        cuda_check(cudaMemcpy(B_host, d_B, xyz_bytes, cudaMemcpyDeviceToHost), "eval_B_grad copy B") ||
        cuda_check(cudaMemcpy(grad_B_host, d_grad_B, grad_bytes, cudaMemcpyDeviceToHost), "eval_B_grad copy grad")) {
        cudaFree(d_xyz); cudaFree(d_B); cudaFree(d_grad_B);
        return 1;
    }
    cudaFree(d_xyz); cudaFree(d_B); cudaFree(d_grad_B);
    set_error("");
    return 0;
}

int sgpu_eval_B_grad_f32(void* handle, const float* xyz_host, float* B_host, float* grad_B_host, int n_points) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !xyz_host || !B_host || !grad_B_host || n_points < 0) {
        set_error("invalid eval_B_grad_f32 arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    float *d_xyz = nullptr, *d_B = nullptr, *d_grad_B = nullptr;
    size_t xyz_bytes = static_cast<size_t>(n_points) * 3 * sizeof(float);
    size_t grad_bytes = static_cast<size_t>(n_points) * 9 * sizeof(float);
    if (cuda_check(cudaMalloc(&d_xyz, xyz_bytes), "eval_B_grad_f32 d_xyz") ||
        cuda_check(cudaMalloc(&d_B, xyz_bytes), "eval_B_grad_f32 d_B") ||
        cuda_check(cudaMalloc(&d_grad_B, grad_bytes), "eval_B_grad_f32 d_grad_B")) {
        cudaFree(d_xyz); cudaFree(d_B); cudaFree(d_grad_B);
        return 1;
    }
    if (cuda_check(cudaMemcpy(d_xyz, xyz_host, xyz_bytes, cudaMemcpyHostToDevice), "eval_B_grad_f32 copy xyz")) {
        cudaFree(d_xyz); cudaFree(d_B); cudaFree(d_grad_B);
        return 1;
    }
    int blocks = (n_points + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t shmem = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    eval_B_grad_kernel<float><<<blocks, THREADS_PER_BLOCK, shmem>>>(
        f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f,
        f->n_segments, d_xyz, d_B, d_grad_B, n_points
    );
    if (cuda_check(cudaGetLastError(), "eval_B_grad_f32 kernel") ||
        cuda_check(cudaDeviceSynchronize(), "eval_B_grad_f32 sync") ||
        cuda_check(cudaMemcpy(B_host, d_B, xyz_bytes, cudaMemcpyDeviceToHost), "eval_B_grad_f32 copy B") ||
        cuda_check(cudaMemcpy(grad_B_host, d_grad_B, grad_bytes, cudaMemcpyDeviceToHost), "eval_B_grad_f32 copy grad")) {
        cudaFree(d_xyz); cudaFree(d_B); cudaFree(d_grad_B);
        return 1;
    }
    cudaFree(d_xyz); cudaFree(d_B); cudaFree(d_grad_B);
    set_error("");
    return 0;
}

int sgpu_internal_eval_B_f32_device(
    void* handle,
    const float* xyz_device,
    float* B_device,
    int n_points
) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !xyz_device || !B_device || n_points < 0) {
        set_error("invalid internal eval_B_f32_device arguments");
        return 1;
    }
    const int blocks = (n_points + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t shmem = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    eval_B_only_kernel<float><<<blocks, THREADS_PER_BLOCK, shmem>>>(
        f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f,
        f->n_segments, xyz_device, B_device, n_points
    );
    return cuda_check(cudaGetLastError(), "internal eval_B_f32_device kernel");
}

int sgpu_internal_eval_B_grad_f32_device(
    void* handle,
    const float* xyz_device,
    float* B_device,
    float* grad_B_device,
    int n_points
) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !xyz_device || !B_device || !grad_B_device || n_points < 0) {
        set_error("invalid internal eval_B_grad_f32_device arguments");
        return 1;
    }
    const int blocks = (n_points + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t shmem = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    eval_B_grad_kernel<float><<<blocks, THREADS_PER_BLOCK, shmem>>>(
        f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f,
        f->n_segments, xyz_device, B_device, grad_B_device, n_points
    );
    return cuda_check(cudaGetLastError(), "internal eval_B_grad_f32_device kernel");
}

} // extern "C"
