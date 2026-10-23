#include "field/device_field.cuh"
#include "coil_field.h"
#include "coil_field_internal.h"

#include <cmath>

namespace coil_eval::field {

__global__ void B_grad_segment_vjp_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const float* __restrict__ xyz,
    const float* __restrict__ adj_B,
    const float* __restrict__ adj_grad_B,
    int npoints,
    float* __restrict__ adj_segment_position,
    float* __restrict__ adj_segment_weight
) {
    const int segment = blockIdx.x;
    if (segment >= nseg) return;
    const float sx = seg_x[segment];
    const float sy = seg_y[segment];
    const float sz = seg_z[segment];
    const float wx = seg_wx[segment];
    const float wy = seg_wy[segment];
    const float wz = seg_wz[segment];
    float accum[6] = {};
    for (int point = threadIdx.x; point < npoints; point += blockDim.x) {
        const float rx = xyz[3 * point] - sx;
        const float ry = xyz[3 * point + 1] - sy;
        const float rz = xyz[3 * point + 2] - sz;
        const float r2 = rx * rx + ry * ry + rz * rz + 1.0e-30f;
        const float invr = rsqrtf(r2);
        const float invr2 = invr * invr;
        const float invr3 = invr * invr2;
        const float invr5 = invr3 * invr2;
        const float invr7 = invr5 * invr2;
        const float ux = wy * rz - wz * ry;
        const float uy = wz * rx - wx * rz;
        const float uz = wx * ry - wy * rx;
        const float abx = adj_B[3 * point];
        const float aby = adj_B[3 * point + 1];
        const float abz = adj_B[3 * point + 2];
        const float* aJ = adj_grad_B + 9 * point;

        float au_x = MU0_OVER_4PI * invr3 * abx;
        float au_y = MU0_OVER_4PI * invr3 * aby;
        float au_z = MU0_OVER_4PI * invr3 * abz;
        float a_f = MU0_OVER_4PI * (abx * ux + aby * uy + abz * uz);

        const float k00 = 0.0f;
        const float k10 = wz;
        const float k20 = -wy;
        const float k01 = -wz;
        const float k11 = 0.0f;
        const float k21 = wx;
        const float k02 = wy;
        const float k12 = -wx;
        const float k22 = 0.0f;
        a_f += MU0_OVER_4PI * (
            aJ[0] * k00 + aJ[3] * k10 + aJ[6] * k20 +
            aJ[1] * k01 + aJ[4] * k11 + aJ[7] * k21 +
            aJ[2] * k02 + aJ[5] * k12 + aJ[8] * k22
        );
        const float aJ_r_x = aJ[0] * rx + aJ[1] * ry + aJ[2] * rz;
        const float aJ_r_y = aJ[3] * rx + aJ[4] * ry + aJ[5] * rz;
        const float aJ_r_z = aJ[6] * rx + aJ[7] * ry + aJ[8] * rz;
        const float factor_u = -3.0f * MU0_OVER_4PI * invr5;
        au_x += factor_u * aJ_r_x;
        au_y += factor_u * aJ_r_y;
        au_z += factor_u * aJ_r_z;
        const float aJT_u_x = aJ[0] * ux + aJ[3] * uy + aJ[6] * uz;
        const float aJT_u_y = aJ[1] * ux + aJ[4] * uy + aJ[7] * uz;
        const float aJT_u_z = aJ[2] * ux + aJ[5] * uy + aJ[8] * uz;
        float ar_x = factor_u * aJT_u_x;
        float ar_y = factor_u * aJT_u_y;
        float ar_z = factor_u * aJT_u_z;
        const float outer_contraction =
            ux * aJ_r_x + uy * aJ_r_y + uz * aJ_r_z;
        const float a_g = -3.0f * MU0_OVER_4PI * outer_contraction;

        const float aK10 = MU0_OVER_4PI * invr3 * aJ[3];
        const float aK20 = MU0_OVER_4PI * invr3 * aJ[6];
        const float aK01 = MU0_OVER_4PI * invr3 * aJ[1];
        const float aK21 = MU0_OVER_4PI * invr3 * aJ[7];
        const float aK02 = MU0_OVER_4PI * invr3 * aJ[2];
        const float aK12 = MU0_OVER_4PI * invr3 * aJ[5];
        float aw_x = aK21 - aK12;
        float aw_y = -aK20 + aK02;
        float aw_z = aK10 - aK01;

        aw_x += ry * au_z - rz * au_y;
        aw_y += rz * au_x - rx * au_z;
        aw_z += rx * au_y - ry * au_x;
        ar_x += au_y * wz - au_z * wy;
        ar_y += au_z * wx - au_x * wz;
        ar_z += au_x * wy - au_y * wx;
        const float radial_factor = -3.0f * a_f * invr5 - 5.0f * a_g * invr7;
        ar_x += radial_factor * rx;
        ar_y += radial_factor * ry;
        ar_z += radial_factor * rz;
        accum[0] -= ar_x;
        accum[1] -= ar_y;
        accum[2] -= ar_z;
        accum[3] += aw_x;
        accum[4] += aw_y;
        accum[5] += aw_z;
    }
    __shared__ float reduction[6 * 256];
    for (int component = 0; component < 6; ++component) {
        reduction[component * blockDim.x + threadIdx.x] = accum[component];
    }
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            for (int component = 0; component < 6; ++component) {
                reduction[component * blockDim.x + threadIdx.x] +=
                    reduction[component * blockDim.x + threadIdx.x + offset];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int component = 0; component < 3; ++component) {
            adj_segment_position[3 * segment + component] = reduction[component * blockDim.x];
            adj_segment_weight[3 * segment + component] = reduction[(component + 3) * blockDim.x];
        }
    }
}

__global__ void B_grad_point_vjp_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const float* __restrict__ xyz,
    const float* __restrict__ adj_B,
    const float* __restrict__ adj_grad_B,
    int npoints,
    float* __restrict__ adj_xyz
) {
    extern __shared__ float shared[];
    float* sx = shared;
    float* sy = sx + SEG_TILE;
    float* sz = sy + SEG_TILE;
    float* swx = sz + SEG_TILE;
    float* swy = swx + SEG_TILE;
    float* swz = swy + SEG_TILE;

    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int point = blockIdx.x * WARPS_PER_BLOCK + tid / WARP_SIZE;
    const bool valid = point < npoints;
    const float px = valid ? xyz[3 * point] : 0.0f;
    const float py = valid ? xyz[3 * point + 1] : 0.0f;
    const float pz = valid ? xyz[3 * point + 2] : 0.0f;
    const float abx = valid ? adj_B[3 * point] : 0.0f;
    const float aby = valid ? adj_B[3 * point + 1] : 0.0f;
    const float abz = valid ? adj_B[3 * point + 2] : 0.0f;
    const float* aJ = valid ? adj_grad_B + 9 * point : nullptr;
    float accum_x = 0.0f;
    float accum_y = 0.0f;
    float accum_z = 0.0f;

    for (int base = 0; base < nseg; base += SEG_TILE) {
        const int count = min(SEG_TILE, nseg - base);
        for (int index = tid; index < count; index += blockDim.x) {
            sx[index] = seg_x[base + index];
            sy[index] = seg_y[base + index];
            sz[index] = seg_z[base + index];
            swx[index] = seg_wx[base + index];
            swy[index] = seg_wy[base + index];
            swz[index] = seg_wz[base + index];
        }
        __syncthreads();
        if (valid) {
            for (int index = lane; index < count; index += WARP_SIZE) {
                const float rx = px - sx[index];
                const float ry = py - sy[index];
                const float rz = pz - sz[index];
                const float wx = swx[index];
                const float wy = swy[index];
                const float wz = swz[index];
                const float r2 = rx * rx + ry * ry + rz * rz + 1.0e-30f;
                const float invr = rsqrtf(r2);
                const float invr2 = invr * invr;
                const float invr3 = invr * invr2;
                const float invr5 = invr3 * invr2;
                const float invr7 = invr5 * invr2;
                const float ux = wy * rz - wz * ry;
                const float uy = wz * rx - wx * rz;
                const float uz = wx * ry - wy * rx;

                float au_x = MU0_OVER_4PI * invr3 * abx;
                float au_y = MU0_OVER_4PI * invr3 * aby;
                float au_z = MU0_OVER_4PI * invr3 * abz;
                float adj_invr3 = MU0_OVER_4PI * (abx * ux + aby * uy + abz * uz);
                adj_invr3 += MU0_OVER_4PI * (
                    aJ[3] * wz - aJ[6] * wy - aJ[1] * wz +
                    aJ[7] * wx + aJ[2] * wy - aJ[5] * wx
                );
                const float aJ_r_x = aJ[0] * rx + aJ[1] * ry + aJ[2] * rz;
                const float aJ_r_y = aJ[3] * rx + aJ[4] * ry + aJ[5] * rz;
                const float aJ_r_z = aJ[6] * rx + aJ[7] * ry + aJ[8] * rz;
                const float factor_u = -3.0f * MU0_OVER_4PI * invr5;
                au_x += factor_u * aJ_r_x;
                au_y += factor_u * aJ_r_y;
                au_z += factor_u * aJ_r_z;
                float ar_x = factor_u * (aJ[0] * ux + aJ[3] * uy + aJ[6] * uz);
                float ar_y = factor_u * (aJ[1] * ux + aJ[4] * uy + aJ[7] * uz);
                float ar_z = factor_u * (aJ[2] * ux + aJ[5] * uy + aJ[8] * uz);
                const float outer = ux * aJ_r_x + uy * aJ_r_y + uz * aJ_r_z;
                const float adj_invr5 = -3.0f * MU0_OVER_4PI * outer;

                ar_x += au_y * wz - au_z * wy;
                ar_y += au_z * wx - au_x * wz;
                ar_z += au_x * wy - au_y * wx;
                const float radial =
                    -3.0f * adj_invr3 * invr5 - 5.0f * adj_invr5 * invr7;
                accum_x += ar_x + radial * rx;
                accum_y += ar_y + radial * ry;
                accum_z += ar_z + radial * rz;
            }
        }
        __syncthreads();
    }
    accum_x = warp_sum_t(accum_x);
    accum_y = warp_sum_t(accum_y);
    accum_z = warp_sum_t(accum_z);
    if (valid && lane == 0) {
        adj_xyz[3 * point] = accum_x;
        adj_xyz[3 * point + 1] = accum_y;
        adj_xyz[3 * point + 2] = accum_z;
    }
}

} // namespace coil_eval::field

// The public ABI stays global and unmangled; implementation state is named.
using namespace coil_eval::field;

extern "C" {

int sgpu_eval_B_grad_point_vjp_f32(
    void* handle,
    const float* xyz_host,
    const float* adj_B_host,
    const float* adj_grad_B_host,
    float* adj_xyz_host,
    int n_points
) {
    auto* field = reinterpret_cast<CoilField*>(handle);
    if (!field || !xyz_host || !adj_B_host || !adj_grad_B_host ||
        !adj_xyz_host || n_points < 0) {
        set_error("invalid eval_B_grad_point_vjp_f32 arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "point VJP cudaSetDevice")) return 1;
    const size_t xyz_bytes = static_cast<size_t>(n_points) * 3 * sizeof(float);
    const size_t grad_bytes = static_cast<size_t>(n_points) * 9 * sizeof(float);
    float* d_xyz = nullptr;
    float* d_adj_B = nullptr;
    float* d_adj_grad_B = nullptr;
    float* d_adj_xyz = nullptr;
    auto cleanup = [&]() {
        cudaFree(d_xyz);
        cudaFree(d_adj_B);
        cudaFree(d_adj_grad_B);
        cudaFree(d_adj_xyz);
    };
    if (cuda_check(cudaMalloc(&d_xyz, xyz_bytes), "point VJP d_xyz") ||
        cuda_check(cudaMalloc(&d_adj_B, xyz_bytes), "point VJP d_adj_B") ||
        cuda_check(cudaMalloc(&d_adj_grad_B, grad_bytes), "point VJP d_adj_grad_B") ||
        cuda_check(cudaMalloc(&d_adj_xyz, xyz_bytes), "point VJP d_adj_xyz") ||
        cuda_check(cudaMemcpy(d_xyz, xyz_host, xyz_bytes, cudaMemcpyHostToDevice), "point VJP copy xyz") ||
        cuda_check(cudaMemcpy(d_adj_B, adj_B_host, xyz_bytes, cudaMemcpyHostToDevice), "point VJP copy adj_B") ||
        cuda_check(cudaMemcpy(d_adj_grad_B, adj_grad_B_host, grad_bytes, cudaMemcpyHostToDevice), "point VJP copy adj_grad_B")) {
        cleanup();
        return 1;
    }
    const int blocks = (n_points + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t shared_bytes = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    B_grad_point_vjp_kernel<<<blocks, THREADS_PER_BLOCK, shared_bytes>>>(
        field->d_x_f, field->d_y_f, field->d_z_f,
        field->d_wx_f, field->d_wy_f, field->d_wz_f,
        field->n_segments, d_xyz, d_adj_B, d_adj_grad_B, n_points, d_adj_xyz
    );
    if (cuda_check(cudaGetLastError(), "point VJP kernel") ||
        cuda_check(cudaDeviceSynchronize(), "point VJP sync") ||
        cuda_check(cudaMemcpy(adj_xyz_host, d_adj_xyz, xyz_bytes, cudaMemcpyDeviceToHost), "point VJP copy output")) {
        cleanup();
        return 1;
    }
    cleanup();
    set_error("");
    return 0;
}

int sgpu_internal_B_grad_segment_vjp_f32_device(
    void* handle,
    const float* xyz_device,
    const float* adj_B_device,
    const float* adj_grad_B_device,
    int n_points,
    float* adj_segment_position_device,
    float* adj_segment_weight_device
) {
    auto* field = reinterpret_cast<CoilField*>(handle);
    if (!field || !xyz_device || !adj_B_device || !adj_grad_B_device ||
        n_points <= 0 || !adj_segment_position_device || !adj_segment_weight_device) {
        set_error("invalid internal B/grad(B) segment VJP arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "internal B/grad(B) VJP cudaSetDevice")) return 1;
    constexpr int threads = 256;
    B_grad_segment_vjp_kernel<<<field->n_segments, threads>>>(
        field->d_x_f, field->d_y_f, field->d_z_f,
        field->d_wx_f, field->d_wy_f, field->d_wz_f,
        field->n_segments, xyz_device, adj_B_device, adj_grad_B_device,
        n_points, adj_segment_position_device, adj_segment_weight_device
    );
    return cuda_check(cudaGetLastError(), "internal B/grad(B) segment VJP kernel");
}

int sgpu_internal_B_grad_point_vjp_f32_device(
    void* handle,
    const float* xyz_device,
    const float* adj_B_device,
    const float* adj_grad_B_device,
    int n_points,
    float* adj_xyz_device
) {
    auto* field = reinterpret_cast<CoilField*>(handle);
    if (!field || !xyz_device || !adj_B_device || !adj_grad_B_device ||
        n_points <= 0 || !adj_xyz_device) {
        set_error("invalid internal B/grad(B) point VJP arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "internal point VJP cudaSetDevice")) return 1;
    const int blocks = (n_points + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const size_t shared_bytes = static_cast<size_t>(SEG_TILE) * 6 * sizeof(float);
    B_grad_point_vjp_kernel<<<blocks, THREADS_PER_BLOCK, shared_bytes>>>(
        field->d_x_f, field->d_y_f, field->d_z_f,
        field->d_wx_f, field->d_wy_f, field->d_wz_f,
        field->n_segments, xyz_device, adj_B_device, adj_grad_B_device,
        n_points, adj_xyz_device
    );
    return cuda_check(cudaGetLastError(), "internal B/grad(B) point VJP kernel");
}

} // extern "C"
