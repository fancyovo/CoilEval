#include "field/device_field.cuh"
#include "coil_field_internal.h"
namespace coil_eval::field {
template <int POINT_WARPS, int POINT_SEG_TILE>
__global__ void eval_A_batch_f32_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const float* __restrict__ xyz,
    float* __restrict__ A,
    int points_per_query,
    int blocks_per_query
) {
    extern __shared__ float shared_A[];
    float* sx = shared_A;
    float* sy = sx + POINT_SEG_TILE;
    float* sz = sy + POINT_SEG_TILE;
    float* swx = sz + POINT_SEG_TILE;
    float* swy = swx + POINT_SEG_TILE;
    float* swz = swy + POINT_SEG_TILE;
    const int query = blockIdx.x / blocks_per_query;
    const int query_block = blockIdx.x - query * blocks_per_query;
    const int lane = threadIdx.x & 31;
    const int point = query_block * POINT_WARPS + threadIdx.x / 32;
    const bool valid = point < points_per_query;
    const size_t index = static_cast<size_t>(query) * points_per_query + point;
    const size_t segment_offset = static_cast<size_t>(query) * nseg;
    const float px = valid ? xyz[3 * index] : 0.0f;
    const float py = valid ? xyz[3 * index + 1] : 0.0f;
    const float pz = valid ? xyz[3 * index + 2] : 0.0f;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    for (int base = 0; base < nseg; base += POINT_SEG_TILE) {
        const int count = min(POINT_SEG_TILE, nseg - base);
        for (int item = threadIdx.x; item < count; item += blockDim.x) {
            const size_t source = segment_offset + base + item;
            sx[item] = seg_x[source]; sy[item] = seg_y[source]; sz[item] = seg_z[source];
            swx[item] = seg_wx[source]; swy[item] = seg_wy[source]; swz[item] = seg_wz[source];
        }
        __syncthreads();
        if (valid) {
            for (int item = lane; item < count; item += 32) {
                const float rx = px - sx[item];
                const float ry = py - sy[item];
                const float rz = pz - sz[item];
                const float invr = rsqrtf(rx * rx + ry * ry + rz * rz + 1.0e-30f);
                ax += swx[item] * invr;
                ay += swy[item] * invr;
                az += swz[item] * invr;
            }
        }
        __syncthreads();
    }
    ax = warp_sum_t(ax) * static_cast<float>(MU0_OVER_4PI);
    ay = warp_sum_t(ay) * static_cast<float>(MU0_OVER_4PI);
    az = warp_sum_t(az) * static_cast<float>(MU0_OVER_4PI);
    if (valid && lane == 0) {
        A[3 * index] = ax;
        A[3 * index + 1] = ay;
        A[3 * index + 2] = az;
    }
}
}
using namespace coil_eval::field;
int sgpu_internal_batch_eval_A_f32_device(
    void* handle,
    const float* xyz_device,
    float* A_device,
    int points_per_query
) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field || !xyz_device || !A_device || points_per_query <= 0) {
        set_error("invalid device batch eval_A arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "device batch eval_A cudaSetDevice")) return 1;
    constexpr int warps = 16;
    const int blocks_per_query = (points_per_query + warps - 1) / warps;
    const size_t shared_bytes = static_cast<size_t>(512) * 6 * sizeof(float);
    eval_A_batch_f32_kernel<warps, 512><<<
        field->query_count * blocks_per_query, warps * 32, shared_bytes
    >>>(
        field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz,
        field->n_segments, xyz_device, A_device, points_per_query, blocks_per_query
    );
    return cuda_check(cudaGetLastError(), "device batch eval_A kernel");
}
