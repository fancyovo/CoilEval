#pragma once

#include "field/state.h"

namespace coil_eval::field {

// Small shared device primitives stay inline so field evaluation, RK4 tracing,
// and psi-design kernels retain their original local arithmetic and reductions.
__device__ inline double warp_sum(double v) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

__device__ inline void eval_B_warp(
    double px, double py, double pz,
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    double* sh,
    double& bx_out,
    double& by_out,
    double& bz_out
) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    double* sx = sh;
    double* sy = sx + SEG_TILE;
    double* sz = sy + SEG_TILE;
    double* swx = sz + SEG_TILE;
    double* swy = swx + SEG_TILE;
    double* swz = swy + SEG_TILE;

    double bx = 0.0;
    double by = 0.0;
    double bz = 0.0;

    for (int base = 0; base < nseg; base += SEG_TILE) {
        int count = min(SEG_TILE, nseg - base);
        for (int j = tid; j < count; j += blockDim.x) {
            int idx = base + j;
            sx[j] = seg_x[idx];
            sy[j] = seg_y[idx];
            sz[j] = seg_z[idx];
            swx[j] = seg_wx[idx];
            swy[j] = seg_wy[idx];
            swz[j] = seg_wz[idx];
        }
        __syncthreads();

        for (int j = lane; j < count; j += WARP_SIZE) {
            double rx = px - sx[j];
            double ry = py - sy[j];
            double rz = pz - sz[j];
            double r2 = rx * rx + ry * ry + rz * rz + 1.0e-300;
            double invr = rsqrt(r2);
            double invr3 = invr * invr * invr;
            double wx = swx[j];
            double wy = swy[j];
            double wz = swz[j];
            bx += (wy * rz - wz * ry) * invr3;
            by += (wz * rx - wx * rz) * invr3;
            bz += (wx * ry - wy * rx) * invr3;
        }
        __syncthreads();
    }

    bx = warp_sum(bx) * MU0_OVER_4PI;
    by = warp_sum(by) * MU0_OVER_4PI;
    bz = warp_sum(bz) * MU0_OVER_4PI;
    bx = __shfl_sync(0xffffffffu, bx, 0);
    by = __shfl_sync(0xffffffffu, by, 0);
    bz = __shfl_sync(0xffffffffu, bz, 0);
    bx_out = bx;
    by_out = by;
    bz_out = bz;
}

__device__ inline void eval_B_block(
    double px, double py, double pz,
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    double* sh,
    double& bx_out,
    double& by_out,
    double& bz_out
) {
    int tid = threadIdx.x;
    double* sx = sh;
    double* sy = sx + SEG_TILE;
    double* sz = sy + SEG_TILE;
    double* swx = sz + SEG_TILE;
    double* swy = swx + SEG_TILE;
    double* swz = swy + SEG_TILE;
    double* rb = swz + SEG_TILE;
    double* rby = rb + blockDim.x;
    double* rbz = rby + blockDim.x;

    double bx = 0.0, by = 0.0, bz = 0.0;
    for (int base = 0; base < nseg; base += SEG_TILE) {
        int count = min(SEG_TILE, nseg - base);
        for (int j = tid; j < count; j += blockDim.x) {
            int idx = base + j;
            sx[j] = seg_x[idx];
            sy[j] = seg_y[idx];
            sz[j] = seg_z[idx];
            swx[j] = seg_wx[idx];
            swy[j] = seg_wy[idx];
            swz[j] = seg_wz[idx];
        }
        __syncthreads();
        for (int j = tid; j < count; j += blockDim.x) {
            double rx = px - sx[j];
            double ry = py - sy[j];
            double rz = pz - sz[j];
            double r2 = rx * rx + ry * ry + rz * rz + 1.0e-300;
            double invr = rsqrt(r2);
            double invr3 = invr * invr * invr;
            double wx = swx[j];
            double wy = swy[j];
            double wz = swz[j];
            bx += (wy * rz - wz * ry) * invr3;
            by += (wz * rx - wx * rz) * invr3;
            bz += (wx * ry - wy * rx) * invr3;
        }
        __syncthreads();
    }
    rb[tid] = bx;
    rby[tid] = by;
    rbz[tid] = bz;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            rb[tid] += rb[tid + stride];
            rby[tid] += rby[tid + stride];
            rbz[tid] += rbz[tid + stride];
        }
        __syncthreads();
    }
    bx_out = rb[0] * MU0_OVER_4PI;
    by_out = rby[0] * MU0_OVER_4PI;
    bz_out = rbz[0] * MU0_OVER_4PI;
}

__device__ inline void eval_B_block_f32(
    float px, float py, float pz,
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    float* sh,
    float& bx_out,
    float& by_out,
    float& bz_out
) {
    int tid = threadIdx.x;
    float* sx = sh;
    float* sy = sx + SEG_TILE;
    float* sz = sy + SEG_TILE;
    float* swx = sz + SEG_TILE;
    float* swy = swx + SEG_TILE;
    float* swz = swy + SEG_TILE;
    float* rb = swz + SEG_TILE;
    float* rby = rb + blockDim.x;
    float* rbz = rby + blockDim.x;

    float bx = 0.0f, by = 0.0f, bz = 0.0f;
    for (int base = 0; base < nseg; base += SEG_TILE) {
        int count = min(SEG_TILE, nseg - base);
        for (int j = tid; j < count; j += blockDim.x) {
            int idx = base + j;
            sx[j] = seg_x[idx];
            sy[j] = seg_y[idx];
            sz[j] = seg_z[idx];
            swx[j] = seg_wx[idx];
            swy[j] = seg_wy[idx];
            swz[j] = seg_wz[idx];
        }
        __syncthreads();
        for (int j = tid; j < count; j += blockDim.x) {
            float rx = px - sx[j];
            float ry = py - sy[j];
            float rz = pz - sz[j];
            float r2 = rx * rx + ry * ry + rz * rz + 1.0e-30f;
            float invr = rsqrtf(r2);
            float invr3 = invr * invr * invr;
            float wx = swx[j];
            float wy = swy[j];
            float wz = swz[j];
            bx += (wy * rz - wz * ry) * invr3;
            by += (wz * rx - wx * rz) * invr3;
            bz += (wx * ry - wy * rx) * invr3;
        }
        __syncthreads();
    }
    rb[tid] = bx;
    rby[tid] = by;
    rbz[tid] = bz;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            rb[tid] += rb[tid + stride];
            rby[tid] += rby[tid + stride];
            rbz[tid] += rbz[tid + stride];
        }
        __syncthreads();
    }
    bx_out = rb[0] * static_cast<float>(MU0_OVER_4PI);
    by_out = rby[0] * static_cast<float>(MU0_OVER_4PI);
    bz_out = rbz[0] * static_cast<float>(MU0_OVER_4PI);
}

template <typename T>
__device__ inline T warp_sum_t(T value) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

} // namespace coil_eval::field
