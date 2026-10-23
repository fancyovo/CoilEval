#include "field/device_field.cuh"
#include "coil_field.h"
#include "nvtx_profile.h"

#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>

namespace coil_eval::field {

__device__ void rhs_cyl(
    double R, double Z, double phi,
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    double* sh,
    double& dR,
    double& dZ
) {
    double cp = cos(phi);
    double sp = sin(phi);
    double bx, by, bz;
    eval_B_warp(R * cp, R * sp, Z, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, bx, by, bz);
    double br = bx * cp + by * sp;
    double bphi = -bx * sp + by * cp;
    double denom = fabs(bphi) > 1.0e-14 ? bphi : copysign(1.0e-14, bphi == 0.0 ? 1.0 : bphi);
    dR = R * br / denom;
    dZ = R * bz / denom;
}

__global__ void trace_period_kernel(
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ R0,
    const double* __restrict__ Z0,
    double* __restrict__ R1,
    double* __restrict__ Z1,
    int nlines,
    int nfp,
    int steps
) {
    extern __shared__ double sh[];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & 31;
    int line = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    double R = (line < nlines) ? R0[line] : 1.0;
    double Z = (line < nlines) ? Z0[line] : 0.0;
    double period = TWOPI / static_cast<double>(nfp);
    double h = period / static_cast<double>(steps);

    for (int s = 0; s < steps; ++s) {
        double phi = h * static_cast<double>(s);
        double k1r, k1z, k2r, k2z, k3r, k3z, k4r, k4z;
        rhs_cyl(R, Z, phi, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k1r, k1z);
        rhs_cyl(R + 0.5 * h * k1r, Z + 0.5 * h * k1z, phi + 0.5 * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k2r, k2z);
        rhs_cyl(R + 0.5 * h * k2r, Z + 0.5 * h * k2z, phi + 0.5 * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k3r, k3z);
        rhs_cyl(R + h * k3r, Z + h * k3z, phi + h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k4r, k4z);
        R += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
        Z += (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
    }
    if (line < nlines && lane == 0) {
        R1[line] = R;
        Z1[line] = Z;
    }
}

__device__ void rhs_cyl_block(
    double R, double Z, double phi,
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    double* sh,
    double& dR,
    double& dZ
) {
    double cp = cos(phi);
    double sp = sin(phi);
    double bx, by, bz;
    eval_B_block(R * cp, R * sp, Z, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, bx, by, bz);
    double br = bx * cp + by * sp;
    double bphi = -bx * sp + by * cp;
    double denom = fabs(bphi) > 1.0e-14 ? bphi : copysign(1.0e-14, bphi == 0.0 ? 1.0 : bphi);
    dR = R * br / denom;
    dZ = R * bz / denom;
}

__device__ void rhs_cyl_block_bf32_state64(
    double R, double Z, double phi,
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    float* sh,
    double& dR,
    double& dZ
) {
    double cp = cos(phi);
    double sp = sin(phi);
    float bx, by, bz;
    eval_B_block_f32(
        static_cast<float>(R * cp), static_cast<float>(R * sp), static_cast<float>(Z),
        seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, bx, by, bz
    );
    double br = static_cast<double>(bx) * cp + static_cast<double>(by) * sp;
    double bphi = -static_cast<double>(bx) * sp + static_cast<double>(by) * cp;
    double denom = fabs(bphi) > 1.0e-14 ? bphi : copysign(1.0e-14, bphi == 0.0 ? 1.0 : bphi);
    dR = R * br / denom;
    dZ = R * static_cast<double>(bz) / denom;
}

__device__ void rhs_cyl_block_f32(
    float R, float Z, float phi,
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    float* sh,
    float& dR,
    float& dZ
) {
    float cp = cosf(phi);
    float sp = sinf(phi);
    float bx, by, bz;
    eval_B_block_f32(R * cp, R * sp, Z, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, bx, by, bz);
    float br = bx * cp + by * sp;
    float bphi = -bx * sp + by * cp;
    float denom = fabsf(bphi) > 1.0e-12f ? bphi : copysignf(1.0e-12f, bphi == 0.0f ? 1.0f : bphi);
    dR = R * br / denom;
    dZ = R * bz / denom;
}

__global__ void trace_period_blockline_kernel(
    const double* __restrict__ seg_x,
    const double* __restrict__ seg_y,
    const double* __restrict__ seg_z,
    const double* __restrict__ seg_wx,
    const double* __restrict__ seg_wy,
    const double* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ R0,
    const double* __restrict__ Z0,
    double* __restrict__ R1,
    double* __restrict__ Z1,
    int nlines,
    int nfp,
    int steps
) {
    extern __shared__ double sh[];
    int line = blockIdx.x;
    double R = (line < nlines) ? R0[line] : 1.0;
    double Z = (line < nlines) ? Z0[line] : 0.0;
    double period = TWOPI / static_cast<double>(nfp);
    double h = period / static_cast<double>(steps);
    for (int s = 0; s < steps; ++s) {
        double phi = h * static_cast<double>(s);
        double k1r, k1z, k2r, k2z, k3r, k3z, k4r, k4z;
        rhs_cyl_block(R, Z, phi, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k1r, k1z);
        rhs_cyl_block(R + 0.5 * h * k1r, Z + 0.5 * h * k1z, phi + 0.5 * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k2r, k2z);
        rhs_cyl_block(R + 0.5 * h * k2r, Z + 0.5 * h * k2z, phi + 0.5 * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k3r, k3z);
        rhs_cyl_block(R + h * k3r, Z + h * k3z, phi + h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k4r, k4z);
        R += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
        Z += (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
    }
    if (line < nlines && threadIdx.x == 0) {
        R1[line] = R;
        Z1[line] = Z;
    }
}

__global__ void trace_period_blockline_bf32_state64_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ R0,
    const double* __restrict__ Z0,
    double* __restrict__ R1,
    double* __restrict__ Z1,
    int nlines,
    int nfp,
    int steps
) {
    extern __shared__ unsigned char shmem_bf32[];
    float* sh = reinterpret_cast<float*>(shmem_bf32);
    int line = blockIdx.x;
    double R = (line < nlines) ? R0[line] : 1.0;
    double Z = (line < nlines) ? Z0[line] : 0.0;
    double period = TWOPI / static_cast<double>(nfp);
    double h = period / static_cast<double>(steps);
    for (int s = 0; s < steps; ++s) {
        double phi = h * static_cast<double>(s);
        double k1r, k1z, k2r, k2z, k3r, k3z, k4r, k4z;
        rhs_cyl_block_bf32_state64(R, Z, phi, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k1r, k1z);
        rhs_cyl_block_bf32_state64(R + 0.5 * h * k1r, Z + 0.5 * h * k1z, phi + 0.5 * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k2r, k2z);
        rhs_cyl_block_bf32_state64(R + 0.5 * h * k2r, Z + 0.5 * h * k2z, phi + 0.5 * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k3r, k3z);
        rhs_cyl_block_bf32_state64(R + h * k3r, Z + h * k3z, phi + h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k4r, k4z);
        R += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
        Z += (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
    }
    if (line < nlines && threadIdx.x == 0) {
        R1[line] = R;
        Z1[line] = Z;
    }
}

__global__ void trace_period_blockline_f32_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ R0,
    const double* __restrict__ Z0,
    double* __restrict__ R1,
    double* __restrict__ Z1,
    int nlines,
    int nfp,
    int steps
) {
    extern __shared__ unsigned char shmem_f32[];
    float* sh = reinterpret_cast<float*>(shmem_f32);
    int line = blockIdx.x;
    float R = (line < nlines) ? static_cast<float>(R0[line]) : 1.0f;
    float Z = (line < nlines) ? static_cast<float>(Z0[line]) : 0.0f;
    float period = static_cast<float>(TWOPI) / static_cast<float>(nfp);
    float h = period / static_cast<float>(steps);
    for (int s = 0; s < steps; ++s) {
        float phi = h * static_cast<float>(s);
        float k1r, k1z, k2r, k2z, k3r, k3z, k4r, k4z;
        rhs_cyl_block_f32(R, Z, phi, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k1r, k1z);
        rhs_cyl_block_f32(R + 0.5f * h * k1r, Z + 0.5f * h * k1z, phi + 0.5f * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k2r, k2z);
        rhs_cyl_block_f32(R + 0.5f * h * k2r, Z + 0.5f * h * k2z, phi + 0.5f * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k3r, k3z);
        rhs_cyl_block_f32(R + h * k3r, Z + h * k3z, phi + h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k4r, k4z);
        R += (h / 6.0f) * (k1r + 2.0f * k2r + 2.0f * k3r + k4r);
        Z += (h / 6.0f) * (k1z + 2.0f * k2z + 2.0f * k3z + k4z);
    }
    if (line < nlines && threadIdx.x == 0) {
        R1[line] = static_cast<double>(R);
        Z1[line] = static_cast<double>(Z);
    }
}

__global__ void trace_period_blockline_f32_state16_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ R0,
    const double* __restrict__ Z0,
    double* __restrict__ R1,
    double* __restrict__ Z1,
    int nlines,
    int nfp,
    int steps
) {
    extern __shared__ unsigned char shmem_f16[];
    float* sh = reinterpret_cast<float*>(shmem_f16);
    int line = blockIdx.x;
    __half Rh = __float2half_rn((line < nlines) ? static_cast<float>(R0[line]) : 1.0f);
    __half Zh = __float2half_rn((line < nlines) ? static_cast<float>(Z0[line]) : 0.0f);
    float period = static_cast<float>(TWOPI) / static_cast<float>(nfp);
    float h = period / static_cast<float>(steps);
    for (int s = 0; s < steps; ++s) {
        float R = __half2float(Rh);
        float Z = __half2float(Zh);
        float phi = h * static_cast<float>(s);
        float k1r, k1z, k2r, k2z, k3r, k3z, k4r, k4z;
        rhs_cyl_block_f32(R, Z, phi, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k1r, k1z);
        rhs_cyl_block_f32(R + 0.5f * h * k1r, Z + 0.5f * h * k1z, phi + 0.5f * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k2r, k2z);
        rhs_cyl_block_f32(R + 0.5f * h * k2r, Z + 0.5f * h * k2z, phi + 0.5f * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k3r, k3z);
        rhs_cyl_block_f32(R + h * k3r, Z + h * k3z, phi + h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k4r, k4z);
        R += (h / 6.0f) * (k1r + 2.0f * k2r + 2.0f * k3r + k4r);
        Z += (h / 6.0f) * (k1z + 2.0f * k2z + k3z * 2.0f + k4z);
        Rh = __float2half_rn(R);
        Zh = __float2half_rn(Z);
    }
    if (line < nlines && threadIdx.x == 0) {
        R1[line] = static_cast<double>(__half2float(Rh));
        Z1[line] = static_cast<double>(__half2float(Zh));
    }
}

__global__ void trace_axis_samples_mixed_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    double R0,
    double Z0,
    int nfp,
    int substeps_per_sample,
    int n_samples,
    double* __restrict__ R_out,
    double* __restrict__ Z_out,
    double* __restrict__ R_phi_out,
    double* __restrict__ Z_phi_out
) {
    extern __shared__ unsigned char shmem_axis[];
    float* sh = reinterpret_cast<float*>(shmem_axis);
    double R = R0;
    double Z = Z0;
    const int total_steps = n_samples * substeps_per_sample;
    const double period = TWOPI / static_cast<double>(nfp);
    const double h = period / static_cast<double>(total_steps);
    for (int sample = 0; sample < n_samples; ++sample) {
        for (int substep = 0; substep < substeps_per_sample; ++substep) {
            const int step = sample * substeps_per_sample + substep;
            const double phi = h * static_cast<double>(step);
            double k1r, k1z, k2r, k2z, k3r, k3z, k4r, k4z;
            rhs_cyl_block_bf32_state64(R, Z, phi, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k1r, k1z);
            if (substep == 0 && threadIdx.x == 0) {
                R_out[sample] = R;
                Z_out[sample] = Z;
                R_phi_out[sample] = k1r;
                Z_phi_out[sample] = k1z;
            }
            rhs_cyl_block_bf32_state64(R + 0.5 * h * k1r, Z + 0.5 * h * k1z, phi + 0.5 * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k2r, k2z);
            rhs_cyl_block_bf32_state64(R + 0.5 * h * k2r, Z + 0.5 * h * k2z, phi + 0.5 * h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k3r, k3z);
            rhs_cyl_block_bf32_state64(R + h * k3r, Z + h * k3z, phi + h, seg_x, seg_y, seg_z, seg_wx, seg_wy, seg_wz, nseg, sh, k4r, k4z);
            R += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
            Z += (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
        }
    }
}

__global__ void trace_period_batch_mixed_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ R0,
    const double* __restrict__ Z0,
    double* __restrict__ R1,
    double* __restrict__ Z1,
    int lines_per_query,
    int nfp,
    int steps
) {
    extern __shared__ float shared[];
    const int query = blockIdx.x / lines_per_query;
    const int line = blockIdx.x - query * lines_per_query;
    const size_t line_index = static_cast<size_t>(query) * lines_per_query + line;
    const size_t segment_offset = static_cast<size_t>(query) * nseg;
    double R = R0[line_index];
    double Z = Z0[line_index];
    const double period = TWOPI / static_cast<double>(nfp);
    const double h = period / static_cast<double>(steps);
    for (int step = 0; step < steps; ++step) {
        const double phi = h * static_cast<double>(step);
        double k1r, k1z, k2r, k2z, k3r, k3z, k4r, k4z;
        rhs_cyl_block_bf32_state64(
            R, Z, phi, seg_x + segment_offset, seg_y + segment_offset,
            seg_z + segment_offset, seg_wx + segment_offset,
            seg_wy + segment_offset, seg_wz + segment_offset, nseg, shared, k1r, k1z
        );
        rhs_cyl_block_bf32_state64(
            R + 0.5 * h * k1r, Z + 0.5 * h * k1z, phi + 0.5 * h,
            seg_x + segment_offset, seg_y + segment_offset, seg_z + segment_offset,
            seg_wx + segment_offset, seg_wy + segment_offset, seg_wz + segment_offset,
            nseg, shared, k2r, k2z
        );
        rhs_cyl_block_bf32_state64(
            R + 0.5 * h * k2r, Z + 0.5 * h * k2z, phi + 0.5 * h,
            seg_x + segment_offset, seg_y + segment_offset, seg_z + segment_offset,
            seg_wx + segment_offset, seg_wy + segment_offset, seg_wz + segment_offset,
            nseg, shared, k3r, k3z
        );
        rhs_cyl_block_bf32_state64(
            R + h * k3r, Z + h * k3z, phi + h,
            seg_x + segment_offset, seg_y + segment_offset, seg_z + segment_offset,
            seg_wx + segment_offset, seg_wy + segment_offset, seg_wz + segment_offset,
            nseg, shared, k4r, k4z
        );
        R += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
        Z += (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
    }
    if (threadIdx.x == 0) {
        R1[line_index] = R;
        Z1[line_index] = Z;
    }
}

__global__ void trace_axis_samples_batch_kernel(
    const float* __restrict__ seg_x,
    const float* __restrict__ seg_y,
    const float* __restrict__ seg_z,
    const float* __restrict__ seg_wx,
    const float* __restrict__ seg_wy,
    const float* __restrict__ seg_wz,
    int nseg,
    const double* __restrict__ R0,
    const double* __restrict__ Z0,
    int nfp,
    int substeps_per_sample,
    int sample_count,
    double* __restrict__ R_out,
    double* __restrict__ Z_out,
    double* __restrict__ R_phi_out,
    double* __restrict__ Z_phi_out
) {
    extern __shared__ float shared[];
    const int query = blockIdx.x;
    const size_t segment_offset = static_cast<size_t>(query) * nseg;
    double R = R0[query];
    double Z = Z0[query];
    const int total_steps = sample_count * substeps_per_sample;
    const double period = TWOPI / static_cast<double>(nfp);
    const double h = period / static_cast<double>(total_steps);
    for (int sample = 0; sample < sample_count; ++sample) {
        for (int substep = 0; substep < substeps_per_sample; ++substep) {
            const int step = sample * substeps_per_sample + substep;
            const double phi = h * static_cast<double>(step);
            double k1r, k1z, k2r, k2z, k3r, k3z, k4r, k4z;
            rhs_cyl_block_bf32_state64(
                R, Z, phi, seg_x + segment_offset, seg_y + segment_offset,
                seg_z + segment_offset, seg_wx + segment_offset,
                seg_wy + segment_offset, seg_wz + segment_offset, nseg, shared, k1r, k1z
            );
            if (substep == 0 && threadIdx.x == 0) {
                const size_t output = static_cast<size_t>(query) * sample_count + sample;
                R_out[output] = R;
                Z_out[output] = Z;
                R_phi_out[output] = k1r;
                Z_phi_out[output] = k1z;
            }
            rhs_cyl_block_bf32_state64(
                R + 0.5 * h * k1r, Z + 0.5 * h * k1z, phi + 0.5 * h,
                seg_x + segment_offset, seg_y + segment_offset, seg_z + segment_offset,
                seg_wx + segment_offset, seg_wy + segment_offset, seg_wz + segment_offset,
                nseg, shared, k2r, k2z
            );
            rhs_cyl_block_bf32_state64(
                R + 0.5 * h * k2r, Z + 0.5 * h * k2z, phi + 0.5 * h,
                seg_x + segment_offset, seg_y + segment_offset, seg_z + segment_offset,
                seg_wx + segment_offset, seg_wy + segment_offset, seg_wz + segment_offset,
                nseg, shared, k3r, k3z
            );
            rhs_cyl_block_bf32_state64(
                R + h * k3r, Z + h * k3z, phi + h,
                seg_x + segment_offset, seg_y + segment_offset, seg_z + segment_offset,
                seg_wx + segment_offset, seg_wy + segment_offset, seg_wz + segment_offset,
                nseg, shared, k4r, k4z
            );
            R += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
            Z += (h / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
        }
    }
}

} // namespace coil_eval::field

// The public ABI stays global and unmangled; implementation state is named.
using namespace coil_eval::field;

extern "C" {

int sgpu_batch_trace_period_mixed(
    void* handle,
    const double* R0_host,
    const double* Z0_host,
    double* R1_host,
    double* Z1_host,
    int lines_per_query,
    int nfp,
    int steps
) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field || !R0_host || !Z0_host || !R1_host || !Z1_host ||
        lines_per_query <= 0 || nfp <= 0 || steps <= 0) {
        set_error("invalid batch trace-period arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "batch trace cudaSetDevice")) return 1;
    const size_t line_count = static_cast<size_t>(field->query_count) * lines_per_query;
    const size_t bytes = line_count * sizeof(double);
    double* d_R0 = nullptr;
    double* d_Z0 = nullptr;
    double* d_R1 = nullptr;
    double* d_Z1 = nullptr;
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R0), bytes), "batch trace R0 allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z0), bytes), "batch trace Z0 allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R1), bytes), "batch trace R1 allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z1), bytes), "batch trace Z1 allocation")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    constexpr int threads = 256;
    const size_t shared_bytes = static_cast<size_t>(SEG_TILE * 6 + threads * 3) * sizeof(float);
    const bool failed =
        cuda_check(cudaMemcpy(d_R0, R0_host, bytes, cudaMemcpyHostToDevice), "batch trace R0 copy") ||
        cuda_check(cudaMemcpy(d_Z0, Z0_host, bytes, cudaMemcpyHostToDevice), "batch trace Z0 copy") ||
        (trace_period_batch_mixed_kernel<<<static_cast<int>(line_count), threads, shared_bytes>>>(
            field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz,
            field->n_segments, d_R0, d_Z0, d_R1, d_Z1, lines_per_query, nfp, steps
        ), false) ||
        cuda_check(cudaGetLastError(), "batch trace kernel") ||
        cuda_check(cudaDeviceSynchronize(), "batch trace sync") ||
        cuda_check(cudaMemcpy(R1_host, d_R1, bytes, cudaMemcpyDeviceToHost), "batch trace R1 copy") ||
        cuda_check(cudaMemcpy(Z1_host, d_Z1, bytes, cudaMemcpyDeviceToHost), "batch trace Z1 copy");
    cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
    if (failed) return 1;
    set_error("");
    return 0;
}

int sgpu_batch_trace_axis_samples(
    void* handle,
    const double* R0_host,
    const double* Z0_host,
    int nfp,
    int integration_steps,
    int sample_count,
    double* R_host,
    double* Z_host,
    double* R_phi_host,
    double* Z_phi_host
) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field || !R0_host || !Z0_host || !R_host || !Z_host || !R_phi_host ||
        !Z_phi_host || nfp <= 0 || integration_steps <= 0 || sample_count <= 0 ||
        integration_steps % sample_count != 0) {
        set_error("invalid batch axis-sample arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(field->device_id), "batch axis cudaSetDevice")) return 1;
    const size_t input_bytes = static_cast<size_t>(field->query_count) * sizeof(double);
    const size_t output_count = static_cast<size_t>(field->query_count) * sample_count;
    const size_t output_bytes = output_count * sizeof(double);
    double* d_R0 = nullptr;
    double* d_Z0 = nullptr;
    double* d_R = nullptr;
    double* d_Z = nullptr;
    double* d_R_phi = nullptr;
    double* d_Z_phi = nullptr;
    auto cleanup = [&] {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R); cudaFree(d_Z);
        cudaFree(d_R_phi); cudaFree(d_Z_phi);
    };
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R0), input_bytes), "batch axis R0 allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z0), input_bytes), "batch axis Z0 allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R), output_bytes), "batch axis R allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z), output_bytes), "batch axis Z allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R_phi), output_bytes), "batch axis R_phi allocation") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z_phi), output_bytes), "batch axis Z_phi allocation")) {
        cleanup();
        return 1;
    }
    constexpr int threads = 256;
    const size_t shared_bytes = static_cast<size_t>(SEG_TILE * 6 + threads * 3) * sizeof(float);
    const bool failed =
        cuda_check(cudaMemcpy(d_R0, R0_host, input_bytes, cudaMemcpyHostToDevice), "batch axis R0 copy") ||
        cuda_check(cudaMemcpy(d_Z0, Z0_host, input_bytes, cudaMemcpyHostToDevice), "batch axis Z0 copy") ||
        (trace_axis_samples_batch_kernel<<<field->query_count, threads, shared_bytes>>>(
            field->d_x, field->d_y, field->d_z, field->d_wx, field->d_wy, field->d_wz,
            field->n_segments, d_R0, d_Z0, nfp, integration_steps / sample_count,
            sample_count, d_R, d_Z, d_R_phi, d_Z_phi
        ), false) ||
        cuda_check(cudaGetLastError(), "batch axis kernel") ||
        cuda_check(cudaDeviceSynchronize(), "batch axis sync") ||
        cuda_check(cudaMemcpy(R_host, d_R, output_bytes, cudaMemcpyDeviceToHost), "batch axis R copy") ||
        cuda_check(cudaMemcpy(Z_host, d_Z, output_bytes, cudaMemcpyDeviceToHost), "batch axis Z copy") ||
        cuda_check(cudaMemcpy(R_phi_host, d_R_phi, output_bytes, cudaMemcpyDeviceToHost), "batch axis R_phi copy") ||
        cuda_check(cudaMemcpy(Z_phi_host, d_Z_phi, output_bytes, cudaMemcpyDeviceToHost), "batch axis Z_phi copy");
    cleanup();
    if (failed) return 1;
    set_error("");
    return 0;
}

int sgpu_batch_refine_axis_hint(
    void* handle,
    const double* hint_R_host,
    const double* hint_Z_host,
    int nfp,
    int trace_steps,
    int newton_iterations,
    double finite_difference_step,
    double maximum_newton_step,
    double residual_tolerance,
    double hint_max_distance,
    double* axis_R_host,
    double* axis_Z_host,
    double* residual_host,
    double* topology_trace_host,
    double* topology_det_host,
    unsigned char* valid_host
) {
    BatchCoilField* field = reinterpret_cast<BatchCoilField*>(handle);
    if (!field || !hint_R_host || !hint_Z_host || nfp <= 0 || trace_steps <= 0 ||
        newton_iterations < 0 || !(finite_difference_step > 0.0) ||
        !(maximum_newton_step > 0.0) || !(residual_tolerance > 0.0) ||
        !(hint_max_distance > 0.0) || !axis_R_host || !axis_Z_host ||
        !residual_host || !topology_trace_host || !topology_det_host || !valid_host) {
        set_error("invalid batch axis-refinement arguments");
        return 1;
    }
    const int query_count = field->query_count;
    std::vector<double> R(hint_R_host, hint_R_host + query_count);
    std::vector<double> Z(hint_Z_host, hint_Z_host + query_count);
    std::vector<double> R_end(query_count), Z_end(query_count);
    std::vector<double> fR(query_count), fZ(query_count), residual(query_count);
    if (sgpu_batch_trace_period_mixed(
            field, R.data(), Z.data(), R_end.data(), Z_end.data(), 1, nfp, trace_steps)) {
        return 1;
    }
    for (int query = 0; query < query_count; ++query) {
        fR[query] = R_end[query] - R[query];
        fZ[query] = Z_end[query] - Z[query];
        residual[query] = std::hypot(fR[query], fZ[query]);
    }
    constexpr int derivative_lines = 4;
    constexpr int line_search_lines = 4;
    const double line_search_scale[line_search_lines] = {1.0, 0.5, 0.25, 0.125};
    for (int iteration = 0; iteration < newton_iterations; ++iteration) {
        bool any_active = false;
        std::vector<double> eval_R(static_cast<size_t>(query_count) * derivative_lines);
        std::vector<double> eval_Z(eval_R.size());
        for (int query = 0; query < query_count; ++query) {
            any_active = any_active || residual[query] > residual_tolerance;
            const size_t base = static_cast<size_t>(query) * derivative_lines;
            eval_R[base] = R[query] + finite_difference_step;
            eval_Z[base] = Z[query];
            eval_R[base + 1] = R[query] - finite_difference_step;
            eval_Z[base + 1] = Z[query];
            eval_R[base + 2] = R[query];
            eval_Z[base + 2] = Z[query] + finite_difference_step;
            eval_R[base + 3] = R[query];
            eval_Z[base + 3] = Z[query] - finite_difference_step;
        }
        if (!any_active) break;
        std::vector<double> eval_R_end(eval_R.size()), eval_Z_end(eval_Z.size());
        if (sgpu_batch_trace_period_mixed(
                field, eval_R.data(), eval_Z.data(), eval_R_end.data(), eval_Z_end.data(),
                derivative_lines, nfp, trace_steps)) {
            return 1;
        }
        std::vector<double> step_R(query_count, 0.0), step_Z(query_count, 0.0);
        for (int query = 0; query < query_count; ++query) {
            if (!(residual[query] > residual_tolerance)) continue;
            const size_t base = static_cast<size_t>(query) * derivative_lines;
            const double residual_R_plus = eval_R_end[base] - eval_R[base];
            const double residual_R_minus = eval_R_end[base + 1] - eval_R[base + 1];
            const double residual_Z_r_plus = eval_Z_end[base] - eval_Z[base];
            const double residual_Z_r_minus = eval_Z_end[base + 1] - eval_Z[base + 1];
            const double residual_R_z_plus = eval_R_end[base + 2] - eval_R[base + 2];
            const double residual_R_z_minus = eval_R_end[base + 3] - eval_R[base + 3];
            const double residual_Z_plus = eval_Z_end[base + 2] - eval_Z[base + 2];
            const double residual_Z_minus = eval_Z_end[base + 3] - eval_Z[base + 3];
            const double inverse_denominator = 0.5 / finite_difference_step;
            const double j11 = (residual_R_plus - residual_R_minus) * inverse_denominator;
            const double j21 = (residual_Z_r_plus - residual_Z_r_minus) * inverse_denominator;
            const double j12 = (residual_R_z_plus - residual_R_z_minus) * inverse_denominator;
            const double j22 = (residual_Z_plus - residual_Z_minus) * inverse_denominator;
            const double determinant = j11 * j22 - j12 * j21;
            if (std::abs(determinant) <= 1.0e-14) continue;
            step_R[query] = (-fR[query] * j22 + j12 * fZ[query]) / determinant;
            step_Z[query] = (j21 * fR[query] - j11 * fZ[query]) / determinant;
            const double norm = std::hypot(step_R[query], step_Z[query]);
            const double scale = std::min(1.0, maximum_newton_step / std::max(norm, 1.0e-300));
            step_R[query] *= scale;
            step_Z[query] *= scale;
        }
        std::vector<double> trial_R(static_cast<size_t>(query_count) * line_search_lines);
        std::vector<double> trial_Z(trial_R.size());
        for (int query = 0; query < query_count; ++query) {
            const size_t base = static_cast<size_t>(query) * line_search_lines;
            for (int line = 0; line < line_search_lines; ++line) {
                trial_R[base + line] = R[query] + line_search_scale[line] * step_R[query];
                trial_Z[base + line] = Z[query] + line_search_scale[line] * step_Z[query];
            }
        }
        std::vector<double> trial_R_end(trial_R.size()), trial_Z_end(trial_Z.size());
        if (sgpu_batch_trace_period_mixed(
                field, trial_R.data(), trial_Z.data(), trial_R_end.data(), trial_Z_end.data(),
                line_search_lines, nfp, trace_steps)) {
            return 1;
        }
        for (int query = 0; query < query_count; ++query) {
            const size_t base = static_cast<size_t>(query) * line_search_lines;
            for (int line = 0; line < line_search_lines; ++line) {
                const double candidate_fR = trial_R_end[base + line] - trial_R[base + line];
                const double candidate_fZ = trial_Z_end[base + line] - trial_Z[base + line];
                const double candidate_residual = std::hypot(candidate_fR, candidate_fZ);
                if (std::isfinite(candidate_residual) && candidate_residual < residual[query]) {
                    R[query] = trial_R[base + line];
                    Z[query] = trial_Z[base + line];
                    fR[query] = candidate_fR;
                    fZ[query] = candidate_fZ;
                    residual[query] = candidate_residual;
                    break;
                }
            }
        }
    }

    std::vector<double> topology_R(static_cast<size_t>(query_count) * derivative_lines);
    std::vector<double> topology_Z(topology_R.size());
    for (int query = 0; query < query_count; ++query) {
        const size_t base = static_cast<size_t>(query) * derivative_lines;
        topology_R[base] = R[query] + finite_difference_step;
        topology_Z[base] = Z[query];
        topology_R[base + 1] = R[query] - finite_difference_step;
        topology_Z[base + 1] = Z[query];
        topology_R[base + 2] = R[query];
        topology_Z[base + 2] = Z[query] + finite_difference_step;
        topology_R[base + 3] = R[query];
        topology_Z[base + 3] = Z[query] - finite_difference_step;
    }
    std::vector<double> topology_R_end(topology_R.size()), topology_Z_end(topology_Z.size());
    if (sgpu_batch_trace_period_mixed(
            field, topology_R.data(), topology_Z.data(), topology_R_end.data(),
            topology_Z_end.data(), derivative_lines, nfp, trace_steps)) {
        return 1;
    }
    for (int query = 0; query < query_count; ++query) {
        const size_t base = static_cast<size_t>(query) * derivative_lines;
        const double inverse_denominator = 0.5 / finite_difference_step;
        const double a = (topology_R_end[base] - topology_R_end[base + 1]) * inverse_denominator;
        const double c = (topology_Z_end[base] - topology_Z_end[base + 1]) * inverse_denominator;
        const double b = (topology_R_end[base + 2] - topology_R_end[base + 3]) * inverse_denominator;
        const double d = (topology_Z_end[base + 2] - topology_Z_end[base + 3]) * inverse_denominator;
        const double trace = a + d;
        const double determinant = a * d - b * c;
        const bool elliptic = determinant > 0.0 && std::isfinite(trace) &&
            std::isfinite(determinant) && std::abs(trace / std::sqrt(determinant)) < 2.0;
        const double hint_distance = std::hypot(R[query] - hint_R_host[query], Z[query] - hint_Z_host[query]);
        axis_R_host[query] = R[query];
        axis_Z_host[query] = Z[query];
        residual_host[query] = residual[query];
        topology_trace_host[query] = trace;
        topology_det_host[query] = determinant;
        valid_host[query] = residual[query] <= residual_tolerance && elliptic &&
            hint_distance <= hint_max_distance;
    }
    set_error("");
    return 0;
}

int sgpu_trace_period(void* handle, const double* R0_host, const double* Z0_host, double* R1_host, double* Z1_host, int n_lines, int nfp, int steps) {
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !R0_host || !Z0_host || !R1_host || !Z1_host || n_lines < 0 || nfp <= 0 || steps <= 0) {
        set_error("invalid trace_period arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    double *d_R0 = nullptr, *d_Z0 = nullptr, *d_R1 = nullptr, *d_Z1 = nullptr;
    size_t bytes = static_cast<size_t>(n_lines) * sizeof(double);
    if (cuda_check(cudaMalloc(&d_R0, bytes), "cudaMalloc R0") ||
        cuda_check(cudaMalloc(&d_Z0, bytes), "cudaMalloc Z0") ||
        cuda_check(cudaMalloc(&d_R1, bytes), "cudaMalloc R1") ||
        cuda_check(cudaMalloc(&d_Z1, bytes), "cudaMalloc Z1")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    if (cuda_check(cudaMemcpy(d_R0, R0_host, bytes, cudaMemcpyHostToDevice), "copy R0") ||
        cuda_check(cudaMemcpy(d_Z0, Z0_host, bytes, cudaMemcpyHostToDevice), "copy Z0")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    int blocks = (n_lines + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t shmem = static_cast<size_t>(SEG_TILE) * 6 * sizeof(double);
    trace_period_kernel<<<blocks, THREADS_PER_BLOCK, shmem>>>(f->d_x, f->d_y, f->d_z, f->d_wx, f->d_wy, f->d_wz, f->n_segments, d_R0, d_Z0, d_R1, d_Z1, n_lines, nfp, steps);
    if (cuda_check(cudaGetLastError(), "trace_period kernel") ||
        cuda_check(cudaDeviceSynchronize(), "trace_period sync") ||
        cuda_check(cudaMemcpy(R1_host, d_R1, bytes, cudaMemcpyDeviceToHost), "copy R1") ||
        cuda_check(cudaMemcpy(Z1_host, d_Z1, bytes, cudaMemcpyDeviceToHost), "copy Z1")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
    set_error("");
    return 0;
}

int sgpu_trace_period_blockline(void* handle, const double* R0_host, const double* Z0_host, double* R1_host, double* Z1_host, int n_lines, int nfp, int steps, int threads_per_line) {
    SGPU_NVTX_RANGE("trace.period.fp64");
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !R0_host || !Z0_host || !R1_host || !Z1_host || n_lines < 0 || nfp <= 0 || steps <= 0) {
        set_error("invalid trace_period_blockline arguments");
        return 1;
    }
    if (!(threads_per_line == 128 || threads_per_line == 256 || threads_per_line == 512 || threads_per_line == 1024)) {
        set_error("threads_per_line must be 128, 256, 512, or 1024");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    double *d_R0 = nullptr, *d_Z0 = nullptr, *d_R1 = nullptr, *d_Z1 = nullptr;
    size_t bytes = static_cast<size_t>(n_lines) * sizeof(double);
    if (cuda_check(cudaMalloc(&d_R0, bytes), "cudaMalloc R0") ||
        cuda_check(cudaMalloc(&d_Z0, bytes), "cudaMalloc Z0") ||
        cuda_check(cudaMalloc(&d_R1, bytes), "cudaMalloc R1") ||
        cuda_check(cudaMalloc(&d_Z1, bytes), "cudaMalloc Z1")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    if (cuda_check(cudaMemcpy(d_R0, R0_host, bytes, cudaMemcpyHostToDevice), "copy R0") ||
        cuda_check(cudaMemcpy(d_Z0, Z0_host, bytes, cudaMemcpyHostToDevice), "copy Z0")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    size_t shmem = static_cast<size_t>(SEG_TILE) * 6 * sizeof(double) + static_cast<size_t>(threads_per_line) * 3 * sizeof(double);
    trace_period_blockline_kernel<<<n_lines, threads_per_line, shmem>>>(f->d_x, f->d_y, f->d_z, f->d_wx, f->d_wy, f->d_wz, f->n_segments, d_R0, d_Z0, d_R1, d_Z1, n_lines, nfp, steps);
    if (cuda_check(cudaGetLastError(), "trace_period_blockline kernel") ||
        cuda_check(cudaDeviceSynchronize(), "trace_period_blockline sync") ||
        cuda_check(cudaMemcpy(R1_host, d_R1, bytes, cudaMemcpyDeviceToHost), "copy R1") ||
        cuda_check(cudaMemcpy(Z1_host, d_Z1, bytes, cudaMemcpyDeviceToHost), "copy Z1")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
    set_error("");
    return 0;
}

int sgpu_trace_period_blockline_mixed(void* handle, const double* R0_host, const double* Z0_host, double* R1_host, double* Z1_host, int n_lines, int nfp, int steps, int threads_per_line, int mode) {
    SgpuNvtxRange trace_range(mode == 1
        ? "trace.period.Bfp32_statefp64"
        : mode == 2
            ? "trace.period.fp32"
            : "trace.period.Bfp32_statefp16");
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !R0_host || !Z0_host || !R1_host || !Z1_host || n_lines < 0 || nfp <= 0 || steps <= 0) {
        set_error("invalid trace_period_blockline_mixed arguments");
        return 1;
    }
    if (threads_per_line != 128 && threads_per_line != 256 && threads_per_line != 512 && threads_per_line != 1024) {
        set_error("threads_per_line must be 128, 256, 512, or 1024");
        return 1;
    }
    if (mode != 1 && mode != 2 && mode != 3) {
        set_error("mixed mode must be 1 (B fp32, state fp64), 2 (B/state fp32), or 3 (B fp32, state fp16)");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "cudaSetDevice")) return 1;
    double *d_R0 = nullptr, *d_Z0 = nullptr, *d_R1 = nullptr, *d_Z1 = nullptr;
    size_t bytes = static_cast<size_t>(n_lines) * sizeof(double);
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R0), bytes), "trace mixed d_R0") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z0), bytes), "trace mixed d_Z0") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R1), bytes), "trace mixed d_R1") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z1), bytes), "trace mixed d_Z1") ||
        cuda_check(cudaMemcpy(d_R0, R0_host, bytes, cudaMemcpyHostToDevice), "trace mixed copy R0") ||
        cuda_check(cudaMemcpy(d_Z0, Z0_host, bytes, cudaMemcpyHostToDevice), "trace mixed copy Z0")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    size_t shmem = static_cast<size_t>(SEG_TILE * 6 + threads_per_line * 3) * sizeof(float);
    if (mode == 1) {
        trace_period_blockline_bf32_state64_kernel<<<n_lines, threads_per_line, shmem>>>(
            f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f,
            f->n_segments, d_R0, d_Z0, d_R1, d_Z1, n_lines, nfp, steps
        );
    } else if (mode == 2) {
        trace_period_blockline_f32_kernel<<<n_lines, threads_per_line, shmem>>>(
            f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f,
            f->n_segments, d_R0, d_Z0, d_R1, d_Z1, n_lines, nfp, steps
        );
    } else {
        trace_period_blockline_f32_state16_kernel<<<n_lines, threads_per_line, shmem>>>(
            f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f,
            f->n_segments, d_R0, d_Z0, d_R1, d_Z1, n_lines, nfp, steps
        );
    }
    if (cuda_check(cudaGetLastError(), "trace_period_blockline_mixed kernel") ||
        cuda_check(cudaDeviceSynchronize(), "trace_period_blockline_mixed sync") ||
        cuda_check(cudaMemcpy(R1_host, d_R1, bytes, cudaMemcpyDeviceToHost), "trace mixed copy R1") ||
        cuda_check(cudaMemcpy(Z1_host, d_Z1, bytes, cudaMemcpyDeviceToHost), "trace mixed copy Z1")) {
        cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
        return 1;
    }
    cudaFree(d_R0); cudaFree(d_Z0); cudaFree(d_R1); cudaFree(d_Z1);
    set_error("");
    return 0;
}

int sgpu_trace_axis_samples(
    void* handle,
    double R0,
    double Z0,
    int nfp,
    int integration_steps,
    int n_samples,
    double* R_host,
    double* Z_host,
    double* R_phi_host,
    double* Z_phi_host
) {
    SGPU_NVTX_RANGE("trace.axis_samples.mixed");
    CoilField* f = reinterpret_cast<CoilField*>(handle);
    if (!f || !R_host || !Z_host || !R_phi_host || !Z_phi_host ||
        nfp <= 0 || integration_steps <= 0 || n_samples <= 1) {
        set_error("invalid trace_axis_samples arguments");
        return 1;
    }
    if (cuda_check(cudaSetDevice(f->device_id), "trace axis cudaSetDevice")) return 1;
    double *d_R = nullptr, *d_Z = nullptr, *d_R_phi = nullptr, *d_Z_phi = nullptr;
    const size_t bytes = static_cast<size_t>(n_samples) * sizeof(double);
    if (cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R), bytes), "trace axis d_R") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z), bytes), "trace axis d_Z") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_R_phi), bytes), "trace axis d_R_phi") ||
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Z_phi), bytes), "trace axis d_Z_phi")) {
        cudaFree(d_R); cudaFree(d_Z); cudaFree(d_R_phi); cudaFree(d_Z_phi);
        return 1;
    }
    const int substeps = (integration_steps + n_samples - 1) / n_samples;
    const int threads = 256;
    const size_t shmem = static_cast<size_t>(SEG_TILE * 6 + threads * 3) * sizeof(float);
    trace_axis_samples_mixed_kernel<<<1, threads, shmem>>>(
        f->d_x_f, f->d_y_f, f->d_z_f, f->d_wx_f, f->d_wy_f, f->d_wz_f,
        f->n_segments, R0, Z0, nfp, substeps, n_samples,
        d_R, d_Z, d_R_phi, d_Z_phi
    );
    if (cuda_check(cudaGetLastError(), "trace axis kernel") ||
        cuda_check(cudaDeviceSynchronize(), "trace axis sync") ||
        cuda_check(cudaMemcpy(R_host, d_R, bytes, cudaMemcpyDeviceToHost), "trace axis copy R") ||
        cuda_check(cudaMemcpy(Z_host, d_Z, bytes, cudaMemcpyDeviceToHost), "trace axis copy Z") ||
        cuda_check(cudaMemcpy(R_phi_host, d_R_phi, bytes, cudaMemcpyDeviceToHost), "trace axis copy R_phi") ||
        cuda_check(cudaMemcpy(Z_phi_host, d_Z_phi, bytes, cudaMemcpyDeviceToHost), "trace axis copy Z_phi")) {
        cudaFree(d_R); cudaFree(d_Z); cudaFree(d_R_phi); cudaFree(d_Z_phi);
        return 1;
    }
    cudaFree(d_R); cudaFree(d_Z); cudaFree(d_R_phi); cudaFree(d_Z_phi);
    set_error("");
    return 0;
}

} // extern "C"
