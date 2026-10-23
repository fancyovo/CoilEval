#pragma once

#include "pipeline/types.cuh"

// Small templates and device-inline numerical primitives shared by kernels.
namespace coil_eval::pipeline {

__device__ inline double atomic_add_double(double* address, double value) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    auto* integer_address = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *integer_address;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(
            integer_address,
            assumed,
            __double_as_longlong(value + __longlong_as_double(assumed))
        );
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

template <typename T>
bool copy_to_device(DeviceBuffer<T>& destination, const std::vector<T>& source) {
    return destination.allocate(source.size()) &&
           cudaMemcpy(destination.data(), source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}

__device__ inline void periodic_hermite_device(
    float phi,
    const float* values,
    const float* derivatives,
    int count,
    float period,
    float& value,
    float& derivative
) {
    float wrapped = fmodf(phi, period);
    if (wrapped < 0.0f) wrapped += period;
    const float position = wrapped * count / period;
    int index0 = static_cast<int>(floorf(position));
    if (index0 >= count) index0 = 0;
    const int index1 = index0 + 1 == count ? 0 : index0 + 1;
    const float t = position - index0;
    const float h = period / count;
    const float h00 = (2.0f * t - 3.0f) * t * t + 1.0f;
    const float h10 = ((t - 2.0f) * t + 1.0f) * t;
    const float h01 = (-2.0f * t + 3.0f) * t * t;
    const float h11 = (t - 1.0f) * t * t;
    value = h00 * values[index0] + h10 * h * derivatives[index0] +
            h01 * values[index1] + h11 * h * derivatives[index1];
    const float dh00 = (6.0f * t - 6.0f) * t;
    const float dh10 = (3.0f * t - 4.0f) * t + 1.0f;
    const float dh01 = (-6.0f * t + 6.0f) * t;
    const float dh11 = (3.0f * t - 2.0f) * t;
    derivative = (dh00 * values[index0] + dh01 * values[index1]) / h +
                 dh10 * derivatives[index0] + dh11 * derivatives[index1];
}

__device__ inline void evaluate_psi_device(
    const float* coeffs,
    const int* mode_a,
    const int* mode_b,
    const int* mode_m,
    const int* mode_kind,
    int n_coeff,
    int nfp,
    int degree,
    int m_tor,
    float a_scale,
    float R,
    float Z,
    float phi,
    float axis_R,
    float axis_Z,
    float axis_R_phi,
    float axis_Z_phi,
    float& value,
    float& grad_R,
    float& grad_Z,
    float& grad_phi
) {
    float xpow[25];
    float zpow[25];
    float cosv[33];
    float sinv[33];
    const float X = (R - axis_R) / a_scale;
    const float Y = (Z - axis_Z) / a_scale;
    xpow[0] = 1.0f;
    zpow[0] = 1.0f;
    for (int power = 1; power <= degree; ++power) {
        xpow[power] = xpow[power - 1] * X;
        zpow[power] = zpow[power - 1] * Y;
    }
    cosv[0] = 1.0f;
    sinv[0] = 0.0f;
    for (int mode = 1; mode <= m_tor; ++mode) {
        sincosf(static_cast<float>(mode * nfp) * phi, &sinv[mode], &cosv[mode]);
    }
    value = X * X;
    grad_R = 2.0f * X / a_scale;
    grad_Z = 0.0f;
    grad_phi = -axis_R_phi * grad_R;
    for (int index = 0; index < n_coeff; ++index) {
        const int ax = mode_a[index];
        const int bz = mode_b[index];
        const int mode = mode_m[index];
        const float trig = mode == 0 ? 1.0f : (mode_kind[index] == 0 ? cosv[mode] : sinv[mode]);
        const float trig_phi = mode == 0 ? 0.0f : static_cast<float>(mode * nfp) *
            (mode_kind[index] == 0 ? -sinv[mode] : cosv[mode]);
        const float mono = xpow[ax] * zpow[bz];
        const float derivative_x = ax > 0 ? ax * xpow[ax - 1] * zpow[bz] / a_scale : 0.0f;
        const float derivative_z = bz > 0 ? bz * xpow[ax] * zpow[bz - 1] / a_scale : 0.0f;
        const float coefficient = coeffs[index];
        value += coefficient * mono * trig;
        grad_R += coefficient * derivative_x * trig;
        grad_Z += coefficient * derivative_z * trig;
        grad_phi += coefficient *
            (mono * trig_phi - axis_R_phi * derivative_x * trig - axis_Z_phi * derivative_z * trig);
    }
}

__device__ inline float evaluate_radial_polynomial(
    const float* coefficients,
    int degree,
    float rho
) {
    float result = coefficients[degree];
    for (int power = degree - 1; power >= 0; --power) result = result * rho + coefficients[power];
    return result;
}
}  // namespace coil_eval::pipeline
