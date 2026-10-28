// Fixed-iteration surface-ray Newton solve. No field-line integration occurs here.
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {
constexpr int maximum_degree = 32;

__device__ double clip(double x, double lower, double upper) {
    // Comparisons preserve NaNs, like torch.clamp; upper wins for inverted bounds.
    const double above_lower = x < lower ? lower : x;
    return above_lower > upper ? upper : above_lower;
}

__device__ void evaluate(const double* coefficients, double radius, int columns,
                         double* powers, double& value, double& derivative) {
    value = 0.; derivative = 0.;
    for (int k = 0; k < columns; ++k) {
        // Integer powers share products within a ray. Keep the polynomial
        // summation and Newton policy while avoiding repeated transcendental pow.
        powers[k] = k ? __dmul_rn(powers[k-1], radius) : 1.;
        value = __dadd_rn(value, __dmul_rn(coefficients[k], powers[k]));
        if (k) derivative = __dadd_rn(derivative,
            __dmul_rn(__dmul_rn(coefficients[k], double(k)), powers[k-1]));
    }
}

__global__ void radial_root_kernel(const double* polynomial, const double* level,
    double* roots, double* final_powers, double* final_derivatives, int* status,
    int rows, int columns, int iterations, double radius_scale,
    double maximum_scale, double tolerance) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    double coefficients[maximum_degree+1], powers[maximum_degree+1];
    for (int k = 0; k < columns; ++k)
        coefficients[k] = polynomial[std::size_t(row)*columns+k];
    if (!(coefficients[2] > 0.)) {
        status[row] = 2; // Preserve the original quadratic-coefficient error.
        roots[row] = final_derivatives[row] = nan("");
        for (int k = 0; k < columns; ++k)
            final_powers[std::size_t(row)*columns+k] = nan("");
        return;
    }
    const double minimum = 1e-12*radius_scale;
    const double maximum = maximum_scale*radius_scale;
    const double target = level[row];
    double radius = clip(sqrt((target < 0. ? 0. : target)/coefficients[2]), minimum, maximum);
    double value, derivative;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        evaluate(coefficients, radius, columns, powers, value, derivative);
        const double denominator = fabs(derivative) > 1e-14 ? derivative :
                                   (derivative < 0. ? -1e-14 : 1e-14);
        const double magnitude = fabs(radius);
        const double limit = .45*(magnitude < 1e-8*radius_scale ? 1e-8*radius_scale : magnitude);
        const double residual = __dadd_rn(value, -target);
        const double step = clip(residual/denominator, -limit, limit);
        if (!(fabs(residual) <= tolerance)) radius = clip(__dadd_rn(radius, -step), minimum, maximum);
    }
    evaluate(coefficients, radius, columns, powers, value, derivative);
    roots[row] = radius;
    final_derivatives[row] = derivative;
    for (int k = 0; k < columns; ++k)
        final_powers[std::size_t(row)*columns+k] = powers[k];
    status[row] = (isfinite(radius) && radius > 0. &&
                  fabs(value-target) < fmax(1e-8, 100*tolerance) &&
                  fabs(derivative) > 1e-12) ? 0 : 1;
}
}

extern "C" int fc_radial_root_f64(const void* polynomial, const void* level,
    void* roots, void* powers, void* derivatives, void* status, int rows,
    int columns, int iterations, double radius_scale, double maximum_scale,
    double tolerance, std::uint64_t stream) {
    if (rows < 0 || columns < 3 || columns > maximum_degree+1 || iterations < 0)
        return int(cudaErrorInvalidValue);
    if (!rows) return 0;
    if (!polynomial || !level || !roots || !powers || !derivatives || !status)
        return int(cudaErrorInvalidValue);
    radial_root_kernel<<<(rows+127)/128, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const double*>(polynomial), static_cast<const double*>(level),
        static_cast<double*>(roots), static_cast<double*>(powers),
        static_cast<double*>(derivatives), static_cast<int*>(status), rows,
        columns, iterations, radius_scale, maximum_scale, tolerance);
    return int(cudaGetLastError());
}
