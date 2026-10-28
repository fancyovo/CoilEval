#pragma once

namespace coil_eval::cpu {
// Runtime-checked AVX2; the shared library itself keeps a portable baseline.
bool avx2_available();
// Exactly eight independent query points, preserving each segment sum order.
void field8_f32(const float* positions, const float* elements, int segments,
                const double* xyz, double* magnetic, double* gradient,
                double* potential);
}
