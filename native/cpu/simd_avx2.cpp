#include "simd_avx2.h"
#include <cstddef>

#if !defined(COIL_EVAL_DISABLE_AVX2) && (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define COIL_EVAL_AVX2 __attribute__((target("avx2,no-fma")))

namespace coil_eval::cpu {
bool avx2_available() {
    // Compiler runtime detection also checks OS support for the YMM state.
    static const bool supported = __builtin_cpu_supports("avx2");
    return supported;
}

namespace {
template<bool Gradient, bool Potential>
COIL_EVAL_AVX2 void evaluate8(const float* positions, const float* elements,
                             int segments, const double* xyz, double* magnetic,
                             double* gradient, double* potential) {
    // Conversion is exactly the scalar float(x[coordinate]) before subtraction.
    const __m256 x[3] = {
        _mm256_setr_ps(float(xyz[0]),float(xyz[3]),float(xyz[6]),float(xyz[9]),
                      float(xyz[12]),float(xyz[15]),float(xyz[18]),float(xyz[21])),
        _mm256_setr_ps(float(xyz[1]),float(xyz[4]),float(xyz[7]),float(xyz[10]),
                      float(xyz[13]),float(xyz[16]),float(xyz[19]),float(xyz[22])),
        _mm256_setr_ps(float(xyz[2]),float(xyz[5]),float(xyz[8]),float(xyz[11]),
                      float(xyz[14]),float(xyz[17]),float(xyz[20]),float(xyz[23]))};
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.f), three = _mm256_set1_ps(3.f);
    const __m256 epsilon = _mm256_set1_ps(1e-30f), sign = _mm256_set1_ps(-0.f);
    __m256 sum[3] = {zero,zero,zero}, jacobian[9], vector_potential[3] = {zero,zero,zero};
    if constexpr(Gradient) for (int j = 0; j < 9; ++j) jacobian[j] = zero;
    for (int segment = 0; segment < segments; ++segment) {
        const std::size_t offset = std::size_t(segment)*3;
        const __m256 r[3] = {
            _mm256_sub_ps(x[0],_mm256_set1_ps(positions[offset])),
            _mm256_sub_ps(x[1],_mm256_set1_ps(positions[offset+1])),
            _mm256_sub_ps(x[2],_mm256_set1_ps(positions[offset+2]))};
        const __m256 w[3] = {_mm256_set1_ps(elements[offset]),
                            _mm256_set1_ps(elements[offset+1]),
                            _mm256_set1_ps(elements[offset+2])};
        const __m256 radius2 = _mm256_add_ps(_mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(r[0],r[0]),_mm256_mul_ps(r[1],r[1])),
            _mm256_mul_ps(r[2],r[2])),epsilon);
        // Use correctly-rounded division/sqrt, not reciprocal approximations.
        const __m256 inverse = _mm256_div_ps(one,_mm256_sqrt_ps(radius2));
        const __m256 inverse3 = _mm256_mul_ps(_mm256_mul_ps(inverse,inverse),inverse);
        const __m256 cross[3] = {
            _mm256_sub_ps(_mm256_mul_ps(w[1],r[2]),_mm256_mul_ps(w[2],r[1])),
            _mm256_sub_ps(_mm256_mul_ps(w[2],r[0]),_mm256_mul_ps(w[0],r[2])),
            _mm256_sub_ps(_mm256_mul_ps(w[0],r[1]),_mm256_mul_ps(w[1],r[0]))};
        for (int i = 0; i < 3; ++i) {
            sum[i] = _mm256_add_ps(sum[i],_mm256_mul_ps(cross[i],inverse3));
            if constexpr(Potential)
                vector_potential[i] = _mm256_add_ps(vector_potential[i],_mm256_mul_ps(w[i],inverse));
        }
        if constexpr(Gradient) {
            const __m256 inverse5 = _mm256_mul_ps(_mm256_mul_ps(inverse3,inverse),inverse);
            const __m256 skew[9] = {zero,_mm256_xor_ps(w[2],sign),w[1],
                                   w[2],zero,_mm256_xor_ps(w[0],sign),
                                   _mm256_xor_ps(w[1],sign),w[0],zero};
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
                const __m256 radial = _mm256_mul_ps(
                    _mm256_mul_ps(_mm256_mul_ps(three,cross[i]),r[j]),inverse5);
                const __m256 contribution = _mm256_sub_ps(_mm256_mul_ps(skew[3*i+j],inverse3),radial);
                jacobian[3*i+j] = _mm256_add_ps(jacobian[3*i+j],contribution);
            }
        }
    }
    const __m256 prefactor = _mm256_set1_ps(1e-7f);
    alignas(32) float lanes[8];
    for (int i = 0; i < 3; ++i) {
        _mm256_store_ps(lanes,_mm256_mul_ps(sum[i],prefactor));
        for (int lane = 0; lane < 8; ++lane) magnetic[3*lane+i] = double(lanes[lane]);
        if constexpr(Potential) {
            _mm256_store_ps(lanes,_mm256_mul_ps(vector_potential[i],prefactor));
            for (int lane = 0; lane < 8; ++lane) potential[3*lane+i] = double(lanes[lane]);
        }
    }
    if constexpr(Gradient) for (int i = 0; i < 9; ++i) {
        _mm256_store_ps(lanes,_mm256_mul_ps(jacobian[i],prefactor));
        for (int lane = 0; lane < 8; ++lane) gradient[9*lane+i] = double(lanes[lane]);
    }
}
}

void field8_f32(const float* positions, const float* elements, int segments,
                const double* xyz, double* magnetic, double* gradient, double* potential) {
    if (gradient) {
        if (potential) evaluate8<true,true>(positions,elements,segments,xyz,magnetic,gradient,potential);
        else evaluate8<true,false>(positions,elements,segments,xyz,magnetic,gradient,nullptr);
    } else {
        if (potential) evaluate8<false,true>(positions,elements,segments,xyz,magnetic,nullptr,potential);
        else evaluate8<false,false>(positions,elements,segments,xyz,magnetic,nullptr,nullptr);
    }
}
}
#else
namespace coil_eval::cpu {
bool avx2_available() { return false; }
void field8_f32(const float*,const float*,int,const double*,double*,double*,double*) {}
}
#endif
