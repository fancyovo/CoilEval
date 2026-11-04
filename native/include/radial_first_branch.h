// The formal sampler and surface screen must use the axis-connected root.
// A small residual and a positive derivative at a candidate are insufficient:
// the candidate may lie on the third branch of a nonmonotone polynomial.
#pragma once
#include <cfloat>

#ifdef __CUDACC__
#define COIL_EVAL_RADIAL_HD __host__ __device__
#else
#define COIL_EVAL_RADIAL_HD
#endif

namespace coil_eval_radial {
constexpr int maximum_degree = 24;
constexpr int subdivision_depth = 10;

COIL_EVAL_RADIAL_HD inline bool finite(double x) {
    return x >= -DBL_MAX && x <= DBL_MAX;
}
COIL_EVAL_RADIAL_HD inline double absolute(double x) { return x < 0 ? -x : x; }
COIL_EVAL_RADIAL_HD inline double value(const double* p, int degree, double u) {
    double result = p[degree];
    for (int j = degree - 1; j >= 0; --j) result = result * u + p[j];
    return result;
}

// Same strict positivity threshold and subdivision depth as target geometry.
// Iterative de Casteljau subdivision avoids device recursion/stack-size changes.
COIL_EVAL_RADIAL_HD inline bool nested(const double* p, int degree, double maximum) {
    if (degree < 2 || degree > maximum_degree || !(p[2] > 1e-12) ||
        !(maximum > 0) || !finite(maximum)) return false;
    const int n = degree - 2;
    double scaled[25], b[25];
    double power = 1;
    for (int j = 0; j <= n; ++j) {
        scaled[j] = (j + 2) * p[j + 2] * power;
        power *= maximum;
    }
    for (int k = 0; k <= n; ++k) {
        double ratio = 1, sum = 0;
        for (int j = 0; j <= k; ++j) {
            sum += scaled[j] * ratio;
            if (j < k) ratio *= double(k - j) / double(n - j);
        }
        if (!finite(sum)) return false;
        b[k] = sum;
    }
    double pending[subdivision_depth][25];
    int pending_depth[subdivision_depth], count = 0, depth = 0;
    for (;;) {
        double low = b[0], high = b[0];
        for (int j = 1; j <= n; ++j) {
            if (b[j] < low) low = b[j];
            if (b[j] > high) high = b[j];
        }
        if (low > 1e-12) {
            if (count == 0) return true;
            --count;
            depth = pending_depth[count];
            for (int j = 0; j <= n; ++j) b[j] = pending[count][j];
            continue;
        }
        if (high <= 1e-12 || depth == subdivision_depth) return false;
        double left[25];
        left[0] = b[0];
        pending[count][n] = b[n];
        for (int k = 1; k <= n; ++k) {
            for (int j = 0; j <= n - k; ++j) b[j] = .5 * (b[j] + b[j + 1]);
            left[k] = b[0];
            pending[count][n - k] = b[n - k];
        }
        pending_depth[count++] = ++depth;
        for (int j = 0; j <= n; ++j) b[j] = left[j];
    }
}

// Preserve already valid Newton results. Otherwise bracket only within the
// certified first monotone branch, never on a later positive-slope branch.
// Coefficients and the returned root use u = radius / psi_a.
template <typename Scalar>
COIL_EVAL_RADIAL_HD inline bool first_root(
    const Scalar* polynomial, int degree, double level, double maximum,
    double tolerance, double& candidate
) {
    if (degree < 2 || degree > maximum_degree || !(level > 0) ||
        !finite(level) || !(maximum > 0) || !finite(maximum)) return false;
    double p[25];
    for (int j = 0; j <= degree; ++j) {
        p[j] = polynomial[j];
        if (!finite(p[j])) return false;
    }
    if (!(p[2] > 1e-12) || p[0] != 0 || p[1] != 0) return false;
    if (candidate > 0 && candidate <= maximum &&
        absolute(value(p, degree, candidate) - level) <= tolerance &&
        nested(p, degree, candidate)) return true;

    double cap = maximum;
    if (!nested(p, degree, cap)) {
        double lo = 0, hi = maximum;
        for (int j = 0; j < 32; ++j) {
            const double mid = .5 * (lo + hi);
            if (nested(p, degree, mid)) lo = mid; else hi = mid;
        }
        cap = lo * (1 - 1e-6);
    }
    if (!(cap > 0) || !nested(p, degree, cap) ||
        !(value(p, degree, cap) >= level)) return false;
    double lo = 0, hi = cap;
    for (int j = 0; j < 48; ++j) {
        const double mid = .5 * (lo + hi);
        if (value(p, degree, mid) < level) lo = mid; else hi = mid;
    }
    candidate = .5 * (lo + hi);
    return finite(candidate) && candidate > 0;
}
}  // namespace coil_eval_radial
#undef COIL_EVAL_RADIAL_HD
