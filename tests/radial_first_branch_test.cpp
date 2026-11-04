#include "radial_first_branch.h"
#include "target_volume_geometry.h"
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    double p[25]{};
    p[2] = 1; p[4] = 4; p[6] = -8;
    double r = std::sqrt(.5);
    if (!coil_eval_radial::first_root(p, 6, .5, 1., 1e-12, r) ||
        std::abs(r - std::pow(2., -.75)) > 1e-12) return 1;
    p[4] = 8; p[6] = -16; p[8] = 8;
    r = 1.; // Third root has a positive slope, so an endpoint guard would pass.
    if (!coil_eval_radial::first_root(p, 8, 1., 1.2, 1e-12, r) ||
        std::abs(r - std::sqrt(.5)) > 1e-12) return 2;
    r = 1.;
    if (coil_eval_radial::first_root(p, 8, 1.1, 1.2, 1e-12, r)) return 3;
    p[4] = p[6] = p[8] = 0;
    r = .456789;
    const double exact_candidate = r;
    if (!coil_eval_radial::first_root(p, 8, r*r, 1., 1e-12, r) ||
        r != exact_candidate) return 4;
    // A first-branch but unconverged Newton candidate also gets a valid bracket.
    r = .9;
    if (!coil_eval_radial::first_root(p, 8, .04, 1., 1e-12, r) ||
        std::abs(r - .2) > 1e-12) return 5;
    p[2] = -1;
    if (coil_eval_radial::first_root(p, 8, .04, 1., 1e-12, r)) return 6;
    p[2] = std::numeric_limits<double>::quiet_NaN();
    if (coil_eval_radial::first_root(p, 8, .04, 1., 1e-12, r)) return 7;
    // Native target geometry deliberately requires strict monotonicity. This
    // nondecreasing polynomial has a flat nonfold at u=.5; the Python exact
    // branch resolver can pass it, while this conservative certificate cannot.
    p[2] = 1; p[4] = -4; p[6] = 16./3.;
    r = 1.;
    if (coil_eval_radial::first_root(p, 8, 7./3., 1.2, 1e-12, r)) return 11;

    int count, degree;
    double level;
    if (!(std::cin >> count >> degree >> level)) return 8;
    for (int ray = 0; ray < count; ++ray) {
        double expected;
        for (int j = 0; j <= degree; ++j) std::cin >> p[j];
        std::cin >> expected;
        r = std::sqrt(level / p[2]);
        if (!coil_eval_radial::first_root(p, degree, level, 1., 1e-11, r) ||
            std::abs(r - expected) > 1e-10) return 9;
        float fp[25];
        for (int j = 0; j <= degree; ++j) fp[j] = p[j];
        r = std::sqrt(float(level) / fp[2]);
        if (!coil_eval_radial::first_root(fp, degree, float(level), 1.2, 1e-8, r) ||
            std::abs(r - expected) > 2e-6) {
            std::cerr << "float coefficient failure at ray " << ray << "\n";
            return 12;
        }
        sgpu_target::Polynomial original{};
        for (int j = 0; j <= degree; ++j) original[j] = p[j];
        // The iterative CUDA-compatible certificate agrees with the host guard
        // on this whole-ray grid, including intervals which contain folds.
        for (int j = 1; j <= 200; ++j) {
            const double bound = double(j) / 200;
            if (coil_eval_radial::nested(p, degree, bound) !=
                sgpu_target::nested(original, degree, bound)) return 10;
        }
    }
    std::cout << "first-branch host regression passed: " << count << " frozen rays\n";
}
