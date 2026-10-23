// Host psi representation, fit/validation orchestration, and device-data upload.
#include "pipeline/types.cuh"
#include "pipeline/device_helpers.cuh"
#include "pipeline/session_state.cuh"
#include "pipeline/common.cuh"
#include "pipeline/psi.cuh"
#include "pipeline/axis.cuh"
#include "nvtx_profile.h"

namespace coil_eval::pipeline {

void periodic_hermite_host(
    double phi,
    const std::vector<double>& values,
    const std::vector<double>& derivatives,
    int nfp,
    double& value,
    double& derivative
) {
    const int count = static_cast<int>(values.size());
    const double period = TWOPI / nfp;
    double wrapped = std::fmod(phi, period);
    if (wrapped < 0.0) wrapped += period;
    const double position = wrapped * count / period;
    const int index0 = std::min(static_cast<int>(std::floor(position)), count - 1);
    const int index1 = (index0 + 1) % count;
    const double t = position - index0;
    const double h = period / count;
    const double h00 = (2.0 * t - 3.0) * t * t + 1.0;
    const double h10 = ((t - 2.0) * t + 1.0) * t;
    const double h01 = (-2.0 * t + 3.0) * t * t;
    const double h11 = (t - 1.0) * t * t;
    value = h00 * values[index0] + h10 * h * derivatives[index0] +
            h01 * values[index1] + h11 * h * derivatives[index1];
    const double dh00 = (6.0 * t - 6.0) * t;
    const double dh10 = (3.0 * t - 4.0) * t + 1.0;
    const double dh01 = (-6.0 * t + 6.0) * t;
    const double dh11 = (3.0 * t - 2.0) * t;
    derivative = (dh00 * values[index0] + dh01 * values[index1]) / h +
                 dh10 * derivatives[index0] + dh11 * derivatives[index1];
}

PsiModes build_psi_modes(int degree, int m_tor) {
    PsiModes modes;
    for (int total = 2; total <= degree; ++total) {
        for (int a = total; a >= 0; --a) {
            const int b = total - a;
            for (int m = 0; m <= m_tor; ++m) {
                if (a == 2 && b == 0 && m == 0) continue;
                modes.a.push_back(a);
                modes.b.push_back(b);
                modes.m.push_back(m);
                modes.kind.push_back(0);
                if (m > 0) {
                    modes.a.push_back(a);
                    modes.b.push_back(b);
                    modes.m.push_back(m);
                    modes.kind.push_back(1);
                }
            }
        }
    }
    return modes;
}

void evaluate_psi_host(
    const PsiData& psi,
    const AxisData& axis,
    const SgpuScoreConfig& config,
    int nfp,
    double R,
    double Z,
    double phi,
    double& value,
    double& grad_R,
    double& grad_Z,
    double& grad_phi
) {
    double axis_R, axis_R_phi, axis_Z, axis_Z_phi;
    periodic_hermite_host(phi, axis.R, axis.R_phi, nfp, axis_R, axis_R_phi);
    periodic_hermite_host(phi, axis.Z, axis.Z_phi, nfp, axis_Z, axis_Z_phi);
    const double X = (R - axis_R) / config.psi_a;
    const double Y = (Z - axis_Z) / config.psi_a;
    std::array<double, 25> xpow{}, ypow{};
    std::array<double, 33> cosv{}, sinv{};
    xpow[0] = 1.0;
    ypow[0] = 1.0;
    for (int power = 1; power <= config.psi_poly_degree; ++power) {
        xpow[power] = xpow[power - 1] * X;
        ypow[power] = ypow[power - 1] * Y;
    }
    cosv[0] = 1.0;
    for (int mode = 1; mode <= config.psi_m_tor; ++mode) {
        const double argument = static_cast<double>(mode * nfp) * phi;
        cosv[mode] = std::cos(argument);
        sinv[mode] = std::sin(argument);
    }
    value = X * X;
    grad_R = 2.0 * X / config.psi_a;
    grad_Z = 0.0;
    grad_phi = -axis_R_phi * grad_R;
    for (size_t k = 0; k < psi.coeffs.size(); ++k) {
        const int a = psi.modes.a[k];
        const int b = psi.modes.b[k];
        const int m = psi.modes.m[k];
        const double trig = psi.modes.kind[k] == 0 ? cosv[m] : sinv[m];
        const double trig_phi = m == 0 ? 0.0 : static_cast<double>(m * nfp) *
            (psi.modes.kind[k] == 0 ? -sinv[m] : cosv[m]);
        const double mono = xpow[a] * ypow[b];
        const double derivative_x = a > 0 ? a * xpow[a - 1] * ypow[b] / config.psi_a : 0.0;
        const double derivative_y = b > 0 ? b * xpow[a] * ypow[b - 1] / config.psi_a : 0.0;
        const double coefficient = psi.coeffs[k];
        value += coefficient * mono * trig;
        grad_R += coefficient * derivative_x * trig;
        grad_Z += coefficient * derivative_y * trig;
        grad_phi += coefficient * (mono * trig_phi - axis_R_phi * derivative_x * trig - axis_Z_phi * derivative_y * trig);
    }
}

bool fit_psi_native(
    void* field,
    const AxisData& axis,
    int nfp,
    const SgpuScoreConfig& config,
    PsiData& psi,
    double& point_generation_time,
    double& fit_time
) {
    SGPU_NVTX_RANGE("psi.fit.total");
    auto started = Clock::now();
    std::vector<double> R, Z, phi;
    const size_t reserve = static_cast<size_t>(config.psi_n_r) * config.psi_n_z * config.psi_n_phi;
    R.reserve(reserve);
    Z.reserve(reserve);
    phi.reserve(reserve);
    std::vector<double> phi_grid(config.psi_n_phi);
    std::vector<double> axis_R_grid(config.psi_n_phi), axis_Z_grid(config.psi_n_phi);
    for (int iphi = 0; iphi < config.psi_n_phi; ++iphi) {
        const double angle = TWOPI * iphi / static_cast<double>(nfp * config.psi_n_phi);
        double axis_R_phi, axis_Z_phi;
        phi_grid[iphi] = angle;
        periodic_hermite_host(
            angle, axis.R, axis.R_phi, nfp, axis_R_grid[iphi], axis_R_phi
        );
        periodic_hermite_host(
            angle, axis.Z, axis.Z_phi, nfp, axis_Z_grid[iphi], axis_Z_phi
        );
    }
    {
        SGPU_NVTX_RANGE("psi.points.cpu");
        for (int ir = 0; ir < config.psi_n_r; ++ir) {
            const double dR = -config.psi_a + 2.0 * config.psi_a * ir / std::max(config.psi_n_r - 1, 1);
            for (int iz = 0; iz < config.psi_n_z; ++iz) {
                const double dZ = -config.psi_a + 2.0 * config.psi_a * iz / std::max(config.psi_n_z - 1, 1);
                const double radius = std::hypot(dR, dZ);
                if (radius < config.psi_rho_min || radius > config.psi_a) continue;
                for (int iphi = 0; iphi < config.psi_n_phi; ++iphi) {
                    R.push_back(axis_R_grid[iphi] + dR);
                    Z.push_back(axis_Z_grid[iphi] + dZ);
                    phi.push_back(phi_grid[iphi]);
                }
            }
        }
    }
    point_generation_time = seconds_since(started);
    psi.modes = build_psi_modes(config.psi_poly_degree, config.psi_m_tor);
    if (g_active_psi_warm_coefficients) {
        if (g_active_psi_warm_coefficients->size() != psi.modes.a.size()) return false;
        psi.coeffs = *g_active_psi_warm_coefficients;
    } else {
        psi.coeffs.resize(psi.modes.a.size());
    }
    std::array<double, 16> stats{};
    started = Clock::now();
    int code = 0;
    {
        SGPU_NVTX_RANGE("psi.fit.fullgpu");
        code = sgpu_fit_psi_fullgpu(
            field, R.data(), Z.data(), phi.data(), static_cast<int>(R.size()),
            axis.R.data(), axis.Z.data(), axis.R_phi.data(), axis.Z_phi.data(),
            static_cast<int>(axis.R.size()),
            psi.modes.a.data(), psi.modes.b.data(), psi.modes.m.data(), psi.modes.kind.data(),
            static_cast<int>(psi.modes.a.size()), nfp, config.psi_a,
            config.psi_poly_degree, config.psi_m_tor, config.psi_ridge,
            g_active_psi_warm_coefficients
                ? (g_active_psi_warm_preconditioned ? 2000 : 1000) + g_active_psi_warm_iterations
                : config.psi_solver_mode,
            config.psi_precision_mode,
            psi.coeffs.data(), &psi.train_rms, stats.data(), static_cast<int>(stats.size())
        );
    }
    fit_time = seconds_since(started);
    return code == 0;
}

bool validate_psi_native(
    void* field,
    const AxisData& axis,
    int nfp,
    const SgpuScoreConfig& config,
    PsiData& psi
) {
    SGPU_NVTX_RANGE("psi.validate.total");
    const int count = config.psi_validation_points;
    std::vector<float> xyz(static_cast<size_t>(count) * 3), B(xyz.size());
    std::vector<double> R(count), Z(count), phi(count), grad_norm(count), residual(count), angle(count);
    std::uint64_t state = 20260704u;
    auto uniform = [&]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 11) * (1.0 / 9007199254740992.0);
    };
    const double u_min = std::pow(config.psi_rho_min / config.psi_a, 2.0);
    {
        SGPU_NVTX_RANGE("psi.validate.points.cpu");
        for (int index = 0; index < count; ++index) {
            phi[index] = uniform() * TWOPI / nfp;
            const double theta = uniform() * TWOPI;
            const double radius = config.psi_a * std::sqrt(u_min + (1.0 - u_min) * uniform());
            double axis_R, axis_R_phi, axis_Z, axis_Z_phi;
            periodic_hermite_host(phi[index], axis.R, axis.R_phi, nfp, axis_R, axis_R_phi);
            periodic_hermite_host(phi[index], axis.Z, axis.Z_phi, nfp, axis_Z, axis_Z_phi);
            R[index] = axis_R + radius * std::cos(theta);
            Z[index] = axis_Z + radius * std::sin(theta);
            xyz[3 * index] = static_cast<float>(R[index] * std::cos(phi[index]));
            xyz[3 * index + 1] = static_cast<float>(R[index] * std::sin(phi[index]));
            xyz[3 * index + 2] = static_cast<float>(Z[index]);
        }
    }
    {
        SGPU_NVTX_RANGE("psi.validate.field.fp32");
        if (sgpu_eval_B_f32(field, xyz.data(), B.data(), count)) return false;
    }
    double residual_norm2 = 0.0;
    double scale_norm2 = 0.0;
    double angle_sum = 0.0;
    SGPU_NVTX_RANGE("psi.validate.reduce.cpu");
    for (int index = 0; index < count; ++index) {
        double value, gR, gZ, gPhi;
        evaluate_psi_host(psi, axis, config, nfp, R[index], Z[index], phi[index], value, gR, gZ, gPhi);
        const double cosine = std::cos(phi[index]);
        const double sine = std::sin(phi[index]);
        const double bx = B[3 * index];
        const double by = B[3 * index + 1];
        const double bz = B[3 * index + 2];
        const double br = bx * cosine + by * sine;
        const double bphi = -bx * sine + by * cosine;
        const double dot = br * gR + bz * gZ + bphi * gPhi / R[index];
        const double bnorm = std::sqrt(bx * bx + by * by + bz * bz);
        const double gnorm = std::sqrt(gR * gR + gZ * gZ + std::pow(gPhi / R[index], 2.0));
        residual[index] = dot;
        grad_norm[index] = bnorm * gnorm;
        angle[index] = std::abs(dot) / std::max(grad_norm[index], 1.0e-14);
        residual_norm2 += dot * dot;
        scale_norm2 += grad_norm[index] * grad_norm[index];
        angle_sum += angle[index];
    }
    psi.angle_mean = angle_sum / count;
    psi.angle_p95 = percentile(angle, 0.95);
    psi.angle_l2 = std::sqrt(residual_norm2 / std::max(scale_norm2, 1.0e-300));
    return true;
}

bool upload_psi_data(
    const PsiData& psi,
    const AxisData& axis,
    DevicePsiData& device
) {
    std::vector<float> coefficients(psi.coeffs.begin(), psi.coeffs.end());
    std::vector<float> axis_R(axis.R.begin(), axis.R.end());
    std::vector<float> axis_Z(axis.Z.begin(), axis.Z.end());
    std::vector<float> axis_R_phi(axis.R_phi.begin(), axis.R_phi.end());
    std::vector<float> axis_Z_phi(axis.Z_phi.begin(), axis.Z_phi.end());
    return copy_to_device(device.coeffs, coefficients) &&
           copy_to_device(device.mode_a, psi.modes.a) &&
           copy_to_device(device.mode_b, psi.modes.b) &&
           copy_to_device(device.mode_m, psi.modes.m) &&
           copy_to_device(device.mode_kind, psi.modes.kind) &&
           copy_to_device(device.axis_R, axis_R) &&
           copy_to_device(device.axis_Z, axis_Z) &&
           copy_to_device(device.axis_R_phi, axis_R_phi) &&
           copy_to_device(device.axis_Z_phi, axis_Z_phi);
}


}  // namespace coil_eval::pipeline
