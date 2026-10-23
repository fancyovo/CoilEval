// Coarse candidate search, axis refinement, topology classification, and circulation.
#include "pipeline/types.cuh"
#include "pipeline/axis.cuh"
#include "pipeline/coil_geometry.cuh"
#include "pipeline/common.cuh"
#include "coil_field_internal.h"
#include "nvtx_profile.h"

namespace coil_eval::pipeline {

namespace {

struct AxisDomain {
    double r_min;
    double r_max;
    double z_min;
    double z_max;
};

}  // namespace


AxisDomain build_axis_domain(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    int n_base_coils,
    int n_coeff,
    const SgpuScoreConfig& config
) {
    std::vector<double> radii;
    std::vector<double> abs_z;
    radii.reserve(static_cast<size_t>(n_base_coils) * 160);
    abs_z.reserve(radii.capacity());
    double center = 0.0;
    for (int coil = 0; coil < n_base_coils; ++coil) {
        center += std::hypot(coeffs_x[static_cast<size_t>(coil) * n_coeff],
                             coeffs_y[static_cast<size_t>(coil) * n_coeff]);
        for (int sample = 0; sample < 160; ++sample) {
            const double t = static_cast<double>(sample) / 160.0;
            double x, dx, ddx, y, dy, ddy, z, dz, ddz;
            eval_fourier_interleaved(coeffs_x + static_cast<size_t>(coil) * n_coeff, n_coeff, t, x, dx, ddx);
            eval_fourier_interleaved(coeffs_y + static_cast<size_t>(coil) * n_coeff, n_coeff, t, y, dy, ddy);
            eval_fourier_interleaved(coeffs_z + static_cast<size_t>(coil) * n_coeff, n_coeff, t, z, dz, ddz);
            radii.push_back(std::hypot(x, y));
            abs_z.push_back(std::abs(z));
        }
    }
    center /= std::max(n_base_coils, 1);
    const double r05 = percentile(radii, 0.05);
    const double r95 = percentile(radii, 0.95);
    const double z95 = percentile(abs_z, 0.95);
    const double r_span = std::max(config.axis_span, 0.45 * (r95 - r05));
    const double z_span = std::max(config.axis_span, 1.15 * z95);
    AxisDomain domain;
    domain.r_min = std::max(config.axis_r_floor, std::min(center - config.axis_span, center - r_span));
    domain.r_max = std::max(center + config.axis_span, center + r_span);
    domain.z_min = -z_span;
    domain.z_max = z_span;
    if (!(domain.r_max > domain.r_min)) domain.r_max = domain.r_min + std::max(config.axis_span, 0.1);
    return domain;
}

bool trace_map(
    void* field,
    const std::vector<double>& R,
    const std::vector<double>& Z,
    int nfp,
    int steps,
    bool verify_fp64,
    std::vector<double>& R_end,
    std::vector<double>& Z_end
) {
    if (R.size() != Z.size() || R.empty()) return false;
    R_end.resize(R.size());
    Z_end.resize(Z.size());
    const int code = verify_fp64
        ? sgpu_trace_period_blockline(
              field, R.data(), Z.data(), R_end.data(), Z_end.data(),
              static_cast<int>(R.size()), nfp, steps, 256)
        : sgpu_trace_period_blockline_mixed(
              field, R.data(), Z.data(), R_end.data(), Z_end.data(),
              static_cast<int>(R.size()), nfp, steps, 256, 1);
    return code == 0;
}

double axis_topology_stability_margin(const AxisCandidate& candidate) {
    if (!(candidate.topology_det > 0.0) ||
        !std::isfinite(candidate.topology_trace) ||
        !std::isfinite(candidate.topology_det)) {
        return -std::numeric_limits<double>::infinity();
    }
    return 2.0 - std::abs(candidate.topology_trace) / std::sqrt(candidate.topology_det);
}

double wrapped_angle_delta(double from, double to) {
    double delta = to - from;
    while (delta <= -PI) delta += TWOPI;
    while (delta > PI) delta -= TWOPI;
    return delta;
}

std::vector<AxisCandidate> find_grid_candidates(
    const std::vector<double>& rs,
    const std::vector<double>& zs,
    const std::vector<double>& dR,
    const std::vector<double>& dZ,
    int max_candidates
) {
    const int grid = static_cast<int>(rs.size());
    std::vector<double> angle(static_cast<size_t>(grid) * grid);
    std::vector<double> residual(angle.size());
    for (size_t index = 0; index < angle.size(); ++index) {
        angle[index] = std::atan2(dZ[index], dR[index]);
        residual[index] = std::hypot(dR[index], dZ[index]);
    }
    std::vector<AxisCandidate> candidates;
    for (int j = 0; j + 1 < grid; ++j) {
        for (int i = 0; i + 1 < grid; ++i) {
            const size_t a = static_cast<size_t>(j) * grid + i;
            const size_t b = a + 1;
            const size_t d = static_cast<size_t>(j + 1) * grid + i;
            const size_t c = d + 1;
            const double winding = wrapped_angle_delta(angle[a], angle[b]) +
                                   wrapped_angle_delta(angle[b], angle[c]) +
                                   wrapped_angle_delta(angle[c], angle[d]) +
                                   wrapped_angle_delta(angle[d], angle[a]);
            if (std::abs(winding) > PI) {
                candidates.push_back({
                    0.5 * (rs[i] + rs[i + 1]),
                    0.5 * (zs[j] + zs[j + 1]),
                    std::min({residual[a], residual[b], residual[c], residual[d]}),
                    true,
                });
            }
        }
    }
    std::vector<AxisCandidate> local_minima;
    for (int j = 1; j + 1 < grid; ++j) {
        for (int i = 1; i + 1 < grid; ++i) {
            const size_t center = static_cast<size_t>(j) * grid + i;
            bool minimum = std::isfinite(residual[center]);
            for (int dj = -1; dj <= 1 && minimum; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    if (residual[center] > residual[static_cast<size_t>(j + dj) * grid + i + di]) {
                        minimum = false;
                        break;
                    }
                }
            }
            if (minimum) local_minima.push_back({rs[i], zs[j], residual[center], false});
        }
    }
    std::sort(local_minima.begin(), local_minima.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.residual < rhs.residual;
    });
    const size_t local_limit = std::min<size_t>(64, local_minima.size());
    candidates.insert(candidates.end(), local_minima.begin(), local_minima.begin() + local_limit);
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.winding != rhs.winding) return lhs.winding > rhs.winding;
        return lhs.residual < rhs.residual;
    });
    const double step = std::max(
        grid > 1 ? rs[1] - rs[0] : 0.0,
        grid > 1 ? zs[1] - zs[0] : 0.0
    );
    std::vector<AxisCandidate> unique;
    for (const auto& candidate : candidates) {
        bool separated = true;
        for (const auto& previous : unique) {
            if (std::hypot(candidate.R - previous.R, candidate.Z - previous.Z) < 0.75 * step) {
                separated = false;
                break;
            }
        }
        if (separated) unique.push_back(candidate);
        if (static_cast<int>(unique.size()) >= max_candidates) break;
    }
    return unique;
}

bool refine_axis_candidates(
    void* field,
    std::vector<AxisCandidate>& candidates,
    const AxisDomain& domain,
    int nfp,
    const SgpuScoreConfig& config,
    int iterations
) {
    if (candidates.empty()) return true;
    std::vector<double> R(candidates.size()), Z(candidates.size()), R_end, Z_end;
    for (size_t i = 0; i < candidates.size(); ++i) {
        R[i] = candidates[i].R;
        Z[i] = candidates[i].Z;
    }
    if (!trace_map(field, R, Z, nfp, config.axis_trace_steps, false, R_end, Z_end)) return false;
    std::vector<double> fR(R.size()), fZ(Z.size()), residual(R.size());
    for (size_t i = 0; i < R.size(); ++i) {
        fR[i] = R_end[i] - R[i];
        fZ[i] = Z_end[i] - Z[i];
        residual[i] = std::hypot(fR[i], fZ[i]);
    }
    const double span = std::max(domain.r_max - domain.r_min, domain.z_max - domain.z_min);
    const double h = std::max(config.axis_fd_absolute, config.axis_fd_relative * span);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<size_t> active;
        for (size_t i = 0; i < R.size(); ++i) {
            if (residual[i] > config.axis_tolerance) active.push_back(i);
        }
        if (active.empty()) break;
        const size_t count = active.size();
        std::vector<double> eval_R(4 * count), eval_Z(4 * count), eval_R_end, eval_Z_end;
        for (size_t j = 0; j < count; ++j) {
            const size_t i = active[j];
            eval_R[j] = std::min(domain.r_max, R[i] + h);
            eval_Z[j] = Z[i];
            eval_R[count + j] = std::max(domain.r_min, R[i] - h);
            eval_Z[count + j] = Z[i];
            eval_R[2 * count + j] = R[i];
            eval_Z[2 * count + j] = std::min(domain.z_max, Z[i] + h);
            eval_R[3 * count + j] = R[i];
            eval_Z[3 * count + j] = std::max(domain.z_min, Z[i] - h);
        }
        if (!trace_map(field, eval_R, eval_Z, nfp, config.axis_trace_steps, false, eval_R_end, eval_Z_end)) return false;
        std::vector<double> step_R(count, 0.0), step_Z(count, 0.0);
        for (size_t j = 0; j < count; ++j) {
            const size_t i = active[j];
            auto map_residual_R = [&](size_t k) { return eval_R_end[k] - eval_R[k]; };
            auto map_residual_Z = [&](size_t k) { return eval_Z_end[k] - eval_Z[k]; };
            const double denom_R = std::max(eval_R[j] - eval_R[count + j], 1.0e-300);
            const double denom_Z = std::max(eval_Z[2 * count + j] - eval_Z[3 * count + j], 1.0e-300);
            const double j11 = (map_residual_R(j) - map_residual_R(count + j)) / denom_R;
            const double j21 = (map_residual_Z(j) - map_residual_Z(count + j)) / denom_R;
            const double j12 = (map_residual_R(2 * count + j) - map_residual_R(3 * count + j)) / denom_Z;
            const double j22 = (map_residual_Z(2 * count + j) - map_residual_Z(3 * count + j)) / denom_Z;
            const double determinant = j11 * j22 - j12 * j21;
            if (std::abs(determinant) > 1.0e-14) {
                step_R[j] = (-fR[i] * j22 + j12 * fZ[i]) / determinant;
                step_Z[j] = (j21 * fR[i] - j11 * fZ[i]) / determinant;
                const double norm = std::hypot(step_R[j], step_Z[j]);
                const double scale = std::min(1.0, 0.25 * span / std::max(norm, 1.0e-300));
                step_R[j] *= scale;
                step_Z[j] *= scale;
            }
        }
        std::vector<unsigned char> accepted(count, 0);
        for (double alpha : {1.0, 0.5, 0.25, 0.125}) {
            std::vector<double> trial_R(count), trial_Z(count), trial_R_end, trial_Z_end;
            for (size_t j = 0; j < count; ++j) {
                const size_t i = active[j];
                trial_R[j] = std::min(domain.r_max, std::max(domain.r_min, R[i] + alpha * step_R[j]));
                trial_Z[j] = std::min(domain.z_max, std::max(domain.z_min, Z[i] + alpha * step_Z[j]));
            }
            if (!trace_map(field, trial_R, trial_Z, nfp, config.axis_trace_steps, false, trial_R_end, trial_Z_end)) return false;
            for (size_t j = 0; j < count; ++j) {
                if (accepted[j]) continue;
                const size_t i = active[j];
                const double trial_fR = trial_R_end[j] - trial_R[j];
                const double trial_fZ = trial_Z_end[j] - trial_Z[j];
                const double trial_residual = std::hypot(trial_fR, trial_fZ);
                if (std::isfinite(trial_residual) && trial_residual < residual[i]) {
                    R[i] = trial_R[j];
                    Z[i] = trial_Z[j];
                    fR[i] = trial_fR;
                    fZ[i] = trial_fZ;
                    residual[i] = trial_residual;
                    accepted[j] = 1;
                }
            }
        }
    }
    for (size_t i = 0; i < candidates.size(); ++i) {
        candidates[i].R = R[i];
        candidates[i].Z = Z[i];
        candidates[i].residual = residual[i];
    }
    return true;
}

void assign_axis_topology(
    AxisCandidate& candidate,
    double r_plus,
    double r_minus,
    double z_plus,
    double z_minus,
    double r_end_r_plus,
    double r_end_r_minus,
    double z_end_r_plus,
    double z_end_r_minus,
    double r_end_z_plus,
    double r_end_z_minus,
    double z_end_z_plus,
    double z_end_z_minus
) {
    const double denom_R = std::max(r_plus - r_minus, 1.0e-300);
    const double denom_Z = std::max(z_plus - z_minus, 1.0e-300);
    const double a = (r_end_r_plus - r_end_r_minus) / denom_R;
    const double c = (z_end_r_plus - z_end_r_minus) / denom_R;
    const double b = (r_end_z_plus - r_end_z_minus) / denom_Z;
    const double d = (z_end_z_plus - z_end_z_minus) / denom_Z;
    const double trace = a + d;
    const double determinant = a * d - b * c;
    candidate.topology_trace = trace;
    candidate.topology_det = determinant;
    candidate.elliptic = determinant > 0.0 &&
        std::abs(trace / std::sqrt(determinant)) < 2.0;
    if (determinant > 0.0) {
        const double scale = std::sqrt(determinant);
        const double q00 = c / scale;
        const double q01 = 0.5 * (d - a) / scale;
        const double q11 = -b / scale;
        const double middle = 0.5 * (q00 + q11);
        const double radius = std::hypot(0.5 * (q00 - q11), q01);
        double eig0 = middle - radius;
        double eig1 = middle + radius;
        if (eig0 < 0.0 && eig1 < 0.0) {
            const double old0 = eig0;
            eig0 = -eig1;
            eig1 = -old0;
        }
        if (eig0 > 0.0 && eig1 >= eig0) {
            candidate.ellipse_aspect = std::sqrt(eig1 / eig0);
        }
    }
}

void classify_axis_topology(
    void* field,
    std::vector<AxisCandidate>& candidates,
    const AxisDomain& domain,
    int nfp,
    const SgpuScoreConfig& config
) {
    if (candidates.empty()) return;
    const size_t count = std::min<size_t>(8, candidates.size());
    const double span = std::max(domain.r_max - domain.r_min, domain.z_max - domain.z_min);
    const double h = std::max(config.axis_fd_absolute, config.axis_fd_relative * span);
    std::vector<double> R(4 * count), Z(4 * count), R_end, Z_end;
    for (size_t j = 0; j < count; ++j) {
        R[j] = std::min(domain.r_max, candidates[j].R + h);
        Z[j] = candidates[j].Z;
        R[count + j] = std::max(domain.r_min, candidates[j].R - h);
        Z[count + j] = candidates[j].Z;
        R[2 * count + j] = candidates[j].R;
        Z[2 * count + j] = std::min(domain.z_max, candidates[j].Z + h);
        R[3 * count + j] = candidates[j].R;
        Z[3 * count + j] = std::max(domain.z_min, candidates[j].Z - h);
    }
    if (!trace_map(field, R, Z, nfp, config.axis_trace_steps, true, R_end, Z_end)) return;
    for (size_t j = 0; j < count; ++j) {
        assign_axis_topology(
            candidates[j],
            R[j], R[count + j], Z[2 * count + j], Z[3 * count + j],
            R_end[j], R_end[count + j], Z_end[j], Z_end[count + j],
            R_end[2 * count + j], R_end[3 * count + j],
            Z_end[2 * count + j], Z_end[3 * count + j]
        );
    }
}

bool search_axis_grid(
    void* field,
    const AxisDomain& domain,
    int nfp,
    const SgpuScoreConfig& config,
    int grid,
    int max_candidates,
    int newton_iters,
    std::vector<AxisCandidate>& candidates,
    bool fallback,
    double* timings
) {
    std::vector<double> rs(grid), zs(grid), R(static_cast<size_t>(grid) * grid), Z(R.size());
    for (int i = 0; i < grid; ++i) {
        rs[i] = domain.r_min + (domain.r_max - domain.r_min) * i / std::max(grid - 1, 1);
        zs[i] = domain.z_min + (domain.z_max - domain.z_min) * i / std::max(grid - 1, 1);
    }
    for (int j = 0; j < grid; ++j) {
        for (int i = 0; i < grid; ++i) {
            const size_t index = static_cast<size_t>(j) * grid + i;
            R[index] = rs[i];
            Z[index] = zs[j];
        }
    }
    std::vector<double> R_end, Z_end;
    auto substage_started = Clock::now();
    if (!trace_map(field, R, Z, nfp, config.axis_trace_steps, false, R_end, Z_end)) return false;
    timings[fallback ? SGPU_SCORE_TIME_AXIS_FALLBACK_GRID_TRACE
                     : SGPU_SCORE_TIME_AXIS_PRIMARY_GRID_TRACE] += seconds_since(substage_started);
    std::vector<double> dR(R.size()), dZ(Z.size());
    for (size_t i = 0; i < R.size(); ++i) {
        dR[i] = R_end[i] - R[i];
        dZ[i] = Z_end[i] - Z[i];
    }
    substage_started = Clock::now();
    candidates = find_grid_candidates(rs, zs, dR, dZ, max_candidates);
    timings[SGPU_SCORE_TIME_AXIS_CANDIDATE_EXTRACT] += seconds_since(substage_started);
    substage_started = Clock::now();
    if (!refine_axis_candidates(field, candidates, domain, nfp, config, newton_iters)) return false;
    timings[SGPU_SCORE_TIME_AXIS_CANDIDATE_REFINE] += seconds_since(substage_started);
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.residual < rhs.residual;
    });
    if (candidates.size() > 8) candidates.resize(8);
    if (!candidates.empty()) {
        std::vector<double> verify_R(candidates.size()), verify_Z(candidates.size()), end_R, end_Z;
        for (size_t i = 0; i < candidates.size(); ++i) {
            verify_R[i] = candidates[i].R;
            verify_Z[i] = candidates[i].Z;
        }
        substage_started = Clock::now();
        if (!trace_map(field, verify_R, verify_Z, nfp, config.axis_trace_steps, true, end_R, end_Z)) return false;
        timings[SGPU_SCORE_TIME_AXIS_FP64_VERIFY] += seconds_since(substage_started);
        for (size_t i = 0; i < candidates.size(); ++i) {
            candidates[i].residual = std::hypot(end_R[i] - verify_R[i], end_Z[i] - verify_Z[i]);
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.residual < rhs.residual;
        });
        substage_started = Clock::now();
        classify_axis_topology(field, candidates, domain, nfp, config);
        timings[SGPU_SCORE_TIME_AXIS_TOPOLOGY] += seconds_since(substage_started);
    }
    return true;
}

bool magnetic_axis_circulation_native(
    void* field,
    const AxisData& axis,
    int nfp,
    double& circulation
) {
    const int count = static_cast<int>(axis.R.size());
    if (!field || nfp <= 0 || count < 16 || axis.Z.size() != axis.R.size() ||
        axis.R_phi.size() != axis.R.size() || axis.Z_phi.size() != axis.R.size()) {
        sgpu_internal_set_error("invalid magnetic-axis circulation input");
        return false;
    }
    std::vector<double> xyz(static_cast<size_t>(count) * 3);
    std::vector<double> B(xyz.size());
    for (int index = 0; index < count; ++index) {
        const double phi = TWOPI * index / static_cast<double>(nfp * count);
        const double cosine = std::cos(phi);
        const double sine = std::sin(phi);
        xyz[3 * index] = axis.R[index] * cosine;
        xyz[3 * index + 1] = axis.R[index] * sine;
        xyz[3 * index + 2] = axis.Z[index];
    }
    if (sgpu_eval_B(field, xyz.data(), B.data(), count)) return false;
    double sum = 0.0;
    for (int index = 0; index < count; ++index) {
        const double phi = TWOPI * index / static_cast<double>(nfp * count);
        const double cosine = std::cos(phi);
        const double sine = std::sin(phi);
        const double tangent_x = axis.R_phi[index] * cosine - axis.R[index] * sine;
        const double tangent_y = axis.R_phi[index] * sine + axis.R[index] * cosine;
        sum += B[3 * index] * tangent_x + B[3 * index + 1] * tangent_y +
            B[3 * index + 2] * axis.Z_phi[index];
    }
    circulation = TWOPI * sum / count;
    if (!std::isfinite(circulation)) {
        sgpu_internal_set_error("magnetic-axis circulation is not finite");
        return false;
    }
    return true;
}

bool vacuum_G_native(
    void* field,
    const AxisData& axis,
    int nfp,
    double edge_toroidal_flux,
    double& G
) {
    if (!std::isfinite(edge_toroidal_flux) || edge_toroidal_flux == 0.0) {
        sgpu_internal_set_error("vacuum G requires nonzero signed toroidal flux");
        return false;
    }
    double circulation = 0.0;
    if (!magnetic_axis_circulation_native(field, axis, nfp, circulation)) return false;
    G = std::copysign(std::abs(circulation) / TWOPI, edge_toroidal_flux);
    return true;
}

bool find_axis_native(
    void* field,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    AxisData& axis,
    double& trace_time,
    double* timings
) {
    SGPU_NVTX_RANGE("axis.total");
    trace_time = 0.0;
    auto substage_started = Clock::now();
    const AxisDomain domain = [&] {
        SGPU_NVTX_RANGE("axis.domain.cpu");
        return build_axis_domain(
            coeffs_x, coeffs_y, coeffs_z, n_base_coils, n_coeff, config
        );
    }();
    timings[SGPU_SCORE_TIME_AXIS_DOMAIN] += seconds_since(substage_started);
    std::vector<AxisCandidate> candidates;
    if (config.axis_hint_enabled) {
        const bool hint_in_domain = config.axis_hint_R >= domain.r_min &&
            config.axis_hint_R <= domain.r_max &&
            config.axis_hint_Z >= domain.z_min && config.axis_hint_Z <= domain.z_max;
        if (!hint_in_domain) {
            axis.hint_distance = std::numeric_limits<double>::infinity();
            if (config.axis_hint_require_continuation) {
                axis.branch_lost = true;
                return true;
            }
        } else {
            candidates.push_back({config.axis_hint_R, config.axis_hint_Z});
        }
    }
    if (config.axis_hint_enabled && !candidates.empty()) {
        substage_started = Clock::now();
        {
            SGPU_NVTX_RANGE("axis.hint.refine");
            if (!refine_axis_candidates(
                    field, candidates, domain, nfp, config, config.axis_newton_iters)) {
                return false;
            }
        }
        timings[SGPU_SCORE_TIME_AXIS_CANDIDATE_REFINE] += seconds_since(substage_started);
        if (!candidates.empty()) {
            const double span = std::max(
                domain.r_max - domain.r_min, domain.z_max - domain.z_min
            );
            const double h = std::max(config.axis_fd_absolute, config.axis_fd_relative * span);
            if (config.axis_hint_require_continuation == 2) {
                std::vector<double> verify_R{
                    std::min(domain.r_max, candidates[0].R + h),
                    std::max(domain.r_min, candidates[0].R - h),
                    candidates[0].R,
                    candidates[0].R,
                };
                std::vector<double> verify_Z{
                    candidates[0].Z,
                    candidates[0].Z,
                    std::min(domain.z_max, candidates[0].Z + h),
                    std::max(domain.z_min, candidates[0].Z - h),
                };
                std::vector<double> end_R, end_Z;
                substage_started = Clock::now();
                {
                    SGPU_NVTX_RANGE("axis.hint.mixed_topology");
                    if (!trace_map(
                            field, verify_R, verify_Z, nfp, config.axis_trace_steps,
                            false, end_R, end_Z)) {
                        return false;
                    }
                }
                timings[SGPU_SCORE_TIME_AXIS_TOPOLOGY] += seconds_since(substage_started);
                assign_axis_topology(
                    candidates[0],
                    verify_R[0], verify_R[1], verify_Z[2], verify_Z[3],
                    end_R[0], end_R[1], end_Z[0], end_Z[1],
                    end_R[2], end_R[3], end_Z[2], end_Z[3]
                );
            } else {
                std::vector<double> verify_R{
                    candidates[0].R,
                    std::min(domain.r_max, candidates[0].R + h),
                    std::max(domain.r_min, candidates[0].R - h),
                    candidates[0].R,
                    candidates[0].R,
                };
                std::vector<double> verify_Z{
                    candidates[0].Z,
                    candidates[0].Z,
                    candidates[0].Z,
                    std::min(domain.z_max, candidates[0].Z + h),
                    std::max(domain.z_min, candidates[0].Z - h),
                };
                std::vector<double> end_R, end_Z;
                substage_started = Clock::now();
                {
                    SGPU_NVTX_RANGE("axis.hint.fp64_verify");
                    if (!trace_map(
                            field, verify_R, verify_Z, nfp, config.axis_trace_steps,
                            true, end_R, end_Z)) {
                        return false;
                    }
                }
                timings[SGPU_SCORE_TIME_AXIS_FP64_VERIFY] += seconds_since(substage_started);
                candidates[0].residual = std::hypot(
                    end_R[0] - verify_R[0], end_Z[0] - verify_Z[0]
                );
                substage_started = Clock::now();
                {
                    SGPU_NVTX_RANGE("axis.hint.topology.cpu");
                    assign_axis_topology(
                        candidates[0],
                        verify_R[1], verify_R[2], verify_Z[3], verify_Z[4],
                        end_R[1], end_R[2], end_Z[1], end_Z[2],
                        end_R[3], end_R[4], end_Z[3], end_Z[4]
                    );
                }
                timings[SGPU_SCORE_TIME_AXIS_TOPOLOGY] += seconds_since(substage_started);
            }
            axis.hint_distance = std::hypot(
                candidates[0].R - config.axis_hint_R,
                candidates[0].Z - config.axis_hint_Z
            );
            axis.used_hint = candidates[0].residual <= config.axis_tolerance &&
                candidates[0].elliptic &&
                axis.hint_distance <= config.axis_hint_max_distance;
        }
        if (!axis.used_hint) {
            candidates.clear();
            if (config.axis_hint_require_continuation) {
                axis.branch_lost = true;
                return true;
            }
        }
    }
    if (!axis.used_hint) {
        if (!search_axis_grid(
                field, domain, nfp, config, config.axis_grid,
                config.axis_max_candidates, config.axis_newton_iters, candidates,
                false, timings)) {
            return false;
        }
    }
    auto eligible = [](const AxisCandidate& candidate, double tolerance) {
        return candidate.residual <= tolerance && candidate.elliptic;
    };
    auto robust = [&](const AxisCandidate& candidate) {
        return eligible(candidate, config.axis_tolerance) &&
            axis_topology_stability_margin(candidate) >= config.axis_topology_margin;
    };
    bool has_robust = std::any_of(candidates.begin(), candidates.end(), [&](const auto& candidate) {
        return robust(candidate);
    });
    if (!axis.used_hint && !has_robust && nfp <= config.axis_fallback_max_nfp) {
        std::vector<AxisCandidate> fallback;
        if (!search_axis_grid(
                field, domain, nfp, config, config.axis_fallback_grid,
                config.axis_fallback_max_candidates, config.axis_fallback_newton_iters,
                fallback, true, timings)) {
            return false;
        }
        axis.candidate_count = static_cast<int>(candidates.size() + fallback.size());
        candidates.insert(candidates.end(), fallback.begin(), fallback.end());
    } else {
        axis.candidate_count = static_cast<int>(candidates.size());
    }
    std::vector<AxisCandidate> elliptic;
    for (const auto& candidate : candidates) {
        if (eligible(candidate, config.axis_tolerance)) elliptic.push_back(candidate);
    }
    if (elliptic.empty()) return true;
    axis.selected = *std::min_element(elliptic.begin(), elliptic.end(), [&](const auto& lhs, const auto& rhs) {
        const bool lhs_robust = robust(lhs);
        const bool rhs_robust = robust(rhs);
        if (lhs_robust != rhs_robust) return lhs_robust > rhs_robust;
        if (lhs.ellipse_aspect != rhs.ellipse_aspect) return lhs.ellipse_aspect < rhs.ellipse_aspect;
        return lhs.residual < rhs.residual;
    });
    axis.R.resize(config.axis_sample_count);
    axis.Z.resize(config.axis_sample_count);
    axis.R_phi.resize(config.axis_sample_count);
    axis.Z_phi.resize(config.axis_sample_count);
    const auto trace_started = Clock::now();
    int trace_code = 0;
    {
        SGPU_NVTX_RANGE("axis.samples");
        trace_code = sgpu_trace_axis_samples(
                field, axis.selected.R, axis.selected.Z, nfp,
                config.axis_trace_steps, config.axis_sample_count,
                axis.R.data(), axis.Z.data(), axis.R_phi.data(), axis.Z_phi.data());
    }
    trace_time = seconds_since(trace_started);
    if (trace_code) {
        return false;
    }
    return true;
}


}  // namespace coil_eval::pipeline
