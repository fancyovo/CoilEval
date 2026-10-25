#pragma once

// CPU orchestration of the original finite-state Newton forward calculation.
// The supplied trace batches independent rows; no arithmetic is shared across
// candidate/query updates, and all four original backtracking trials remain.
#include <algorithm>
#include <cmath>
#include <vector>

namespace sgpu_axis_refinement {

template<class Candidate, class Domain, class Config, class Trace>
bool refine(std::vector<std::vector<Candidate>>& candidates,
            const std::vector<Domain>& domains, const std::vector<Config>& configs,
            Trace&& trace) {
    const size_t queries = candidates.size();
    std::vector<std::vector<double>> R(queries), Z(queries), R_end, Z_end;
    std::vector<std::vector<double>> fR(queries), fZ(queries), residual(queries);
    std::vector<double> spans(queries), hs(queries);
    int iterations = 0;
    bool any = false;
    for (size_t q = 0; q < queries; ++q) {
        const auto count = candidates[q].size();
        R[q].resize(count); Z[q].resize(count);
        for (size_t i = 0; i < count; ++i) { R[q][i] = candidates[q][i].R; Z[q][i] = candidates[q][i].Z; }
        any = any || count != 0;
        const auto& domain = domains[q];
        spans[q] = std::max(domain.r_max - domain.r_min, domain.z_max - domain.z_min);
        hs[q] = std::max(configs[q].axis_fd_absolute, configs[q].axis_fd_relative * spans[q]);
        iterations = std::max(iterations, configs[q].axis_newton_iters);
    }
    if (!any) return true;
    if (!trace(R, Z, R_end, Z_end)) return false;
    for (size_t q = 0; q < queries; ++q) {
        fR[q].resize(R[q].size()); fZ[q].resize(R[q].size()); residual[q].resize(R[q].size());
        for (size_t i = 0; i < R[q].size(); ++i) {
            fR[q][i] = R_end[q][i] - R[q][i];
            fZ[q][i] = Z_end[q][i] - Z[q][i];
            residual[q][i] = std::hypot(fR[q][i], fZ[q][i]);
        }
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<std::vector<size_t>> active(queries);
        std::vector<std::vector<double>> eval_R(queries), eval_Z(queries), eval_R_end, eval_Z_end;
        any = false;
        for (size_t q = 0; q < queries; ++q) {
            if (iteration >= configs[q].axis_newton_iters) continue;
            for (size_t i = 0; i < R[q].size(); ++i) {
                if (residual[q][i] > configs[q].axis_tolerance) active[q].push_back(i);
            }
            const size_t count = active[q].size();
            any = any || count != 0;
            eval_R[q].resize(4 * count); eval_Z[q].resize(4 * count);
            const auto& domain = domains[q];
            const double h = hs[q];
            for (size_t j = 0; j < count; ++j) {
                const size_t i = active[q][j];
                eval_R[q][j] = std::min(domain.r_max, R[q][i] + h);
                eval_Z[q][j] = Z[q][i];
                eval_R[q][count + j] = std::max(domain.r_min, R[q][i] - h);
                eval_Z[q][count + j] = Z[q][i];
                eval_R[q][2 * count + j] = R[q][i];
                eval_Z[q][2 * count + j] = std::min(domain.z_max, Z[q][i] + h);
                eval_R[q][3 * count + j] = R[q][i];
                eval_Z[q][3 * count + j] = std::max(domain.z_min, Z[q][i] - h);
            }
        }
        if (!any) break;
        if (!trace(eval_R, eval_Z, eval_R_end, eval_Z_end)) return false;
        std::vector<std::vector<double>> step_R(queries), step_Z(queries);
        std::vector<std::vector<unsigned char>> accepted(queries);
        for (size_t q = 0; q < queries; ++q) {
            const size_t count = active[q].size();
            step_R[q].assign(count, 0.0); step_Z[q].assign(count, 0.0); accepted[q].assign(count, 0);
            const double span = spans[q];
            for (size_t j = 0; j < count; ++j) {
                const size_t i = active[q][j];
                auto map_residual_R = [&](size_t k) { return eval_R_end[q][k] - eval_R[q][k]; };
                auto map_residual_Z = [&](size_t k) { return eval_Z_end[q][k] - eval_Z[q][k]; };
                const double denom_R = std::max(eval_R[q][j] - eval_R[q][count + j], 1.0e-300);
                const double denom_Z = std::max(eval_Z[q][2 * count + j] - eval_Z[q][3 * count + j], 1.0e-300);
                const double j11 = (map_residual_R(j) - map_residual_R(count + j)) / denom_R;
                const double j21 = (map_residual_Z(j) - map_residual_Z(count + j)) / denom_R;
                const double j12 = (map_residual_R(2 * count + j) - map_residual_R(3 * count + j)) / denom_Z;
                const double j22 = (map_residual_Z(2 * count + j) - map_residual_Z(3 * count + j)) / denom_Z;
                const double determinant = j11 * j22 - j12 * j21;
                if (std::abs(determinant) > 1.0e-14) {
                    step_R[q][j] = (-fR[q][i] * j22 + j12 * fZ[q][i]) / determinant;
                    step_Z[q][j] = (j21 * fR[q][i] - j11 * fZ[q][i]) / determinant;
                    const double norm = std::hypot(step_R[q][j], step_Z[q][j]);
                    const double scale = std::min(1.0, 0.25 * span / std::max(norm, 1.0e-300));
                    step_R[q][j] *= scale;
                    step_Z[q][j] *= scale;
                }
            }
        }
        for (double alpha : {1.0, 0.5, 0.25, 0.125}) {
            std::vector<std::vector<double>> trial_R(queries), trial_Z(queries), trial_R_end, trial_Z_end;
            for (size_t q = 0; q < queries; ++q) {
                const size_t count = active[q].size();
                trial_R[q].resize(count); trial_Z[q].resize(count);
                const auto& domain = domains[q];
                for (size_t j = 0; j < count; ++j) {
                    const size_t i = active[q][j];
                    trial_R[q][j] = std::min(domain.r_max, std::max(domain.r_min, R[q][i] + alpha * step_R[q][j]));
                    trial_Z[q][j] = std::min(domain.z_max, std::max(domain.z_min, Z[q][i] + alpha * step_Z[q][j]));
                }
            }
            if (!trace(trial_R, trial_Z, trial_R_end, trial_Z_end)) return false;
            for (size_t q = 0; q < queries; ++q) {
                for (size_t j = 0; j < active[q].size(); ++j) {
                    if (accepted[q][j]) continue;
                    const size_t i = active[q][j];
                    const double trial_fR = trial_R_end[q][j] - trial_R[q][j];
                    const double trial_fZ = trial_Z_end[q][j] - trial_Z[q][j];
                    const double trial_residual = std::hypot(trial_fR, trial_fZ);
                    if (std::isfinite(trial_residual) && trial_residual < residual[q][i]) {
                        R[q][i] = trial_R[q][j];
                        Z[q][i] = trial_Z[q][j];
                        fR[q][i] = trial_fR;
                        fZ[q][i] = trial_fZ;
                        residual[q][i] = trial_residual;
                        accepted[q][j] = 1;
                    }
                }
            }
        }
    }
    for (size_t q = 0; q < queries; ++q) {
        for (size_t i = 0; i < candidates[q].size(); ++i) {
            candidates[q][i].R = R[q][i]; candidates[q][i].Z = Z[q][i];
            candidates[q][i].residual = residual[q][i];
        }
    }
    return true;
}

}  // namespace sgpu_axis_refinement
