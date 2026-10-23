// Radial level-set roots and legacy surface confidence/long-horizon diagnostics.
#include "pipeline/types.cuh"
#include "pipeline/axis.cuh"
#include "pipeline/common.cuh"
#include "pipeline/psi.cuh"
#include "pipeline/surface.cuh"
#include "nvtx_profile.h"

namespace coil_eval::pipeline {

void ray_polynomial_phi0(
    const PsiData& psi,
    const SgpuScoreConfig& config,
    double theta,
    std::array<double, 25>& polynomial
) {
    polynomial.fill(0.0);
    const double cosine = std::cos(theta);
    const double sine = std::sin(theta);
    std::array<double, 25> cpow{}, spow{};
    cpow[0] = 1.0;
    spow[0] = 1.0;
    for (int degree = 1; degree <= config.psi_poly_degree; ++degree) {
        cpow[degree] = cpow[degree - 1] * cosine;
        spow[degree] = spow[degree - 1] * sine;
    }
    polynomial[2] = cosine * cosine;
    for (size_t k = 0; k < psi.coeffs.size(); ++k) {
        if (psi.modes.kind[k] != 0) continue;
        polynomial[psi.modes.a[k] + psi.modes.b[k]] +=
            psi.coeffs[k] * cpow[psi.modes.a[k]] * spow[psi.modes.b[k]];
    }
}

double solve_ray_radius(
    const std::array<double, 25>& polynomial,
    double level,
    const SgpuScoreConfig& config
) {
    const double max_radius = config.surface_max_radius_scale * config.psi_a;
    const double q = polynomial[2];
    double radius = q > 1.0e-10
        ? config.psi_a * std::sqrt(std::max(level, 0.0) / q)
        : config.psi_a * std::sqrt(std::max(level, 1.0e-16));
    radius = std::min(max_radius, std::max(1.0e-12 * config.psi_a, radius));
    for (int iteration = 0; iteration < config.surface_newton_iters; ++iteration) {
        const double u = radius / config.psi_a;
        double value = 0.0;
        double derivative = 0.0;
        double power = u * u;
        double previous_power = u;
        for (int degree = 2; degree <= config.psi_poly_degree; ++degree) {
            if (degree > 2) {
                previous_power = power;
                power *= u;
            }
            value += polynomial[degree] * power;
            derivative += degree * polynomial[degree] * previous_power / config.psi_a;
        }
        const double residual = value - level;
        if (std::abs(residual) <= config.surface_newton_tolerance) break;
        const double denominator = std::abs(derivative) > 1.0e-14
            ? derivative : std::copysign(1.0e-14, derivative == 0.0 ? 1.0 : derivative);
        const double limit = 0.45 * std::max(std::abs(radius), 1.0e-8 * config.psi_a);
        const double step = std::min(limit, std::max(-limit, residual / denominator));
        radius = std::min(max_radius, std::max(1.0e-12 * config.psi_a, radius - step));
    }
    return radius;
}

bool screen_surfaces_native(
    void* field,
    const AxisData& axis,
    const PsiData& psi,
    int nfp,
    const SgpuScoreConfig& config,
    std::vector<SurfaceScreen>& screens,
    double* timings
) {
    const int levels = config.surface_level_count;
    const int theta_count = config.surface_theta_count;
    const size_t count = static_cast<size_t>(levels) * theta_count;
    std::vector<double> R(count), Z(count), radii(count), theta(count), R_end, Z_end;
    auto substage_started = Clock::now();
    for (int level_index = 0; level_index < levels; ++level_index) {
        for (int theta_index = 0; theta_index < theta_count; ++theta_index) {
            const size_t index = static_cast<size_t>(level_index) * theta_count + theta_index;
            theta[index] = TWOPI * theta_index / theta_count;
            std::array<double, 25> polynomial;
            ray_polynomial_phi0(psi, config, theta[index], polynomial);
            radii[index] = solve_ray_radius(polynomial, config.surface_levels[level_index], config);
            R[index] = axis.R[0] + radii[index] * std::cos(theta[index]);
            Z[index] = axis.Z[0] + radii[index] * std::sin(theta[index]);
        }
    }
    timings[SGPU_SCORE_TIME_SURFACE_RAY_ROOTS] += seconds_since(substage_started);
    substage_started = Clock::now();
    if (!trace_map(field, R, Z, nfp, config.surface_trace_steps, false, R_end, Z_end)) return false;
    timings[SGPU_SCORE_TIME_SURFACE_MIXED_TRACE] += seconds_since(substage_started);
    substage_started = Clock::now();
    screens.resize(levels);
    for (int level_index = 0; level_index < levels; ++level_index) {
        std::vector<double> distances;
        distances.reserve(theta_count);
        double radius_sum = 0.0;
        double radius_max = 0.0;
        for (int theta_index = 0; theta_index < theta_count; ++theta_index) {
            const size_t index = static_cast<size_t>(level_index) * theta_count + theta_index;
            double value, gR, gZ, gPhi;
            evaluate_psi_host(
                psi, axis, config, nfp, R_end[index], Z_end[index], TWOPI / nfp,
                value, gR, gZ, gPhi
            );
            const double gradient_norm = std::sqrt(gR * gR + gZ * gZ + std::pow(gPhi / R_end[index], 2.0));
            distances.push_back(std::abs(value - config.surface_levels[level_index]) /
                                std::max(gradient_norm, 1.0e-14));
            radius_sum += radii[index];
            radius_max = std::max(radius_max, radii[index]);
        }
        SurfaceScreen& screen = screens[level_index];
        screen.level = config.surface_levels[level_index];
        screen.drift_p95 = percentile(distances, 0.95);
        screen.radius_mean = radius_sum / theta_count;
        screen.radius_max = radius_max;
        screen.relative_drift_p95 = screen.drift_p95 / std::max(screen.radius_mean, 1.0e-14);
        screen.one_period_relative_drift_p95 = screen.relative_drift_p95;
        screen.stable = screen.drift_p95 <= config.surface_drift_absolute_tolerance &&
                        screen.relative_drift_p95 <= 0.30 &&
                        screen.radius_max < config.surface_max_radius_scale * config.psi_a * 0.999;
        screen.strict = screen.stable &&
                        screen.relative_drift_p95 <= config.surface_drift_relative_tolerance;
    }
    timings[SGPU_SCORE_TIME_SURFACE_MIXED_REDUCE] += seconds_since(substage_started);

    std::vector<int> verify_indices;
    // Keep the expensive long-horizon path bounded: only the six largest
    // one-period candidates participate in high-precision verification.
    for (int index = levels - 1; index >= 0 && verify_indices.size() < 6; --index) {
        if (screens[index].strict) verify_indices.push_back(index);
    }
    if (!verify_indices.empty()) {
        std::vector<double> verify_R, verify_Z, verify_R_end, verify_Z_end;
        verify_R.reserve(static_cast<size_t>(verify_indices.size()) * theta_count);
        verify_Z.reserve(verify_R.capacity());
        for (int level_index : verify_indices) {
            const size_t offset = static_cast<size_t>(level_index) * theta_count;
            verify_R.insert(verify_R.end(), R.begin() + offset, R.begin() + offset + theta_count);
            verify_Z.insert(verify_Z.end(), Z.begin() + offset, Z.begin() + offset + theta_count);
        }
        substage_started = Clock::now();
        if (!trace_map(field, verify_R, verify_Z, nfp, config.surface_trace_steps, true, verify_R_end, verify_Z_end)) return false;
        timings[SGPU_SCORE_TIME_SURFACE_FP64_TRACE] += seconds_since(substage_started);
        substage_started = Clock::now();
        for (size_t block = 0; block < verify_indices.size(); ++block) {
            const int level_index = verify_indices[block];
            std::vector<double> distances;
            distances.reserve(theta_count);
            for (int theta_index = 0; theta_index < theta_count; ++theta_index) {
                const size_t index = block * theta_count + theta_index;
                double value, gR, gZ, gPhi;
                evaluate_psi_host(
                    psi, axis, config, nfp, verify_R_end[index], verify_Z_end[index], TWOPI / nfp,
                    value, gR, gZ, gPhi
                );
                const double gradient_norm = std::sqrt(gR * gR + gZ * gZ + std::pow(gPhi / verify_R_end[index], 2.0));
                distances.push_back(std::abs(value - screens[level_index].level) /
                                    std::max(gradient_norm, 1.0e-14));
            }
            screens[level_index].drift_p95 = percentile(distances, 0.95);
            screens[level_index].relative_drift_p95 = screens[level_index].drift_p95 /
                std::max(screens[level_index].radius_mean, 1.0e-14);
            screens[level_index].one_period_relative_drift_p95 =
                screens[level_index].relative_drift_p95;
            screens[level_index].stable =
                screens[level_index].drift_p95 <= config.surface_drift_absolute_tolerance &&
                screens[level_index].relative_drift_p95 <= 0.30 &&
                screens[level_index].radius_max < config.surface_max_radius_scale * config.psi_a * 0.999;
            screens[level_index].strict = screens[level_index].stable &&
                screens[level_index].relative_drift_p95 <= config.surface_drift_relative_tolerance;
            screens[level_index].verified = true;
        }
        timings[SGPU_SCORE_TIME_SURFACE_FP64_REDUCE] += seconds_since(substage_started);
    }
    return true;
}

double smooth_tail_risk(const std::vector<double>& values, double temperature) {
    if (values.empty()) return std::numeric_limits<double>::infinity();
    const double maximum = *std::max_element(values.begin(), values.end());
    if (!std::isfinite(maximum)) return std::numeric_limits<double>::infinity();
    double sum = 0.0;
    for (double value : values) {
        sum += std::exp((value - maximum) / temperature);
    }
    return maximum + temperature * std::log(sum / values.size());
}

double logistic_confidence(double risk, const SgpuScoreConfig& config) {
    if (!std::isfinite(risk)) return 0.0;
    const double argument = (risk - config.surface_confidence_drift_center) /
        config.surface_confidence_drift_temperature;
    if (argument >= 40.0) return 0.0;
    if (argument <= -40.0) return 1.0;
    return 1.0 / (1.0 + std::exp(argument));
}

void project_nonincreasing_confidence(
    std::vector<double>& confidence,
    const SgpuScoreConfig& config
) {
    struct Block {
        int begin;
        int end;
        double weight;
        double mean;
    };
    std::vector<Block> blocks;
    blocks.reserve(confidence.size());
    double previous_level = 0.0;
    for (int index = 0; index < static_cast<int>(confidence.size()); ++index) {
        const double level = config.surface_levels[index];
        const double weight = std::max(level - previous_level, 1.0e-12);
        previous_level = level;
        blocks.push_back({index, index + 1, weight, confidence[index]});
        while (blocks.size() >= 2 &&
               blocks[blocks.size() - 2].mean < blocks.back().mean) {
            const Block right = blocks.back();
            blocks.pop_back();
            Block& left = blocks.back();
            const double total_weight = left.weight + right.weight;
            left.mean = (left.weight * left.mean + right.weight * right.mean) /
                total_weight;
            left.weight = total_weight;
            left.end = right.end;
        }
    }
    for (const Block& block : blocks) {
        for (int index = block.begin; index < block.end; ++index) {
            confidence[index] = block.mean;
        }
    }
}

bool screen_surface_confidence_native(
    void* field,
    const AxisData& axis,
    const PsiData& psi,
    int nfp,
    const SgpuScoreConfig& config,
    std::vector<SurfaceScreen>& screens,
    SgpuScoreResult& result
) {
    SGPU_NVTX_RANGE("surface.total");
    const int levels = config.surface_level_count;
    const int theta_count = config.surface_theta_count;
    const size_t count = static_cast<size_t>(levels) * theta_count;
    std::vector<double> R(count), Z(count), radii(count), radius_mean(levels, 0.0);
    std::vector<double> radius_max(levels, 0.0), R_end, Z_end;
    auto substage_started = Clock::now();
    {
        SGPU_NVTX_RANGE("surface.ray_roots.cpu");
        for (int level_index = 0; level_index < levels; ++level_index) {
            for (int theta_index = 0; theta_index < theta_count; ++theta_index) {
                const size_t index = static_cast<size_t>(level_index) * theta_count + theta_index;
                const double theta = TWOPI * theta_index / theta_count;
                std::array<double, 25> polynomial;
                ray_polynomial_phi0(psi, config, theta, polynomial);
                radii[index] = solve_ray_radius(
                    polynomial, config.surface_levels[level_index], config
                );
                radius_mean[level_index] += radii[index] / theta_count;
                radius_max[level_index] = std::max(radius_max[level_index], radii[index]);
                R[index] = axis.R[0] + radii[index] * std::cos(theta);
                Z[index] = axis.Z[0] + radii[index] * std::sin(theta);
            }
        }
    }
    result.timings[SGPU_SCORE_TIME_SURFACE_RAY_ROOTS] += seconds_since(substage_started);

    std::vector<double> maximum_relative_drift(count, 0.0);
    for (int period = 0; period < config.surface_confidence_periods; ++period) {
        substage_started = Clock::now();
        {
            SGPU_NVTX_RANGE("surface.trace.mixed");
            if (!trace_map(field, R, Z, nfp, config.surface_trace_steps, false, R_end, Z_end)) {
                return false;
            }
        }
        result.timings[SGPU_SCORE_TIME_SURFACE_MIXED_TRACE] += seconds_since(substage_started);
        substage_started = Clock::now();
        {
            SGPU_NVTX_RANGE("surface.reduce.cpu");
            for (int level_index = 0; level_index < levels; ++level_index) {
                for (int theta_index = 0; theta_index < theta_count; ++theta_index) {
                    const size_t index = static_cast<size_t>(level_index) * theta_count + theta_index;
                    double value, gR, gZ, gPhi;
                    evaluate_psi_host(
                        psi, axis, config, nfp, R_end[index], Z_end[index], TWOPI / nfp,
                        value, gR, gZ, gPhi
                    );
                    const double gradient_norm = std::sqrt(
                        gR * gR + gZ * gZ + std::pow(gPhi / R_end[index], 2.0)
                    );
                    const double distance = std::abs(value - config.surface_levels[level_index]) /
                        std::max(gradient_norm, 1.0e-14);
                    const double relative = distance / std::max(radius_mean[level_index], 1.0e-14);
                    maximum_relative_drift[index] = std::max(
                        maximum_relative_drift[index],
                        std::isfinite(relative) ? relative : std::numeric_limits<double>::infinity()
                    );
                }
            }
        }
        R.swap(R_end);
        Z.swap(Z_end);
        result.timings[SGPU_SCORE_TIME_SURFACE_MIXED_REDUCE] += seconds_since(substage_started);
    }

    substage_started = Clock::now();
    SGPU_NVTX_RANGE("surface.confidence.cpu");
    std::vector<double> risk(levels), confidence(levels);
    for (int level_index = 0; level_index < levels; ++level_index) {
        const size_t offset = static_cast<size_t>(level_index) * theta_count;
        std::vector<double> level_values(
            maximum_relative_drift.begin() + offset,
            maximum_relative_drift.begin() + offset + theta_count
        );
        risk[level_index] = smooth_tail_risk(
            level_values, config.surface_confidence_smoothmax_temperature
        );
        const double radius_limit = config.surface_max_radius_scale * config.psi_a;
        const double radius_margin = (0.995 * radius_limit - radius_max[level_index]) /
            std::max(0.003 * radius_limit, 1.0e-14);
        const double radius_confidence = radius_margin >= 40.0
            ? 1.0
            : radius_margin <= -40.0
                ? 0.0
                : 1.0 / (1.0 + std::exp(-radius_margin));
        confidence[level_index] = logistic_confidence(risk[level_index], config) *
            radius_confidence;
    }
    project_nonincreasing_confidence(confidence, config);

    double effective_level = 0.0;
    double previous_level = 0.0;
    double previous_confidence = 1.0;
    for (int level_index = 0; level_index < levels; ++level_index) {
        const double level = config.surface_levels[level_index];
        effective_level += 0.5 * (previous_confidence + confidence[level_index]) *
            (level - previous_level);
        previous_level = level;
        previous_confidence = confidence[level_index];
    }
    effective_level = std::min(
        config.surface_levels[levels - 1], std::max(0.0, effective_level)
    );
    const auto interpolate = [&](const std::vector<double>& values, double initial) {
        if (effective_level <= config.surface_levels[0]) {
            const double weight = effective_level / config.surface_levels[0];
            return initial + weight * (values[0] - initial);
        }
        for (int index = 1; index < levels; ++index) {
            if (effective_level <= config.surface_levels[index]) {
                const double weight = (effective_level - config.surface_levels[index - 1]) /
                    (config.surface_levels[index] - config.surface_levels[index - 1]);
                return values[index - 1] + weight * (values[index] - values[index - 1]);
            }
        }
        return values.back();
    };
    const double effective_risk = interpolate(risk, 0.0);
    const double effective_confidence = logistic_confidence(effective_risk, config);
    result.surface_confidence_mean = effective_level / config.surface_levels[levels - 1];
    result.surface_confidence_edge = effective_confidence;
    result.surface_effective_level = effective_level;
    result.surface_confidence_risk = effective_risk;

    SurfaceScreen screen;
    screen.level = std::max(effective_level, config.surface_levels[0]);
    double mean_radius = 0.0;
    double maximum_radius = 0.0;
    for (int theta_index = 0; theta_index < theta_count; ++theta_index) {
        const double theta = TWOPI * theta_index / theta_count;
        std::array<double, 25> polynomial;
        ray_polynomial_phi0(psi, config, theta, polynomial);
        const double radius = solve_ray_radius(polynomial, screen.level, config);
        mean_radius += radius / theta_count;
        maximum_radius = std::max(maximum_radius, radius);
    }
    screen.radius_mean = mean_radius;
    screen.radius_max = maximum_radius;
    screen.relative_drift_p95 = effective_risk;
    screen.one_period_relative_drift_p95 = effective_risk;
    screen.drift_p95 = effective_risk * mean_radius;
    screen.long_trace_periods_completed = config.surface_confidence_periods;
    screen.long_verified = false;
    screen.stable = effective_level >= config.surface_levels[0];
    screen.strict = screen.stable;
    screen.verified = true;
    screens.assign(1, screen);
    result.surface_confidence_edge = effective_confidence;
    result.timings[SGPU_SCORE_TIME_SURFACE_CONFIDENCE] += seconds_since(substage_started);
    return true;
}

bool verify_surface_long_horizon(
    void* field,
    const AxisData& axis,
    const PsiData& psi,
    int nfp,
    const SgpuScoreConfig& config,
    SurfaceScreen& screen,
    double* timings
) {
    screen.long_verified = true;
    screen.long_trace_periods_completed = 1;
    if (!screen.strict || config.surface_long_trace_periods <= 1) return true;

    const int theta_count = config.surface_theta_count;
    std::vector<double> R(theta_count), Z(theta_count), R_end, Z_end;
    for (int theta_index = 0; theta_index < theta_count; ++theta_index) {
        const double theta = TWOPI * theta_index / theta_count;
        std::array<double, 25> polynomial;
        ray_polynomial_phi0(psi, config, theta, polynomial);
        const double radius = solve_ray_radius(polynomial, screen.level, config);
        R[theta_index] = axis.R[0] + radius * std::cos(theta);
        Z[theta_index] = axis.Z[0] + radius * std::sin(theta);
    }

    double maximum_relative_drift = screen.relative_drift_p95;
    for (int period = 1; period <= config.surface_long_trace_periods; ++period) {
        auto substage_started = Clock::now();
        if (!trace_map(field, R, Z, nfp, config.surface_trace_steps, false, R_end, Z_end)) {
            return false;
        }
        timings[SGPU_SCORE_TIME_SURFACE_LONG_TRACE] += seconds_since(substage_started);
        substage_started = Clock::now();
        std::vector<double> distances;
        distances.reserve(theta_count);
        for (int theta_index = 0; theta_index < theta_count; ++theta_index) {
            double value, gR, gZ, gPhi;
            evaluate_psi_host(
                psi, axis, config, nfp, R_end[theta_index], Z_end[theta_index], TWOPI / nfp,
                value, gR, gZ, gPhi
            );
            const double gradient_norm = std::sqrt(
                gR * gR + gZ * gZ + std::pow(gPhi / R_end[theta_index], 2.0)
            );
            const double distance = std::abs(value - screen.level) /
                std::max(gradient_norm, 1.0e-14);
            distances.push_back(std::isfinite(distance)
                ? distance : std::numeric_limits<double>::infinity());
        }
        const double relative_drift = percentile(distances, 0.95) /
            std::max(screen.radius_mean, 1.0e-14);
        maximum_relative_drift = std::max(maximum_relative_drift, relative_drift);
        screen.long_trace_periods_completed = period;
        R.swap(R_end);
        Z.swap(Z_end);
        timings[SGPU_SCORE_TIME_SURFACE_LONG_REDUCE] += seconds_since(substage_started);
        if (maximum_relative_drift > config.surface_long_trace_relative_tolerance) break;
    }
    screen.relative_drift_p95 = maximum_relative_drift;
    screen.drift_p95 = maximum_relative_drift * screen.radius_mean;
    screen.strict = screen.strict &&
        maximum_relative_drift <= config.surface_long_trace_relative_tolerance &&
        screen.long_trace_periods_completed == config.surface_long_trace_periods;
    return true;
}


}  // namespace coil_eval::pipeline
