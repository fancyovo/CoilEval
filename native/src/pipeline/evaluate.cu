// Formal center downstream pipeline: flux, volume points, alpha/iota, and QS.
#include "pipeline/types.cuh"
#include "pipeline/reverse_capture.cuh"
#include "pipeline/session_state.cuh"
#include "pipeline/alpha.cuh"
#include "pipeline/axis.cuh"
#include "pipeline/common.cuh"
#include "pipeline/evaluate.cuh"
#include "pipeline/flux.cuh"
#include "pipeline/psi.cuh"
#include "pipeline/qs.cuh"
#include "pipeline/surface.cuh"
#include "pipeline/target_volume.cuh"
#include "pipeline/volume.cuh"
#include "pipeline/gradients.cuh"
#include "coil_field_internal.h"

namespace coil_eval::pipeline {

bool run_downstream_gpu(
    void* field,
    const double* currents_a,
    int n_base_coils,
    int nfp,
    const AxisData& axis,
    const PsiData& psi,
    std::vector<SurfaceScreen>& screens,
    const SgpuScoreConfig& config,
    SgpuScoreResult& result,
    bool verify_long_horizon
) {
    DevicePsiData device_psi;
    if (!upload_psi_data(psi, axis, device_psi)) {
        fail_result(&result, "psi/axis upload for downstream pipeline failed");
        return false;
    }
    std::vector<SurfaceScreen*> candidates;
    for (auto& screen : screens) {
        if (screen.strict && screen.verified) candidates.push_back(&screen);
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto* lhs, const auto* rhs) {
        return lhs->level > rhs->level;
    });
    auto started = Clock::now();
    FluxCalibrationNative flux;
    SurfaceScreen* selected_surface = nullptr;
    SurfaceScreen* closest_rejected_surface = nullptr;
    SurfaceScreen continuous_surface;
    bool flux_ok = false;
    auto calibrate_level = [&](double level, FluxCalibrationNative& trial) {
        ++result.flux_attempt_count;
        const auto calibration_started = Clock::now();
        const bool trial_ok = calibrate_flux_native(
            field, device_psi, axis, psi, nfp, level, config, trial, result
        );
        result.timings[SGPU_SCORE_TIME_FLUX_CALIBRATION] += seconds_since(calibration_started);
        return trial_ok;
    };
    if (config.target_volume>0 && !candidates.empty()) {
        continuous_surface=*candidates.front();selected_surface=&continuous_surface;
        flux_ok=calibrate_level(continuous_surface.level,flux);
    } else if (config.surface_selection_mode == 1 && !candidates.empty()) {
        continuous_surface = *candidates.front();
        double high = continuous_surface.level;
        FluxCalibrationNative high_trial;
        const bool high_ok = calibrate_level(high, high_trial);
        if (result.status == SGPU_SCORE_INTERNAL_ERROR && result.error_message[0]) return false;
        if (high_ok) {
            flux = std::move(high_trial);
            flux_ok = true;
        } else {
            double low = std::numeric_limits<double>::quiet_NaN();
            FluxCalibrationNative low_trial;
            bool low_ok = false;
            for (int level_index = config.surface_level_count - 1;
                 level_index >= 0; --level_index) {
                const double probe = config.surface_levels[level_index];
                if (!(probe < high)) continue;
                FluxCalibrationNative probe_trial;
                const bool probe_ok = calibrate_level(probe, probe_trial);
                if (result.status == SGPU_SCORE_INTERNAL_ERROR && result.error_message[0]) {
                    return false;
                }
                if (probe_ok) {
                    low = probe;
                    low_trial = std::move(probe_trial);
                    low_ok = true;
                    break;
                }
                low_trial = std::move(probe_trial);
            }
            if (low_ok) {
                flux = std::move(low_trial);
                for (int iteration = 0; iteration < config.surface_flux_bisection_iters; ++iteration) {
                    const double middle = 0.5 * (low + high);
                    FluxCalibrationNative middle_trial;
                    const bool middle_ok = calibrate_level(middle, middle_trial);
                    if (result.status == SGPU_SCORE_INTERNAL_ERROR && result.error_message[0]) return false;
                    if (middle_ok) {
                        low = middle;
                        flux = std::move(middle_trial);
                    } else {
                        high = middle;
                    }
                }
                continuous_surface.level = low;
                flux_ok = true;
            } else {
                flux = std::move(low_trial);
                continuous_surface.level = config.surface_levels[0];
            }
        }
        selected_surface = &continuous_surface;
        result.surface_effective_level = continuous_surface.level;
    } else {
        for (SurfaceScreen* candidate : candidates) {
            const auto trace_started = Clock::now();
            if (verify_long_horizon) {
                if (!verify_surface_long_horizon(
                        field, axis, psi, nfp, config, *candidate, result.timings)) {
                    fail_from_backend(&result, "long-horizon surface trace");
                    return false;
                }
                result.timings[SGPU_SCORE_TIME_SURFACE_SCREEN] += seconds_since(trace_started);
            }
            if (!candidate->strict) {
                ++result.surface_long_trace_rejected_count;
                if (!closest_rejected_surface ||
                    candidate->relative_drift_p95 < closest_rejected_surface->relative_drift_p95) {
                    closest_rejected_surface = candidate;
                }
                continue;
            }
            FluxCalibrationNative trial;
            const bool trial_ok = calibrate_level(candidate->level, trial);
            if (result.status == SGPU_SCORE_INTERNAL_ERROR && result.error_message[0]) return false;
            if (!selected_surface || trial_ok) {
                selected_surface = candidate;
                flux = std::move(trial);
            }
            if (trial_ok) {
                flux_ok = true;
                break;
            }
        }
    }
    result.timings[SGPU_SCORE_TIME_FLUX] = seconds_since(started);
    if (!selected_surface) {
        if (closest_rejected_surface) {
            result.surface_level = closest_rejected_surface->level;
            result.surface_drift_relative_p95 = closest_rejected_surface->relative_drift_p95;
            result.surface_one_period_drift_relative_p95 =
                closest_rejected_surface->one_period_relative_drift_p95;
            result.surface_long_trace_periods_completed =
                closest_rejected_surface->long_trace_periods_completed;
        }
        result.status = SGPU_SCORE_DRIFT_REJECTED;
        return false;
    }
    result.surface_level = selected_surface->level;
    result.surface_drift_relative_p95 = selected_surface->relative_drift_p95;
    result.surface_one_period_drift_relative_p95 =
        selected_surface->one_period_relative_drift_p95;
    result.surface_long_trace_periods_completed =
        selected_surface->long_trace_periods_completed;
    result.flux_edge = flux.edge_flux;
    result.flux_fit_relative_rms = flux.fit_relative_rms;
    result.flux_section_relative_std_edge = flux.section_relative_std_edge;
    result.flux_boundary_residual_max = flux.boundary_residual_max;
    result.flux_derivative_min = flux.derivative_min;
    result.flux_derivative_max = flux.derivative_max;
    result.surface_volume = flux.volume;
    result.surface_effective_minor_radius = flux.effective_minor_radius;
    result.surface_inverse_aspect_ratio = flux.effective_minor_radius /
        std::max(flux.major_radius, 1.0e-30);
    target_quality(config,result);
    if(config.target_volume>0 && (!std::isfinite(result.target_volume_relative_error) ||
       std::abs(result.target_volume_relative_error)>config.target_volume_forward_tolerance)) {
        result.status=SGPU_SCORE_FLUX_REJECTED;
        std::snprintf(result.error_message,sizeof(result.error_message),"target-volume forward precision tolerance exceeded");
        return false;
    }
    result.stage_completed = SCORE_STAGE_FLUX;
    if (!flux_ok) {
        result.status = SGPU_SCORE_FLUX_REJECTED;
        return false;
    }

    DeviceVolumePoints points;
    started = Clock::now();
    if (!build_volume_points_native(device_psi, axis, psi, flux, nfp, config, points, result)) {
        return false;
    }
    result.timings[SGPU_SCORE_TIME_VOLUME_POINTS] = seconds_since(started);
    DeviceBuffer<float> d_B(static_cast<size_t>(points.count) * 3);
    DeviceBuffer<float> d_grad_B(static_cast<size_t>(points.count) * 9);
    if (!d_B.data() || !d_grad_B.data()) {
        fail_result(&result, "volume field allocation failed");
        return false;
    }
    started = Clock::now();
    if (sgpu_internal_eval_B_grad_f32_device(
            field, points.xyz.data(), d_B.data(), d_grad_B.data(), points.count) ||
        !cuda_stage_ok(cudaDeviceSynchronize(), result, "volume B/grad(B) evaluation")) {
        if (result.status != SGPU_SCORE_INTERNAL_ERROR || !result.error_message[0]) {
            fail_from_backend(&result, "volume B/grad(B) evaluation");
        }
        return false;
    }
    result.timings[SGPU_SCORE_TIME_FIELD_VOLUME] = seconds_since(started);
    AlphaFitNative alpha;
    if (!fit_alpha_native(points, d_B.data(), nfp, config, alpha, result)) {
        if (result.status != SGPU_SCORE_INTERNAL_ERROR) result.status = SGPU_SCORE_ALPHA_FAILED;
        return false;
    }
    result.alpha_relative_l2 = alpha.relative_l2;
    result.alpha_normal_B_relative_l2 = alpha.normal_B_relative_l2;
    g_boundary_iota = 0.0;
    for (float coefficient : alpha.iota_coefficients) g_boundary_iota += coefficient;
    result.iota_min = std::numeric_limits<double>::infinity();
    result.iota_max = -std::numeric_limits<double>::infinity();
    const double u_min = config.volume_rho_min * config.volume_rho_min;
    for (int sample = 0; sample <= 256; ++sample) {
        const double u = u_min + (1.0 - u_min) * sample / 256.0;
        double iota = 0.0;
        double power = 1.0;
        for (float coefficient : alpha.iota_coefficients) {
            iota += coefficient * power;
            power *= u;
        }
        result.iota_min = std::min(result.iota_min, iota);
        result.iota_max = std::max(result.iota_max, iota);
    }
    result.stage_completed = SCORE_STAGE_ALPHA;
    started = Clock::now();
    std::array<double, 9> qs_sums{};
    double G = 0.0;
    if (!vacuum_G_native(field, axis, nfp, flux.edge_flux, G)) {
        fail_from_backend(&result, "magnetic-axis Ampere circulation");
        return false;
    }
    if (!compute_qs_metric_native(
            points, d_B.data(), d_grad_B.data(), alpha, nfp, G, config, result,
            g_active_g2_cache ? &qs_sums : nullptr)) {
        return false;
    }
    result.timings[SGPU_SCORE_TIME_QS_METRICS] = seconds_since(started);
    const double section_score = q_down(
        result.flux_section_relative_std_edge, config.score_flux_section_std_scale, 1.0
    );
    const double boundary_score = q_down(
        result.flux_boundary_residual_max, config.score_flux_boundary_residual_scale, 1.0
    );
    const double flux_score = blend({{0.75, section_score}, {0.25, boundary_score}});
    const double normal_score = q_down(
        result.alpha_normal_B_relative_l2, config.score_alpha_normal_B_scale, 1.0
    );
    const double alpha_score = q_down(
        result.alpha_relative_l2, config.score_alpha_relative_l2_scale, 1.0
    );
    result.components[SGPU_SCORE_COMPONENT_COORDINATE] = blend({
        {0.35, flux_score}, {0.35, normal_score}, {0.20, alpha_score}, {0.10, 1.0},
    });
    const bool qh_target = config.target_M != 0 && config.target_N != 0;
    const double iota_score = !qh_target ? 1.0 : std::pow(
        clip01(minimum_absolute_iota(result.iota_min, result.iota_max) /
               config.score_qh_iota_threshold),
        config.score_qh_iota_power
    );
    result.score_iota = iota_score;
    result.components[SGPU_SCORE_COMPONENT_IOTA] = iota_score;
    // f_C is linear in the helicity pair, so score its magnitude per unit (M, N).
    const double global_score = q_down(
        result.qs_target_global_error_per_helicity, config.score_qs_global_scale, 0.9
    );
    const double edge_score = q_down(
        result.qs_target_edge_error_per_helicity, config.score_qs_edge_scale, 0.9, global_score
    );
    const double residual_score = blend({{0.80, global_score}, {0.20, edge_score}});
    const double competitor_error = std::min(
        result.qs_qa_global_error_per_helicity, result.qs_qp_global_error_per_helicity
    );
    const double helicity_advantage = !qh_target ? 1.0 :
        competitor_error / std::max(
            result.qs_target_global_error_per_helicity + competitor_error, 1.0e-300
        );
    const double size_score = q_saturating_up(
        result.surface_inverse_aspect_ratio, config.score_surface_inverse_aspect_saturation
    );
    const double size_factor = config.score_volume_qs_size_floor +
        (1.0 - config.score_volume_qs_size_floor) * size_score;
    const double iota_factor = !qh_target ? 1.0 :
        config.score_volume_qs_iota_floor +
        (1.0 - config.score_volume_qs_iota_floor) * iota_score;
    result.score_surface_size = size_score;
    result.score_qs_residual = residual_score;
    result.score_qh_helicity_advantage = clip01(helicity_advantage);
    result.score_volume_qs_size_factor = size_factor;
    result.score_volume_qs_iota_factor = iota_factor;
    result.components[SGPU_SCORE_COMPONENT_VOLUME_QS] =
        residual_score * size_factor * iota_factor;
    result.status = SGPU_SCORE_OK;
    result.stage_completed = SCORE_STAGE_QS;
    if (g_active_g2_cache) {
        if (g_reverse_capture_flags) g_active_g2_cache->reverse_flux_coefficients = flux.coefficients;
        g_active_g2_cache->points = std::move(points);
        g_active_g2_cache->B = std::move(d_B);
        g_active_g2_cache->grad_B = std::move(d_grad_B);
        g_active_g2_cache->alpha = std::move(alpha);
        g_active_g2_cache->qs_sums = qs_sums;
        g_active_g2_cache->G = result.qs_vacuum_G;
        g_active_g2_cache->ready = true;
    }
    return true;
}


}  // namespace coil_eval::pipeline
