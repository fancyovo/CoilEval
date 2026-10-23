// Configuration validation and the retained legacy composite score policy.
#include "pipeline/types.cuh"
#include "pipeline/axis.cuh"
#include "pipeline/coil_geometry.cuh"
#include "pipeline/common.cuh"
#include "pipeline/policy.cuh"
#include "pipeline/psi.cuh"
#include "pipeline/surface.cuh"

namespace coil_eval::pipeline {

bool validate_config(const SgpuScoreConfig& config, std::string& reason) {
    if (config.abi_version != SGPU_SCORE_ABI_VERSION || config.struct_size != sizeof(SgpuScoreConfig)) {
        reason = "score config ABI version or size mismatch";
        return false;
    }
    if (config.device_id < 0 || config.segments_per_coil <= 0 ||
        config.axis_grid < 8 || config.axis_fallback_grid < config.axis_grid ||
        config.axis_fallback_max_nfp < 1 ||
        config.axis_sample_count < 16 || config.psi_poly_degree < 2 ||
        config.psi_poly_degree > 24 || config.psi_m_tor < 0 || config.psi_m_tor > 32 ||
        (config.psi_solver_mode != 1 && config.psi_solver_mode != 2) ||
        (config.psi_precision_mode != 1 && config.psi_precision_mode != 2) ||
        config.surface_level_count <= 0 ||
        config.surface_level_count > SGPU_SCORE_MAX_SURFACE_LEVELS ||
        config.surface_theta_count < 16 || config.surface_long_trace_periods < 1 ||
        config.surface_long_trace_periods > 64 ||
        !(config.surface_long_trace_relative_tolerance > 0.0) ||
        config.volume_point_count <= 0 ||
        config.alpha_fit_point_count <= 0 ||
        config.alpha_fit_point_count > config.volume_point_count ||
        (config.alpha_solver_mode != 1 && config.alpha_solver_mode != 2) ||
        !(config.score_surface_inverse_aspect_saturation > 0.0) ||
        !(config.score_qh_iota_threshold > 0.0) ||
        !(config.score_qh_iota_power > 0.0) ||
        config.score_volume_qs_size_floor < 0.0 ||
        config.score_volume_qs_size_floor > 1.0 ||
        config.score_volume_qs_iota_floor < 0.0 ||
        config.score_volume_qs_iota_floor > 1.0 ||
        config.score_qh_total_iota_floor < 0.0 ||
        config.score_qh_total_iota_floor > 1.0 ||
        config.score_qh_total_helicity_floor < 0.0 ||
        config.score_qh_total_helicity_floor > 1.0 ||
        !(config.score_qh_helicity_good > config.score_qh_helicity_bad) ||
        config.axis_topology_margin < 0.0 || config.axis_topology_margin >= 2.0 ||
        config.score_qh_helicity_exploration_fraction < 0.0 ||
        config.score_qh_helicity_exploration_fraction > 1.0 ||
        config.surface_selection_mode < 0 || config.surface_selection_mode > 1 ||
        config.surface_confidence_periods < 1 || config.surface_confidence_periods > 16 ||
        config.surface_flux_bisection_iters < 0 || config.surface_flux_bisection_iters > 12 ||
        !(config.surface_confidence_drift_center > 0.0) ||
        !(config.surface_confidence_drift_temperature > 0.0) ||
        !(config.surface_confidence_smoothmax_temperature > 0.0) ||
        config.surface_confidence_minimum < 0.0 || config.surface_confidence_minimum > 1.0 ||
        config.axis_hint_enabled < 0 || config.axis_hint_enabled > 1 ||
        config.axis_hint_require_continuation < 0 || config.axis_hint_require_continuation > 2 ||
        (config.axis_hint_require_continuation == 2 && !config.axis_hint_enabled) ||
        !(config.axis_hint_max_distance > 0.0)) {
        reason = "invalid score configuration dimensions";
        return false;
    }
    if (!std::isfinite(config.target_volume) || config.target_volume<0 ||
        !(config.target_volume_relative_tolerance>0) || !(config.target_volume_neighbor_tolerance>0) ||
        !(config.target_volume_forward_tolerance>0) ||
        !(config.target_quality_threshold>0) || !(config.target_quality_temperature>0) ||
        !(config.target_quality_weight>=0) || !std::isfinite(config.target_quality_weight)) {
        reason="invalid ABI-12 target volume or quality configuration";return false;
    }
    return true;
}

void fill_early_components(
    const SgpuScoreConfig& config,
    const CoilMetrics& coil,
    const AxisData& axis,
    const PsiData& psi,
    const std::vector<SurfaceScreen>& screens,
    SgpuScoreResult& result
) {
    const double axis_residual = q_down(axis.selected.residual, config.score_axis_residual_scale, 0.8);
    const double stability_margin = axis_topology_stability_margin(axis.selected);
    const double topology = !axis.selected.elliptic
        ? 0.1
        : config.axis_topology_margin > 0.0
            ? q_saturating_up(stability_margin, config.axis_topology_margin)
            : 1.0;
    const double aspect = q_down(std::max(axis.selected.ellipse_aspect - 1.0, 0.0), 1.0, 1.2, 0.8);
    result.components[SGPU_SCORE_COMPONENT_AXIS] = blend({{0.70, axis_residual}, {0.20, topology}, {0.10, aspect}});
    result.components[SGPU_SCORE_COMPONENT_PSI] = blend({
        {0.58, q_down(psi.angle_p95, config.score_psi_angle_p95_scale, 1.1)},
        {0.32, q_down(psi.angle_l2, config.score_psi_angle_l2_scale, 1.1)},
        {0.10, q_down(psi.train_rms, 5.0e-3, 1.0, 0.45)},
    });
    double minimum_drift = std::numeric_limits<double>::infinity();
    int strict_count = 0;
    for (const auto& screen : screens) {
        if (std::isfinite(screen.relative_drift_p95)) minimum_drift = std::min(minimum_drift, screen.relative_drift_p95);
        if (screen.strict) ++strict_count;
    }
    if (std::isfinite(result.surface_drift_relative_p95)) {
        minimum_drift = result.surface_drift_relative_p95;
    }
    const double size = q_saturating_up(
        result.surface_inverse_aspect_ratio, config.score_surface_inverse_aspect_saturation
    );
    result.score_surface_size = size;
    if (config.surface_selection_mode == 1) {
        const double selected_level = std::isfinite(result.surface_level)
            ? result.surface_level : result.surface_effective_level;
        const double extent = q_saturating_up(
            selected_level, config.surface_levels[config.surface_level_count - 1]
        );
        // The selected level already combines short-horizon confidence with the
        // continuously calibrated flux boundary. Do not reuse risk measured at
        // the larger proposal after flux calibration has moved the edge inward.
        result.components[SGPU_SCORE_COMPONENT_SURFACE] = blend({
            {0.65, size}, {0.35, extent},
        });
        result.components[SGPU_SCORE_COMPONENT_COIL] = coil_component(coil);
        return;
    }
    const double drift = q_down(minimum_drift, config.score_surface_drift_scale, 1.0, 0.15);
    const double count = q_up(strict_count, 2.0, 1.0);
    result.components[SGPU_SCORE_COMPONENT_SURFACE] = blend({{0.65, size}, {0.25, drift}, {0.10, count}});
    result.components[SGPU_SCORE_COMPONENT_COIL] = coil_component(coil);
}

void finalize_score(const SgpuScoreConfig& config, SgpuScoreResult& result) {
    if (result.status == SGPU_SCORE_INTERNAL_ERROR) return;
    if (result.stage_completed < SCORE_STAGE_ALPHA) {
        result.components[SGPU_SCORE_COMPONENT_COORDINATE] = 0.08;
    }
    if (result.stage_completed < SCORE_STAGE_QS) {
        result.components[SGPU_SCORE_COMPONENT_VOLUME_QS] = 0.04;
    }
    if (result.stage_completed < SCORE_STAGE_ALPHA) {
        result.components[SGPU_SCORE_COMPONENT_IOTA] = 0.04;
    }
    double total_weight = 0.0;
    double weighted_score = 0.0;
    const double iota_score = clip01(result.components[SGPU_SCORE_COMPONENT_IOTA]);
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        const double value = clip01(result.components[component]);
        total_weight += std::max(config.score_weights[component], 0.0);
        weighted_score += std::max(config.score_weights[component], 0.0) * value;
        result.components[component] = 100.0 * value;
    }
    result.score_before_qh_iota_gate =
        total_weight > 0.0 ? 100.0 * clip01(weighted_score / total_weight) : 0.0;
    const bool qh_target = config.target_M != 0 && config.target_N != 0;
    result.score_qh_total_iota_factor = qh_target
        ? config.score_qh_total_iota_floor +
            (1.0 - config.score_qh_total_iota_floor) * iota_score
        : 1.0;
    const double helicity_position = clip01(
        (result.score_qh_helicity_advantage - config.score_qh_helicity_bad) /
        (config.score_qh_helicity_good - config.score_qh_helicity_bad)
    );
    const double helicity_linear = clip01(
        result.score_qh_helicity_advantage / config.score_qh_helicity_good
    );
    const double helicity_window =
        helicity_position * helicity_position * (3.0 - 2.0 * helicity_position);
    result.score_qh_helicity_quality = qh_target
        ? config.score_qh_helicity_exploration_fraction * helicity_linear +
            (1.0 - config.score_qh_helicity_exploration_fraction) * helicity_window
        : 1.0;
    result.score_qh_total_helicity_factor = qh_target
        ? config.score_qh_total_helicity_floor +
            (1.0 - config.score_qh_total_helicity_floor) *
                result.score_qh_helicity_quality
        : 1.0;
    result.score = result.score_before_qh_iota_gate *
        result.score_qh_total_iota_factor * result.score_qh_total_helicity_factor;
    if (result.status == SGPU_SCORE_OK) result.stage_completed = SCORE_STAGE_COMPLETE;
}


}  // namespace coil_eval::pipeline
