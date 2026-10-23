// Result initialization, numerical guards, timing, and robust summary statistics.
#include "pipeline/types.cuh"
#include "pipeline/common.cuh"
#include "coil_field_internal.h"

namespace coil_eval::pipeline {

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double finite_or(double value, double fallback) {
    return std::isfinite(value) ? value : fallback;
}

double clip01(double value) {
    return std::min(1.0, std::max(0.0, finite_or(value, 0.0)));
}

double q_down(double value, double scale, double power, double fallback) {
    if (!std::isfinite(value) || !(scale > 0.0)) return fallback;
    const double x = std::max(value, 0.0) / scale;
    return 1.0 / (1.0 + std::pow(x, power));
}

double q_up(double value, double scale, double power, double fallback) {
    if (!std::isfinite(value) || value <= 0.0 || !(scale > 0.0)) return fallback;
    return 1.0 / (1.0 + std::pow(scale / value, power));
}

double q_saturating_up(double value, double saturation, double fallback) {
    if (!std::isfinite(value) || value <= 0.0 || !(saturation > 0.0)) return fallback;
    const double x = clip01(value / saturation);
    return x * x * (3.0 - 2.0 * x);
}

double minimum_absolute_iota(double iota_min, double iota_max) {
    if (!std::isfinite(iota_min) || !std::isfinite(iota_max)) return 0.0;
    if (iota_min <= 0.0 && iota_max >= 0.0) return 0.0;
    return std::min(std::abs(iota_min), std::abs(iota_max));
}

double blend(std::initializer_list<std::pair<double, double>> values) {
    double total = 0.0;
    double weighted = 0.0;
    for (const auto& item : values) {
        if (item.first > 0.0) {
            total += item.first;
            weighted += item.first * clip01(item.second);
        }
    }
    return total > 0.0 ? weighted / total : 0.0;
}

void initialize_result(SgpuScoreResult* result, int device_id) {
    std::memset(result, 0, sizeof(*result));
    result->abi_version = SGPU_SCORE_ABI_VERSION;
    result->struct_size = sizeof(*result);
    result->status = SGPU_SCORE_INTERNAL_ERROR;
    result->stage_completed = SCORE_STAGE_NONE;
    result->device_id = device_id;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    result->score = nan;
    result->axis_R = nan;
    result->axis_Z = nan;
    result->axis_residual = nan;
    result->axis_topology_trace = nan;
    result->axis_topology_det = nan;
    result->axis_ellipse_aspect = nan;
    result->psi_train_rms = nan;
    result->psi_angle_mean = nan;
    result->psi_angle_p95 = nan;
    result->psi_angle_l2 = nan;
    result->surface_level = nan;
    result->surface_drift_relative_p95 = nan;
    result->surface_one_period_drift_relative_p95 = nan;
    result->surface_effective_minor_radius = nan;
    result->surface_inverse_aspect_ratio = nan;
    result->surface_volume = nan;
    result->flux_edge = nan;
    result->flux_fit_relative_rms = nan;
    result->flux_section_relative_std_edge = nan;
    result->flux_boundary_residual_max = nan;
    result->flux_derivative_min = nan;
    result->flux_derivative_max = nan;
    result->alpha_relative_l2 = nan;
    result->alpha_normal_B_relative_l2 = nan;
    result->iota_min = nan;
    result->iota_max = nan;
    result->score_surface_size = nan;
    result->score_iota = nan;
    result->score_qs_residual = nan;
    result->score_volume_qs_size_factor = nan;
    result->score_volume_qs_iota_factor = nan;
    result->score_before_qh_iota_gate = nan;
    result->score_qh_total_iota_factor = nan;
    result->score_qh_helicity_advantage = 0.0;
    result->score_qh_helicity_quality = 0.0;
    result->score_qh_total_helicity_factor = nan;
    result->qs_global_error = nan;
    result->qs_edge_error = nan;
    result->qs_qa_global_error = nan;
    result->qs_qp_global_error = nan;
    result->qs_vacuum_G = nan;
    result->qs_target_global_error_per_helicity = nan;
    result->qs_target_edge_error_per_helicity = nan;
    result->qs_qa_global_error_per_helicity = nan;
    result->qs_qp_global_error_raw = nan;
    result->qs_qp_global_error_per_helicity = nan;
    result->qs_abs_p95 = nan;
    result->qs_abs_p95_per_helicity = nan;
    result->volume_valid_fraction = nan;
    result->volume_weight_effective_fraction = nan;
    result->edge_weight_effective_fraction = nan;
    result->surface_confidence_mean = nan;
    result->surface_confidence_edge = nan;
    result->surface_effective_level = nan;
    result->surface_confidence_risk = nan;
    result->axis_hint_distance = nan;
    result->coil_length_mean = nan;
    result->coil_curvature_p95 = nan;
    result->coil_curvature_max = nan;
    result->coil_min_intercoil_distance = nan;
    result->coil_min_axis_distance = nan;
    result->coil_high_mode_energy_fraction = nan;
    result->coil_current_abs_max_a = nan;
}

int fail_result(SgpuScoreResult* result, const char* message) {
    result->status = SGPU_SCORE_INTERNAL_ERROR;
    std::snprintf(result->error_message, sizeof(result->error_message), "%s", message ? message : "unknown error");
    sgpu_internal_set_error(result->error_message);
    return 1;
}

int fail_from_backend(SgpuScoreResult* result, const char* stage) {
    const char* backend = sgpu_last_error();
    std::string message = stage ? stage : "GPU backend failure";
    if (backend && backend[0]) {
        message += ": ";
        message += backend;
    }
    return fail_result(result, message.c_str());
}

double percentile(std::vector<double> values, double fraction) {
    values.erase(
        std::remove_if(values.begin(), values.end(), [](double x) { return !std::isfinite(x); }),
        values.end()
    );
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double position = clip01(fraction) * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(position));
    const size_t hi = static_cast<size_t>(std::ceil(position));
    const double alpha = position - static_cast<double>(lo);
    return values[lo] * (1.0 - alpha) + values[hi] * alpha;
}

bool cuda_stage_ok(cudaError_t error, SgpuScoreResult& result, const char* stage) {
    if (error == cudaSuccess) return true;
    std::string message = stage ? stage : "CUDA stage";
    message += ": ";
    message += cudaGetErrorString(error);
    fail_result(&result, message.c_str());
    return false;
}


}  // namespace coil_eval::pipeline
