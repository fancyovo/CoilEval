// C ABI for configuration, formal/local evaluation, and center capture.
#include "pipeline/types.cuh"
#include "pipeline/session_state.cuh"
#include "pipeline/axis.cuh"
#include "pipeline/coil_geometry.cuh"
#include "pipeline/common.cuh"
#include "pipeline/evaluate.cuh"
#include "pipeline/local_batch.cuh"
#include "pipeline/policy.cuh"
#include "pipeline/psi.cuh"
#include "pipeline/surface.cuh"
#include "pipeline/target_volume.cuh"
#include "pipeline/alpha.cuh"
#include "pipeline/gradients.cuh"
#include "coil_field_internal.h"
#include "nvtx_profile.h"

using namespace coil_eval::pipeline;

extern "C" {

int sgpu_get_boundary_iota(int query, double* value) {
    if (!value) return 1;
    if (query < 0) *value = g_boundary_iota;
    else if (static_cast<size_t>(query) < g_batch_boundary_iota.size())
        *value = g_batch_boundary_iota[query];
    else return 1;
    return std::isfinite(*value) ? 0 : 1;
}

std::size_t sgpu_score_config_size() {
    return sizeof(SgpuScoreConfig);
}

std::size_t sgpu_score_result_size() {
    return sizeof(SgpuScoreResult);
}

int sgpu_score_coils_local_batch(
    void* batch_field_handle,
    const double* currents_a,
    int n_base_coils,
    int nfp,
    const double* axis_R,
    const double* axis_Z,
    const double* axis_R_phi,
    const double* axis_Z_phi,
    int axis_count,
    const double* axis_residual,
    const double* axis_topology_trace,
    const double* axis_topology_det,
    const double* psi_coefficients,
    int psi_coefficient_count,
    const double* psi_train_rms,
    const double* coil_components,
    const SgpuScoreResult* center_result,
    const SgpuScoreConfig* config,
    int surface_theta_count,
    int alpha_iterations,
    SgpuScoreResult* query_results,
    double* stats_out,
    int stats_len
) {
    g_batch_boundary_iota.clear();
    if (!batch_field_handle || !center_result || !config || !query_results) {
        sgpu_internal_set_error("invalid local batch score handle or result");
        return 1;
    }
    std::string reason;
    if (!validate_config(*config, reason)) {
        sgpu_internal_set_error(reason.c_str());
        return 1;
    }
    if (cudaSetDevice(config->device_id) != cudaSuccess) {
        sgpu_internal_set_error("local batch cudaSetDevice failed");
        return 1;
    }
    return score_local_batch_impl(
        batch_field_handle, currents_a, n_base_coils, nfp,
        axis_R, axis_Z, axis_R_phi, axis_Z_phi, axis_count,
        axis_residual, axis_topology_trace, axis_topology_det,
        psi_coefficients, psi_coefficient_count, psi_train_rms,
        coil_components, *center_result, *config, surface_theta_count,
        alpha_iterations, query_results, stats_out, stats_len
    ) ? 0 : 1;
}

int sgpu_default_score_config(SgpuScoreConfig* config) {
    if (!config) {
        sgpu_internal_set_error("score config pointer is null");
        return 1;
    }
    std::memset(config, 0, sizeof(*config));
    config->abi_version = SGPU_SCORE_ABI_VERSION;
    config->struct_size = sizeof(*config);
    config->device_id = 0;
    config->segments_per_coil = 256;
    config->target_M = 1;
    config->target_N = 0;
    config->axis_grid = 48;
    config->axis_fallback_grid = 64;
    config->axis_max_candidates = 16;
    config->axis_fallback_max_candidates = 48;
    config->axis_newton_iters = 6;
    config->axis_fallback_newton_iters = 6;
    config->axis_trace_steps = 960;
    config->axis_sample_count = 240;
    config->axis_fallback_max_nfp = 7;
    config->axis_span = 0.5;
    config->axis_tolerance = 1.0e-7;
    config->axis_r_floor = 1.0e-4;
    config->axis_fd_relative = 2.0e-4;
    config->axis_fd_absolute = 2.0e-6;
    config->axis_topology_margin = 2.0e-2;
    config->psi_poly_degree = 10;
    config->psi_m_tor = 12;
    config->psi_n_r = 48;
    config->psi_n_z = 48;
    config->psi_n_phi = 48;
    config->psi_validation_points = 4000;
    config->psi_solver_mode = 2;
    config->psi_precision_mode = 2;
    config->psi_a = 0.05;
    config->psi_rho_min = 0.002;
    config->psi_ridge = 1.0e-6;
    const double default_levels[] = {
        0.001, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08,
        0.16, 0.25, 0.36, 0.49, 0.64, 0.81,
    };
    config->surface_level_count = static_cast<int>(sizeof(default_levels) / sizeof(default_levels[0]));
    std::copy(default_levels, default_levels + config->surface_level_count, config->surface_levels);
    config->surface_theta_count = 256;
    config->surface_trace_steps = 800;
    config->surface_newton_iters = 20;
    config->surface_newton_tolerance = 1.0e-12;
    config->surface_max_radius_scale = 1.0;
    config->surface_drift_relative_tolerance = 0.05;
    config->surface_drift_absolute_tolerance = 5.0e-4;
    config->surface_long_trace_periods = 16;
    config->surface_long_trace_relative_tolerance = 0.05;
    config->flux_level_count = 11;
    config->flux_phi_count = 8;
    config->flux_theta_count = 256;
    config->flux_radial_quadrature = 24;
    config->flux_polynomial_degree = 4;
    config->flux_boundary_tolerance = 2.0e-6;
    config->flux_section_relative_std_tolerance = 0.02;
    config->volume_point_count = 100000;
    config->volume_phi_count = 96;
    config->volume_theta_count = 32;
    config->alpha_fit_point_count = 30000;
    config->alpha_radial_order = 12;
    config->alpha_poloidal_order = 12;
    config->alpha_toroidal_order = 12;
    // iota is a flux function; rho^2 is the normalized radial surface label.
    config->iota_degree = 3;
    config->radial_bin_count = 10;
    config->alpha_solver_mode = 2;
    config->volume_rho_min = 0.08;
    config->alpha_ridge = 1.0e-7;
    const double weights[] = {10.0, 10.0, 10.0, 10.0, 42.0, 10.0, 8.0};
    std::copy(weights, weights + SGPU_SCORE_COMPONENT_COUNT, config->score_weights);
    config->score_axis_residual_scale = 1.0e-5;
    config->score_psi_angle_p95_scale = 3.0e-3;
    config->score_psi_angle_l2_scale = 1.0e-3;
    config->score_surface_inverse_aspect_saturation = 0.03;
    config->score_surface_drift_scale = 0.02;
    config->score_flux_section_std_scale = 0.01;
    config->score_flux_boundary_residual_scale = 2.0e-6;
    config->score_alpha_normal_B_scale = 1.0e-4;
    config->score_alpha_relative_l2_scale = 0.25;
    config->score_qs_global_scale = 0.05;
    config->score_qs_edge_scale = 0.07;
    config->score_qh_iota_threshold = 1.0;
    config->score_qh_iota_power = 2.0;
    config->score_volume_qs_size_floor = 0.65;
    config->score_volume_qs_iota_floor = 0.50;
    config->score_qh_total_iota_floor = 0.10;
    config->score_qh_total_helicity_floor = 0.10;
    config->score_qh_helicity_bad = 0.10;
    config->score_qh_helicity_good = 0.30;
    config->score_qh_helicity_exploration_fraction = 0.20;
    config->surface_selection_mode = 0;
    config->surface_confidence_periods = 2;
    config->surface_flux_bisection_iters = 6;
    config->target_volume=0;
    config->target_volume_relative_tolerance=1e-7;
    config->target_volume_neighbor_tolerance=0.01;
    config->target_volume_forward_tolerance=1e-4;
    config->target_quality_threshold=0.01;
    config->target_quality_temperature=0.0025;
    config->target_quality_weight=10000.0;
    config->surface_confidence_drift_center = 0.05;
    config->surface_confidence_drift_temperature = 0.0125;
    config->surface_confidence_smoothmax_temperature = 0.005;
    config->surface_confidence_minimum = 0.05;
    config->axis_hint_enabled = 0;
    config->axis_hint_require_continuation = 0;
    config->axis_hint_R = 0.0;
    config->axis_hint_Z = 0.0;
    config->axis_hint_max_distance = 0.10;
    sgpu_internal_set_error("");
    return 0;
}

int sgpu_score_coils(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* result
) {
    if (!result) {
        sgpu_internal_set_error("score result pointer is null");
        return 1;
    }
    const int device_id = config ? config->device_id : -1;
    initialize_result(result, device_id);
    g_boundary_iota = std::numeric_limits<double>::quiet_NaN();
    SGPU_NVTX_RANGE("score.total");
    auto total_started = Clock::now();
    if (!coeffs_x || !coeffs_y || !coeffs_z || !currents_a || !config ||
        n_base_coils <= 0 || n_coeff < 3 || (n_coeff % 2) == 0 || nfp <= 0) {
        return fail_result(result, "invalid score input arrays or dimensions");
    }
    std::string config_error;
    if (!validate_config(*config, config_error)) return fail_result(result, config_error.c_str());
    result->target_volume_requested=config->target_volume;
    if (cudaSetDevice(config->device_id) != cudaSuccess) return fail_result(result, "cudaSetDevice failed");

    auto stage_started = Clock::now();
    const CoilMetrics coil = [&] {
        SGPU_NVTX_RANGE("score.coil_geometry.cpu");
        return compute_coil_metrics(
            coeffs_x, coeffs_y, coeffs_z, currents_a, n_base_coils, n_coeff, nfp
        );
    }();
    result->timings[SGPU_SCORE_TIME_COIL_GEOMETRY] = seconds_since(stage_started);
    result->coil_length_mean = coil.length_mean;
    result->coil_curvature_p95 = coil.curvature_p95;
    result->coil_curvature_max = coil.curvature_max;
    result->coil_min_intercoil_distance = coil.min_intercoil_distance;
    result->coil_min_axis_distance = coil.min_axis_distance;
    result->coil_high_mode_energy_fraction = coil.high_mode_fraction;
    result->coil_current_abs_max_a = coil.current_abs_max;

    void* field = nullptr;
    stage_started = Clock::now();
    {
        SGPU_NVTX_RANGE("score.field_create");
        if (sgpu_create_field(
                coeffs_x, coeffs_y, coeffs_z, currents_a, n_base_coils, n_coeff,
                nfp, config->segments_per_coil, config->device_id, &field)) {
            return fail_from_backend(result, "field creation");
        }
    }
    result->timings[SGPU_SCORE_TIME_FIELD_CREATE] = seconds_since(stage_started);
    result->stage_completed = SCORE_STAGE_FIELD;

    int return_code = 0;
    do {
        AxisData axis;
        stage_started = Clock::now();
        double axis_trace_time = 0.0;
        {
            SGPU_NVTX_RANGE("score.axis");
            if (!find_axis_native(
                    field, coeffs_x, coeffs_y, coeffs_z, n_base_coils, n_coeff,
                    nfp, *config, axis, axis_trace_time, result->timings)) {
                return_code = fail_from_backend(result, "axis search");
                break;
            }
        }
        result->timings[SGPU_SCORE_TIME_AXIS_TRACE] = axis_trace_time;
        result->timings[SGPU_SCORE_TIME_AXIS_SEARCH] =
            std::max(0.0, seconds_since(stage_started) - axis_trace_time);
        result->axis_candidate_count = axis.candidate_count;
        result->axis_used_hint = axis.used_hint ? 1 : 0;
        result->axis_hint_distance = axis.hint_distance;
        if (axis.R.empty()) {
            result->status = axis.branch_lost ? SGPU_SCORE_BRANCH_LOST : SGPU_SCORE_NO_AXIS;
            result->stage_completed = SCORE_STAGE_AXIS;
            result->components[SGPU_SCORE_COMPONENT_COIL] = coil_component(coil);
            break;
        }
        result->axis_R = axis.selected.R;
        result->axis_Z = axis.selected.Z;
        result->axis_residual = axis.selected.residual;
        result->axis_topology_trace = axis.selected.topology_trace;
        result->axis_topology_det = axis.selected.topology_det;
        result->axis_ellipse_aspect = axis.selected.ellipse_aspect;
        result->stage_completed = SCORE_STAGE_AXIS;

        PsiData psi;
        {
            SGPU_NVTX_RANGE("score.psi_fit");
            if (!fit_psi_native(
                    field, axis, nfp, *config, psi,
                    result->timings[SGPU_SCORE_TIME_PSI_POINTS],
                    result->timings[SGPU_SCORE_TIME_PSI_FIT])) {
                return_code = fail_from_backend(result, "psi fit");
                break;
            }
        }
        stage_started = Clock::now();
        {
            SGPU_NVTX_RANGE("score.psi_validate");
            if (!validate_psi_native(field, axis, nfp, *config, psi)) {
                return_code = fail_from_backend(result, "psi validation");
                break;
            }
        }
        result->timings[SGPU_SCORE_TIME_PSI_VALIDATE] = seconds_since(stage_started);
        result->psi_train_rms = psi.train_rms;
        result->psi_angle_mean = psi.angle_mean;
        result->psi_angle_p95 = psi.angle_p95;
        result->psi_angle_l2 = psi.angle_l2;
        result->stage_completed = SCORE_STAGE_PSI;

        std::vector<SurfaceScreen> screens;
        stage_started = Clock::now();
        bool screen_ok = false;
        {
            SGPU_NVTX_RANGE("score.surface_screen");
            screen_ok = config->target_volume>0
                ? screen_target_volume_native(field,axis,psi,nfp,*config,screens,*result)
                : config->surface_selection_mode == 1
                ? screen_surface_confidence_native(
                    field, axis, psi, nfp, *config, screens, *result)
                : screen_surfaces_native(
                    field, axis, psi, nfp, *config, screens, result->timings);
        }
        if (!screen_ok) {
            return_code = fail_from_backend(result, "surface screen");
            break;
        }
        result->timings[SGPU_SCORE_TIME_SURFACE_SCREEN] = seconds_since(stage_started);
        const auto best_strict = std::max_element(screens.begin(), screens.end(), [](const auto& lhs, const auto& rhs) {
            return (lhs.strict && lhs.verified ? lhs.level : -1.0) <
                   (rhs.strict && rhs.verified ? rhs.level : -1.0);
        });
        result->stable_surface_count = static_cast<int>(std::count_if(screens.begin(), screens.end(), [](const auto& screen) {
            return screen.stable;
        }));
        result->stage_completed = SCORE_STAGE_SURFACE;
        if (result->stable_surface_count == 0) {
            result->status = SGPU_SCORE_NO_SURFACE;
            fill_early_components(*config, coil, axis, psi, screens, *result);
            break;
        }
        if (best_strict == screens.end() || !best_strict->strict || !best_strict->verified) {
            result->status = SGPU_SCORE_DRIFT_REJECTED;
            fill_early_components(*config, coil, axis, psi, screens, *result);
            break;
        }
        bool downstream_ok = false;
        {
            SGPU_NVTX_RANGE("score.downstream");
            downstream_ok = run_downstream_gpu(
                    field, currents_a, n_base_coils, nfp, axis, psi,
                    screens, *config, *result, config->surface_selection_mode == 0);
        }
        fill_early_components(*config, coil, axis, psi, screens, *result);
        if (!downstream_ok) {
            if (result->status == SGPU_SCORE_INTERNAL_ERROR) return_code = 1;
            break;
        }
        if (g_active_g2_cache && g_active_g2_cache->ready) {
            g_active_g2_cache->axis = axis;
            g_active_g2_cache->psi = psi;
        }
    } while (false);

    if (g_active_g2_cache && g_active_g2_cache->ready) {
        g_active_g2_cache->field = field;
        field = nullptr;
    }
    if (field) sgpu_destroy_field(field);
    stage_started = Clock::now();
    {
        SGPU_NVTX_RANGE("score.finalize.cpu");
        finalize_score(*config, *result);
    }
    result->timings[SGPU_SCORE_TIME_SCORE] = seconds_since(stage_started);
    result->timings[SGPU_SCORE_TIME_TOTAL] = seconds_since(total_started);
    return return_code;
}

int sgpu_get_captured_iota_profile(double* coefficients, int capacity, int* count) {
    if (!coefficients || !count || g_captured_iota_profile.empty() ||
        capacity < static_cast<int>(g_captured_iota_profile.size())) return 1;
    *count = static_cast<int>(g_captured_iota_profile.size());
    std::copy(g_captured_iota_profile.begin(), g_captured_iota_profile.end(), coefficients);
    return 0;
}

int sgpu_score_coils_capture_psi_center(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* result,
    double* psi_coefficients,
    int psi_coefficient_capacity,
    int* psi_coefficient_count
) {
    if (!coeffs_x || !coeffs_y || !coeffs_z || !currents_a || !config || !result ||
        !psi_coefficients || psi_coefficient_capacity <= 0 || !psi_coefficient_count) {
        sgpu_internal_set_error("invalid psi-center capture input");
        return 1;
    }
    *psi_coefficient_count = 0;
    g_captured_iota_profile.clear();
    g_alpha_warm_preconditioner = AlphaWarmPreconditionerCache{};
    sgpu_clear_psi_warm_preconditioner();
    sgpu_set_psi_warm_preconditioner_capture(1);
    FixedFrontG2Cache cache;
    g_active_gradient_group = 3;
    g_active_g2_cache = &cache;
    const int code = sgpu_score_coils(
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp, config, result
    );
    g_active_g2_cache = nullptr;
    g_active_gradient_group = 0;
    sgpu_set_psi_warm_preconditioner_capture(0);
    if (code != 0 || result->status != SGPU_SCORE_OK || !cache.ready ||
        cache.psi.coeffs.empty() || !sgpu_has_psi_warm_preconditioner(
            static_cast<int>(cache.psi.coeffs.size())) || !cache.alpha.adjoint_ready) {
        if (cache.field) sgpu_destroy_field(cache.field);
        sgpu_clear_psi_warm_preconditioner();
        sgpu_internal_set_error("psi-center capture score is not ok");
        return code != 0 ? code : 1;
    }
    g_alpha_warm_preconditioner.column_count = cache.alpha.column_count;
    g_alpha_warm_preconditioner.qr_rows = cache.alpha.qr_rows;
    g_alpha_warm_preconditioner.upper_factor.resize(
        static_cast<size_t>(cache.alpha.column_count) * cache.alpha.column_count
    );
    g_alpha_warm_preconditioner.center_scale.resize(cache.alpha.column_count);
    g_alpha_warm_preconditioner.center_scaled_solution.resize(cache.alpha.column_count);
    if (cudaMemcpy2D(
            g_alpha_warm_preconditioner.upper_factor.data(),
            static_cast<size_t>(cache.alpha.column_count) * sizeof(float),
            cache.alpha.qr_matrix.data(),
            static_cast<size_t>(cache.alpha.qr_rows) * sizeof(float),
            static_cast<size_t>(cache.alpha.column_count) * sizeof(float),
            cache.alpha.column_count,
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(
            g_alpha_warm_preconditioner.center_scale.data(), cache.alpha.scales.data(),
            static_cast<size_t>(cache.alpha.column_count) * sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(
            g_alpha_warm_preconditioner.center_scaled_solution.data(),
            cache.alpha.scaled_solution.data(),
            static_cast<size_t>(cache.alpha.column_count) * sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
        if (cache.field) sgpu_destroy_field(cache.field);
        sgpu_clear_psi_warm_preconditioner();
        g_alpha_warm_preconditioner = AlphaWarmPreconditionerCache{};
        sgpu_internal_set_error("alpha-center preconditioner capture failed");
        return 1;
    }
    if (static_cast<int>(cache.psi.coeffs.size()) > psi_coefficient_capacity) {
        if (cache.field) sgpu_destroy_field(cache.field);
        sgpu_clear_psi_warm_preconditioner();
        sgpu_internal_set_error("psi-center capture output capacity is too small");
        return 1;
    }
    g_captured_iota_profile = cache.alpha.iota_coefficients;
    g_boundary_iota = 0.0;
    for (float coefficient : g_captured_iota_profile) g_boundary_iota += coefficient;
    std::copy(cache.psi.coeffs.begin(), cache.psi.coeffs.end(), psi_coefficients);
    *psi_coefficient_count = static_cast<int>(cache.psi.coeffs.size());
    if (cache.field) sgpu_destroy_field(cache.field);
    sgpu_internal_set_error("");
    return 0;
}

}  // extern "C"
