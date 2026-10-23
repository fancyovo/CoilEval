// Retained legacy composite-score derivative and frozen-batch C ABI.
#include "pipeline/types.cuh"
#include "pipeline/session_state.cuh"
#include "pipeline/coil_geometry.cuh"
#include "pipeline/common.cuh"
#include "pipeline/gradients.cuh"
#include "coil_field_internal.h"

using namespace coil_eval::pipeline;

extern "C" {

std::size_t sgpu_score_gradient_result_size() {
    return sizeof(SgpuScoreGradientResult);
}

int sgpu_coil_component_gradient(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    double* component_value,
    double* gradient_x,
    double* gradient_y,
    double* gradient_z,
    double* gradient_current
) {
    if (!component_value || !gradient_x || !gradient_y || !gradient_z || !gradient_current) {
        sgpu_internal_set_error("coil-component gradient output pointer is null");
        return 1;
    }
    CoilComponentGradient gradient;
    std::string error;
    if (!compute_coil_component_gradient_impl(
            coeffs_x, coeffs_y, coeffs_z, currents_a,
            n_base_coils, n_coeff, nfp, gradient, error)) {
        sgpu_internal_set_error(error.c_str());
        return 1;
    }
    *component_value = gradient.value;
    std::copy(gradient.x.begin(), gradient.x.end(), gradient_x);
    std::copy(gradient.y.begin(), gradient.y.end(), gradient_y);
    std::copy(gradient.z.begin(), gradient.z.end(), gradient_z);
    std::copy(gradient.current.begin(), gradient.current.end(), gradient_current);
    sgpu_internal_set_error("");
    return 0;
}

int sgpu_coil_metric_gradient(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    double* metric_values,
    double* metric_derivatives,
    int* spacing_pair_indices
) {
    if (!metric_values || !metric_derivatives) {
        sgpu_internal_set_error("coil metric gradient output pointer is null");
        return 1;
    }
    CoilComponentGradient gradient;
    CoilMetricDerivativeOutput metrics;
    std::string error;
    if (!compute_coil_component_gradient_impl(
            coeffs_x, coeffs_y, coeffs_z, currents_a,
            n_base_coils, n_coeff, nfp, gradient, error, &metrics)) {
        sgpu_internal_set_error(error.c_str());
        return 1;
    }
    metric_values[0] = metrics.length_mean;
    metric_values[1] = metrics.curvature_p95;
    metric_values[2] = metrics.curvature_max;
    metric_values[3] = metrics.min_intercoil_distance;
    metric_values[4] = metrics.min_axis_distance;
    metric_values[5] = metrics.high_mode_fraction;
    metric_values[6] = metrics.current_abs_max;
    const std::vector<double>* blocks[6][3] = {
        {&metrics.d_length[0], &metrics.d_length[1], &metrics.d_length[2]},
        {&metrics.d_p95[0], &metrics.d_p95[1], &metrics.d_p95[2]},
        {&metrics.d_max[0], &metrics.d_max[1], &metrics.d_max[2]},
        {&metrics.d_spacing[0], &metrics.d_spacing[1], &metrics.d_spacing[2]},
        {&metrics.d_radius[0], &metrics.d_radius[1], &metrics.d_radius[2]},
        {&metrics.d_high_mode[0], &metrics.d_high_mode[1], &metrics.d_high_mode[2]},
    };
    const int parameter_count = n_base_coils * n_coeff;
    for (int metric = 0; metric < 6; ++metric) {
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            const std::vector<double>& source = *blocks[metric][coordinate];
            if (static_cast<int>(source.size()) != parameter_count) {
                sgpu_internal_set_error("coil metric derivative block size mismatch");
                return 1;
            }
            std::copy(
                source.begin(), source.end(),
                metric_derivatives + static_cast<size_t>(metric * 3 + coordinate) * parameter_count
            );
        }
    }
    if (spacing_pair_indices) {
        spacing_pair_indices[0] = metrics.spacing_pair[0];
        spacing_pair_indices[1] = metrics.spacing_pair[1];
    }
    sgpu_internal_set_error("");
    return 0;
}

int sgpu_score_coils_g1_gradient(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* score_result,
    double* gradient_x,
    double* gradient_y,
    double* gradient_z,
    double* gradient_current,
    SgpuScoreGradientResult* gradient_result
) {
    if (!gradient_result) {
        sgpu_internal_set_error("score gradient result pointer is null");
        return 1;
    }
    std::memset(gradient_result, 0, sizeof(*gradient_result));
    gradient_result->abi_version = SGPU_SCORE_GRADIENT_ABI_VERSION;
    gradient_result->struct_size = sizeof(*gradient_result);
    gradient_result->status = 1;
    gradient_result->gradient_group = 1;
    if (!score_result || !gradient_x || !gradient_y || !gradient_z || !gradient_current || !config) {
        std::snprintf(
            gradient_result->error_message,
            sizeof(gradient_result->error_message),
            "%s",
            "G1 gradient input/output pointer is null"
        );
        sgpu_internal_set_error(gradient_result->error_message);
        return 1;
    }
    const auto forward_started = Clock::now();
    const int score_code = sgpu_score_coils(
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp, config, score_result
    );
    gradient_result->forward_wall_s = seconds_since(forward_started);
    if (score_code != 0) {
        std::snprintf(
            gradient_result->error_message,
            sizeof(gradient_result->error_message),
            "forward score failed: %.220s",
            score_result->error_message
        );
        sgpu_internal_set_error(gradient_result->error_message);
        return score_code;
    }
    const auto gradient_started = Clock::now();
    CoilComponentGradient component_gradient;
    std::string error;
    if (!compute_coil_component_gradient_impl(
            coeffs_x, coeffs_y, coeffs_z, currents_a,
            n_base_coils, n_coeff, nfp, component_gradient, error)) {
        std::snprintf(
            gradient_result->error_message,
            sizeof(gradient_result->error_message),
            "%s",
            error.c_str()
        );
        sgpu_internal_set_error(gradient_result->error_message);
        return 1;
    }
    double total_weight = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        total_weight += std::max(config->score_weights[component], 0.0);
    }
    const double gate = score_result->score_qh_total_iota_factor *
        score_result->score_qh_total_helicity_factor;
    const double score_scale = total_weight > 0.0
        ? std::max(config->score_weights[SGPU_SCORE_COMPONENT_COIL], 0.0) / total_weight * gate
        : 0.0;
    double score_norm2 = 0.0;
    double component_norm2 = 0.0;
    size_t gradient_count = 0;
    const int parameter_count = n_base_coils * n_coeff;
    const std::vector<double>* sources[] = {
        &component_gradient.x, &component_gradient.y, &component_gradient.z,
    };
    double* destinations[] = {gradient_x, gradient_y, gradient_z};
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        for (int parameter = 0; parameter < parameter_count; ++parameter) {
            const double component_value = (*sources[coordinate])[parameter];
            destinations[coordinate][parameter] = score_scale * component_value;
            component_norm2 += component_value * component_value;
            score_norm2 += destinations[coordinate][parameter] * destinations[coordinate][parameter];
            ++gradient_count;
        }
    }
    for (int coil = 0; coil < n_base_coils; ++coil) {
        gradient_current[coil] = score_scale * component_gradient.current[coil];
        component_norm2 += component_gradient.current[coil] * component_gradient.current[coil];
        score_norm2 += gradient_current[coil] * gradient_current[coil];
        ++gradient_count;
    }
    gradient_result->gradient_wall_s = seconds_since(gradient_started);
    gradient_result->score_gradient_rms = std::sqrt(score_norm2 / std::max<size_t>(gradient_count, 1));
    gradient_result->coil_component_gradient_rms =
        std::sqrt(component_norm2 / std::max<size_t>(gradient_count, 1));
    gradient_result->status = 0;
    sgpu_internal_set_error("");
    return 0;
}

int sgpu_score_coils_g2_gradient(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* score_result,
    double* gradient_x,
    double* gradient_y,
    double* gradient_z,
    double* gradient_current,
    SgpuScoreGradientResult* gradient_result
) {
    if (!gradient_result) {
        sgpu_internal_set_error("score gradient result pointer is null");
        return 1;
    }
    std::memset(gradient_result, 0, sizeof(*gradient_result));
    gradient_result->abi_version = SGPU_SCORE_GRADIENT_ABI_VERSION;
    gradient_result->struct_size = sizeof(*gradient_result);
    gradient_result->status = 1;
    gradient_result->gradient_group = 2;
    if (!score_result || !gradient_x || !gradient_y || !gradient_z ||
        !gradient_current || !config || !coeffs_x || !coeffs_y || !coeffs_z ||
        !currents_a) {
        std::snprintf(
            gradient_result->error_message,
            sizeof(gradient_result->error_message),
            "%s",
            "G2 gradient input/output pointer is null"
        );
        sgpu_internal_set_error(gradient_result->error_message);
        return 1;
    }
    FixedFrontG2Cache cache;
    const auto forward_started = Clock::now();
    g_active_gradient_group = 2;
    g_active_g2_cache = &cache;
    const int score_code = sgpu_score_coils(
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp, config, score_result
    );
    g_active_g2_cache = nullptr;
    g_active_gradient_group = 0;
    gradient_result->forward_wall_s = seconds_since(forward_started);
    if (score_code != 0) {
        if (cache.field) sgpu_destroy_field(cache.field);
        std::snprintf(
            gradient_result->error_message,
            sizeof(gradient_result->error_message),
            "G2 score forward failed: %.210s",
            score_result->error_message
        );
        sgpu_internal_set_error(gradient_result->error_message);
        return score_code;
    }
    if (score_result->status != SGPU_SCORE_OK) {
        if (cache.field) sgpu_destroy_field(cache.field);
        const size_t parameter_count =
            static_cast<size_t>(n_base_coils) * static_cast<size_t>(n_coeff);
        std::fill_n(gradient_x, parameter_count, 0.0);
        std::fill_n(gradient_y, parameter_count, 0.0);
        std::fill_n(gradient_z, parameter_count, 0.0);
        std::fill_n(gradient_current, static_cast<size_t>(n_base_coils), 0.0);
        gradient_result->status = 1;
        sgpu_internal_set_error("");
        return 0;
    }
    if (!cache.ready || !cache.field) {
        if (cache.field) sgpu_destroy_field(cache.field);
        std::snprintf(
            gradient_result->error_message,
            sizeof(gradient_result->error_message),
            "%s",
            "G2 cache is incomplete after an ok score"
        );
        sgpu_internal_set_error(gradient_result->error_message);
        return 1;
    }
    const auto gradient_started = Clock::now();
    CoilComponentGradient g1;
    FixedFrontG2Gradient g2;
    std::string error;
    if (!compute_coil_component_gradient_impl(
            coeffs_x, coeffs_y, coeffs_z, currents_a,
            n_base_coils, n_coeff, nfp, g1, error) ||
        !compute_fixed_front_g2_gradient(
            cache, coeffs_x, coeffs_y, coeffs_z, currents_a,
            n_base_coils, n_coeff, nfp, *config, *score_result, g2)) {
        sgpu_destroy_field(cache.field);
        if (!score_result->error_message[0]) {
            std::snprintf(
                gradient_result->error_message,
                sizeof(gradient_result->error_message),
                "%s",
                error.c_str()
            );
        } else {
            std::snprintf(
                gradient_result->error_message,
                sizeof(gradient_result->error_message),
                "%.250s",
                score_result->error_message
            );
        }
        sgpu_internal_set_error(gradient_result->error_message);
        return 1;
    }
    sgpu_destroy_field(cache.field);
    cache.field = nullptr;

    double total_weight = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        total_weight += std::max(config->score_weights[component], 0.0);
    }
    const double g1_scale = total_weight > 0.0
        ? std::max(config->score_weights[SGPU_SCORE_COMPONENT_COIL], 0.0) / total_weight *
            score_result->score_qh_total_iota_factor * score_result->score_qh_total_helicity_factor
        : 0.0;
    const int parameter_count = n_base_coils * n_coeff;
    const std::vector<double>* g1_sources[] = {&g1.x, &g1.y, &g1.z};
    const std::vector<double>* g2_sources[] = {&g2.x, &g2.y, &g2.z};
    double* destinations[] = {gradient_x, gradient_y, gradient_z};
    double score_norm2 = 0.0;
    double component_norm2 = 0.0;
    size_t gradient_count = 0;
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        for (int parameter = 0; parameter < parameter_count; ++parameter) {
            const double g1_value = (*g1_sources[coordinate])[parameter];
            destinations[coordinate][parameter] =
                g1_scale * g1_value + (*g2_sources[coordinate])[parameter];
            score_norm2 += destinations[coordinate][parameter] * destinations[coordinate][parameter];
            component_norm2 += g1_value * g1_value;
            ++gradient_count;
        }
    }
    for (int coil = 0; coil < n_base_coils; ++coil) {
        gradient_current[coil] = g1_scale * g1.current[coil] + g2.current[coil];
        score_norm2 += gradient_current[coil] * gradient_current[coil];
        component_norm2 += g1.current[coil] * g1.current[coil];
        ++gradient_count;
    }
    gradient_result->point_vjp_s = g2.point_vjp_s;
    gradient_result->field_vjp_s = g2.field_vjp_s;
    gradient_result->parameter_map_s = g2.parameter_map_s;
    gradient_result->gradient_wall_s = seconds_since(gradient_started);
    gradient_result->score_gradient_rms =
        std::sqrt(score_norm2 / std::max<size_t>(gradient_count, 1));
    gradient_result->coil_component_gradient_rms =
        std::sqrt(component_norm2 / std::max<size_t>(gradient_count, 1));
    gradient_result->status = 0;
    sgpu_internal_set_error("");
    return 0;
}

int sgpu_score_coils_g2_frozen_batch(
    const double* center_coeffs_x,
    const double* center_coeffs_y,
    const double* center_coeffs_z,
    const double* center_currents_a,
    const double* query_coeffs_x,
    const double* query_coeffs_y,
    const double* query_coeffs_z,
    const double* query_currents_a,
    int query_count,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* center_score_result,
    double* frozen_scores,
    double* volume_components,
    double* coil_components,
    double* target_errors,
    double* qa_errors,
    double* qp_errors
) {
    if (!center_coeffs_x || !center_coeffs_y || !center_coeffs_z ||
        !center_currents_a || !query_coeffs_x || !query_coeffs_y ||
        !query_coeffs_z || !query_currents_a || query_count <= 0 ||
        n_base_coils <= 0 || n_coeff <= 0 || nfp <= 0 || !config ||
        !center_score_result || !frozen_scores || !volume_components ||
        !coil_components || !target_errors || !qa_errors || !qp_errors) {
        sgpu_internal_set_error("invalid G2 frozen-batch input");
        return 1;
    }
    FixedFrontG2Cache cache;
    g_active_gradient_group = 2;
    g_active_g2_cache = &cache;
    const int score_code = sgpu_score_coils(
        center_coeffs_x, center_coeffs_y, center_coeffs_z, center_currents_a,
        n_base_coils, n_coeff, nfp, config, center_score_result
    );
    g_active_g2_cache = nullptr;
    g_active_gradient_group = 0;
    if (score_code != 0 || center_score_result->status != SGPU_SCORE_OK ||
        !cache.ready || !cache.field) {
        if (cache.field) sgpu_destroy_field(cache.field);
        if (score_code == 0 && center_score_result->status != SGPU_SCORE_OK) {
            sgpu_internal_set_error("G2 frozen-batch center score is not ok");
            return 1;
        }
        if (score_code == 0) {
            sgpu_internal_set_error("G2 frozen-batch center cache is incomplete");
            return 1;
        }
        return score_code;
    }

    const size_t coefficient_stride =
        static_cast<size_t>(n_base_coils) * static_cast<size_t>(n_coeff);
    const size_t current_stride = static_cast<size_t>(n_base_coils);
    std::string error;
    for (int query = 0; query < query_count; ++query) {
        if (!evaluate_fixed_front_g2_scalar(
                cache,
                query_coeffs_x + query * coefficient_stride,
                query_coeffs_y + query * coefficient_stride,
                query_coeffs_z + query * coefficient_stride,
                query_currents_a + query * current_stride,
                n_base_coils, n_coeff, nfp, *config, *center_score_result,
                frozen_scores[query], volume_components[query],
                coil_components[query], target_errors[query],
                qa_errors[query], qp_errors[query], error)) {
            sgpu_destroy_field(cache.field);
            std::snprintf(
                center_score_result->error_message,
                sizeof(center_score_result->error_message),
                "G2 frozen query %d failed: %.215s",
                query,
                error.c_str()
            );
            center_score_result->status = SGPU_SCORE_INTERNAL_ERROR;
            sgpu_internal_set_error(center_score_result->error_message);
            return 1;
        }
    }
    sgpu_destroy_field(cache.field);
    cache.field = nullptr;
    sgpu_internal_set_error("");
    return 0;
}

int sgpu_score_coils_g3_frozen_batch(
    const double* center_coeffs_x,
    const double* center_coeffs_y,
    const double* center_coeffs_z,
    const double* center_currents_a,
    const double* query_coeffs_x,
    const double* query_coeffs_y,
    const double* query_coeffs_z,
    const double* query_currents_a,
    int query_count,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* center_score_result,
    double* frozen_scores,
    double* volume_components,
    double* coordinate_components,
    double* iota_components,
    double* coil_components,
    double* target_errors,
    double* qa_errors,
    double* qp_errors,
    double* iota_minima,
    double* iota_maxima
) {
    if (!center_coeffs_x || !center_coeffs_y || !center_coeffs_z ||
        !center_currents_a || !query_coeffs_x || !query_coeffs_y ||
        !query_coeffs_z || !query_currents_a || query_count <= 0 ||
        n_base_coils <= 0 || n_coeff <= 0 || nfp <= 0 || !config ||
        !center_score_result || !frozen_scores || !volume_components ||
        !coordinate_components || !iota_components || !coil_components ||
        !target_errors || !qa_errors || !qp_errors || !iota_minima ||
        !iota_maxima) {
        sgpu_internal_set_error("invalid G3 frozen-batch input");
        return 1;
    }
    FixedFrontG2Cache cache;
    g_active_gradient_group = 3;
    g_active_g2_cache = &cache;
    const int score_code = sgpu_score_coils(
        center_coeffs_x, center_coeffs_y, center_coeffs_z, center_currents_a,
        n_base_coils, n_coeff, nfp, config, center_score_result
    );
    g_active_g2_cache = nullptr;
    g_active_gradient_group = 0;
    if (score_code != 0 || center_score_result->status != SGPU_SCORE_OK ||
        !cache.ready || !cache.field || !cache.alpha.adjoint_ready) {
        if (cache.field) sgpu_destroy_field(cache.field);
        if (score_code == 0 && center_score_result->status != SGPU_SCORE_OK) {
            sgpu_internal_set_error("G3 frozen-batch center score is not ok");
            return 1;
        }
        if (score_code == 0) {
            sgpu_internal_set_error("G3 frozen-batch center cache is incomplete");
            return 1;
        }
        return score_code;
    }

    const size_t coefficient_stride =
        static_cast<size_t>(n_base_coils) * static_cast<size_t>(n_coeff);
    const size_t current_stride = static_cast<size_t>(n_base_coils);
    std::string error;
    for (int query = 0; query < query_count; ++query) {
        if (!evaluate_fixed_front_g3_scalar(
                cache,
                query_coeffs_x + query * coefficient_stride,
                query_coeffs_y + query * coefficient_stride,
                query_coeffs_z + query * coefficient_stride,
                query_currents_a + query * current_stride,
                n_base_coils, n_coeff, nfp, *config, *center_score_result,
                frozen_scores[query], volume_components[query],
                coordinate_components[query], iota_components[query],
                coil_components[query], target_errors[query], qa_errors[query],
                qp_errors[query], iota_minima[query], iota_maxima[query], error)) {
            sgpu_destroy_field(cache.field);
            std::snprintf(
                center_score_result->error_message,
                sizeof(center_score_result->error_message),
                "G3 frozen query %d failed: %.215s",
                query,
                error.c_str()
            );
            center_score_result->status = SGPU_SCORE_INTERNAL_ERROR;
            sgpu_internal_set_error(center_score_result->error_message);
            return 1;
        }
    }
    sgpu_destroy_field(cache.field);
    cache.field = nullptr;
    sgpu_internal_set_error("");
    return 0;
}

int sgpu_score_coils_g4_fixed_branch_batch(
    const double* center_coeffs_x,
    const double* center_coeffs_y,
    const double* center_coeffs_z,
    const double* center_currents_a,
    const double* query_coeffs_x,
    const double* query_coeffs_y,
    const double* query_coeffs_z,
    const double* query_currents_a,
    int query_count,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* center_score_result,
    SgpuScoreResult* query_score_results
) {
    if (!center_coeffs_x || !center_coeffs_y || !center_coeffs_z ||
        !center_currents_a || !query_coeffs_x || !query_coeffs_y ||
        !query_coeffs_z || !query_currents_a || query_count <= 0 ||
        n_base_coils <= 0 || n_coeff <= 0 || nfp <= 0 || !config ||
        !center_score_result || !query_score_results) {
        sgpu_internal_set_error("invalid G4 fixed-branch batch input");
        return 1;
    }
    FixedFrontG2Cache cache;
    g_active_gradient_group = 2;
    g_active_g2_cache = &cache;
    const int score_code = sgpu_score_coils(
        center_coeffs_x, center_coeffs_y, center_coeffs_z, center_currents_a,
        n_base_coils, n_coeff, nfp, config, center_score_result
    );
    g_active_g2_cache = nullptr;
    g_active_gradient_group = 0;
    if (score_code != 0 || center_score_result->status != SGPU_SCORE_OK ||
        !cache.ready || cache.axis.R.empty()) {
        if (cache.field) sgpu_destroy_field(cache.field);
        sgpu_internal_set_error("G4 fixed-branch center score is not ok");
        return score_code != 0 ? score_code : 1;
    }
    if (cache.field) {
        sgpu_destroy_field(cache.field);
        cache.field = nullptr;
    }

    const size_t coefficient_stride = static_cast<size_t>(n_base_coils) * n_coeff;
    for (int query_index = 0; query_index < query_count; ++query_index) {
        std::string error;
        if (!evaluate_fixed_branch_g4_scalar(
                cache,
                query_coeffs_x + query_index * coefficient_stride,
                query_coeffs_y + query_index * coefficient_stride,
                query_coeffs_z + query_index * coefficient_stride,
                query_currents_a + static_cast<size_t>(query_index) * n_base_coils,
                n_base_coils, n_coeff, nfp, *config, *center_score_result,
                query_score_results[query_index], error)) {
            std::snprintf(
                center_score_result->error_message,
                sizeof(center_score_result->error_message),
                "G4 fixed-branch query %d failed: %.205s",
                query_index,
                error.c_str()
            );
            center_score_result->status = SGPU_SCORE_INTERNAL_ERROR;
            sgpu_internal_set_error(center_score_result->error_message);
            return 1;
        }
    }
    sgpu_internal_set_error("");
    return 0;
}

int sgpu_score_coils_psi_warm_batch(
    const double* center_coeffs_x,
    const double* center_coeffs_y,
    const double* center_coeffs_z,
    const double* center_currents_a,
    const double* query_coeffs_x,
    const double* query_coeffs_y,
    const double* query_coeffs_z,
    const double* query_currents_a,
    int query_count,
    int n_base_coils,
    int n_coeff,
    int nfp,
    int warm_iterations,
    int use_center_qr_preconditioner,
    const SgpuScoreConfig* config,
    SgpuScoreResult* center_score_result,
    SgpuScoreResult* query_score_results
) {
    if (!center_coeffs_x || !center_coeffs_y || !center_coeffs_z ||
        !center_currents_a || !query_coeffs_x || !query_coeffs_y ||
        !query_coeffs_z || !query_currents_a || query_count <= 0 ||
        n_base_coils <= 0 || n_coeff <= 0 || nfp <= 0 || warm_iterations < 0 ||
        !config || !center_score_result || !query_score_results) {
        sgpu_internal_set_error("invalid psi warm-batch input");
        return 1;
    }

    sgpu_clear_psi_warm_preconditioner();
    if (use_center_qr_preconditioner) sgpu_set_psi_warm_preconditioner_capture(1);
    FixedFrontG2Cache cache;
    g_active_gradient_group = 2;
    g_active_g2_cache = &cache;
    const int center_code = sgpu_score_coils(
        center_coeffs_x, center_coeffs_y, center_coeffs_z, center_currents_a,
        n_base_coils, n_coeff, nfp, config, center_score_result
    );
    sgpu_set_psi_warm_preconditioner_capture(0);
    g_active_g2_cache = nullptr;
    g_active_gradient_group = 0;
    if (center_code != 0 || center_score_result->status != SGPU_SCORE_OK ||
        !cache.ready || cache.psi.coeffs.empty()) {
        if (cache.field) sgpu_destroy_field(cache.field);
        sgpu_clear_psi_warm_preconditioner();
        sgpu_internal_set_error("psi warm-batch center score is not ok");
        return center_code != 0 ? center_code : 1;
    }
    if (use_center_qr_preconditioner && !sgpu_has_psi_warm_preconditioner(
            static_cast<int>(cache.psi.coeffs.size()))) {
        if (cache.field) sgpu_destroy_field(cache.field);
        sgpu_clear_psi_warm_preconditioner();
        sgpu_internal_set_error("psi warm-batch failed to capture the center QR factor");
        return 1;
    }
    if (cache.field) {
        sgpu_destroy_field(cache.field);
        cache.field = nullptr;
    }

    const size_t coefficient_stride = static_cast<size_t>(n_base_coils) * n_coeff;
    const size_t current_stride = static_cast<size_t>(n_base_coils);
    g_active_psi_warm_coefficients = &cache.psi.coeffs;
    g_active_psi_warm_iterations = warm_iterations;
    g_active_psi_warm_preconditioned = use_center_qr_preconditioner != 0;
    int return_code = 0;
    for (int query = 0; query < query_count; ++query) {
        return_code = sgpu_score_coils(
            query_coeffs_x + query * coefficient_stride,
            query_coeffs_y + query * coefficient_stride,
            query_coeffs_z + query * coefficient_stride,
            query_currents_a + query * current_stride,
            n_base_coils, n_coeff, nfp, config, &query_score_results[query]
        );
        if (return_code != 0) break;
    }
    g_active_psi_warm_coefficients = nullptr;
    g_active_psi_warm_iterations = 0;
    g_active_psi_warm_preconditioned = false;
    sgpu_clear_psi_warm_preconditioner();
    return return_code;
}

int sgpu_score_coils_g3_gradient(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* score_result,
    double* gradient_x,
    double* gradient_y,
    double* gradient_z,
    double* gradient_current,
    SgpuScoreGradientResult* gradient_result
) {
    if (!gradient_result) {
        sgpu_internal_set_error("score gradient result pointer is null");
        return 1;
    }
    std::memset(gradient_result, 0, sizeof(*gradient_result));
    gradient_result->abi_version = SGPU_SCORE_GRADIENT_ABI_VERSION;
    gradient_result->struct_size = sizeof(*gradient_result);
    gradient_result->status = 1;
    gradient_result->gradient_group = 3;
    if (!score_result || !gradient_x || !gradient_y || !gradient_z ||
        !gradient_current || !config || !coeffs_x || !coeffs_y || !coeffs_z ||
        !currents_a) {
        std::snprintf(
            gradient_result->error_message,
            sizeof(gradient_result->error_message),
            "%s",
            "G3 gradient input/output pointer is null"
        );
        sgpu_internal_set_error(gradient_result->error_message);
        return 1;
    }
    FixedFrontG2Cache cache;
    const auto forward_started = Clock::now();
    g_active_gradient_group = 3;
    g_active_g2_cache = &cache;
    const int score_code = sgpu_score_coils(
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp, config, score_result
    );
    g_active_g2_cache = nullptr;
    g_active_gradient_group = 0;
    gradient_result->forward_wall_s = seconds_since(forward_started);
    if (score_code != 0 || score_result->status != SGPU_SCORE_OK || !cache.ready ||
        !cache.field || !cache.alpha.adjoint_ready) {
        if (cache.field) sgpu_destroy_field(cache.field);
        std::snprintf(
            gradient_result->error_message,
            sizeof(gradient_result->error_message),
            "G3 requires an ok complete QR score: %.180s",
            score_result->error_message
        );
        sgpu_internal_set_error(gradient_result->error_message);
        return score_code != 0 ? score_code : 1;
    }
    const auto gradient_started = Clock::now();
    CoilComponentGradient g1;
    FixedFrontG2Gradient g2;
    FixedFrontG2Gradient g3;
    std::string error;
    if (!compute_coil_component_gradient_impl(
            coeffs_x, coeffs_y, coeffs_z, currents_a,
            n_base_coils, n_coeff, nfp, g1, error) ||
        !compute_fixed_front_g2_gradient(
            cache, coeffs_x, coeffs_y, coeffs_z, currents_a,
            n_base_coils, n_coeff, nfp, *config, *score_result, g2) ||
        !compute_alpha_iota_g3_gradient(
            cache, coeffs_x, coeffs_y, coeffs_z, currents_a,
            n_base_coils, n_coeff, nfp, *config, *score_result, g3)) {
        sgpu_destroy_field(cache.field);
        if (!score_result->error_message[0]) {
            std::snprintf(
                gradient_result->error_message,
                sizeof(gradient_result->error_message),
                "%s",
                error.c_str()
            );
        } else {
            std::snprintf(
                gradient_result->error_message,
                sizeof(gradient_result->error_message),
                "%.250s",
                score_result->error_message
            );
        }
        sgpu_internal_set_error(gradient_result->error_message);
        return 1;
    }
    sgpu_destroy_field(cache.field);
    cache.field = nullptr;

    double total_weight = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        total_weight += std::max(config->score_weights[component], 0.0);
    }
    const double g1_scale = total_weight > 0.0
        ? std::max(config->score_weights[SGPU_SCORE_COMPONENT_COIL], 0.0) / total_weight *
            score_result->score_qh_total_iota_factor * score_result->score_qh_total_helicity_factor
        : 0.0;
    const int parameter_count = n_base_coils * n_coeff;
    const std::vector<double>* g1_sources[] = {&g1.x, &g1.y, &g1.z};
    const std::vector<double>* g2_sources[] = {&g2.x, &g2.y, &g2.z};
    const std::vector<double>* g3_sources[] = {&g3.x, &g3.y, &g3.z};
    double* destinations[] = {gradient_x, gradient_y, gradient_z};
    double score_norm2 = 0.0;
    double component_norm2 = 0.0;
    size_t gradient_count = 0;
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        for (int parameter = 0; parameter < parameter_count; ++parameter) {
            const double g1_value = (*g1_sources[coordinate])[parameter];
            destinations[coordinate][parameter] =
                g1_scale * g1_value + (*g2_sources[coordinate])[parameter] +
                (*g3_sources[coordinate])[parameter];
            score_norm2 += destinations[coordinate][parameter] * destinations[coordinate][parameter];
            component_norm2 += g1_value * g1_value;
            ++gradient_count;
        }
    }
    for (int coil = 0; coil < n_base_coils; ++coil) {
        gradient_current[coil] = g1_scale * g1.current[coil] + g2.current[coil] + g3.current[coil];
        score_norm2 += gradient_current[coil] * gradient_current[coil];
        component_norm2 += g1.current[coil] * g1.current[coil];
        ++gradient_count;
    }
    gradient_result->point_vjp_s = g2.point_vjp_s + g3.point_vjp_s;
    gradient_result->field_vjp_s = g2.field_vjp_s + g3.field_vjp_s;
    gradient_result->parameter_map_s = g2.parameter_map_s + g3.parameter_map_s;
    gradient_result->gradient_wall_s = seconds_since(gradient_started);
    gradient_result->score_gradient_rms =
        std::sqrt(score_norm2 / std::max<size_t>(gradient_count, 1));
    gradient_result->coil_component_gradient_rms =
        std::sqrt(component_norm2 / std::max<size_t>(gradient_count, 1));
    gradient_result->status = 0;
    sgpu_internal_set_error("");
    return 0;
}

}  // extern "C"
