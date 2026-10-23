// Retained fixed-front analytical derivatives and frozen-branch value evaluators.
#include "pipeline/types.cuh"
#include "pipeline/device_helpers.cuh"
#include "pipeline/alpha.cuh"
#include "pipeline/axis.cuh"
#include "pipeline/coil_geometry.cuh"
#include "pipeline/common.cuh"
#include "pipeline/evaluate.cuh"
#include "pipeline/gradients.cuh"
#include "pipeline/policy.cuh"
#include "pipeline/psi.cuh"
#include "pipeline/qs.cuh"
#include <cublas_v2.h>
#include "pipeline/surface.cuh"
#include "coil_field_internal.h"

namespace coil_eval::pipeline {

namespace {

struct QsErrorAdjoints {
    double target = 0.0;
    double target_edge = 0.0;
    double qa = 0.0;
    double qp = 0.0;
};

}  // namespace


double helicity_quality_derivative(double advantage, const SgpuScoreConfig& config) {
    double derivative = 0.0;
    const double linear_position = advantage / config.score_qh_helicity_good;
    if (linear_position > 0.0 && linear_position < 1.0) {
        derivative += config.score_qh_helicity_exploration_fraction /
            config.score_qh_helicity_good;
    }
    const double window_position =
        (advantage - config.score_qh_helicity_bad) /
        (config.score_qh_helicity_good - config.score_qh_helicity_bad);
    if (window_position > 0.0 && window_position < 1.0) {
        derivative += (1.0 - config.score_qh_helicity_exploration_fraction) *
            6.0 * window_position * (1.0 - window_position) /
            (config.score_qh_helicity_good - config.score_qh_helicity_bad);
    }
    return derivative;
}

QsErrorAdjoints score_qs_error_adjoints(
    const SgpuScoreConfig& config,
    const SgpuScoreResult& result
) {
    QsErrorAdjoints output;
    double total_weight = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        total_weight += std::max(config.score_weights[component], 0.0);
    }
    if (!(total_weight > 0.0)) return output;
    const double d_global = q_down_derivative(
        result.qs_target_global_error_per_helicity, config.score_qs_global_scale, 0.9
    );
    const double d_edge = q_down_derivative(
        result.qs_target_edge_error_per_helicity, config.score_qs_edge_scale, 0.9
    );
    const double volume_scale = result.score_volume_qs_size_factor *
        result.score_volume_qs_iota_factor;
    const double score_before_scale =
        100.0 * std::max(config.score_weights[SGPU_SCORE_COMPONENT_VOLUME_QS], 0.0) /
        total_weight * volume_scale;
    const double d_before_target = score_before_scale * 0.80 * d_global;
    const double d_before_edge = score_before_scale * 0.20 * d_edge;
    const double iota_gate = result.score_qh_total_iota_factor;
    const double helicity_gate = result.score_qh_total_helicity_factor;
    output.target = iota_gate * helicity_gate * d_before_target;
    output.target_edge = iota_gate * helicity_gate * d_before_edge;

    const bool qh_target = config.target_M != 0 && config.target_N != 0;
    if (!qh_target) return output;
    const double target_error = result.qs_target_global_error_per_helicity;
    const bool qa_competitor =
        result.qs_qa_global_error_per_helicity <= result.qs_qp_global_error_per_helicity;
    const double competitor = qa_competitor
        ? result.qs_qa_global_error_per_helicity
        : result.qs_qp_global_error_per_helicity;
    const double denominator = target_error + competitor;
    if (!(denominator > 1.0e-30)) return output;
    const double raw_advantage = competitor / denominator;
    if (!(raw_advantage > 0.0 && raw_advantage < 1.0)) return output;
    const double d_quality = helicity_quality_derivative(raw_advantage, config);
    const double d_gate_d_advantage =
        (1.0 - config.score_qh_total_helicity_floor) * d_quality;
    const double common = iota_gate * result.score_before_qh_iota_gate *
        d_gate_d_advantage;
    output.target += common * (-competitor / (denominator * denominator));
    const double d_competitor = common * target_error / (denominator * denominator);
    if (qa_competitor) {
        output.qa = d_competitor;
    } else {
        output.qp = d_competitor;
    }
    return output;
}

bool map_fixed_front_field_adjoint(
    FixedFrontG2Cache& cache,
    const DeviceBuffer<float>& d_adj_B,
    const DeviceBuffer<float>& d_adj_grad_B,
    const DeviceBuffer<double>& d_adj_G,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    SgpuScoreResult& result,
    FixedFrontG2Gradient& output
) {
    const int segment_count = sgpu_segment_count(cache.field);
    DeviceBuffer<float> d_adj_segment_position(static_cast<size_t>(segment_count) * 3);
    DeviceBuffer<float> d_adj_segment_weight(static_cast<size_t>(segment_count) * 3);
    if (segment_count <= 0 || !d_adj_segment_position.data() || !d_adj_segment_weight.data()) {
        fail_result(&result, "fixed-front segment-adjoint allocation failed");
        return false;
    }
    auto started = Clock::now();
    if (sgpu_internal_B_grad_segment_vjp_f32_device(
            cache.field, cache.points.xyz.data(), d_adj_B.data(), d_adj_grad_B.data(),
            cache.points.count, d_adj_segment_position.data(), d_adj_segment_weight.data()) ||
        !cuda_stage_ok(cudaDeviceSynchronize(), result, "fixed-front Biot-Savart VJP")) {
        if (!result.error_message[0]) fail_from_backend(&result, "fixed-front Biot-Savart VJP");
        return false;
    }
    output.field_vjp_s = seconds_since(started);

    started = Clock::now();
    std::vector<float> adj_segment_position(static_cast<size_t>(segment_count) * 3);
    std::vector<float> adj_segment_weight(static_cast<size_t>(segment_count) * 3);
    double adj_G = 0.0;
    if (cudaMemcpy(
            adj_segment_position.data(), d_adj_segment_position.data(),
            adj_segment_position.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(
            adj_segment_weight.data(), d_adj_segment_weight.data(),
            adj_segment_weight.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&adj_G, d_adj_G.data(), sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail_result(&result, "fixed-front segment-adjoint copy failed");
        return false;
    }
    const int parameter_count = n_base_coils * n_coeff;
    output.x.assign(parameter_count, 0.0);
    output.y.assign(parameter_count, 0.0);
    output.z.assign(parameter_count, 0.0);
    output.current.assign(n_base_coils, 0.0);
    std::vector<double>* coordinate_outputs[] = {&output.x, &output.y, &output.z};
    int segment = 0;
    for (int coil = 0; coil < n_base_coils; ++coil) {
        for (int sample = 0; sample < config.segments_per_coil; ++sample) {
            const double t = (static_cast<double>(sample) + 0.5) / config.segments_per_coil;
            double px, vx, unused, py, vy, pz, vz;
            eval_fourier_interleaved(
                coeffs_x + static_cast<size_t>(coil) * n_coeff, n_coeff, t, px, vx, unused
            );
            eval_fourier_interleaved(
                coeffs_y + static_cast<size_t>(coil) * n_coeff, n_coeff, t, py, vy, unused
            );
            eval_fourier_interleaved(
                coeffs_z + static_cast<size_t>(coil) * n_coeff, n_coeff, t, pz, vz, unused
            );
            const Vec3d base_velocity{
                vx / config.segments_per_coil,
                vy / config.segments_per_coil,
                vz / config.segments_per_coil,
            };
            for (int period = 0; period < nfp; ++period) {
                for (int reflected = 0; reflected < 2; ++reflected) {
                    if (segment >= segment_count) {
                        fail_result(&result, "fixed-front segment ordering mismatch");
                        return false;
                    }
                    const Vec3d adj_position{
                        adj_segment_position[3 * segment],
                        adj_segment_position[3 * segment + 1],
                        adj_segment_position[3 * segment + 2],
                    };
                    const Vec3d adj_weight{
                        adj_segment_weight[3 * segment],
                        adj_segment_weight[3 * segment + 1],
                        adj_segment_weight[3 * segment + 2],
                    };
                    const double current_sign = reflected ? -1.0 : 1.0;
                    for (int coordinate = 0; coordinate < 3; ++coordinate) {
                        for (int coefficient = 0; coefficient < n_coeff; ++coefficient) {
                            double basis, first, second;
                            fourier_basis(n_coeff, t, coefficient, basis, first, second);
                            const Vec3d d_position = transformed_basis_vector(
                                coordinate, basis, reflected, period, nfp
                            );
                            Vec3d d_weight = transformed_basis_vector(
                                coordinate,
                                first / config.segments_per_coil,
                                reflected,
                                period,
                                nfp
                            );
                            d_weight.x *= current_sign * currents_a[coil];
                            d_weight.y *= current_sign * currents_a[coil];
                            d_weight.z *= current_sign * currents_a[coil];
                            (*coordinate_outputs[coordinate])[coil * n_coeff + coefficient] +=
                                vec_dot(adj_position, d_position) + vec_dot(adj_weight, d_weight);
                        }
                    }
                    Vec3d d_weight_current = base_velocity;
                    if (reflected) {
                        d_weight_current.y = -d_weight_current.y;
                        d_weight_current.z = -d_weight_current.z;
                    }
                    d_weight_current = rotate_z(
                        d_weight_current, TWOPI * static_cast<double>(period) / nfp
                    );
                    d_weight_current.x *= current_sign;
                    d_weight_current.y *= current_sign;
                    d_weight_current.z *= current_sign;
                    output.current[coil] += vec_dot(adj_weight, d_weight_current);
                    ++segment;
                }
            }
        }
    }
    if (segment != segment_count) {
        fail_result(&result, "fixed-front segment count mismatch");
        return false;
    }
    if (adj_G != 0.0) {
        std::vector<double> circulation_per_amp(n_base_coils, 0.0);
        std::vector<double> unit_currents(n_base_coils, 0.0);
        for (int coil = 0; coil < n_base_coils; ++coil) {
            unit_currents[coil] = 1.0;
            void* unit_field = nullptr;
            if (sgpu_create_field(
                    coeffs_x, coeffs_y, coeffs_z, unit_currents.data(),
                    n_base_coils, n_coeff, nfp, config.segments_per_coil,
                    config.device_id, &unit_field)) {
                fail_from_backend(&result, "vacuum-G current derivative field creation");
                return false;
            }
            const bool circulation_ok = magnetic_axis_circulation_native(
                unit_field, cache.axis, nfp, circulation_per_amp[coil]
            );
            sgpu_destroy_field(unit_field);
            unit_currents[coil] = 0.0;
            if (!circulation_ok) {
                fail_from_backend(&result, "vacuum-G current derivative circulation");
                return false;
            }
        }
        double total_circulation = 0.0;
        for (int coil = 0; coil < n_base_coils; ++coil) {
            total_circulation += currents_a[coil] * circulation_per_amp[coil];
        }
        if (std::abs(total_circulation) > 1.0e-30) {
            const double sign = std::copysign(1.0, result.flux_edge) *
                std::copysign(1.0, total_circulation);
            for (int coil = 0; coil < n_base_coils; ++coil) {
                output.current[coil] += adj_G * sign *
                    circulation_per_amp[coil] / TWOPI;
            }
        }
    }
    output.parameter_map_s = seconds_since(started);
    return true;
}

bool compute_fixed_front_g2_gradient(
    FixedFrontG2Cache& cache,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    SgpuScoreResult& result,
    FixedFrontG2Gradient& output
) {
    if (!cache.ready || !cache.field || cache.points.count <= 0) {
        fail_result(&result, "fixed-front G2 cache is incomplete");
        return false;
    }
    const int point_count = cache.points.count;
    DeviceBuffer<float> d_iota;
    DeviceBuffer<float> d_adj_B(static_cast<size_t>(point_count) * 3);
    DeviceBuffer<float> d_adj_grad_B(static_cast<size_t>(point_count) * 9);
    DeviceBuffer<double> d_adj_G(1);
    if (!copy_to_device(d_iota, cache.alpha.iota_coefficients) ||
        !d_adj_B.data() || !d_adj_grad_B.data() || !d_adj_G.data() ||
        cudaMemset(d_adj_G.data(), 0, sizeof(double)) != cudaSuccess) {
        fail_result(&result, "fixed-front G2 point-adjoint allocation failed");
        return false;
    }
    const QsErrorAdjoints errors = score_qs_error_adjoints(config, result);
    const double target_norm = std::max(
        std::hypot(static_cast<double>(config.target_M), static_cast<double>(config.target_N)),
        1.0
    );
    const double qp_norm = std::max(std::abs(nfp), 1);
    const float edge_threshold = static_cast<float>(
        config.volume_rho_min + (1.0 - config.volume_rho_min) *
        (config.radial_bin_count - 1.0) / config.radial_bin_count
    );
    auto started = Clock::now();
    constexpr int threads = 256;
    compute_qs_point_adjoint_kernel<<<(point_count + threads - 1) / threads, threads>>>(
        cache.B.data(), cache.grad_B.data(), cache.points.grad_s.data(),
        cache.points.flux_derivative.data(), cache.points.rho.data(),
        cache.points.volume_weight.data(), point_count, d_iota.data(), config.iota_degree,
        config.target_M, config.target_N, nfp, cache.G, edge_threshold,
        cache.qs_sums[0], cache.qs_sums[2], result.qs_global_error,
        result.qs_edge_error, result.qs_qa_global_error,
        result.qs_qp_global_error_per_helicity * qp_norm,
        errors.target, errors.target_edge, errors.qa, errors.qp,
        d_adj_B.data(), d_adj_grad_B.data(), d_adj_G.data()
    );
    if (!cuda_stage_ok(cudaDeviceSynchronize(), result, "fixed-front QS point VJP")) return false;
    output.point_vjp_s = seconds_since(started);
    return map_fixed_front_field_adjoint(
        cache, d_adj_B, d_adj_grad_B, d_adj_G,
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp, config, result, output
    );
}

bool evaluate_fixed_front_g2_scalar(
    const FixedFrontG2Cache& cache,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    const SgpuScoreResult& center,
    double& frozen_score,
    double& volume_component,
    double& coil_score,
    double& target_error,
    double& qa_error,
    double& qp_error,
    std::string& error
) {
    void* field = nullptr;
    if (sgpu_create_field(
            coeffs_x, coeffs_y, coeffs_z, currents_a, n_base_coils, n_coeff,
            nfp, config.segments_per_coil, config.device_id, &field)) {
        error = "fixed-front query field creation failed";
        return false;
    }
    DeviceBuffer<float> d_B(static_cast<size_t>(cache.points.count) * 3);
    DeviceBuffer<float> d_grad_B(static_cast<size_t>(cache.points.count) * 9);
    SgpuScoreResult query;
    initialize_result(&query, config.device_id);
    bool ok = d_B.data() && d_grad_B.data();
    if (!ok) {
        error = "fixed-front query field allocation failed";
    } else if (sgpu_internal_eval_B_grad_f32_device(
                   field, cache.points.xyz.data(), d_B.data(), d_grad_B.data(),
                   cache.points.count) ||
               !cuda_stage_ok(cudaDeviceSynchronize(), query, "fixed-front query B/grad(B)")) {
        error = query.error_message[0]
            ? query.error_message : "fixed-front query B/grad(B) failed";
        ok = false;
    } else {
        double G = 0.0;
        if (!vacuum_G_native(field, cache.axis, nfp, center.flux_edge, G)) {
            error = "fixed-front query magnetic-axis circulation failed";
            ok = false;
        } else if (!compute_qs_metric_native(
                       cache.points, d_B.data(), d_grad_B.data(), cache.alpha,
                       nfp, G, config, query, nullptr)) {
            error = query.error_message[0]
                ? query.error_message : "fixed-front query QS metric failed";
            ok = false;
        }
    }
    sgpu_destroy_field(field);
    if (!ok) return false;

    const CoilMetrics metrics = compute_coil_metrics(
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp
    );
    const double coil_unit = coil_component(metrics);
    const double global_score = q_down(
        query.qs_target_global_error_per_helicity,
        config.score_qs_global_scale,
        0.9
    );
    const double edge_score = q_down(
        query.qs_target_edge_error_per_helicity,
        config.score_qs_edge_scale,
        0.9,
        global_score
    );
    const double residual_score = blend({{0.80, global_score}, {0.20, edge_score}});
    const double volume_unit = residual_score *
        center.score_volume_qs_size_factor * center.score_volume_qs_iota_factor;
    const bool qh_target = config.target_M != 0 && config.target_N != 0;
    const double competitor_error = std::min(
        query.qs_qa_global_error_per_helicity,
        query.qs_qp_global_error_per_helicity
    );
    const double advantage = !qh_target ? 1.0 : competitor_error / std::max(
        query.qs_target_global_error_per_helicity + competitor_error,
        1.0e-300
    );
    const double helicity_position = clip01(
        (advantage - config.score_qh_helicity_bad) /
        (config.score_qh_helicity_good - config.score_qh_helicity_bad)
    );
    const double helicity_linear = clip01(
        advantage / config.score_qh_helicity_good
    );
    const double helicity_window = helicity_position * helicity_position *
        (3.0 - 2.0 * helicity_position);
    const double helicity_quality = !qh_target ? 1.0 :
        config.score_qh_helicity_exploration_fraction * helicity_linear +
        (1.0 - config.score_qh_helicity_exploration_fraction) * helicity_window;
    const double helicity_factor = !qh_target ? 1.0 :
        config.score_qh_total_helicity_floor +
        (1.0 - config.score_qh_total_helicity_floor) * helicity_quality;

    double total_weight = 0.0;
    double weighted_score = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        const double weight = std::max(config.score_weights[component], 0.0);
        double value = center.components[component] / 100.0;
        if (component == SGPU_SCORE_COMPONENT_VOLUME_QS) value = volume_unit;
        if (component == SGPU_SCORE_COMPONENT_COIL) value = coil_unit;
        total_weight += weight;
        weighted_score += weight * clip01(value);
    }
    const double score_before = total_weight > 0.0
        ? 100.0 * clip01(weighted_score / total_weight) : 0.0;
    frozen_score = score_before * center.score_qh_total_iota_factor * helicity_factor;
    volume_component = 100.0 * clip01(volume_unit);
    coil_score = 100.0 * clip01(coil_unit);
    target_error = query.qs_target_global_error_per_helicity;
    qa_error = query.qs_qa_global_error_per_helicity;
    qp_error = query.qs_qp_global_error_per_helicity;
    return true;
}

bool evaluate_fixed_front_g3_scalar(
    const FixedFrontG2Cache& cache,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    const SgpuScoreResult& center,
    double& frozen_score,
    double& volume_component,
    double& coordinate_component,
    double& iota_component,
    double& coil_score,
    double& target_error,
    double& qa_error,
    double& qp_error,
    double& iota_minimum,
    double& iota_maximum,
    std::string& error
) {
    void* field = nullptr;
    if (sgpu_create_field(
            coeffs_x, coeffs_y, coeffs_z, currents_a, n_base_coils, n_coeff,
            nfp, config.segments_per_coil, config.device_id, &field)) {
        error = "fixed-front G3 query field creation failed";
        return false;
    }
    DeviceBuffer<float> d_B(static_cast<size_t>(cache.points.count) * 3);
    DeviceBuffer<float> d_grad_B(static_cast<size_t>(cache.points.count) * 9);
    SgpuScoreResult query;
    initialize_result(&query, config.device_id);
    AlphaFitNative alpha;
    bool ok = d_B.data() && d_grad_B.data();
    if (!ok) {
        error = "fixed-front G3 query field allocation failed";
    } else if (sgpu_internal_eval_B_grad_f32_device(
                   field, cache.points.xyz.data(), d_B.data(), d_grad_B.data(),
                   cache.points.count) ||
               !cuda_stage_ok(cudaDeviceSynchronize(), query, "fixed-front G3 query B/grad(B)")) {
        error = query.error_message[0]
            ? query.error_message : "fixed-front G3 query B/grad(B) failed";
        ok = false;
    } else if (!fit_alpha_native(cache.points, d_B.data(), nfp, config, alpha, query)) {
        error = query.error_message[0]
            ? query.error_message : "fixed-front G3 query alpha/iota fit failed";
        ok = false;
    } else {
        double G = 0.0;
        if (!vacuum_G_native(field, cache.axis, nfp, center.flux_edge, G)) {
            error = "fixed-front G3 query magnetic-axis circulation failed";
            ok = false;
        } else if (!compute_qs_metric_native(
                       cache.points, d_B.data(), d_grad_B.data(), alpha,
                       nfp, G, config, query, nullptr)) {
            error = query.error_message[0]
                ? query.error_message : "fixed-front G3 query QS metric failed";
            ok = false;
        }
    }
    sgpu_destroy_field(field);
    if (!ok) return false;

    iota_minimum = std::numeric_limits<double>::infinity();
    iota_maximum = -std::numeric_limits<double>::infinity();
    const double u_min = config.volume_rho_min * config.volume_rho_min;
    for (int sample = 0; sample <= 256; ++sample) {
        const double u = u_min + (1.0 - u_min) * sample / 256.0;
        double iota = 0.0;
        double power = 1.0;
        for (float coefficient : alpha.iota_coefficients) {
            iota += coefficient * power;
            power *= u;
        }
        iota_minimum = std::min(iota_minimum, iota);
        iota_maximum = std::max(iota_maximum, iota);
    }

    const bool qh_target = config.target_M != 0 && config.target_N != 0;
    const double iota_unit = !qh_target ? 1.0 : std::pow(
        clip01(minimum_absolute_iota(iota_minimum, iota_maximum) /
               config.score_qh_iota_threshold),
        config.score_qh_iota_power
    );
    const double iota_factor = !qh_target ? 1.0 :
        config.score_volume_qs_iota_floor +
        (1.0 - config.score_volume_qs_iota_floor) * iota_unit;
    const double total_iota_factor = !qh_target ? 1.0 :
        config.score_qh_total_iota_floor +
        (1.0 - config.score_qh_total_iota_floor) * iota_unit;
    const double global_score = q_down(
        query.qs_target_global_error_per_helicity,
        config.score_qs_global_scale,
        0.9
    );
    const double edge_score = q_down(
        query.qs_target_edge_error_per_helicity,
        config.score_qs_edge_scale,
        0.9,
        global_score
    );
    const double residual_score = blend({{0.80, global_score}, {0.20, edge_score}});
    const double volume_unit = residual_score *
        center.score_volume_qs_size_factor * iota_factor;
    const double alpha_score = q_down(
        alpha.relative_l2, config.score_alpha_relative_l2_scale, 1.0
    );
    const double center_alpha_score = q_down(
        center.alpha_relative_l2, config.score_alpha_relative_l2_scale, 1.0
    );
    const double coordinate_unit = center.components[SGPU_SCORE_COMPONENT_COORDINATE] / 100.0 +
        0.20 * (alpha_score - center_alpha_score);
    const CoilMetrics metrics = compute_coil_metrics(
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp
    );
    const double coil_unit = coil_component(metrics);
    const double competitor_error = std::min(
        query.qs_qa_global_error_per_helicity,
        query.qs_qp_global_error_per_helicity
    );
    const double advantage = !qh_target ? 1.0 : competitor_error / std::max(
        query.qs_target_global_error_per_helicity + competitor_error,
        1.0e-300
    );
    const double helicity_position = clip01(
        (advantage - config.score_qh_helicity_bad) /
        (config.score_qh_helicity_good - config.score_qh_helicity_bad)
    );
    const double helicity_linear = clip01(
        advantage / config.score_qh_helicity_good
    );
    const double helicity_window = helicity_position * helicity_position *
        (3.0 - 2.0 * helicity_position);
    const double helicity_quality = !qh_target ? 1.0 :
        config.score_qh_helicity_exploration_fraction * helicity_linear +
        (1.0 - config.score_qh_helicity_exploration_fraction) * helicity_window;
    const double helicity_factor = !qh_target ? 1.0 :
        config.score_qh_total_helicity_floor +
        (1.0 - config.score_qh_total_helicity_floor) * helicity_quality;

    double total_weight = 0.0;
    double weighted_score = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        const double weight = std::max(config.score_weights[component], 0.0);
        double value = center.components[component] / 100.0;
        if (component == SGPU_SCORE_COMPONENT_COORDINATE) value = coordinate_unit;
        if (component == SGPU_SCORE_COMPONENT_VOLUME_QS) value = volume_unit;
        if (component == SGPU_SCORE_COMPONENT_IOTA) value = iota_unit;
        if (component == SGPU_SCORE_COMPONENT_COIL) value = coil_unit;
        total_weight += weight;
        weighted_score += weight * clip01(value);
    }
    const double score_before = total_weight > 0.0
        ? 100.0 * clip01(weighted_score / total_weight) : 0.0;
    frozen_score = score_before * total_iota_factor * helicity_factor;
    volume_component = 100.0 * clip01(volume_unit);
    coordinate_component = 100.0 * clip01(coordinate_unit);
    iota_component = 100.0 * clip01(iota_unit);
    coil_score = 100.0 * clip01(coil_unit);
    target_error = query.qs_target_global_error_per_helicity;
    qa_error = query.qs_qa_global_error_per_helicity;
    qp_error = query.qs_qp_global_error_per_helicity;
    return true;
}

bool evaluate_fixed_branch_g4_scalar(
    const FixedFrontG2Cache& cache,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    const SgpuScoreResult& center,
    SgpuScoreResult& query,
    std::string& error
) {
    initialize_result(&query, config.device_id);
    const auto total_started = Clock::now();
    const CoilMetrics coil = compute_coil_metrics(
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp
    );
    query.coil_length_mean = coil.length_mean;
    query.coil_curvature_p95 = coil.curvature_p95;
    query.coil_curvature_max = coil.curvature_max;
    query.coil_min_intercoil_distance = coil.min_intercoil_distance;
    query.coil_min_axis_distance = coil.min_axis_distance;
    query.coil_high_mode_energy_fraction = coil.high_mode_fraction;
    query.coil_current_abs_max_a = coil.current_abs_max;
    query.axis_R = center.axis_R;
    query.axis_Z = center.axis_Z;
    query.axis_residual = center.axis_residual;
    query.axis_topology_trace = center.axis_topology_trace;
    query.axis_topology_det = center.axis_topology_det;
    query.axis_ellipse_aspect = center.axis_ellipse_aspect;
    query.axis_candidate_count = center.axis_candidate_count;
    query.stable_surface_count = center.stable_surface_count;
    query.stage_completed = SCORE_STAGE_AXIS;

    void* field = nullptr;
    auto started = Clock::now();
    if (sgpu_create_field(
            coeffs_x, coeffs_y, coeffs_z, currents_a, n_base_coils, n_coeff,
            nfp, config.segments_per_coil, config.device_id, &field)) {
        error = "fixed-branch G4 query field creation failed";
        return false;
    }
    query.timings[SGPU_SCORE_TIME_FIELD_CREATE] = seconds_since(started);
    query.stage_completed = SCORE_STAGE_FIELD;

    PsiData psi;
    bool ok = fit_psi_native(
        field, cache.axis, nfp, config, psi,
        query.timings[SGPU_SCORE_TIME_PSI_POINTS],
        query.timings[SGPU_SCORE_TIME_PSI_FIT]
    );
    if (!ok) {
        error = "fixed-branch G4 query psi fit failed";
    } else {
        started = Clock::now();
        ok = validate_psi_native(field, cache.axis, nfp, config, psi);
        query.timings[SGPU_SCORE_TIME_PSI_VALIDATE] = seconds_since(started);
        if (!ok) error = "fixed-branch G4 query psi validation failed";
    }
    if (ok) {
        query.psi_train_rms = psi.train_rms;
        query.psi_angle_mean = psi.angle_mean;
        query.psi_angle_p95 = psi.angle_p95;
        query.psi_angle_l2 = psi.angle_l2;
        query.stage_completed = SCORE_STAGE_PSI;

        SurfaceScreen selected;
        selected.level = center.surface_level;
        selected.relative_drift_p95 = center.surface_drift_relative_p95;
        selected.one_period_relative_drift_p95 = center.surface_one_period_drift_relative_p95;
        selected.long_trace_periods_completed = center.surface_long_trace_periods_completed;
        selected.stable = true;
        selected.strict = true;
        selected.verified = true;
        selected.long_verified = true;
        std::vector<SurfaceScreen> screens{selected};
        ok = run_downstream_gpu(
            field, currents_a, n_base_coils, nfp, cache.axis, psi,
            screens, config, query, false
        );
        fill_early_components(config, coil, cache.axis, psi, screens, query);

        // Axis, drift and discrete surface-count terms define the fixed branch.
        // Only the continuously recomputed flux volume changes the surface-size term.
        query.components[SGPU_SCORE_COMPONENT_AXIS] =
            center.components[SGPU_SCORE_COMPONENT_AXIS] / 100.0;
        query.components[SGPU_SCORE_COMPONENT_SURFACE] =
            center.components[SGPU_SCORE_COMPONENT_SURFACE] / 100.0 +
            0.65 * (query.score_surface_size - center.score_surface_size);
        if (!ok && query.status == SGPU_SCORE_INTERNAL_ERROR) {
            error = query.error_message[0]
                ? query.error_message : "fixed-branch G4 downstream query failed";
        }
    }
    sgpu_destroy_field(field);
    if (!ok && query.status == SGPU_SCORE_INTERNAL_ERROR) return false;

    if (query.status == SGPU_SCORE_OK &&
        center.volume_candidate_count == center.volume_available_count &&
        query.volume_candidate_count != query.volume_available_count) {
        query.status = SGPU_SCORE_FLUX_REJECTED;
        std::snprintf(
            query.error_message,
            sizeof(query.error_message),
            "%s",
            "fixed-branch G4 volume active set changed"
        );
    }
    started = Clock::now();
    finalize_score(config, query);
    query.timings[SGPU_SCORE_TIME_SCORE] = seconds_since(started);
    query.timings[SGPU_SCORE_TIME_TOTAL] = seconds_since(total_started);
    return true;
}

void add_iota_score_solution_adjoint(
    const AlphaFitNative& alpha,
    const SgpuScoreConfig& config,
    const SgpuScoreResult& result,
    std::vector<double>& adj_solution
) {
    if (config.target_M == 0 || config.target_N == 0 || alpha.iota_coefficients.empty()) return;
    double iota_min = std::numeric_limits<double>::infinity();
    double iota_max = -std::numeric_limits<double>::infinity();
    double u_at_min = 0.0;
    double u_at_max = 0.0;
    const double u_min = config.volume_rho_min * config.volume_rho_min;
    for (int sample = 0; sample <= 256; ++sample) {
        const double u = u_min + (1.0 - u_min) * sample / 256.0;
        double value = 0.0;
        double power = 1.0;
        for (float coefficient : alpha.iota_coefficients) {
            value += coefficient * power;
            power *= u;
        }
        if (value < iota_min) {
            iota_min = value;
            u_at_min = u;
        }
        if (value > iota_max) {
            iota_max = value;
            u_at_max = u;
        }
    }
    if (iota_min <= 0.0 && iota_max >= 0.0) return;
    const bool use_min = std::abs(iota_min) <= std::abs(iota_max);
    const double active_iota = use_min ? iota_min : iota_max;
    const double active_u = use_min ? u_at_min : u_at_max;
    const double magnitude = std::abs(active_iota);
    if (!(magnitude > 0.0) || magnitude >= config.score_qh_iota_threshold) return;
    const double ratio = magnitude / config.score_qh_iota_threshold;
    const double d_iota_score_d_magnitude = config.score_qh_iota_power *
        std::pow(ratio, config.score_qh_iota_power - 1.0) /
        config.score_qh_iota_threshold;
    double total_weight = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        total_weight += std::max(config.score_weights[component], 0.0);
    }
    if (!(total_weight > 0.0)) return;
    const double iota_component_active =
        result.components[SGPU_SCORE_COMPONENT_IOTA] > 0.0 &&
        result.components[SGPU_SCORE_COMPONENT_IOTA] < 100.0;
    const double volume_component_active =
        result.components[SGPU_SCORE_COMPONENT_VOLUME_QS] > 0.0 &&
        result.components[SGPU_SCORE_COMPONENT_VOLUME_QS] < 100.0;
    const double d_before = 100.0 / total_weight * (
        (iota_component_active ? std::max(config.score_weights[SGPU_SCORE_COMPONENT_IOTA], 0.0) : 0.0) +
        (volume_component_active
            ? std::max(config.score_weights[SGPU_SCORE_COMPONENT_VOLUME_QS], 0.0) *
                result.score_qs_residual * result.score_volume_qs_size_factor *
                (1.0 - config.score_volume_qs_iota_floor)
            : 0.0)
    );
    const double d_score_d_iota_score = result.score_qh_total_helicity_factor * (
        result.score_qh_total_iota_factor * d_before +
        result.score_before_qh_iota_gate * (1.0 - config.score_qh_total_iota_floor)
    );
    const double active_scale = d_score_d_iota_score * d_iota_score_d_magnitude *
        std::copysign(1.0, active_iota);
    double power = 1.0;
    for (size_t degree = 0; degree < alpha.iota_coefficients.size(); ++degree) {
        adj_solution[alpha.mode_count + degree] += active_scale * power;
        power *= active_u;
    }
}

bool compute_alpha_iota_g3_gradient(
    FixedFrontG2Cache& cache,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    SgpuScoreResult& result,
    FixedFrontG2Gradient& output
) {
    AlphaFitNative& alpha = cache.alpha;
    if (!cache.ready || !cache.field || !alpha.adjoint_ready || config.alpha_solver_mode != 2 ||
        alpha.fit_count <= 0 || alpha.column_count <= 0 || alpha.qr_rows <= 0) {
        fail_result(&result, "G3 alpha QR adjoint cache is incomplete");
        return false;
    }
    const auto started = Clock::now();
    constexpr int threads = 256;
    DeviceBuffer<float> d_iota;
    DeviceBuffer<double> d_adj_iota(config.iota_degree + 1);
    if (!copy_to_device(d_iota, alpha.iota_coefficients) || !d_adj_iota.data() ||
        cudaMemset(d_adj_iota.data(), 0, d_adj_iota.size() * sizeof(double)) != cudaSuccess) {
        fail_result(&result, "G3 iota adjoint allocation failed");
        return false;
    }
    const QsErrorAdjoints errors = score_qs_error_adjoints(config, result);
    const float edge_threshold = static_cast<float>(
        config.volume_rho_min + (1.0 - config.volume_rho_min) *
        (config.radial_bin_count - 1.0) / config.radial_bin_count
    );
    compute_qs_iota_adjoint_kernel<<<
        (cache.points.count + threads - 1) / threads, threads
    >>>(
        cache.B.data(), cache.grad_B.data(), cache.points.grad_s.data(),
        cache.points.flux_derivative.data(), cache.points.rho.data(),
        cache.points.volume_weight.data(), cache.points.count, d_iota.data(),
        config.iota_degree, config.target_M, config.target_N, cache.G, edge_threshold,
        cache.qs_sums[0], cache.qs_sums[2], result.qs_global_error,
        result.qs_edge_error, result.qs_qa_global_error,
        errors.target, errors.target_edge, errors.qa, d_adj_iota.data()
    );
    std::vector<double> adj_iota(config.iota_degree + 1, 0.0);
    if (!cuda_stage_ok(cudaMemcpy(
            adj_iota.data(), d_adj_iota.data(),
            adj_iota.size() * sizeof(double), cudaMemcpyDeviceToHost
        ), result, "G3 iota adjoint copy")) {
        return false;
    }
    std::vector<double> adj_solution(alpha.column_count, 0.0);
    for (size_t degree = 0; degree < adj_iota.size(); ++degree) {
        adj_solution[alpha.mode_count + degree] = adj_iota[degree];
    }
    add_iota_score_solution_adjoint(alpha, config, result, adj_solution);

    double total_weight = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        total_weight += std::max(config.score_weights[component], 0.0);
    }
    const bool coordinate_active = result.components[SGPU_SCORE_COMPONENT_COORDINATE] > 0.0 &&
        result.components[SGPU_SCORE_COMPONENT_COORDINATE] < 100.0;
    const double coordinate_scale = coordinate_active && total_weight > 0.0
        ? result.score_qh_total_iota_factor * result.score_qh_total_helicity_factor *
            100.0 * std::max(config.score_weights[SGPU_SCORE_COMPONENT_COORDINATE], 0.0) /
            total_weight
        : 0.0;
    const float adj_relative = static_cast<float>(
        coordinate_scale * 0.20 * q_down_derivative(
            result.alpha_relative_l2, config.score_alpha_relative_l2_scale, 1.0
        )
    );
    // The normal-field residual is covariant with the fitted psi geometry. Its
    // frozen-surface partial is singular near B.normal = 0 and must be added
    // together with the compensating psi/surface-motion terms in G4.
    constexpr float adj_normal = 0.0f;
    std::vector<float> adj_solution_f(alpha.column_count);
    std::transform(adj_solution.begin(), adj_solution.end(), adj_solution_f.begin(), [](double value) {
        return static_cast<float>(value);
    });
    DeviceBuffer<float> d_adj_solution;
    DeviceBuffer<float> d_adj_residual(alpha.fit_count), d_adj_rhs(alpha.fit_count);
    DeviceBuffer<float> d_adj_scaled_solution(alpha.column_count);
    DeviceBuffer<float> d_multiplier(alpha.column_count);
    DeviceBuffer<float> d_matrix_multiplier(alpha.fit_count);
    DeviceBuffer<float> d_dot_adjoint_matrix(alpha.column_count);
    DeviceBuffer<float> d_adj_weight(alpha.fit_count), d_adj_b_theta(alpha.fit_count), d_adj_b_phi(alpha.fit_count);
    DeviceBuffer<int> d_mode_m, d_mode_n, d_mode_kind;
    DeviceBuffer<float> d_radial_coefficients;
    if (!copy_to_device(d_adj_solution, adj_solution_f) ||
        !copy_to_device(d_mode_m, alpha.modes.m) || !copy_to_device(d_mode_n, alpha.modes.n) ||
        !copy_to_device(d_mode_kind, alpha.modes.kind) ||
        !copy_to_device(d_radial_coefficients, alpha.modes.radial_coefficients) ||
        !d_adj_residual.data() || !d_adj_rhs.data() || !d_adj_scaled_solution.data() ||
        !d_multiplier.data() || !d_matrix_multiplier.data() || !d_dot_adjoint_matrix.data() ||
        !d_adj_weight.data() || !d_adj_b_theta.data() || !d_adj_b_phi.data()) {
        fail_result(&result, "G3 alpha LS adjoint allocation failed");
        return false;
    }
    alpha_residual_adjoint_kernel<<<(alpha.fit_count + threads - 1) / threads, threads>>>(
        alpha.residual.data(), alpha.rhs_reference.data(), alpha.fit_count,
        static_cast<float>(alpha.residual_norm), static_cast<float>(alpha.rhs_norm),
        adj_relative, d_adj_residual.data(), d_adj_rhs.data()
    );
    cublasHandle_t blas = nullptr;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
        fail_result(&result, "G3 cuBLAS initialization failed");
        return false;
    }
    const float one = 1.0f;
    const float zero = 0.0f;
    cublasStatus_t status = cublasSgemv(
        blas, CUBLAS_OP_T, alpha.fit_count, alpha.column_count,
        &one, alpha.matrix_reference.data(), alpha.fit_count,
        d_adj_residual.data(), 1, &zero, d_adj_scaled_solution.data(), 1
    );
    add_alpha_solution_adjoint_kernel<<<
        (alpha.column_count + threads - 1) / threads, threads
    >>>(
        d_adj_solution.data(), alpha.scales.data(), alpha.column_count,
        d_adj_scaled_solution.data()
    );
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasScopy(
            blas, alpha.column_count, d_adj_scaled_solution.data(), 1,
            d_multiplier.data(), 1
        );
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasStrsv(
            blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            alpha.column_count, alpha.qr_matrix.data(), alpha.qr_rows,
            d_multiplier.data(), 1
        );
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasStrsv(
            blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            alpha.column_count, alpha.qr_matrix.data(), alpha.qr_rows,
            d_multiplier.data(), 1
        );
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasSgemv(
            blas, CUBLAS_OP_N, alpha.fit_count, alpha.column_count,
            &one, alpha.matrix_reference.data(), alpha.fit_count,
            d_multiplier.data(), 1, &zero, d_matrix_multiplier.data(), 1
        );
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasSaxpy(
            blas, alpha.fit_count, &one, d_matrix_multiplier.data(), 1,
            d_adj_rhs.data(), 1
        );
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(blas);
        fail_result(&result, "G3 alpha QR triangular adjoint failed");
        return false;
    }
    alpha_column_dot_kernel<<<alpha.column_count, threads, threads * sizeof(double)>>>(
        alpha.matrix_reference.data(), alpha.fit_count, d_adj_residual.data(),
        alpha.residual.data(), d_multiplier.data(), d_matrix_multiplier.data(),
        alpha.scaled_solution.data(), d_dot_adjoint_matrix.data()
    );
    alpha_row_adjoint_kernel<<<(alpha.fit_count + threads - 1) / threads, threads>>>(
        alpha.matrix_reference.data(), alpha.fit_count, alpha.column_count,
        d_adj_residual.data(), alpha.residual.data(), d_adj_rhs.data(),
        d_multiplier.data(), d_matrix_multiplier.data(), alpha.scaled_solution.data(),
        d_adj_solution.data(), alpha.scales.data(), d_dot_adjoint_matrix.data(),
        d_mode_m.data(), d_mode_n.data(), d_mode_kind.data(), d_radial_coefficients.data(),
        alpha.mode_count, config.alpha_radial_order, config.iota_degree, nfp,
        alpha.rho.data(), alpha.theta.data(), alpha.phi.data(), alpha.b_theta.data(),
        alpha.b_phi.data(), alpha.weights.data(), d_adj_weight.data(),
        d_adj_b_theta.data(), d_adj_b_phi.data()
    );
    float adj_weight_dot_weight = 0.0f;
    status = cublasSdot(
        blas, alpha.fit_count, d_adj_weight.data(), 1,
        alpha.weights.data(), 1, &adj_weight_dot_weight
    );
    cublasDestroy(blas);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fail_result(&result, "G3 alpha weight normalization adjoint failed");
        return false;
    }
    DeviceBuffer<float> d_adj_B(static_cast<size_t>(cache.points.count) * 3);
    DeviceBuffer<float> d_adj_grad_B(static_cast<size_t>(cache.points.count) * 9);
    DeviceBuffer<double> d_adj_G(1);
    if (!d_adj_B.data() || !d_adj_grad_B.data() || !d_adj_G.data() ||
        cudaMemset(d_adj_B.data(), 0, d_adj_B.size() * sizeof(float)) != cudaSuccess ||
        cudaMemset(d_adj_grad_B.data(), 0, d_adj_grad_B.size() * sizeof(float)) != cudaSuccess ||
        cudaMemset(d_adj_G.data(), 0, sizeof(double)) != cudaSuccess) {
        fail_result(&result, "G3 alpha field-adjoint allocation failed");
        return false;
    }
    const float weight_scale = static_cast<float>(
        std::sqrt(alpha.fit_count / std::max(alpha.weight_norm2, 1.0e-300))
    );
    alpha_B_adjoint_kernel<<<(alpha.fit_count + threads - 1) / threads, threads>>>(
        cache.B.data(), cache.points.grad_s.data(), cache.points.grad_theta.data(),
        cache.points.grad_phi.data(), cache.points.count, alpha.fit_count,
        alpha.weights.data(), d_adj_weight.data(), d_adj_b_theta.data(), d_adj_b_phi.data(),
        weight_scale, adj_weight_dot_weight, adj_normal,
        static_cast<float>(alpha.normal_B_relative_l2),
        static_cast<float>(alpha.normal_numerator),
        static_cast<float>(alpha.field_denominator), d_adj_B.data()
    );
    if (!cuda_stage_ok(cudaDeviceSynchronize(), result, "G3 alpha/iota point VJP")) return false;
    output.point_vjp_s = seconds_since(started);
    return map_fixed_front_field_adjoint(
        cache, d_adj_B, d_adj_grad_B, d_adj_G,
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp, config, result, output
    );
}


}  // namespace coil_eval::pipeline
