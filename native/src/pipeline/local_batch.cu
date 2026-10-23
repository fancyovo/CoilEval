// Local endpoint pipeline using the captured center and its preconditioner.
#include "pipeline/types.cuh"
#include "pipeline/device_helpers.cuh"
#include "pipeline/session_state.cuh"
#include "pipeline/alpha.cuh"
#include "pipeline/common.cuh"
#include "pipeline/flux.cuh"
#include "pipeline/stokes.cuh"
#include "pipeline/local_batch.cuh"
#include "pipeline/local_kernels.cuh"
#include "pipeline/policy.cuh"
#include "pipeline/psi.cuh"
#include "pipeline/surface.cuh"
#include "pipeline/target_volume.cuh"
#include <cublas_v2.h>
#include "pipeline/axis.cuh"
#include "pipeline/coil_geometry.cuh"
#include "coil_field_internal.h"

namespace coil_eval::pipeline {

bool vacuum_G_batch_native(
    void* batch_field,
    const double* axis_R,
    const double* axis_Z,
    const double* axis_R_phi,
    const double* axis_Z_phi,
    int query_count,
    int axis_count,
    int nfp,
    const std::vector<float>& edge_flux,
    std::vector<double>& G
) {
    if (!batch_field || !axis_R || !axis_Z || !axis_R_phi || !axis_Z_phi ||
        query_count <= 0 || axis_count < 16 || nfp <= 0 ||
        static_cast<int>(edge_flux.size()) != query_count) {
        sgpu_internal_set_error("invalid local-batch magnetic-axis circulation input");
        return false;
    }
    const size_t total = static_cast<size_t>(query_count) * axis_count;
    std::vector<float> xyz(total * 3);
    std::vector<float> B(xyz.size());
    for (int query = 0; query < query_count; ++query) {
        for (int index = 0; index < axis_count; ++index) {
            const size_t flat = static_cast<size_t>(query) * axis_count + index;
            const double phi = TWOPI * index / static_cast<double>(nfp * axis_count);
            const double cosine = std::cos(phi);
            const double sine = std::sin(phi);
            xyz[3 * flat] = static_cast<float>(axis_R[flat] * cosine);
            xyz[3 * flat + 1] = static_cast<float>(axis_R[flat] * sine);
            xyz[3 * flat + 2] = static_cast<float>(axis_Z[flat]);
        }
    }
    if (sgpu_batch_eval_B_f32(batch_field, xyz.data(), B.data(), axis_count)) return false;
    G.resize(query_count);
    for (int query = 0; query < query_count; ++query) {
        if (!std::isfinite(edge_flux[query]) || edge_flux[query] == 0.0f) {
            sgpu_internal_set_error("local-batch vacuum G requires nonzero signed toroidal flux");
            return false;
        }
        double sum = 0.0;
        for (int index = 0; index < axis_count; ++index) {
            const size_t flat = static_cast<size_t>(query) * axis_count + index;
            const double phi = TWOPI * index / static_cast<double>(nfp * axis_count);
            const double cosine = std::cos(phi);
            const double sine = std::sin(phi);
            const double tangent_x = axis_R_phi[flat] * cosine - axis_R[flat] * sine;
            const double tangent_y = axis_R_phi[flat] * sine + axis_R[flat] * cosine;
            sum += B[3 * flat] * tangent_x + B[3 * flat + 1] * tangent_y +
                B[3 * flat + 2] * axis_Z_phi[flat];
        }
        const double circulation = TWOPI * sum / axis_count;
        if (!std::isfinite(circulation)) {
            sgpu_internal_set_error("local-batch magnetic-axis circulation is not finite");
            return false;
        }
        G[query] = std::copysign(std::abs(circulation) / TWOPI, edge_flux[query]);
    }
    return true;
}

bool score_local_batch_impl(
    void* batch_field,
    const double* currents_a,
    int n_base_coils,
    int nfp,
    const double* axis_R_host,
    const double* axis_Z_host,
    const double* axis_R_phi_host,
    const double* axis_Z_phi_host,
    int axis_count,
    const double* axis_residual,
    const double* axis_topology_trace,
    const double* axis_topology_det,
    const double* psi_coefficients_host,
    int psi_coefficient_count,
    const double* psi_train_rms,
    const double* coil_components,
    const SgpuScoreResult& center,
    const SgpuScoreConfig& config,
    int surface_theta_count,
    int alpha_iterations,
    SgpuScoreResult* results,
    double* stats_out,
    int stats_len
) {
    using LocalClock = std::chrono::steady_clock;
    const auto total_started = LocalClock::now();
    const int query_count = sgpu_internal_batch_query_count(batch_field);
    if (query_count <= 0 || !currents_a || n_base_coils <= 0 || nfp <= 0 || !axis_R_host ||
        !axis_Z_host || !axis_R_phi_host || !axis_Z_phi_host || axis_count <= 1 ||
        !axis_residual || !axis_topology_trace || !axis_topology_det ||
        !psi_coefficients_host || psi_coefficient_count <= 0 || !psi_train_rms ||
        !coil_components || surface_theta_count < 8 || alpha_iterations < 0 || !results) {
        sgpu_internal_set_error("invalid local batch score input");
        return false;
    }
    const PsiModes psi_modes = build_psi_modes(config.psi_poly_degree, config.psi_m_tor);
    if (static_cast<int>(psi_modes.a.size()) != psi_coefficient_count) {
        sgpu_internal_set_error("local batch psi mode count mismatch");
        return false;
    }
    const int alpha_column_count = static_cast<int>(
        build_clebsch_modes_native(
            config.alpha_radial_order, config.alpha_poloidal_order,
            config.alpha_toroidal_order
        ).m.size()
    ) + config.iota_degree + 1;
    if (g_alpha_warm_preconditioner.column_count != alpha_column_count ||
        g_alpha_warm_preconditioner.upper_factor.size() !=
            static_cast<size_t>(alpha_column_count) * alpha_column_count) {
        sgpu_internal_set_error("local batch alpha center preconditioner is unavailable");
        return false;
    }
    std::vector<AxisData> axes(query_count);
    std::vector<PsiData> psis(query_count);
    for (int query = 0; query < query_count; ++query) {
        AxisData& axis = axes[query];
        const size_t axis_offset = static_cast<size_t>(query) * axis_count;
        axis.R.assign(axis_R_host + axis_offset, axis_R_host + axis_offset + axis_count);
        axis.Z.assign(axis_Z_host + axis_offset, axis_Z_host + axis_offset + axis_count);
        axis.R_phi.assign(axis_R_phi_host + axis_offset, axis_R_phi_host + axis_offset + axis_count);
        axis.Z_phi.assign(axis_Z_phi_host + axis_offset, axis_Z_phi_host + axis_offset + axis_count);
        axis.selected.R = axis.R[0];
        axis.selected.Z = axis.Z[0];
        axis.selected.residual = axis_residual[query];
        axis.selected.topology_trace = axis_topology_trace[query];
        axis.selected.topology_det = axis_topology_det[query];
        axis.selected.elliptic = axis_topology_det[query] > 0.0 &&
            std::abs(axis_topology_trace[query]) /
                std::sqrt(axis_topology_det[query]) < 2.0;
        axis.selected.ellipse_aspect = center.axis_ellipse_aspect;
        if(config.target_volume>0 && (!axis.selected.elliptic ||
            !std::isfinite(axis.selected.residual) || axis.selected.residual>config.axis_tolerance)) {
            sgpu_internal_set_error("target-volume neighbor axis is invalid; gradient unavailable");return false;
        }
        PsiData& psi = psis[query];
        psi.modes = psi_modes;
        const size_t psi_offset = static_cast<size_t>(query) * psi_coefficient_count;
        psi.coeffs.assign(
            psi_coefficients_host + psi_offset,
            psi_coefficients_host + psi_offset + psi_coefficient_count
        );
        psi.train_rms = psi_train_rms[query];
        initialize_result(&results[query], config.device_id);
    }

    const auto surface_started = LocalClock::now();
    std::vector<double> query_levels(query_count,center.surface_level);
    if(config.target_volume>0) {
        if(!predict_target_levels_gpu(axes,psi_modes,psi_coefficients_host,
                psi_coefficient_count,nfp,config,center,query_levels))return false;
        for(int query=0;query<query_count;++query)
            results[query].target_volume_cache_id=center.target_volume_cache_id;
    }
    const auto prediction_finished=LocalClock::now();
    std::vector<double> surface_R(static_cast<size_t>(query_count) * surface_theta_count);
    std::vector<double> surface_Z(surface_R.size()), surface_radius(surface_R.size());
    for (int query = 0; query < query_count; ++query) {
        const double edge_level = query_levels[query];
        for (int theta_index = 0; theta_index < surface_theta_count; ++theta_index) {
            const size_t index = static_cast<size_t>(query) * surface_theta_count + theta_index;
            const double theta = TWOPI * theta_index / surface_theta_count;
            std::array<double, 25> polynomial;
            ray_polynomial_phi0(psis[query], config, theta, polynomial);
            const double radius = solve_ray_radius(polynomial, edge_level, config);
            surface_radius[index] = radius;
            surface_R[index] = axes[query].R[0] + radius * std::cos(theta);
            surface_Z[index] = axes[query].Z[0] + radius * std::sin(theta);
        }
    }
    const auto roots_finished=LocalClock::now();
    std::vector<double> surface_R_end(surface_R.size()), surface_Z_end(surface_Z.size());
    if (sgpu_batch_trace_period_mixed(
            batch_field, surface_R.data(), surface_Z.data(), surface_R_end.data(),
            surface_Z_end.data(), surface_theta_count, nfp, config.surface_trace_steps)) {
        return false;
    }
    const auto trace_finished=LocalClock::now();
    std::vector<double> surface_drift(query_count, 0.0);
    for (int query = 0; query < query_count; ++query) {
        const double edge_level = query_levels[query];
        std::vector<double> relative(surface_theta_count);
        double radius_mean = 0.0;
        for (int theta_index = 0; theta_index < surface_theta_count; ++theta_index) {
            radius_mean += surface_radius[static_cast<size_t>(query) * surface_theta_count + theta_index] /
                surface_theta_count;
        }
        for (int theta_index = 0; theta_index < surface_theta_count; ++theta_index) {
            const size_t index = static_cast<size_t>(query) * surface_theta_count + theta_index;
            double value, grad_R, grad_Z, grad_phi;
            evaluate_psi_host(
                psis[query], axes[query], config, nfp,
                surface_R_end[index], surface_Z_end[index], TWOPI / nfp,
                value, grad_R, grad_Z, grad_phi
            );
            const double gradient_norm = std::sqrt(
                grad_R * grad_R + grad_Z * grad_Z +
                std::pow(grad_phi / surface_R_end[index], 2.0)
            );
            relative[theta_index] = std::abs(value - edge_level) /
                std::max(gradient_norm * radius_mean, 1.0e-14);
        }
        surface_drift[query] = percentile(relative, 0.95);
    }
    const double surface_seconds = std::chrono::duration<double>(
        LocalClock::now() - surface_started
    ).count();

    std::vector<float> axis_R(static_cast<size_t>(query_count) * axis_count);
    std::vector<float> axis_Z(axis_R.size()), axis_R_phi(axis_R.size()), axis_Z_phi(axis_R.size());
    std::vector<float> psi_coefficients(static_cast<size_t>(query_count) * psi_coefficient_count);
    for (size_t index = 0; index < axis_R.size(); ++index) {
        axis_R[index] = static_cast<float>(axis_R_host[index]);
        axis_Z[index] = static_cast<float>(axis_Z_host[index]);
        axis_R_phi[index] = static_cast<float>(axis_R_phi_host[index]);
        axis_Z_phi[index] = static_cast<float>(axis_Z_phi_host[index]);
    }
    for (size_t index = 0; index < psi_coefficients.size(); ++index) {
        psi_coefficients[index] = static_cast<float>(psi_coefficients_host[index]);
    }
    DeviceBuffer<float> d_axis_R, d_axis_Z, d_axis_R_phi, d_axis_Z_phi, d_psi_coefficients;
    DeviceBuffer<int> d_psi_a, d_psi_b, d_psi_m, d_psi_kind;
    if (!copy_to_device(d_axis_R, axis_R) || !copy_to_device(d_axis_Z, axis_Z) ||
        !copy_to_device(d_axis_R_phi, axis_R_phi) || !copy_to_device(d_axis_Z_phi, axis_Z_phi) ||
        !copy_to_device(d_psi_coefficients, psi_coefficients) ||
        !copy_to_device(d_psi_a, psi_modes.a) || !copy_to_device(d_psi_b, psi_modes.b) ||
        !copy_to_device(d_psi_m, psi_modes.m) || !copy_to_device(d_psi_kind, psi_modes.kind)) {
        sgpu_internal_set_error("local batch common upload failed");
        return false;
    }

    constexpr int threads = 256;
    auto synchronize_stage = [](const char* stage) {
        const cudaError_t error = cudaDeviceSynchronize();
        if (error == cudaSuccess) return true;
        std::string message(stage);
        message += ": ";
        message += cudaGetErrorString(error);
        sgpu_internal_set_error(message.c_str());
        return false;
    };

    // Calibrate toroidal flux for every query at the center-selected surface
    // level. Only the tiny 11-by-4 polynomial fits return to the host.
    const auto flux_started = LocalClock::now();
    const int flux_phi = config.flux_phi_count;
    const int flux_levels = config.flux_level_count;
    const int flux_theta = config.flux_theta_count;
    const int flux_radial = config.flux_radial_quadrature;
    const bool stokes_flux = g_stokes_flux && config.target_volume > 0;
    const int flux_rays = flux_phi * flux_theta;
    const int flux_points_per_query = flux_phi * flux_levels * flux_theta * (stokes_flux ? 1 : flux_radial);
    std::vector<float> levels(static_cast<size_t>(query_count)*flux_levels);
    for(int query=0;query<query_count;++query) for (int index = 0; index < flux_levels; ++index) {
        const double fraction = static_cast<double>(index + 1) / flux_levels;
        levels[static_cast<size_t>(query)*flux_levels+index] = static_cast<float>(query_levels[query] * fraction * fraction);
    }
    std::vector<float> gauss_nodes, gauss_weights;
    gauss_legendre(flux_radial, gauss_nodes, gauss_weights);
    std::vector<float> flux_coefficients(
        static_cast<size_t>(query_count) * config.flux_polynomial_degree
    );
    std::vector<float> edge_flux(query_count);
    {
        DeviceBuffer<float> d_levels, d_gauss_nodes, d_gauss_weights;
        DeviceBuffer<float> d_ray_coefficients(
            static_cast<size_t>(query_count) * flux_rays * (config.psi_poly_degree + 1)
        );
        DeviceBuffer<float> d_radii(
            static_cast<size_t>(query_count) * flux_levels * flux_rays
        );
        DeviceBuffer<float> d_residuals(d_radii.size());
        DeviceBuffer<float> d_xyz(
            static_cast<size_t>(query_count) * flux_points_per_query * 3
        );
        DeviceBuffer<float> d_B(d_xyz.size());
        DeviceBuffer<double> d_sections(
            static_cast<size_t>(query_count) * flux_phi * flux_levels
        );
        if (!copy_to_device(d_levels, levels) ||
            !copy_to_device(d_gauss_nodes, gauss_nodes) ||
            !copy_to_device(d_gauss_weights, gauss_weights) ||
            !d_ray_coefficients.data() || !d_radii.data() || !d_residuals.data() ||
            !d_xyz.data() || !d_B.data() || !d_sections.data()) {
            sgpu_internal_set_error("local batch flux allocation failed");
            return false;
        }
        const size_t ray_total = static_cast<size_t>(query_count) * flux_rays;
        local_batch_build_ray_coefficients_kernel<<<
            static_cast<unsigned int>((ray_total + threads - 1) / threads), threads
        >>>(
            d_psi_coefficients.data(), d_psi_a.data(), d_psi_b.data(), d_psi_m.data(),
            d_psi_kind.data(), psi_coefficient_count, query_count, nfp,
            config.psi_poly_degree, config.psi_m_tor, flux_phi, flux_theta,
            0.0f, 0.0f, config.target_volume>0 ? 0.0f : 0.217f, d_ray_coefficients.data()
        );
        const size_t radius_total = ray_total * flux_levels;
        local_batch_solve_boundary_radii_kernel<<<
            static_cast<unsigned int>((radius_total + threads - 1) / threads), threads
        >>>(
            d_ray_coefficients.data(), flux_rays, query_count, config.psi_poly_degree,
            static_cast<float>(config.psi_a), d_levels.data(), flux_levels,
            config.surface_newton_iters,
            static_cast<float>(config.surface_newton_tolerance),
            static_cast<float>(config.surface_max_radius_scale),
            d_radii.data(), d_residuals.data()
        );
        const size_t point_total =
            static_cast<size_t>(query_count) * flux_points_per_query;
        if(config.target_volume>0) {
            DeviceBuffer<double> d_derivatives(query_count);
            if(!d_derivatives.data())return false;
            target_boundary_guard_kernel<<<query_count,threads>>>(d_ray_coefficients.data(),
                d_radii.data(),d_residuals.data(),d_axis_R.data(),d_axis_R_phi.data(),axis_count,
                flux_rays,flux_theta,flux_levels,config.psi_poly_degree,nfp,
                static_cast<float>(config.psi_a),static_cast<float>(config.flux_boundary_tolerance),d_derivatives.data());
            std::vector<double> derivatives(query_count);
            if(cudaGetLastError()!=cudaSuccess || cudaMemcpy(derivatives.data(),d_derivatives.data(),
                    query_count*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)return false;
            for(int query=0;query<query_count;++query) {
                if(!(std::isfinite(derivatives[query])&&derivatives[query]>0)) {
                    sgpu_internal_set_error("target-volume neighbor boundary is invalid; gradient unavailable");return false;
                }
                results[query].target_volume_dVds=derivatives[query];
            }
        }
        if (stokes_flux) {
            DeviceBuffer<float> d_tangent(d_xyz.size());
            DeviceBuffer<unsigned int> d_stokes_invalid(query_count);
            if (!d_tangent.data() || !d_stokes_invalid.data() || cudaMemset(
                    d_stokes_invalid.data(), 0, query_count * sizeof(unsigned int)) != cudaSuccess) {
                sgpu_internal_set_error("Stokes guard allocation failed");
                return false;
            }
            local_batch_stokes_points_kernel<<<
                static_cast<unsigned int>((point_total + threads - 1) / threads), threads
            >>>(d_radii.data(), query_count, flux_phi, flux_levels, flux_theta, nfp,
                d_axis_R.data(), d_axis_Z.data(), d_axis_R_phi.data(), d_axis_Z_phi.data(),
                axis_count, d_psi_coefficients.data(), d_psi_a.data(), d_psi_b.data(),
                d_psi_m.data(), d_psi_kind.data(), psi_coefficient_count,
                config.psi_poly_degree, config.psi_m_tor, static_cast<float>(config.psi_a),
                d_xyz.data(), d_tangent.data(), d_stokes_invalid.data());
            std::vector<unsigned int> stokes_invalid(query_count);
            if (cudaGetLastError() != cudaSuccess || cudaMemcpy(stokes_invalid.data(),
                    d_stokes_invalid.data(), query_count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost) != cudaSuccess) {
                sgpu_internal_set_error("Stokes boundary guard failed");
                return false;
            }
            for (int query = 0; query < query_count; ++query) {
                if (stokes_invalid[query]) {
                    char error[160];
                    std::snprintf(error, sizeof(error),
                        "Stokes boundary invalid at query %d (flags=%u); gradient unavailable",
                        query, stokes_invalid[query]);
                    sgpu_internal_set_error(error);
                    return false;
                }
            }
            if (cudaGetLastError() != cudaSuccess ||
                sgpu_internal_batch_eval_A_f32_device(
                    batch_field, d_xyz.data(), d_B.data(), flux_points_per_query)) {
                sgpu_internal_set_error("Stokes potential evaluation failed");
                return false;
            }
            local_batch_stokes_reduce_kernel<<<
                query_count * flux_phi * flux_levels, threads, threads * sizeof(double)
            >>>(d_B.data(), d_tangent.data(), flux_theta, d_sections.data());
            if (!synchronize_stage("Stokes flux")) return false;
        } else {
        local_batch_generate_flux_points_kernel<<<
            static_cast<unsigned int>((point_total + threads - 1) / threads), threads
        >>>(
            d_radii.data(), d_gauss_nodes.data(), query_count, flux_phi, flux_levels,
            flux_theta, flux_radial, nfp, d_axis_R.data(), d_axis_Z.data(),
            d_axis_R_phi.data(), d_axis_Z_phi.data(), axis_count, d_xyz.data()
        );
        if (cudaGetLastError() != cudaSuccess ||
            sgpu_internal_batch_eval_B_f32_device(
                batch_field, d_xyz.data(), d_B.data(), flux_points_per_query)) {
            sgpu_internal_set_error("local batch flux field evaluation failed");
            return false;
        }
        local_batch_reduce_flux_sections_kernel<<<
            query_count * flux_phi * flux_levels, threads, threads * sizeof(double)
        >>>(
            d_B.data(), d_radii.data(), d_gauss_nodes.data(), d_gauss_weights.data(),
            query_count, flux_phi, flux_levels, flux_theta, flux_radial, nfp,
            d_sections.data()
        );
        if (!synchronize_stage("local batch flux reduction")) return false;
        }

        std::vector<float> radii(d_radii.size()), residuals(d_residuals.size());
        std::vector<double> sections(d_sections.size());
        if (cudaMemcpy(radii.data(), d_radii.data(), radii.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(residuals.data(), d_residuals.data(), residuals.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(sections.data(), d_sections.data(), sections.size() * sizeof(double),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            sgpu_internal_set_error("local batch flux result copy failed");
            return false;
        }
        if (stokes_flux) {
            for (double value : sections) if (!std::isfinite(value)) {
                sgpu_internal_set_error("nonfinite Stokes section"); return false;
            }
        }
        const int polynomial_degree = config.flux_polynomial_degree;
        for (int query = 0; query < query_count; ++query) {
            const double edge_level=query_levels[query];
            const float* query_level_values=levels.data()+static_cast<size_t>(query)*flux_levels;
            std::vector<double> matrix(static_cast<size_t>(flux_levels)*polynomial_degree);
            for(int row=0;row<flux_levels;++row){double power=query_level_values[row];
                for(int col=0;col<polynomial_degree;++col){matrix[static_cast<size_t>(row)*polynomial_degree+col]=power;power*=query_level_values[row];}}
            SgpuScoreResult& result = results[query];
            std::vector<double> mean(flux_levels, 0.0), section_std(flux_levels, 0.0);
            const size_t section_offset = static_cast<size_t>(query) * flux_phi * flux_levels;
            for (int level = 0; level < flux_levels; ++level) {
                for (int phi_index = 0; phi_index < flux_phi; ++phi_index) {
                    mean[level] += sections[
                        section_offset + static_cast<size_t>(phi_index) * flux_levels + level
                    ] / flux_phi;
                }
                for (int phi_index = 0; phi_index < flux_phi; ++phi_index) {
                    const double delta = sections[
                        section_offset + static_cast<size_t>(phi_index) * flux_levels + level
                    ] - mean[level];
                    section_std[level] += delta * delta / flux_phi;
                }
                section_std[level] = std::sqrt(section_std[level]);
            }
            std::vector<double> coefficients;
            if (!solve_small_least_squares(
                    matrix, mean, flux_levels, polynomial_degree, coefficients)) {
                sgpu_internal_set_error("local batch flux polynomial fit failed");
                return false;
            }
            double fit_error2 = 0.0;
            for (int column = 0; column < polynomial_degree; ++column) {
                flux_coefficients[
                    static_cast<size_t>(query) * polynomial_degree + column
                ] = static_cast<float>(coefficients[column]);
            }
            for (int row = 0; row < flux_levels; ++row) {
                double fitted = 0.0, power = query_level_values[row];
                for (int column = 0; column < polynomial_degree; ++column) {
                    fitted += coefficients[column] * power;
                    power *= query_level_values[row];
                }
                const double delta = fitted - mean[row];
                fit_error2 += delta * delta / flux_levels;
            }
            double derivative_min = std::numeric_limits<double>::infinity();
            double derivative_max = -std::numeric_limits<double>::infinity();
            for (int sample = 0; sample <= 200; ++sample) {
                const double level = edge_level * sample / 200.0;
                double derivative = 0.0, power = 1.0;
                for (int column = 0; column < polynomial_degree; ++column) {
                    derivative += (column + 1) * coefficients[column] * power;
                    power *= level;
                }
                derivative_min = std::min(derivative_min, derivative);
                derivative_max = std::max(derivative_max, derivative);
            }
            double edge = 0.0, edge_power = edge_level;
            for (double coefficient : coefficients) {
                edge += coefficient * edge_power;
                edge_power *= edge_level;
            }
            edge_flux[query] = static_cast<float>(edge);
            const size_t radius_offset = static_cast<size_t>(query) * flux_levels * flux_rays;
            result.flux_boundary_residual_max = 0.0;
            for (size_t index = 0; index < static_cast<size_t>(flux_levels) * flux_rays; ++index) {
                result.flux_boundary_residual_max = std::max(
                    result.flux_boundary_residual_max,
                    static_cast<double>(residuals[radius_offset + index])
                );
            }
            double volume_integrand = 0.0, major_radius = 0.0;
            for (int phi_index = 0; phi_index < flux_phi; ++phi_index) {
                const double phi = TWOPI * phi_index / static_cast<double>(nfp * flux_phi);
                double axis_radius, axis_radius_phi_value, axis_z, axis_z_phi_value;
                periodic_hermite_host(
                    phi, axes[query].R, axes[query].R_phi, nfp,
                    axis_radius, axis_radius_phi_value
                );
                periodic_hermite_host(
                    phi, axes[query].Z, axes[query].Z_phi, nfp,
                    axis_z, axis_z_phi_value
                );
                major_radius += axis_radius / flux_phi;
                for (int theta_index = 0; theta_index < flux_theta; ++theta_index) {
                    const int ray = phi_index * flux_theta + theta_index;
                    const double radius = radii[
                        radius_offset + static_cast<size_t>(flux_levels - 1) * flux_rays + ray
                    ];
                    const double theta = TWOPI * theta_index / flux_theta;
                    volume_integrand += (
                        0.5 * axis_radius * radius * radius +
                        std::cos(theta) * radius * radius * radius / 3.0
                    ) / static_cast<double>(flux_phi * flux_theta);
                }
            }
            const double volume = TWOPI * TWOPI * volume_integrand;
            const double minor_radius = std::sqrt(
                std::max(volume, 0.0) /
                std::max(2.0 * PI * PI * major_radius, 1.0e-30)
            );
            result.surface_level = edge_level;
            result.surface_effective_level = edge_level;
            result.surface_drift_relative_p95 = surface_drift[query];
            result.surface_one_period_drift_relative_p95 = surface_drift[query];
            result.surface_volume = volume;
            result.surface_effective_minor_radius = minor_radius;
            result.surface_inverse_aspect_ratio = minor_radius / std::max(major_radius, 1.0e-30);
            result.flux_edge = edge;
            result.flux_fit_relative_rms =
                std::sqrt(fit_error2) / std::max(std::abs(mean.back()), 1.0e-30);
            result.flux_section_relative_std_edge =
                section_std.back() / std::max(std::abs(mean.back()), 1.0e-30);
            result.flux_derivative_min = derivative_min;
            result.flux_derivative_max = derivative_max;
            result.flux_attempt_count = 1;
            result.stage_completed = SCORE_STAGE_FLUX;
        }
    }
    const double flux_seconds = std::chrono::duration<double>(
        LocalClock::now() - flux_started
    ).count();

    // Build one full-density physical-volume cloud per query. Local endpoints
    // inherit the center's feasible active set, so no per-query compaction or
    // branch search is performed in this gradient-only oracle.
    const auto volume_started = LocalClock::now();
    const int volume_phi = config.volume_phi_count;
    const int volume_theta_count = config.volume_theta_count;
    const int volume_rays = volume_phi * volume_theta_count;
    const int candidate_target = static_cast<int>(std::ceil(config.volume_point_count * 1.25));
    const int volume_radial = (candidate_target + volume_rays - 1) / volume_rays;
    const int volume_points = config.volume_point_count;
    const size_t volume_total = static_cast<size_t>(query_count) * volume_points;
    DeviceBuffer<float> d_volume_xyz(volume_total * 3), d_volume_grad_s(volume_total * 3);
    DeviceBuffer<float> d_volume_grad_theta(volume_total * 3), d_volume_grad_phi(volume_total * 3);
    DeviceBuffer<float> d_volume_rho(volume_total), d_volume_theta_values(volume_total);
    DeviceBuffer<float> d_volume_phi_values(volume_total), d_volume_weight(volume_total);
    DeviceBuffer<float> d_volume_flux_derivative(volume_total);
    DeviceBuffer<float> d_volume_B(volume_total * 3), d_volume_grad_B(volume_total * 9);
    {
        std::vector<float> edge_levels(query_levels.begin(),query_levels.end());
        DeviceBuffer<float> d_edge_levels, d_flux_coefficients, d_edge_flux;
        DeviceBuffer<float> d_volume_ray_coefficients(
            static_cast<size_t>(query_count) * volume_rays * (config.psi_poly_degree + 1)
        );
        DeviceBuffer<float> d_volume_boundary(static_cast<size_t>(query_count) * volume_rays);
        DeviceBuffer<float> d_volume_boundary_residual(d_volume_boundary.size());
        if (!copy_to_device(d_edge_levels, edge_levels) ||
            !copy_to_device(d_flux_coefficients, flux_coefficients) ||
            !copy_to_device(d_edge_flux, edge_flux) ||
            !d_volume_ray_coefficients.data() || !d_volume_boundary.data() ||
            !d_volume_boundary_residual.data() || !d_volume_xyz.data() ||
            !d_volume_grad_s.data() || !d_volume_grad_theta.data() ||
            !d_volume_grad_phi.data() || !d_volume_rho.data() ||
            !d_volume_theta_values.data() || !d_volume_phi_values.data() ||
            !d_volume_weight.data() || !d_volume_flux_derivative.data() ||
            !d_volume_B.data() || !d_volume_grad_B.data()) {
            sgpu_internal_set_error("local batch volume allocation failed");
            return false;
        }
        const size_t ray_total = static_cast<size_t>(query_count) * volume_rays;
        local_batch_build_ray_coefficients_kernel<<<
            static_cast<unsigned int>((ray_total + threads - 1) / threads), threads
        >>>(
            d_psi_coefficients.data(), d_psi_a.data(), d_psi_b.data(), d_psi_m.data(),
            d_psi_kind.data(), psi_coefficient_count, query_count, nfp,
            config.psi_poly_degree, config.psi_m_tor, volume_phi, volume_theta_count,
            0.417f, 0.613f, 0.217f, d_volume_ray_coefficients.data()
        );
        local_batch_solve_boundary_radii_kernel<<<
            static_cast<unsigned int>((ray_total + threads - 1) / threads), threads
        >>>(
            d_volume_ray_coefficients.data(), volume_rays, query_count,
            config.psi_poly_degree, static_cast<float>(config.psi_a),
            d_edge_levels.data(), 1, config.surface_newton_iters,
            static_cast<float>(config.surface_newton_tolerance),
            static_cast<float>(config.surface_max_radius_scale),
            d_volume_boundary.data(), d_volume_boundary_residual.data()
        );
        local_batch_generate_volume_points_kernel<<<
            static_cast<unsigned int>((volume_total + threads - 1) / threads), threads
        >>>(
            d_volume_boundary.data(), query_count, volume_phi, volume_theta_count,
            volume_radial, volume_points, nfp,
            static_cast<float>(config.volume_rho_min), static_cast<float>(center.surface_level),
            d_flux_coefficients.data(), config.flux_polynomial_degree, d_edge_flux.data(),
            d_psi_coefficients.data(), d_psi_a.data(), d_psi_b.data(), d_psi_m.data(),
            d_psi_kind.data(), psi_coefficient_count, config.psi_poly_degree,
            config.psi_m_tor, static_cast<float>(config.psi_a),
            d_axis_R.data(), d_axis_Z.data(), d_axis_R_phi.data(), d_axis_Z_phi.data(),
            axis_count, d_volume_xyz.data(), d_volume_grad_s.data(),
            d_volume_grad_theta.data(), d_volume_grad_phi.data(), d_volume_rho.data(),
            d_volume_theta_values.data(), d_volume_phi_values.data(), d_volume_weight.data(),
            d_volume_flux_derivative.data()
        );
        if (cudaGetLastError() != cudaSuccess ||
            sgpu_internal_batch_eval_B_grad_f32_device(
                batch_field, d_volume_xyz.data(), d_volume_B.data(),
                d_volume_grad_B.data(), volume_points) ||
            !synchronize_stage("local batch volume field")) {
            sgpu_internal_set_error("local batch volume field evaluation failed");
            return false;
        }
    }
    const double volume_seconds = std::chrono::duration<double>(
        LocalClock::now() - volume_started
    ).count();

    // Fit alpha/iota in bounded query tiles. The center Householder R is used
    // only as a right preconditioner; every endpoint has its own design matrix.
    const auto alpha_started = LocalClock::now();
    const int fit_count = std::min(config.alpha_fit_point_count, volume_points);
    const ClebschModesNative alpha_modes = build_clebsch_modes_native(
        config.alpha_radial_order, config.alpha_poloidal_order,
        config.alpha_toroidal_order
    );
    const int mode_count = static_cast<int>(alpha_modes.m.size());
    const int column_count = mode_count + config.iota_degree + 1;
    const int radial_bins = config.radial_bin_count;
    const size_t fit_total = static_cast<size_t>(query_count) * fit_count;
    DeviceBuffer<int> d_alpha_m, d_alpha_n, d_alpha_kind;
    DeviceBuffer<int> d_alpha_bin_counts(static_cast<size_t>(query_count) * radial_bins);
    DeviceBuffer<float> d_alpha_radial;
    DeviceBuffer<float> d_b_theta(fit_total), d_b_phi(fit_total), d_alpha_weights(fit_total);
    DeviceBuffer<float> d_alpha_rho(fit_total), d_alpha_theta(fit_total), d_alpha_phi(fit_total);
    DeviceBuffer<float> d_alpha_rhs(fit_total);
    DeviceBuffer<double> d_alpha_diagnostics(static_cast<size_t>(query_count) * 3);
    DeviceBuffer<float> d_center_R, d_center_scale, d_center_solution;
    DeviceBuffer<float> d_iota(static_cast<size_t>(query_count) * (config.iota_degree + 1));
    if (!copy_to_device(d_alpha_m, alpha_modes.m) ||
        !copy_to_device(d_alpha_n, alpha_modes.n) ||
        !copy_to_device(d_alpha_kind, alpha_modes.kind) ||
        !copy_to_device(d_alpha_radial, alpha_modes.radial_coefficients) ||
        !copy_to_device(d_center_R, g_alpha_warm_preconditioner.upper_factor) ||
        !copy_to_device(d_center_scale, g_alpha_warm_preconditioner.center_scale) ||
        !copy_to_device(d_center_solution, g_alpha_warm_preconditioner.center_scaled_solution) ||
        !d_alpha_bin_counts.data() || !d_b_theta.data() || !d_b_phi.data() ||
        !d_alpha_weights.data() || !d_alpha_rho.data() || !d_alpha_theta.data() ||
        !d_alpha_phi.data() || !d_alpha_rhs.data() || !d_alpha_diagnostics.data() ||
        !d_iota.data() ||
        cudaMemset(d_alpha_bin_counts.data(), 0,
                   d_alpha_bin_counts.size() * sizeof(int)) != cudaSuccess ||
        cudaMemset(d_alpha_diagnostics.data(), 0,
                   d_alpha_diagnostics.size() * sizeof(double)) != cudaSuccess) {
        sgpu_internal_set_error("local batch alpha common allocation failed");
        return false;
    }
    local_batch_alpha_histogram_kernel<<<
        static_cast<unsigned int>((fit_total + threads - 1) / threads), threads
    >>>(
        d_volume_rho.data(), query_count, volume_points, fit_count, radial_bins,
        d_alpha_bin_counts.data()
    );
    local_batch_alpha_prepare_kernel<<<
        static_cast<unsigned int>((fit_total + threads - 1) / threads), threads
    >>>(
        d_volume_B.data(), d_volume_grad_s.data(), d_volume_grad_theta.data(),
        d_volume_grad_phi.data(), d_volume_rho.data(), d_volume_theta_values.data(),
        d_volume_phi_values.data(), query_count, volume_points, fit_count, radial_bins,
        d_alpha_bin_counts.data(), d_b_theta.data(), d_b_phi.data(),
        d_alpha_weights.data(), d_alpha_rho.data(), d_alpha_theta.data(),
        d_alpha_phi.data(), d_alpha_diagnostics.data()
    );
    local_batch_normalize_alpha_kernel<<<
        static_cast<unsigned int>((fit_total + threads - 1) / threads), threads
    >>>(
        d_alpha_weights.data(), d_b_theta.data(), d_alpha_diagnostics.data(),
        query_count, fit_count, d_alpha_rhs.data()
    );
    if (!synchronize_stage("local batch alpha preparation")) return false;
    std::vector<double> alpha_diagnostics(d_alpha_diagnostics.size());
    if (cudaMemcpy(
            alpha_diagnostics.data(), d_alpha_diagnostics.data(),
            alpha_diagnostics.size() * sizeof(double), cudaMemcpyDeviceToHost
        ) != cudaSuccess) {
        sgpu_internal_set_error("local batch alpha diagnostics copy failed");
        return false;
    }

    cublasHandle_t blas = nullptr;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
        sgpu_internal_set_error("local batch alpha cuBLAS initialization failed");
        return false;
    }
    cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH);
    const float one = 1.0f, zero = 0.0f;
    const float ridge = static_cast<float>(std::max(config.alpha_ridge, 0.0));
    const int alpha_tile_size = std::min(query_count, 8);
    bool alpha_ok = true;
    for (int query_begin = 0; query_begin < query_count && alpha_ok;
         query_begin += alpha_tile_size) {
        const int tile = std::min(alpha_tile_size, query_count - query_begin);
        const size_t matrix_stride = static_cast<size_t>(fit_count) * column_count;
        DeviceBuffer<float> d_matrix(static_cast<size_t>(tile) * matrix_stride);
        DeviceBuffer<float> d_coefficients(static_cast<size_t>(tile) * column_count);
        DeviceBuffer<float> d_prediction(static_cast<size_t>(tile) * fit_count);
        DeviceBuffer<float> d_residual(static_cast<size_t>(tile) * fit_count);
        DeviceBuffer<float> d_normal(static_cast<size_t>(tile) * column_count);
        DeviceBuffer<float> d_direction(static_cast<size_t>(tile) * column_count);
        DeviceBuffer<float> d_transformed(static_cast<size_t>(tile) * column_count);
        DeviceBuffer<float> d_image(static_cast<size_t>(tile) * fit_count);
        DeviceBuffer<float> d_gamma(tile), d_gamma_new(tile), d_delta(tile);
        if (!d_matrix.data() || !d_coefficients.data() || !d_prediction.data() ||
            !d_residual.data() || !d_normal.data() || !d_direction.data() ||
            !d_transformed.data() || !d_image.data() || !d_gamma.data() ||
            !d_gamma_new.data() || !d_delta.data()) {
            alpha_ok = false;
            break;
        }
        dim3 alpha_grid(column_count, tile);
        local_batch_assemble_alpha_tile_kernel<<<alpha_grid, threads>>>(
            d_alpha_m.data(), d_alpha_n.data(), d_alpha_kind.data(),
            d_alpha_radial.data(), mode_count, config.alpha_radial_order,
            config.iota_degree, nfp, d_alpha_rho.data(), d_alpha_theta.data(),
            d_alpha_phi.data(), d_b_theta.data(), d_b_phi.data(),
            d_alpha_weights.data(), fit_count, query_begin, tile,
            d_center_scale.data(), d_matrix.data()
        );
        local_batch_initialize_alpha_coefficients_kernel<<<
            (tile * column_count + threads - 1) / threads, threads
        >>>(d_center_solution.data(), column_count, tile, d_coefficients.data());
        cublasStatus_t status = cublasSgemmStridedBatched(
            blas, CUBLAS_OP_N, CUBLAS_OP_N, fit_count, 1, column_count,
            &one, d_matrix.data(), fit_count, static_cast<long long>(matrix_stride),
            d_coefficients.data(), column_count, column_count,
            &zero, d_prediction.data(), fit_count, fit_count, tile
        );
        local_batch_initialize_alpha_residual_kernel<<<
            (tile * fit_count + threads - 1) / threads, threads
        >>>(
            d_alpha_rhs.data(), d_prediction.data(), fit_count, query_begin,
            tile, d_residual.data()
        );
        if (status == CUBLAS_STATUS_SUCCESS) {
            status = cublasSgemmStridedBatched(
                blas, CUBLAS_OP_T, CUBLAS_OP_N, column_count, 1, fit_count,
                &one, d_matrix.data(), fit_count, static_cast<long long>(matrix_stride),
                d_residual.data(), fit_count, fit_count,
                &zero, d_normal.data(), column_count, column_count, tile
            );
        }
        local_batch_alpha_normal_ridge_kernel<<<
            (tile * column_count + threads - 1) / threads, threads
        >>>(
            d_normal.data(), d_coefficients.data(), ridge,
            static_cast<size_t>(tile) * column_count
        );
        if (status == CUBLAS_STATUS_SUCCESS) {
            status = cublasStrsm(
                blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT, column_count, tile, &one,
                d_center_R.data(), column_count, d_normal.data(), column_count
            );
        }
        if (cudaMemcpy(
                d_direction.data(), d_normal.data(),
                d_direction.size() * sizeof(float), cudaMemcpyDeviceToDevice
            ) != cudaSuccess) status = CUBLAS_STATUS_EXECUTION_FAILED;
        local_batch_column_norm_kernel<<<tile, threads, threads * sizeof(float)>>>(
            d_normal.data(), column_count, tile, d_gamma.data()
        );
        for (int iteration = 0; iteration < alpha_iterations &&
             status == CUBLAS_STATUS_SUCCESS; ++iteration) {
            if (cudaMemcpy(
                    d_transformed.data(), d_direction.data(),
                    d_transformed.size() * sizeof(float), cudaMemcpyDeviceToDevice
                ) != cudaSuccess) {
                status = CUBLAS_STATUS_EXECUTION_FAILED;
                break;
            }
            status = cublasStrsm(
                blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, column_count, tile, &one,
                d_center_R.data(), column_count, d_transformed.data(), column_count
            );
            if (status == CUBLAS_STATUS_SUCCESS) {
                status = cublasSgemmStridedBatched(
                    blas, CUBLAS_OP_N, CUBLAS_OP_N, fit_count, 1, column_count,
                    &one, d_matrix.data(), fit_count, static_cast<long long>(matrix_stride),
                    d_transformed.data(), column_count, column_count,
                    &zero, d_image.data(), fit_count, fit_count, tile
                );
            }
            local_batch_alpha_delta_kernel<<<tile, threads, threads * sizeof(float)>>>(
                d_image.data(), fit_count, d_transformed.data(), column_count,
                tile, ridge, d_delta.data()
            );
            local_batch_alpha_update_kernel<<<
                (std::max(tile * fit_count, tile * column_count) + threads - 1) / threads,
                threads
            >>>(
                d_coefficients.data(), d_residual.data(), d_transformed.data(),
                d_image.data(), d_gamma.data(), d_delta.data(), column_count,
                fit_count, tile
            );
            if (status == CUBLAS_STATUS_SUCCESS) {
                status = cublasSgemmStridedBatched(
                    blas, CUBLAS_OP_T, CUBLAS_OP_N, column_count, 1, fit_count,
                    &one, d_matrix.data(), fit_count, static_cast<long long>(matrix_stride),
                    d_residual.data(), fit_count, fit_count,
                    &zero, d_normal.data(), column_count, column_count, tile
                );
            }
            local_batch_alpha_normal_ridge_kernel<<<
                (tile * column_count + threads - 1) / threads, threads
            >>>(
                d_normal.data(), d_coefficients.data(), ridge,
                static_cast<size_t>(tile) * column_count
            );
            if (status == CUBLAS_STATUS_SUCCESS) {
                status = cublasStrsm(
                    blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                    CUBLAS_DIAG_NON_UNIT, column_count, tile, &one,
                    d_center_R.data(), column_count, d_normal.data(), column_count
                );
            }
            local_batch_column_norm_kernel<<<tile, threads, threads * sizeof(float)>>>(
                d_normal.data(), column_count, tile, d_gamma_new.data()
            );
            local_batch_direction_update_kernel<<<
                (tile * column_count + threads - 1) / threads, threads
            >>>(
                d_direction.data(), d_normal.data(), d_gamma.data(),
                d_gamma_new.data(), column_count, tile
            );
            std::swap(d_gamma, d_gamma_new);
        }
        local_batch_extract_iota_kernel<<<
            (tile * (config.iota_degree + 1) + threads - 1) / threads, threads
        >>>(
            d_coefficients.data(), d_center_scale.data(), column_count, mode_count,
            config.iota_degree + 1, query_begin, tile, d_iota.data()
        );
        if (status != CUBLAS_STATUS_SUCCESS ||
            !synchronize_stage("local batch alpha tile")) {
            alpha_ok = false;
        }
    }
    cublasDestroy(blas);
    if (!alpha_ok) {
        sgpu_internal_set_error("local batch alpha PCGLS failed");
        return false;
    }
    const double alpha_seconds = std::chrono::duration<double>(
        LocalClock::now() - alpha_started
    ).count();

    const auto qs_started = LocalClock::now();
    std::vector<double> G;
    if (!vacuum_G_batch_native(
            batch_field, axis_R_host, axis_Z_host, axis_R_phi_host,
            axis_Z_phi_host, query_count, axis_count, nfp, edge_flux, G)) {
        return false;
    }
    constexpr int histogram_bins = 256;
    DeviceBuffer<double> d_G, d_qs_sums(static_cast<size_t>(query_count) * 9);
    DeviceBuffer<double> d_psi_sums(static_cast<size_t>(query_count) * 2);
    DeviceBuffer<unsigned int> d_psi_histogram(
        static_cast<size_t>(query_count) * histogram_bins
    );
    if (!copy_to_device(d_G, G) || !d_qs_sums.data() || !d_psi_sums.data() ||
        !d_psi_histogram.data() ||
        cudaMemset(d_qs_sums.data(), 0, d_qs_sums.size() * sizeof(double)) != cudaSuccess ||
        cudaMemset(d_psi_sums.data(), 0, d_psi_sums.size() * sizeof(double)) != cudaSuccess ||
        cudaMemset(d_psi_histogram.data(), 0,
                   d_psi_histogram.size() * sizeof(unsigned int)) != cudaSuccess) {
        sgpu_internal_set_error("local batch QS allocation failed");
        return false;
    }
    const float edge_threshold = static_cast<float>(
        config.volume_rho_min + (1.0 - config.volume_rho_min) *
        (config.radial_bin_count - 1.0) / config.radial_bin_count
    );
    local_batch_qs_metric_kernel<<<
        static_cast<unsigned int>((volume_total + threads - 1) / threads), threads
    >>>(
        d_volume_B.data(), d_volume_grad_B.data(), d_volume_grad_s.data(),
        d_volume_flux_derivative.data(), d_volume_rho.data(), d_volume_weight.data(),
        query_count, volume_points, d_iota.data(), config.iota_degree,
        config.target_M, config.target_N, nfp, d_G.data(), edge_threshold,
        d_qs_sums.data(), d_psi_histogram.data(), histogram_bins, d_psi_sums.data()
    );
    if (!synchronize_stage("local batch QS reduction")) return false;
    std::vector<double> qs_sums(d_qs_sums.size()), psi_sums(d_psi_sums.size());
    std::vector<unsigned int> psi_histogram(d_psi_histogram.size());
    std::vector<float> iota(static_cast<size_t>(query_count) * (config.iota_degree + 1));
    if (cudaMemcpy(qs_sums.data(), d_qs_sums.data(), qs_sums.size() * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(psi_sums.data(), d_psi_sums.data(), psi_sums.size() * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(psi_histogram.data(), d_psi_histogram.data(),
                   psi_histogram.size() * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(iota.data(), d_iota.data(), iota.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        sgpu_internal_set_error("local batch QS result copy failed");
        return false;
    }
    const double qs_seconds = std::chrono::duration<double>(
        LocalClock::now() - qs_started
    ).count();

    const double target_helicity_norm = std::max(
        std::hypot(static_cast<double>(config.target_M), static_cast<double>(config.target_N)),
        1.0
    );
    const double qp_helicity_norm = std::max(std::abs(nfp), 1);
    const bool qh_target = config.target_M != 0 && config.target_N != 0;
    const double u_min = config.volume_rho_min * config.volume_rho_min;
    for (int query = 0; query < query_count; ++query) {
        const double edge_level = query_levels[query];
        SgpuScoreResult& result = results[query];
        result.axis_R = axes[query].selected.R;
        result.axis_Z = axes[query].selected.Z;
        result.axis_residual = axes[query].selected.residual;
        result.axis_topology_trace = axes[query].selected.topology_trace;
        result.axis_topology_det = axes[query].selected.topology_det;
        result.axis_ellipse_aspect = axes[query].selected.ellipse_aspect;
        result.axis_candidate_count = 1;
        result.axis_used_hint = 1;
        result.psi_train_rms = psis[query].train_rms;
        result.psi_angle_l2 = std::sqrt(
            psi_sums[static_cast<size_t>(query) * 2] /
            std::max(psi_sums[static_cast<size_t>(query) * 2 + 1], 1.0e-300)
        );
        const unsigned int percentile_target =
            static_cast<unsigned int>(std::ceil(0.95 * volume_points));
        unsigned int cumulative = 0;
        int percentile_bin = histogram_bins - 1;
        for (int bin = 0; bin < histogram_bins; ++bin) {
            cumulative += psi_histogram[static_cast<size_t>(query) * histogram_bins + bin];
            if (cumulative >= percentile_target) {
                percentile_bin = bin;
                break;
            }
        }
        result.psi_angle_p95 = std::pow(
            10.0, -10.0 + 8.0 * (percentile_bin + 1.0) / histogram_bins
        );
        result.psi_angle_mean = std::numeric_limits<double>::quiet_NaN();
        psis[query].angle_p95 = result.psi_angle_p95;
        psis[query].angle_l2 = result.psi_angle_l2;
        psis[query].angle_mean = result.psi_angle_mean;
        result.stable_surface_count = 1;
        result.surface_long_trace_periods_completed = 1;
        result.volume_candidate_count = volume_phi * volume_theta_count * volume_radial;
        result.volume_available_count = result.volume_candidate_count;
        result.volume_point_count = volume_points;
        result.volume_valid_fraction = 1.0;
        result.alpha_column_count = column_count;
        result.alpha_normal_B_relative_l2 = std::sqrt(
            alpha_diagnostics[static_cast<size_t>(query) * 3 + 1] /
            std::max(alpha_diagnostics[static_cast<size_t>(query) * 3 + 2], 1.0e-300)
        );
        // The coordinate component is deliberately frozen in this derivative proxy.
        result.alpha_relative_l2 = center.alpha_relative_l2;
        if (g_batch_boundary_iota.size() <= static_cast<size_t>(query))
            g_batch_boundary_iota.resize(query + 1, std::numeric_limits<double>::quiet_NaN());
        double edge_iota = 0.0;
        for (int degree = 0; degree <= config.iota_degree; ++degree)
            edge_iota += iota[static_cast<size_t>(query) * (config.iota_degree + 1) + degree];
        g_batch_boundary_iota[query] = edge_iota;
        result.iota_min = std::numeric_limits<double>::infinity();
        result.iota_max = -std::numeric_limits<double>::infinity();
        for (int sample = 0; sample <= 256; ++sample) {
            const double u = u_min + (1.0 - u_min) * sample / 256.0;
            double value = 0.0, power = 1.0;
            for (int degree = 0; degree <= config.iota_degree; ++degree) {
                value += iota[
                    static_cast<size_t>(query) * (config.iota_degree + 1) + degree
                ] * power;
                power *= u;
            }
            result.iota_min = std::min(result.iota_min, value);
            result.iota_max = std::max(result.iota_max, value);
        }
        const double* sums = qs_sums.data() + static_cast<size_t>(query) * 9;
        result.qs_global_error = std::sqrt(sums[1] / std::max(sums[0], 1.0e-300));
        result.qs_edge_error = std::sqrt(sums[3] / std::max(sums[2], 1.0e-300));
        result.qs_qa_global_error = std::sqrt(sums[7] / std::max(sums[0], 1.0e-300));
        result.qs_qp_global_error_raw = std::sqrt(sums[8] / std::max(sums[0], 1.0e-300));
        result.qs_vacuum_G = G[query];
        result.qs_target_global_error_per_helicity = result.qs_global_error / target_helicity_norm;
        result.qs_target_edge_error_per_helicity = result.qs_edge_error / target_helicity_norm;
        result.qs_qa_global_error_per_helicity = result.qs_qa_global_error;
        result.qs_qp_global_error_per_helicity = result.qs_qp_global_error_raw / qp_helicity_norm;
        result.qs_qp_global_error = result.qs_qp_global_error_per_helicity;
        result.volume_weight_effective_fraction =
            sums[0] * sums[0] / std::max(volume_points * sums[4], 1.0e-300);
        result.edge_weight_effective_fraction =
            sums[2] * sums[2] / std::max(sums[6] * sums[5], 1.0e-300);

        std::vector<SurfaceScreen> screens(1);
        screens[0].level = edge_level;
        screens[0].relative_drift_p95 = surface_drift[query];
        screens[0].one_period_relative_drift_p95 = surface_drift[query];
        screens[0].stable = true;
        screens[0].strict = true;
        screens[0].verified = true;
        CoilMetrics dummy_coil;
        fill_early_components(config, dummy_coil, axes[query], psis[query], screens, result);
        result.components[SGPU_SCORE_COMPONENT_COIL] = clip01(coil_components[query] / 100.0);
        result.components[SGPU_SCORE_COMPONENT_COORDINATE] =
            clip01(center.components[SGPU_SCORE_COMPONENT_COORDINATE] / 100.0);
        const double iota_score = !qh_target ? 1.0 : std::pow(
            clip01(minimum_absolute_iota(result.iota_min, result.iota_max) /
                   config.score_qh_iota_threshold),
            config.score_qh_iota_power
        );
        result.score_iota = iota_score;
        result.components[SGPU_SCORE_COMPONENT_IOTA] = iota_score;
        const double global_score = q_down(
            result.qs_target_global_error_per_helicity,
            config.score_qs_global_scale, 0.9
        );
        const double edge_score = q_down(
            result.qs_target_edge_error_per_helicity,
            config.score_qs_edge_scale, 0.9, global_score
        );
        const double residual_score = blend({{0.80, global_score}, {0.20, edge_score}});
        const double competitor_error = std::min(
            result.qs_qa_global_error_per_helicity,
            result.qs_qp_global_error_per_helicity
        );
        const double advantage = !qh_target ? 1.0 : competitor_error / std::max(
            result.qs_target_global_error_per_helicity + competitor_error, 1.0e-300
        );
        const double size_score = q_saturating_up(
            result.surface_inverse_aspect_ratio,
            config.score_surface_inverse_aspect_saturation
        );
        const double size_factor = config.score_volume_qs_size_floor +
            (1.0 - config.score_volume_qs_size_floor) * size_score;
        const double iota_factor = !qh_target ? 1.0 :
            config.score_volume_qs_iota_floor +
            (1.0 - config.score_volume_qs_iota_floor) * iota_score;
        result.score_surface_size = size_score;
        result.score_qs_residual = residual_score;
        result.score_qh_helicity_advantage = clip01(advantage);
        result.score_volume_qs_size_factor = size_factor;
        result.score_volume_qs_iota_factor = iota_factor;
        result.components[SGPU_SCORE_COMPONENT_VOLUME_QS] =
            residual_score * size_factor * iota_factor;
        target_quality(config,result);
        if(config.target_volume>0 && (!std::isfinite(result.target_quality_penalty) ||
            !std::isfinite(result.qs_target_global_error_per_helicity) ||
            !(result.flux_edge>=0?result.flux_derivative_min>0:result.flux_derivative_max<0) ||
            result.flux_boundary_residual_max>config.flux_boundary_tolerance ||
            std::abs(result.target_volume_relative_error)>config.target_volume_neighbor_tolerance)) {
            sgpu_internal_set_error("target-volume neighbor numerical/geometry validation failed; gradient unavailable");return false;
        }
        result.status = SGPU_SCORE_OK;
        result.stage_completed = SCORE_STAGE_QS;
        result.timings[SGPU_SCORE_TIME_SURFACE_SCREEN] = surface_seconds;
        result.timings[SGPU_SCORE_TIME_FLUX] = flux_seconds;
        result.timings[SGPU_SCORE_TIME_VOLUME_POINTS] = volume_seconds;
        result.timings[SGPU_SCORE_TIME_ALPHA_QR] = alpha_seconds;
        result.timings[SGPU_SCORE_TIME_QS_METRICS] = qs_seconds;
        finalize_score(config, result);
    }
    const double total_seconds = std::chrono::duration<double>(
        LocalClock::now() - total_started
    ).count();
    for (int query = 0; query < query_count; ++query) {
        const double edge_level = query_levels[query];
        results[query].timings[SGPU_SCORE_TIME_TOTAL] = total_seconds;
    }
    if (stats_out && stats_len > 0) {
        const double values[] = {
            static_cast<double>(query_count), surface_seconds, flux_seconds,
            volume_seconds, alpha_seconds, qs_seconds, total_seconds,
            static_cast<double>(alpha_tile_size), static_cast<double>(volume_points),
            static_cast<double>(flux_points_per_query),
            std::chrono::duration<double>(prediction_finished-surface_started).count(),
            std::chrono::duration<double>(roots_finished-prediction_finished).count(),
            std::chrono::duration<double>(trace_finished-roots_finished).count(),
            surface_seconds-std::chrono::duration<double>(trace_finished-surface_started).count(),
        };
        const int count = std::min<int>(
            stats_len, static_cast<int>(sizeof(values) / sizeof(values[0]))
        );
        std::copy(values, values + count, stats_out);
    }
    sgpu_internal_set_error("");
    return true;
}


}  // namespace coil_eval::pipeline
