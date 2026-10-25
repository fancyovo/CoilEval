#include "pipeline/reverse_capture.cuh"
#include "pipeline/session_state.cuh"
#include "pipeline/target_volume.cuh"
#include "coil_field.h"
#include "coil_field_internal.h"
#include <cstring>
namespace coil_eval::pipeline {
thread_local int g_reverse_capture_flags = 0;
// Experimental, explicitly enabled host snapshots for the GPU reverse path.
// All matrices are exported in mathematical row-major order. No GPU pointers
// escape this interface, so Python copies survive later native center captures.
thread_local std::vector<ReverseCapturedArray> g_reverse_snapshot;

template <typename T>
void reverse_add(std::vector<ReverseCapturedArray>& arrays, const char* name,
                 const std::vector<T>& values, std::vector<std::int64_t> shape = {}) {
    if (shape.empty()) shape = {static_cast<std::int64_t>(values.size())};
    arrays.push_back({name, std::move(shape), std::vector<double>(values.begin(), values.end())});
}

void reverse_scalar(std::vector<ReverseCapturedArray>& arrays, const char* name, double value) {
    arrays.push_back({name, {}, {value}});
}

template <typename T>
bool reverse_device(std::vector<ReverseCapturedArray>& arrays, const char* name,
                    const DeviceBuffer<T>& buffer, std::vector<std::int64_t> shape = {}) {
    std::vector<T> values(buffer.size());
    if (!values.empty() && cudaMemcpy(values.data(), buffer.data(), values.size() * sizeof(T),
                                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        sgpu_internal_set_error("reverse-state device export failed");
        return false;
    }
    reverse_add(arrays, name, values, std::move(shape));
    return true;
}

bool capture_reverse_snapshot(const FixedFrontG2Cache& cache, int nfp,
                              const SgpuScoreConfig& config, const SgpuScoreResult& result) {
    std::vector<ReverseCapturedArray> arrays;
    const int na = static_cast<int>(cache.axis.R.size());
    const int np = static_cast<int>(cache.psi.coeffs.size());
    const int nc = cache.alpha.column_count;
    reverse_scalar(arrays, "capture_version", 1);
    reverse_scalar(arrays, "nfp", nfp);
    reverse_add(arrays, "axis_R", cache.axis.R);
    reverse_add(arrays, "axis_Z", cache.axis.Z);
    reverse_add(arrays, "axis_R_phi", cache.axis.R_phi);
    reverse_add(arrays, "axis_Z_phi", cache.axis.Z_phi);
    std::vector<double> axis_samples(static_cast<size_t>(na) * 4);
    for (int i = 0; i < na; ++i) {
        axis_samples[4*i] = cache.axis.R[i];
        axis_samples[4*i+1] = cache.axis.Z[i];
        axis_samples[4*i+2] = cache.axis.R_phi[i];
        axis_samples[4*i+3] = cache.axis.Z_phi[i];
    }
    reverse_add(arrays, "axis_samples", axis_samples, {na, 4});
    reverse_add(arrays, "psi_coefficients", cache.psi.coeffs);
    reverse_add(arrays, "psi_mode_a", cache.psi.modes.a);
    reverse_add(arrays, "psi_mode_b", cache.psi.modes.b);
    reverse_add(arrays, "psi_mode_m", cache.psi.modes.m);
    reverse_add(arrays, "psi_mode_kind", cache.psi.modes.kind);
    std::vector<double> psi_upper(static_cast<size_t>(np) * np), psi_scale(np);
    if (sgpu_copy_psi_warm_preconditioner(psi_upper.data(), psi_scale.data(), np)) return false;
    reverse_add(arrays, "psi_upper", psi_upper, {np, np});
    reverse_add(arrays, "psi_scale", psi_scale);
    const auto& alpha = cache.alpha;
    reverse_add(arrays, "alpha_mode_l", alpha.modes.l);
    reverse_add(arrays, "alpha_mode_m", alpha.modes.m);
    reverse_add(arrays, "alpha_mode_n", alpha.modes.n);
    reverse_add(arrays, "alpha_mode_kind", alpha.modes.kind);
    reverse_add(arrays, "alpha_radial_coefficients", alpha.modes.radial_coefficients,
                {alpha.mode_count, alpha.modes.radial_order + 1});
    reverse_add(arrays, "alpha_iota_coefficients", alpha.iota_coefficients);
    reverse_scalar(arrays, "alpha_fit_count", alpha.fit_count);
    reverse_scalar(arrays, "alpha_column_count", nc);
    reverse_scalar(arrays, "alpha_qr_rows", alpha.qr_rows);
    reverse_scalar(arrays, "alpha_radial_weighting_mode", alpha.radial_weighting_mode);
    reverse_scalar(arrays, "alpha_weight_sigma", alpha.radial_weight_sigma);
    reverse_scalar(arrays, "alpha_count_floor", alpha.radial_count_floor);
    if (!alpha.radial_soft_counts.empty())
        reverse_add(arrays, "alpha_radial_soft_counts", alpha.radial_soft_counts);
    reverse_scalar(arrays, "alpha_residual_norm", alpha.residual_norm);
    reverse_scalar(arrays, "alpha_rhs_norm", alpha.rhs_norm);
    std::vector<double> alpha_upper(static_cast<size_t>(nc) * nc, 0.0);
    for (int row = 0; row < nc; ++row)
        for (int column = row; column < nc; ++column)
            alpha_upper[static_cast<size_t>(row) * nc + column] =
                g_alpha_warm_preconditioner.upper_factor[static_cast<size_t>(column) * nc + row];
    reverse_add(arrays, "alpha_upper", alpha_upper, {nc, nc});
    reverse_add(arrays, "alpha_scale", g_alpha_warm_preconditioner.center_scale);
    reverse_add(arrays, "alpha_center_scaled_solution", g_alpha_warm_preconditioner.center_scaled_solution);
    reverse_add(arrays, "flux_coefficients", cache.reverse_flux_coefficients);
    reverse_scalar(arrays, "flux_edge", result.flux_edge);
    reverse_scalar(arrays, "vacuum_G", cache.G);
    reverse_add(arrays, "qs_sums", std::vector<double>(cache.qs_sums.begin(), cache.qs_sums.end()));
    if (config.target_volume > 0.0) {
        // Populate through the exact same GPU contraction used by the local
        // fast oracle. This also preserves its signed quadrature residual.
        std::vector<AxisData> axes{cache.axis};
        std::vector<double> predicted_levels(1);
        if (!predict_target_levels_gpu(axes, cache.psi.modes, cache.psi.coeffs.data(), np,
                                       nfp, config, result, predicted_levels)) return false;
        const auto& target = g_target_volume_cache;
        reverse_scalar(arrays, "target_level", target.level);
        reverse_scalar(arrays, "target_dVds", target.dVds);
        reverse_scalar(arrays, "target_volume", target.volume);
        reverse_scalar(arrays, "target_requested_volume", target.target);
        reverse_scalar(arrays, "target_adjoint_residual", target.adjoint_residual);
        reverse_scalar(arrays, "target_center_predicted_level", predicted_levels[0]);
        reverse_add(arrays, "target_adjoint_weights", target.adjoint_weights);
        reverse_add(arrays, "target_axis_weights", target.axis_weights);
        reverse_add(arrays, "target_reference_coefficients", target.coefficients);
        std::vector<double> axis_R(config.flux_phi_count), axis_Z(axis_R.size()), phi(axis_R.size());
        for (int i = 0; i < config.flux_phi_count; ++i) {
            const auto& ray = target.geometry.rays[static_cast<size_t>(i) * config.flux_theta_count];
            axis_R[i] = ray.R; axis_Z[i] = ray.Z; phi[i] = ray.phi;
        }
        reverse_add(arrays, "target_axis_R", axis_R);
        reverse_add(arrays, "target_axis_Z", axis_Z);
        reverse_add(arrays, "target_phi", phi);
        if (g_reverse_capture_flags & 2)
            reverse_add(arrays, "target_radii", target.radii, {config.flux_phi_count, config.flux_theta_count});
    }
    if (g_reverse_capture_flags & 2) {
        const auto& p = cache.points;
        const int n = p.count;
        if (!reverse_device(arrays, "points_xyz", p.xyz, {n,3}) ||
            !reverse_device(arrays, "points_grad_s", p.grad_s, {n,3}) ||
            !reverse_device(arrays, "points_grad_theta", p.grad_theta, {n,3}) ||
            !reverse_device(arrays, "points_grad_phi", p.grad_phi, {n,3}) ||
            !reverse_device(arrays, "points_s", p.s) ||
            !reverse_device(arrays, "points_rho", p.rho) ||
            !reverse_device(arrays, "points_theta", p.theta) ||
            !reverse_device(arrays, "points_phi", p.phi) ||
            !reverse_device(arrays, "points_R", p.R) ||
            !reverse_device(arrays, "points_volume_weight", p.volume_weight) ||
            !reverse_device(arrays, "points_flux_derivative", p.flux_derivative) ||
            !reverse_device(arrays, "points_B", cache.B, {n,3}) ||
            !reverse_device(arrays, "points_grad_B", cache.grad_B, {n,3,3}) ||
            !reverse_device(arrays, "alpha_weights", alpha.weights) ||
            !reverse_device(arrays, "alpha_rho", alpha.rho) ||
            !reverse_device(arrays, "alpha_theta", alpha.theta) ||
            !reverse_device(arrays, "alpha_phi", alpha.phi) ||
            !reverse_device(arrays, "alpha_b_theta", alpha.b_theta) ||
            !reverse_device(arrays, "alpha_b_phi", alpha.b_phi) ||
            !reverse_device(arrays, "alpha_rhs_reference", alpha.rhs_reference) ||
            !reverse_device(arrays, "alpha_residual", alpha.residual)) return false;
    }
    for (const auto& array : arrays) {
        std::int64_t count = 1;
        for (const auto size : array.shape) count *= size;
        if (count != static_cast<std::int64_t>(array.values.size())) {
            sgpu_internal_set_error("reverse-state snapshot shape mismatch");
            return false;
        }
    }
    g_reverse_snapshot.swap(arrays);
    return true;
}

}
using namespace coil_eval::pipeline;
extern "C" {
int sgpu_set_reverse_capture(int flags) {
    if (flags < 0 || flags > 3 || flags == 2) {
        sgpu_internal_set_error("reverse capture flags must be 0, 1 or 3");
        return 1;
    }
    g_reverse_capture_flags = flags;
    g_reverse_snapshot.clear();
    return 0;
}

int sgpu_reverse_capture_array_count() {
    return static_cast<int>(g_reverse_snapshot.size());
}

int sgpu_reverse_capture_array_info(int index, char* name, int name_capacity,
                                  std::int64_t* shape, int shape_capacity, int* ndim) {
    if (index < 0 || index >= static_cast<int>(g_reverse_snapshot.size()) || !name || !shape || !ndim) return 1;
    const auto& array = g_reverse_snapshot[index];
    if (name_capacity <= static_cast<int>(array.name.size()) || shape_capacity < static_cast<int>(array.shape.size())) return 1;
    std::memcpy(name, array.name.c_str(), array.name.size() + 1);
    std::copy(array.shape.begin(), array.shape.end(), shape);
    *ndim = static_cast<int>(array.shape.size());
    return 0;
}

int sgpu_reverse_capture_copy(int index, double* values, std::int64_t capacity) {
    if (index < 0 || index >= static_cast<int>(g_reverse_snapshot.size()) || !values || capacity < 0) return 1;
    const auto& array = g_reverse_snapshot[index];
    if (static_cast<std::uint64_t>(capacity) < array.values.size()) return 1;
    std::copy(array.values.begin(), array.values.end(), values);
    return 0;
}
}
