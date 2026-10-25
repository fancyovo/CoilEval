from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np

TRACE_PRECISION_ALIASES = {
    "mixed64": "mixed64",
    "bf32_state64": "mixed64",
    "blockline_mixed64": "mixed64",
    "fp64": "fp64",
    "float64": "fp64",
    "blockline": "fp64",
    "fp32": "fp32",
    "float32": "fp32",
    "f32": "fp32",
    "blockline_f32": "fp32",
}

_UTILITY_LIB_CACHE: dict[str, ctypes.CDLL] = {}
_NATIVE_SCORE_LIB_CACHE: dict[str, ctypes.CDLL] = {}


class GpuError(RuntimeError):
    pass


SGPU_SCORE_ABI_VERSION = 12
SGPU_SCORE_COMPONENT_NAMES = (
    "axis",
    "psi",
    "surface",
    "coordinate",
    "volume_qs",
    "iota",
    "coil",
)
SGPU_SCORE_TIMING_NAMES = (
    "total_s",
    "field_create_s",
    "coil_geometry_s",
    "axis_search_s",
    "axis_trace_s",
    "psi_points_s",
    "psi_fit_s",
    "psi_validate_s",
    "surface_screen_s",
    "flux_s",
    "volume_points_s",
    "field_volume_s",
    "alpha_assemble_s",
    "alpha_solve_s",
    "qs_metrics_s",
    "score_s",
    "axis_domain_s",
    "axis_primary_grid_trace_s",
    "axis_fallback_grid_trace_s",
    "axis_candidate_extract_s",
    "axis_candidate_refine_s",
    "axis_fp64_verify_s",
    "axis_topology_s",
    "surface_ray_roots_s",
    "surface_mixed_trace_s",
    "surface_mixed_reduce_s",
    "surface_fp64_trace_s",
    "surface_fp64_reduce_s",
    "surface_long_trace_s",
    "surface_long_reduce_s",
    "flux_calibration_s",
    "surface_confidence_s",
)
SGPU_SCORE_STATUS_NAMES = {
    0: "ok",
    1: "no_axis",
    2: "no_surface",
    3: "drift_rejected",
    4: "flux_rejected",
    5: "alpha_failed",
    6: "branch_lost",
    100: "internal_error",
}


class _SgpuScoreConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("device_id", ctypes.c_int32),
        ("segments_per_coil", ctypes.c_int32),
        ("target_M", ctypes.c_int32),
        ("target_N", ctypes.c_int32),
        ("axis_grid", ctypes.c_int32),
        ("axis_fallback_grid", ctypes.c_int32),
        ("axis_max_candidates", ctypes.c_int32),
        ("axis_fallback_max_candidates", ctypes.c_int32),
        ("axis_newton_iters", ctypes.c_int32),
        ("axis_fallback_newton_iters", ctypes.c_int32),
        ("axis_trace_steps", ctypes.c_int32),
        ("axis_sample_count", ctypes.c_int32),
        ("axis_fallback_max_nfp", ctypes.c_int32),
        ("axis_span", ctypes.c_double),
        ("axis_tolerance", ctypes.c_double),
        ("axis_r_floor", ctypes.c_double),
        ("axis_fd_relative", ctypes.c_double),
        ("axis_fd_absolute", ctypes.c_double),
        ("axis_topology_margin", ctypes.c_double),
        ("psi_poly_degree", ctypes.c_int32),
        ("psi_m_tor", ctypes.c_int32),
        ("psi_n_r", ctypes.c_int32),
        ("psi_n_z", ctypes.c_int32),
        ("psi_n_phi", ctypes.c_int32),
        ("psi_validation_points", ctypes.c_int32),
        ("psi_solver_mode", ctypes.c_int32),
        ("psi_precision_mode", ctypes.c_int32),
        ("psi_a", ctypes.c_double),
        ("psi_rho_min", ctypes.c_double),
        ("psi_ridge", ctypes.c_double),
        ("surface_level_count", ctypes.c_int32),
        ("surface_theta_count", ctypes.c_int32),
        ("surface_trace_steps", ctypes.c_int32),
        ("surface_newton_iters", ctypes.c_int32),
        ("surface_levels", ctypes.c_double * 16),
        ("surface_newton_tolerance", ctypes.c_double),
        ("surface_max_radius_scale", ctypes.c_double),
        ("surface_drift_relative_tolerance", ctypes.c_double),
        ("surface_drift_absolute_tolerance", ctypes.c_double),
        ("flux_level_count", ctypes.c_int32),
        ("flux_phi_count", ctypes.c_int32),
        ("flux_theta_count", ctypes.c_int32),
        ("flux_radial_quadrature", ctypes.c_int32),
        ("flux_polynomial_degree", ctypes.c_int32),
        ("flux_boundary_tolerance", ctypes.c_double),
        ("flux_section_relative_std_tolerance", ctypes.c_double),
        ("volume_point_count", ctypes.c_int32),
        ("volume_phi_count", ctypes.c_int32),
        ("volume_theta_count", ctypes.c_int32),
        ("alpha_fit_point_count", ctypes.c_int32),
        ("alpha_radial_order", ctypes.c_int32),
        ("alpha_poloidal_order", ctypes.c_int32),
        ("alpha_toroidal_order", ctypes.c_int32),
        ("iota_degree", ctypes.c_int32),
        ("radial_bin_count", ctypes.c_int32),
        ("alpha_solver_mode", ctypes.c_int32),
        ("volume_rho_min", ctypes.c_double),
        ("alpha_ridge", ctypes.c_double),
        ("score_weights", ctypes.c_double * 7),
        ("score_axis_residual_scale", ctypes.c_double),
        ("score_psi_angle_p95_scale", ctypes.c_double),
        ("score_psi_angle_l2_scale", ctypes.c_double),
        ("score_surface_inverse_aspect_saturation", ctypes.c_double),
        ("score_surface_drift_scale", ctypes.c_double),
        ("score_flux_section_std_scale", ctypes.c_double),
        ("score_flux_boundary_residual_scale", ctypes.c_double),
        ("score_alpha_normal_B_scale", ctypes.c_double),
        ("score_alpha_relative_l2_scale", ctypes.c_double),
        ("score_qs_global_scale", ctypes.c_double),
        ("score_qs_edge_scale", ctypes.c_double),
        ("score_qh_iota_threshold", ctypes.c_double),
        ("score_qh_iota_power", ctypes.c_double),
        ("score_volume_qs_size_floor", ctypes.c_double),
        ("score_volume_qs_iota_floor", ctypes.c_double),
        ("score_qh_total_iota_floor", ctypes.c_double),
        ("surface_long_trace_periods", ctypes.c_int32),
        ("surface_long_trace_relative_tolerance", ctypes.c_double),
        ("score_qh_total_helicity_floor", ctypes.c_double),
        ("score_qh_helicity_bad", ctypes.c_double),
        ("score_qh_helicity_good", ctypes.c_double),
        ("score_qh_helicity_exploration_fraction", ctypes.c_double),
        ("surface_selection_mode", ctypes.c_int32),
        ("surface_confidence_periods", ctypes.c_int32),
        ("surface_flux_bisection_iters", ctypes.c_int32),
        ("surface_confidence_drift_center", ctypes.c_double),
        ("surface_confidence_drift_temperature", ctypes.c_double),
        ("surface_confidence_smoothmax_temperature", ctypes.c_double),
        ("surface_confidence_minimum", ctypes.c_double),
        ("axis_hint_enabled", ctypes.c_int32),
        ("axis_hint_require_continuation", ctypes.c_int32),
        ("axis_hint_R", ctypes.c_double),
        ("axis_hint_Z", ctypes.c_double),
        ("axis_hint_max_distance", ctypes.c_double),
        ("target_volume", ctypes.c_double),
        ("target_volume_relative_tolerance", ctypes.c_double),
        ("target_volume_neighbor_tolerance", ctypes.c_double),
        ("target_volume_forward_tolerance", ctypes.c_double),
        ("target_quality_threshold", ctypes.c_double),
        ("target_quality_temperature", ctypes.c_double),
        ("target_quality_weight", ctypes.c_double),

    ]


class _SgpuScoreResult(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("status", ctypes.c_int32),
        ("stage_completed", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
        ("flux_attempt_count", ctypes.c_int32),
        ("score", ctypes.c_double),
        ("components", ctypes.c_double * 7),
        ("timings", ctypes.c_double * 32),
        ("axis_R", ctypes.c_double),
        ("axis_Z", ctypes.c_double),
        ("axis_residual", ctypes.c_double),
        ("axis_topology_trace", ctypes.c_double),
        ("axis_topology_det", ctypes.c_double),
        ("axis_ellipse_aspect", ctypes.c_double),
        ("psi_train_rms", ctypes.c_double),
        ("psi_angle_mean", ctypes.c_double),
        ("psi_angle_p95", ctypes.c_double),
        ("psi_angle_l2", ctypes.c_double),
        ("surface_level", ctypes.c_double),
        ("surface_drift_relative_p95", ctypes.c_double),
        ("surface_one_period_drift_relative_p95", ctypes.c_double),
        ("surface_effective_minor_radius", ctypes.c_double),
        ("surface_inverse_aspect_ratio", ctypes.c_double),
        ("surface_volume", ctypes.c_double),
        ("flux_edge", ctypes.c_double),
        ("flux_fit_relative_rms", ctypes.c_double),
        ("flux_section_relative_std_edge", ctypes.c_double),
        ("flux_boundary_residual_max", ctypes.c_double),
        ("flux_derivative_min", ctypes.c_double),
        ("flux_derivative_max", ctypes.c_double),
        ("alpha_relative_l2", ctypes.c_double),
        ("alpha_normal_B_relative_l2", ctypes.c_double),
        ("iota_min", ctypes.c_double),
        ("iota_max", ctypes.c_double),
        ("score_surface_size", ctypes.c_double),
        ("score_iota", ctypes.c_double),
        ("score_qs_residual", ctypes.c_double),
        ("score_volume_qs_size_factor", ctypes.c_double),
        ("score_volume_qs_iota_factor", ctypes.c_double),
        ("score_before_qh_iota_gate", ctypes.c_double),
        ("score_qh_total_iota_factor", ctypes.c_double),
        ("score_qh_helicity_advantage", ctypes.c_double),
        ("score_qh_helicity_quality", ctypes.c_double),
        ("score_qh_total_helicity_factor", ctypes.c_double),
        ("qs_global_error", ctypes.c_double),
        ("qs_edge_error", ctypes.c_double),
        ("qs_qa_global_error", ctypes.c_double),
        ("qs_qp_global_error", ctypes.c_double),
        ("qs_vacuum_G", ctypes.c_double),
        ("qs_target_global_error_per_helicity", ctypes.c_double),
        ("qs_target_edge_error_per_helicity", ctypes.c_double),
        ("qs_qa_global_error_per_helicity", ctypes.c_double),
        ("qs_qp_global_error_raw", ctypes.c_double),
        ("qs_qp_global_error_per_helicity", ctypes.c_double),
        ("qs_abs_p95", ctypes.c_double),
        ("qs_abs_p95_per_helicity", ctypes.c_double),
        ("volume_valid_fraction", ctypes.c_double),
        ("volume_weight_effective_fraction", ctypes.c_double),
        ("edge_weight_effective_fraction", ctypes.c_double),
        ("surface_confidence_mean", ctypes.c_double),
        ("surface_confidence_edge", ctypes.c_double),
        ("surface_effective_level", ctypes.c_double),
        ("surface_confidence_risk", ctypes.c_double),
        ("axis_hint_distance", ctypes.c_double),
        ("coil_length_mean", ctypes.c_double),
        ("coil_curvature_p95", ctypes.c_double),
        ("coil_curvature_max", ctypes.c_double),
        ("coil_min_intercoil_distance", ctypes.c_double),
        ("coil_min_axis_distance", ctypes.c_double),
        ("coil_high_mode_energy_fraction", ctypes.c_double),
        ("coil_current_abs_max_a", ctypes.c_double),
        ("axis_candidate_count", ctypes.c_int32),
        ("stable_surface_count", ctypes.c_int32),
        ("volume_candidate_count", ctypes.c_int32),
        ("volume_available_count", ctypes.c_int32),
        ("volume_point_count", ctypes.c_int32),
        ("alpha_column_count", ctypes.c_int32),
        ("surface_long_trace_periods_completed", ctypes.c_int32),
        ("surface_long_trace_rejected_count", ctypes.c_int32),
        ("axis_used_hint", ctypes.c_int32),
        ("error_message", ctypes.c_char * 256),
        ("target_volume_requested", ctypes.c_double),
        ("target_volume_relative_error", ctypes.c_double),
        ("target_volume_dVds", ctypes.c_double),
        ("target_volume_maximum", ctypes.c_double),
        ("target_quality_error", ctypes.c_double),
        ("target_quality_penalty", ctypes.c_double),
        ("target_volume_cache_id", ctypes.c_uint64),
        ("target_volume_iterations", ctypes.c_int32),
    ]


class _SgpuScoreGradientResult(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("status", ctypes.c_int32),
        ("gradient_group", ctypes.c_int32),
        ("forward_wall_s", ctypes.c_double),
        ("gradient_wall_s", ctypes.c_double),
        ("point_vjp_s", ctypes.c_double),
        ("field_vjp_s", ctypes.c_double),
        ("parameter_map_s", ctypes.c_double),
        ("score_gradient_rms", ctypes.c_double),
        ("coil_component_gradient_rms", ctypes.c_double),
        ("error_message", ctypes.c_char * 256),
    ]


def _bind_native_score(lib: ctypes.CDLL) -> None:
    lib.sgpu_score_config_size.restype = ctypes.c_size_t
    lib.sgpu_score_config_size.argtypes = []
    lib.sgpu_score_result_size.restype = ctypes.c_size_t
    lib.sgpu_score_result_size.argtypes = []
    lib.sgpu_default_score_config.restype = ctypes.c_int
    lib.sgpu_default_score_config.argtypes = [ctypes.POINTER(_SgpuScoreConfig)]
    lib.sgpu_score_coils.restype = ctypes.c_int
    lib.sgpu_score_coils.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(_SgpuScoreConfig),
        ctypes.POINTER(_SgpuScoreResult),
    ]
    lib.sgpu_last_error.restype = ctypes.c_char_p
    if lib.sgpu_score_config_size() != ctypes.sizeof(_SgpuScoreConfig):
        raise GpuError("native score config ABI size mismatch")
    if lib.sgpu_score_result_size() != ctypes.sizeof(_SgpuScoreResult):
        raise GpuError("native score result ABI size mismatch")


def _bind_native_gradient(lib: ctypes.CDLL) -> None:
    if getattr(lib, "_sgpu_gradient_bound", False):
        return
    try:
        lib.sgpu_score_gradient_result_size.restype = ctypes.c_size_t
        lib.sgpu_score_gradient_result_size.argtypes = []
        lib.sgpu_coil_component_gradient.restype = ctypes.c_int
        lib.sgpu_coil_component_gradient.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.sgpu_score_coils_g1_gradient.restype = ctypes.c_int
        lib.sgpu_score_coils_g1_gradient.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_SgpuScoreConfig),
            ctypes.POINTER(_SgpuScoreResult),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(_SgpuScoreGradientResult),
        ]
        lib.sgpu_score_coils_g2_gradient.restype = ctypes.c_int
        lib.sgpu_score_coils_g2_gradient.argtypes = lib.sgpu_score_coils_g1_gradient.argtypes
        lib.sgpu_score_coils_g3_gradient.restype = ctypes.c_int
        lib.sgpu_score_coils_g3_gradient.argtypes = lib.sgpu_score_coils_g1_gradient.argtypes
    except AttributeError as exc:
        raise GpuError("native library does not provide the experimental G1 gradient API") from exc
    if lib.sgpu_score_gradient_result_size() != ctypes.sizeof(_SgpuScoreGradientResult):
        raise GpuError("native score gradient result ABI size mismatch")
    lib._sgpu_gradient_bound = True


def _bind_native_metric_gradient(lib: ctypes.CDLL) -> bool:
    """Bind the optional separated-metric entry point. Old libraries return False."""
    if getattr(lib, "_sgpu_metric_gradient_bound", None) is not None:
        return bool(lib._sgpu_metric_gradient_bound)
    if not hasattr(lib, "sgpu_coil_metric_gradient"):
        lib._sgpu_metric_gradient_bound = False
        return False
    lib.sgpu_coil_metric_gradient.restype = ctypes.c_int
    lib.sgpu_coil_metric_gradient.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib._sgpu_metric_gradient_bound = True
    return True


def _bind_native_g2_frozen(lib: ctypes.CDLL) -> None:
    if getattr(lib, "_sgpu_g2_frozen_bound", False):
        return
    try:
        lib.sgpu_score_coils_g2_frozen_batch.restype = ctypes.c_int
        lib.sgpu_score_coils_g2_frozen_batch.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_SgpuScoreConfig),
            ctypes.POINTER(_SgpuScoreResult),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
    except AttributeError as exc:
        raise GpuError("native library does not provide the G2 frozen-front oracle") from exc
    lib._sgpu_g2_frozen_bound = True


def _bind_native_g3_frozen(lib: ctypes.CDLL) -> None:
    if getattr(lib, "_sgpu_g3_frozen_bound", False):
        return
    try:
        lib.sgpu_score_coils_g3_frozen_batch.restype = ctypes.c_int
        lib.sgpu_score_coils_g3_frozen_batch.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_SgpuScoreConfig),
            ctypes.POINTER(_SgpuScoreResult),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
    except AttributeError as exc:
        raise GpuError("native library does not provide the G3 frozen-geometry oracle") from exc
    lib._sgpu_g3_frozen_bound = True


def _bind_native_g4_fixed_branch(lib: ctypes.CDLL) -> None:
    if getattr(lib, "_sgpu_g4_fixed_branch_bound", False):
        return
    try:
        lib.sgpu_score_coils_g4_fixed_branch_batch.restype = ctypes.c_int
        lib.sgpu_score_coils_g4_fixed_branch_batch.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_SgpuScoreConfig),
            ctypes.POINTER(_SgpuScoreResult),
            ctypes.POINTER(_SgpuScoreResult),
        ]
    except AttributeError as exc:
        raise GpuError("native library does not provide the G4 fixed-branch oracle") from exc
    lib._sgpu_g4_fixed_branch_bound = True


def _bind_native_psi_warm_batch(lib: ctypes.CDLL) -> None:
    if getattr(lib, "_sgpu_psi_warm_batch_bound", False):
        return
    try:
        lib.sgpu_score_coils_psi_warm_batch.restype = ctypes.c_int
        lib.sgpu_score_coils_psi_warm_batch.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_SgpuScoreConfig),
            ctypes.POINTER(_SgpuScoreResult),
            ctypes.POINTER(_SgpuScoreResult),
        ]
    except AttributeError as exc:
        raise GpuError("native library does not provide the psi warm-batch oracle") from exc
    lib._sgpu_psi_warm_batch_bound = True


def _coerce_score_inputs(coeffs_x, coeffs_y, coeffs_z, currents_a):
    coeffs_x = np.ascontiguousarray(np.atleast_2d(coeffs_x), dtype=np.float64)
    coeffs_y = np.ascontiguousarray(np.atleast_2d(coeffs_y), dtype=np.float64)
    coeffs_z = np.ascontiguousarray(np.atleast_2d(coeffs_z), dtype=np.float64)
    currents_a = np.ascontiguousarray(currents_a, dtype=np.float64).ravel()
    if not (coeffs_x.shape == coeffs_y.shape == coeffs_z.shape):
        raise ValueError("coeffs_x/y/z must have the same shape")
    if currents_a.size != coeffs_x.shape[0]:
        raise ValueError("currents size must equal n_base_coils")
    return coeffs_x, coeffs_y, coeffs_z, currents_a


def _apply_score_config_overrides(config: _SgpuScoreConfig, overrides: dict | None) -> None:
    for name, value in (overrides or {}).items():
        if not hasattr(config, name):
            raise ValueError(f"unknown native score config field {name!r}")
        target = getattr(config, name)
        if isinstance(target, ctypes.Array):
            values = tuple(value)
            if len(values) != len(target):
                raise ValueError(
                    f"native score config array {name!r} requires {len(target)} values"
                )
            target[:] = values
        else:
            setattr(config, name, value)


def _score_result_dict(result: _SgpuScoreResult) -> dict:
    diagnostics = {
        name: getattr(result, name)
        for name, _ in _SgpuScoreResult._fields_
        if name not in {"components", "timings", "error_message"}
    }
    diagnostics["error_message"] = bytes(result.error_message).split(b"\0", 1)[0].decode("utf-8", "replace")
    return {
        "score": float(result.score),
        "status": SGPU_SCORE_STATUS_NAMES.get(int(result.status), f"unknown_{result.status}"),
        "components": {
            name: float(result.components[index])
            for index, name in enumerate(SGPU_SCORE_COMPONENT_NAMES)
        },
        "timing": {
            name: float(result.timings[index])
            for index, name in enumerate(SGPU_SCORE_TIMING_NAMES)
        },
        "diagnostics": diagnostics,
    }


def _with_boundary_iota(lib, result, query=-1):
    getter = getattr(lib, 'sgpu_get_boundary_iota', None)
    if getter is not None:
        getter.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        getter.restype = ctypes.c_int
        value = ctypes.c_double()
        if getter(query, ctypes.byref(value)) == 0:
            result['diagnostics']['iota_target_surface'] = value.value
    return result


def score_coils_native(
    lib_path: str | Path,
    coeffs_x,
    coeffs_y,
    coeffs_z,
    currents_a,
    nfp: int,
    *,
    device_id: int = 0,
    target_helicity: tuple[int, int] = (1, 0),
    config_overrides: dict | None = None,
) -> dict:
    """Call the all-native coil-to-score pipeline as a single black box."""
    path = str(Path(lib_path).resolve())
    lib = _NATIVE_SCORE_LIB_CACHE.get(path)
    if lib is None:
        lib = ctypes.CDLL(path)
        _bind_native_score(lib)
        _NATIVE_SCORE_LIB_CACHE[path] = lib
    config = _SgpuScoreConfig()
    _check_lib_code(lib, lib.sgpu_default_score_config(ctypes.byref(config)))
    config.device_id = int(device_id)
    config.target_M = int(target_helicity[0])
    config.target_N = int(target_helicity[1])
    _apply_score_config_overrides(config, config_overrides)

    coeffs_x = np.ascontiguousarray(np.atleast_2d(coeffs_x), dtype=np.float64)
    coeffs_y = np.ascontiguousarray(np.atleast_2d(coeffs_y), dtype=np.float64)
    coeffs_z = np.ascontiguousarray(np.atleast_2d(coeffs_z), dtype=np.float64)
    currents_a = np.ascontiguousarray(currents_a, dtype=np.float64).ravel()
    if not (coeffs_x.shape == coeffs_y.shape == coeffs_z.shape):
        raise ValueError("coeffs_x/y/z must have the same shape")
    if currents_a.size != coeffs_x.shape[0]:
        raise ValueError("currents size must equal n_base_coils")
    result = _SgpuScoreResult()
    code = lib.sgpu_score_coils(
        coeffs_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coeffs_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coeffs_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        currents_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(coeffs_x.shape[0]),
        ctypes.c_int(coeffs_x.shape[1]),
        ctypes.c_int(int(nfp)),
        ctypes.byref(config),
        ctypes.byref(result),
    )
    _check_lib_code(lib, code)
    diagnostics = {
        name: getattr(result, name)
        for name, _ in _SgpuScoreResult._fields_
        if name not in {"components", "timings", "error_message"}
    }
    diagnostics["error_message"] = bytes(result.error_message).split(b"\0", 1)[0].decode("utf-8", "replace")
    return _with_torsion(lib, _with_boundary_iota(lib, {
        "score": float(result.score),
        "status": SGPU_SCORE_STATUS_NAMES.get(int(result.status), f"unknown_{result.status}"),
        "components": {
            name: float(result.components[index])
            for index, name in enumerate(SGPU_SCORE_COMPONENT_NAMES)
        },
        "timing": {
            name: float(result.timings[index])
            for index, name in enumerate(SGPU_SCORE_TIMING_NAMES)
        },
        "diagnostics": diagnostics,
    }), coeffs_x, coeffs_y, coeffs_z)


COIL_METRIC_NAMES = (
    "length_mean",
    "curvature_p95",
    "curvature_max",
    "min_intercoil_distance",
    "min_axis_distance",
    "high_mode_fraction",
    "current_abs_max",
)


def coil_metric_gradient_native(
    lib_path: str | Path,
    coeffs_x,
    coeffs_y,
    coeffs_z,
    currents_a,
    nfp: int,
) -> dict:
    """Return the separated coil metrics and their analytical derivatives.

    Values follow COIL_METRIC_NAMES. Derivatives have shape
    (6, 3, n_base_coils, n_coeff) with metric order length_mean, curvature_p95,
    curvature_max, min_intercoil_distance, min_axis_distance, high_mode_fraction
    and coordinate order x, y, z.
    """
    path = str(Path(lib_path).resolve())
    lib = _NATIVE_SCORE_LIB_CACHE.get(path)
    if lib is None:
        lib = ctypes.CDLL(path)
        _bind_native_score(lib)
        _NATIVE_SCORE_LIB_CACHE[path] = lib
    if not _bind_native_metric_gradient(lib):
        raise GpuError("native library does not export sgpu_coil_metric_gradient")
    coeffs_x, coeffs_y, coeffs_z, currents_a = _coerce_score_inputs(
        coeffs_x, coeffs_y, coeffs_z, currents_a
    )
    n_base_coils, n_coeff = coeffs_x.shape
    values = np.zeros(len(COIL_METRIC_NAMES), dtype=np.float64)
    derivatives = np.zeros((6, 3, n_base_coils, n_coeff), dtype=np.float64)
    spacing_pair = np.full(2, -1, dtype=np.int32)
    code = lib.sgpu_coil_metric_gradient(
        coeffs_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coeffs_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coeffs_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        currents_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n_base_coils)),
        ctypes.c_int(int(n_coeff)),
        ctypes.c_int(int(nfp)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        derivatives.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        spacing_pair.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    _check_lib_code(lib, code)
    return {
        "values": {name: float(values[index]) for index, name in enumerate(COIL_METRIC_NAMES)},
        "derivatives": derivatives,
        "spacing_pair": [int(spacing_pair[0]), int(spacing_pair[1])],
    }


def coil_component_gradient_native(
    lib_path: str | Path,
    coeffs_x,
    coeffs_y,
    coeffs_z,
    currents_a,
    nfp: int,
) -> dict:
    """Return the 0--100 coil component and its active-branch analytical gradient."""
    path = str(Path(lib_path).resolve())
    lib = _NATIVE_SCORE_LIB_CACHE.get(path)
    if lib is None:
        lib = ctypes.CDLL(path)
        _bind_native_score(lib)
        _NATIVE_SCORE_LIB_CACHE[path] = lib
    _bind_native_gradient(lib)
    coeffs_x, coeffs_y, coeffs_z, currents_a = _coerce_score_inputs(
        coeffs_x, coeffs_y, coeffs_z, currents_a
    )
    gradient_x = np.empty_like(coeffs_x)
    gradient_y = np.empty_like(coeffs_y)
    gradient_z = np.empty_like(coeffs_z)
    gradient_current = np.empty_like(currents_a)
    component = ctypes.c_double()
    code = lib.sgpu_coil_component_gradient(
        coeffs_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coeffs_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coeffs_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        currents_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(coeffs_x.shape[0]),
        ctypes.c_int(coeffs_x.shape[1]),
        ctypes.c_int(int(nfp)),
        ctypes.byref(component),
        gradient_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gradient_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gradient_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gradient_current.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    _check_lib_code(lib, code)
    return {
        "component": float(component.value),
        "gradient": {
            "x": gradient_x,
            "y": gradient_y,
            "z": gradient_z,
            "current": gradient_current,
        },
    }


def score_coils_g1_gradient_native(
    lib_path: str | Path,
    coeffs_x,
    coeffs_y,
    coeffs_z,
    currents_a,
    nfp: int,
    *,
    device_id: int = 0,
    target_helicity: tuple[int, int] = (1, 0),
    config_overrides: dict | None = None,
) -> dict:
    """Run the exact score and opt-in G1 score gradient as a separate path."""
    path = str(Path(lib_path).resolve())
    lib = _NATIVE_SCORE_LIB_CACHE.get(path)
    if lib is None:
        lib = ctypes.CDLL(path)
        _bind_native_score(lib)
        _NATIVE_SCORE_LIB_CACHE[path] = lib
    _bind_native_gradient(lib)
    config = _SgpuScoreConfig()
    _check_lib_code(lib, lib.sgpu_default_score_config(ctypes.byref(config)))
    config.device_id = int(device_id)
    config.target_M = int(target_helicity[0])
    config.target_N = int(target_helicity[1])
    _apply_score_config_overrides(config, config_overrides)
    coeffs_x, coeffs_y, coeffs_z, currents_a = _coerce_score_inputs(
        coeffs_x, coeffs_y, coeffs_z, currents_a
    )
    gradient_x = np.empty_like(coeffs_x)
    gradient_y = np.empty_like(coeffs_y)
    gradient_z = np.empty_like(coeffs_z)
    gradient_current = np.empty_like(currents_a)
    score_result = _SgpuScoreResult()
    gradient_result = _SgpuScoreGradientResult()
    code = lib.sgpu_score_coils_g1_gradient(
        coeffs_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coeffs_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coeffs_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        currents_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(coeffs_x.shape[0]),
        ctypes.c_int(coeffs_x.shape[1]),
        ctypes.c_int(int(nfp)),
        ctypes.byref(config),
        ctypes.byref(score_result),
        gradient_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gradient_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gradient_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gradient_current.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(gradient_result),
    )
    _check_lib_code(lib, code)
    gradient_diagnostics = {
        name: getattr(gradient_result, name)
        for name, _ in _SgpuScoreGradientResult._fields_
        if name != "error_message"
    }
    gradient_diagnostics["error_message"] = bytes(gradient_result.error_message).split(b"\0", 1)[0].decode(
        "utf-8", "replace"
    )
    return {
        "score_result": _with_torsion(lib, _score_result_dict(score_result), coeffs_x, coeffs_y, coeffs_z),
        "gradient": {
            "x": gradient_x,
            "y": gradient_y,
            "z": gradient_z,
            "current": gradient_current,
        },
        "gradient_diagnostics": gradient_diagnostics,
    }


def score_coils_g2_gradient_native(
    lib_path: str | Path,
    coeffs_x,
    coeffs_y,
    coeffs_z,
    currents_a,
    nfp: int,
    *,
    device_id: int = 0,
    target_helicity: tuple[int, int] = (1, 0),
    config_overrides: dict | None = None,
) -> dict:
    """Run the exact score and opt-in cumulative fixed-front G1+G2 gradient.

    Valid non-OK scores are returned with zero gradients and a nonzero gradient
    diagnostic status, allowing callers to reject or backtrack the candidate.
    """
    path = str(Path(lib_path).resolve())
    lib = _NATIVE_SCORE_LIB_CACHE.get(path)
    if lib is None:
        lib = ctypes.CDLL(path)
        _bind_native_score(lib)
        _NATIVE_SCORE_LIB_CACHE[path] = lib
    _bind_native_gradient(lib)
    config = _SgpuScoreConfig()
    _check_lib_code(lib, lib.sgpu_default_score_config(ctypes.byref(config)))
    config.device_id = int(device_id)
    config.target_M = int(target_helicity[0])
    config.target_N = int(target_helicity[1])
    _apply_score_config_overrides(config, config_overrides)
    coeffs_x, coeffs_y, coeffs_z, currents_a = _coerce_score_inputs(
        coeffs_x, coeffs_y, coeffs_z, currents_a
    )
    gradient_x = np.empty_like(coeffs_x)
    gradient_y = np.empty_like(coeffs_y)
    gradient_z = np.empty_like(coeffs_z)
    gradient_current = np.empty_like(currents_a)
    score_result = _SgpuScoreResult()
    gradient_result = _SgpuScoreGradientResult()
    code = lib.sgpu_score_coils_g2_gradient(
        coeffs_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coeffs_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coeffs_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        currents_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(coeffs_x.shape[0]),
        ctypes.c_int(coeffs_x.shape[1]),
        ctypes.c_int(int(nfp)),
        ctypes.byref(config),
        ctypes.byref(score_result),
        gradient_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gradient_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gradient_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gradient_current.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(gradient_result),
    )
    _check_lib_code(lib, code)
    gradient_diagnostics = {
        name: getattr(gradient_result, name)
        for name, _ in _SgpuScoreGradientResult._fields_
        if name != "error_message"
    }
    gradient_diagnostics["error_message"] = bytes(gradient_result.error_message).split(b"\0", 1)[0].decode(
        "utf-8", "replace"
    )
    return {
        "score_result": _with_torsion(lib, _score_result_dict(score_result), coeffs_x, coeffs_y, coeffs_z),
        "gradient": {
            "x": gradient_x,
            "y": gradient_y,
            "z": gradient_z,
            "current": gradient_current,
        },
        "gradient_diagnostics": gradient_diagnostics,
    }


def score_coils_g2_frozen_batch_native(
    lib_path: str | Path,
    center_coeffs_x,
    center_coeffs_y,
    center_coeffs_z,
    center_currents_a,
    query_coeffs_x,
    query_coeffs_y,
    query_coeffs_z,
    query_currents_a,
    nfp: int,
    *,
    device_id: int = 0,
    target_helicity: tuple[int, int] = (1, 0),
    config_overrides: dict | None = None,
) -> dict:
    """Evaluate the exact scalar closure oracle for the fixed-front G2 VJP."""
    path = str(Path(lib_path).resolve())
    lib = _NATIVE_SCORE_LIB_CACHE.get(path)
    if lib is None:
        lib = ctypes.CDLL(path)
        _bind_native_score(lib)
        _NATIVE_SCORE_LIB_CACHE[path] = lib
    _bind_native_g2_frozen(lib)
    config = _SgpuScoreConfig()
    _check_lib_code(lib, lib.sgpu_default_score_config(ctypes.byref(config)))
    config.device_id = int(device_id)
    config.target_M = int(target_helicity[0])
    config.target_N = int(target_helicity[1])
    _apply_score_config_overrides(config, config_overrides)
    center_x, center_y, center_z, center_current = _coerce_score_inputs(
        center_coeffs_x, center_coeffs_y, center_coeffs_z, center_currents_a
    )
    query_x = np.ascontiguousarray(query_coeffs_x, dtype=np.float64)
    query_y = np.ascontiguousarray(query_coeffs_y, dtype=np.float64)
    query_z = np.ascontiguousarray(query_coeffs_z, dtype=np.float64)
    query_current = np.ascontiguousarray(query_currents_a, dtype=np.float64)
    if query_x.ndim == 2:
        query_x = query_x[None]
        query_y = query_y[None]
        query_z = query_z[None]
    if query_current.ndim == 1:
        query_current = query_current[None]
    expected_shape = (query_x.shape[0], *center_x.shape)
    if not (query_x.shape == query_y.shape == query_z.shape == expected_shape):
        raise ValueError(f"query coeffs must all have shape {expected_shape}")
    if query_current.shape != (query_x.shape[0], center_x.shape[0]):
        raise ValueError("query currents must have shape (query_count, n_base_coils)")
    outputs = [np.empty(query_x.shape[0], dtype=np.float64) for _ in range(6)]
    center_result = _SgpuScoreResult()
    code = lib.sgpu_score_coils_g2_frozen_batch(
        center_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        center_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        center_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        center_current.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        query_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        query_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        query_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        query_current.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(query_x.shape[0]),
        ctypes.c_int(center_x.shape[0]),
        ctypes.c_int(center_x.shape[1]),
        ctypes.c_int(int(nfp)),
        ctypes.byref(config),
        ctypes.byref(center_result),
        *(value.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for value in outputs),
    )
    _check_lib_code(lib, code)
    names = (
        "frozen_score",
        "volume_qs_component",
        "coil_component",
        "target_error",
        "qa_error",
        "qp_error",
    )
    return {
        "center_score_result": _with_torsion(lib, _score_result_dict(center_result), center_x, center_y, center_z),
        **{name: value for name, value in zip(names, outputs, strict=True)},
    }


def score_coils_g3_frozen_batch_native(
    lib_path: str | Path,
    center_coeffs_x,
    center_coeffs_y,
    center_coeffs_z,
    center_currents_a,
    query_coeffs_x,
    query_coeffs_y,
    query_coeffs_z,
    query_currents_a,
    nfp: int,
    *,
    device_id: int = 0,
    target_helicity: tuple[int, int] = (1, 0),
    config_overrides: dict | None = None,
) -> dict:
    """Evaluate the exact frozen-geometry scalar for cumulative G1+G2+G3."""
    path = str(Path(lib_path).resolve())
    lib = _NATIVE_SCORE_LIB_CACHE.get(path)
    if lib is None:
        lib = ctypes.CDLL(path)
        _bind_native_score(lib)
        _NATIVE_SCORE_LIB_CACHE[path] = lib
    _bind_native_g3_frozen(lib)
    config = _SgpuScoreConfig()
    _check_lib_code(lib, lib.sgpu_default_score_config(ctypes.byref(config)))
    config.device_id = int(device_id)
    config.target_M = int(target_helicity[0])
    config.target_N = int(target_helicity[1])
    _apply_score_config_overrides(config, config_overrides)
    center_x, center_y, center_z, center_current = _coerce_score_inputs(
        center_coeffs_x, center_coeffs_y, center_coeffs_z, center_currents_a
    )
    query_x = np.ascontiguousarray(query_coeffs_x, dtype=np.float64)
    query_y = np.ascontiguousarray(query_coeffs_y, dtype=np.float64)
    query_z = np.ascontiguousarray(query_coeffs_z, dtype=np.float64)
    query_current = np.ascontiguousarray(query_currents_a, dtype=np.float64)
    if query_x.ndim == 2:
        query_x = query_x[None]
        query_y = query_y[None]
        query_z = query_z[None]
    if query_current.ndim == 1:
        query_current = query_current[None]
    expected_shape = (query_x.shape[0], *center_x.shape)
    if not (query_x.shape == query_y.shape == query_z.shape == expected_shape):
        raise ValueError(f"query coeffs must all have shape {expected_shape}")
    if query_current.shape != (query_x.shape[0], center_x.shape[0]):
        raise ValueError("query currents must have shape (query_count, n_base_coils)")
    outputs = [np.empty(query_x.shape[0], dtype=np.float64) for _ in range(10)]
    center_result = _SgpuScoreResult()
    code = lib.sgpu_score_coils_g3_frozen_batch(
        center_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        center_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        center_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        center_current.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        query_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        query_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        query_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        query_current.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(query_x.shape[0]),
        ctypes.c_int(center_x.shape[0]),
        ctypes.c_int(center_x.shape[1]),
        ctypes.c_int(int(nfp)),
        ctypes.byref(config),
        ctypes.byref(center_result),
        *(value.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for value in outputs),
    )
    _check_lib_code(lib, code)
    names = (
        "frozen_score",
        "volume_qs_component",
        "coordinate_component",
        "iota_component",
        "coil_component",
        "target_error",
        "qa_error",
        "qp_error",
        "iota_min",
        "iota_max",
    )
    return {
        "center_score_result": _with_torsion(lib, _score_result_dict(center_result), center_x, center_y, center_z),
        **{name: value for name, value in zip(names, outputs, strict=True)},
    }


def score_coils_g4_fixed_branch_batch_native(
    lib_path: str | Path,
    center_coeffs_x,
    center_coeffs_y,
    center_coeffs_z,
    center_currents_a,
    query_coeffs_x,
    query_coeffs_y,
    query_coeffs_z,
    query_currents_a,
    nfp: int,
    *,
    device_id: int = 0,
    target_helicity: tuple[int, int] = (1, 0),
    config_overrides: dict | None = None,
) -> dict:
    """Recompute the continuous G4 front while fixing the center axis/level."""
    path = str(Path(lib_path).resolve())
    lib = _NATIVE_SCORE_LIB_CACHE.get(path)
    if lib is None:
        lib = ctypes.CDLL(path)
        _bind_native_score(lib)
        _NATIVE_SCORE_LIB_CACHE[path] = lib
    _bind_native_g4_fixed_branch(lib)
    config = _SgpuScoreConfig()
    _check_lib_code(lib, lib.sgpu_default_score_config(ctypes.byref(config)))
    config.device_id = int(device_id)
    config.target_M = int(target_helicity[0])
    config.target_N = int(target_helicity[1])
    _apply_score_config_overrides(config, config_overrides)
    center_x, center_y, center_z, center_current = _coerce_score_inputs(
        center_coeffs_x, center_coeffs_y, center_coeffs_z, center_currents_a
    )
    query_x = np.ascontiguousarray(query_coeffs_x, dtype=np.float64)
    query_y = np.ascontiguousarray(query_coeffs_y, dtype=np.float64)
    query_z = np.ascontiguousarray(query_coeffs_z, dtype=np.float64)
    query_current = np.ascontiguousarray(query_currents_a, dtype=np.float64)
    if query_x.ndim == 2:
        query_x = query_x[None]
        query_y = query_y[None]
        query_z = query_z[None]
    if query_current.ndim == 1:
        query_current = query_current[None]
    expected_shape = (query_x.shape[0], *center_x.shape)
    if not (query_x.shape == query_y.shape == query_z.shape == expected_shape):
        raise ValueError(f"query coeffs must all have shape {expected_shape}")
    if query_current.shape != (query_x.shape[0], center_x.shape[0]):
        raise ValueError("query currents must have shape (query_count, n_base_coils)")
    center_result = _SgpuScoreResult()
    query_results = (_SgpuScoreResult * query_x.shape[0])()
    code = lib.sgpu_score_coils_g4_fixed_branch_batch(
        center_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        center_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        center_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        center_current.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        query_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        query_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        query_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        query_current.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(query_x.shape[0]),
        ctypes.c_int(center_x.shape[0]),
        ctypes.c_int(center_x.shape[1]),
        ctypes.c_int(int(nfp)),
        ctypes.byref(config),
        ctypes.byref(center_result),
        query_results,
    )
    _check_lib_code(lib, code)
    return {
        "center_score_result": _with_torsion(lib, _score_result_dict(center_result), center_x, center_y, center_z),
        "query_score_results": [
            _with_torsion(lib, _score_result_dict(query_results[index]), query_x[index], query_y[index], query_z[index])
            for index in range(query_x.shape[0])
        ],
    }


def score_coils_psi_warm_batch_native(
    lib_path: str | Path,
    center_coeffs_x,
    center_coeffs_y,
    center_coeffs_z,
    center_currents_a,
    query_coeffs_x,
    query_coeffs_y,
    query_coeffs_z,
    query_currents_a,
    nfp: int,
    warm_iterations: int,
    *,
    use_center_qr_preconditioner: bool = False,
    device_id: int = 0,
    target_helicity: tuple[int, int] = (1, 0),
    config_overrides: dict | None = None,
) -> dict:
    """Recompute complete query scores with a center-warm full-density psi fit."""
    path = str(Path(lib_path).resolve())
    lib = _NATIVE_SCORE_LIB_CACHE.get(path)
    if lib is None:
        lib = ctypes.CDLL(path)
        _bind_native_score(lib)
        _NATIVE_SCORE_LIB_CACHE[path] = lib
    _bind_native_psi_warm_batch(lib)
    config = _SgpuScoreConfig()
    _check_lib_code(lib, lib.sgpu_default_score_config(ctypes.byref(config)))
    config.device_id = int(device_id)
    config.target_M = int(target_helicity[0])
    config.target_N = int(target_helicity[1])
    _apply_score_config_overrides(config, config_overrides)
    center_x, center_y, center_z, center_current = _coerce_score_inputs(
        center_coeffs_x, center_coeffs_y, center_coeffs_z, center_currents_a
    )
    query_x = np.ascontiguousarray(query_coeffs_x, dtype=np.float64)
    query_y = np.ascontiguousarray(query_coeffs_y, dtype=np.float64)
    query_z = np.ascontiguousarray(query_coeffs_z, dtype=np.float64)
    query_current = np.ascontiguousarray(query_currents_a, dtype=np.float64)
    if query_x.ndim == 2:
        query_x = query_x[None]
        query_y = query_y[None]
        query_z = query_z[None]
    if query_current.ndim == 1:
        query_current = query_current[None]
    expected_shape = (query_x.shape[0], *center_x.shape)
    if not (query_x.shape == query_y.shape == query_z.shape == expected_shape):
        raise ValueError(f"query coeffs must all have shape {expected_shape}")
    if query_current.shape != (query_x.shape[0], center_x.shape[0]):
        raise ValueError("query currents must have shape (query_count, n_base_coils)")
    if int(warm_iterations) < 0:
        raise ValueError("warm_iterations must be nonnegative")
    center_result = _SgpuScoreResult()
    query_results = (_SgpuScoreResult * query_x.shape[0])()
    code = lib.sgpu_score_coils_psi_warm_batch(
        center_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        center_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        center_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        center_current.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        query_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        query_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        query_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        query_current.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(query_x.shape[0]),
        ctypes.c_int(center_x.shape[0]),
        ctypes.c_int(center_x.shape[1]),
        ctypes.c_int(int(nfp)),
        ctypes.c_int(int(warm_iterations)),
        ctypes.c_int(int(bool(use_center_qr_preconditioner))),
        ctypes.byref(config),
        ctypes.byref(center_result),
        query_results,
    )
    _check_lib_code(lib, code)
    return {
        "center_score_result": _with_torsion(lib, _score_result_dict(center_result), center_x, center_y, center_z),
        "query_score_results": [
            _with_torsion(lib, _score_result_dict(query_results[index]), query_x[index], query_y[index], query_z[index])
            for index in range(query_x.shape[0])
        ],
    }


def score_coils_g3_gradient_native(
    lib_path: str | Path,
    coeffs_x,
    coeffs_y,
    coeffs_z,
    currents_a,
    nfp: int,
    *,
    device_id: int = 0,
    target_helicity: tuple[int, int] = (1, 0),
    config_overrides: dict | None = None,
) -> dict:
    """Run the exact score and cumulative fixed-front G1+G2+G3 gradient."""
    path = str(Path(lib_path).resolve())
    lib = _NATIVE_SCORE_LIB_CACHE.get(path)
    if lib is None:
        lib = ctypes.CDLL(path)
        _bind_native_score(lib)
        _NATIVE_SCORE_LIB_CACHE[path] = lib
    _bind_native_gradient(lib)
    config = _SgpuScoreConfig()
    _check_lib_code(lib, lib.sgpu_default_score_config(ctypes.byref(config)))
    config.device_id = int(device_id)
    config.target_M = int(target_helicity[0])
    config.target_N = int(target_helicity[1])
    _apply_score_config_overrides(config, config_overrides)
    coeffs_x, coeffs_y, coeffs_z, currents_a = _coerce_score_inputs(
        coeffs_x, coeffs_y, coeffs_z, currents_a
    )
    gradient_x = np.empty_like(coeffs_x)
    gradient_y = np.empty_like(coeffs_y)
    gradient_z = np.empty_like(coeffs_z)
    gradient_current = np.empty_like(currents_a)
    score_result = _SgpuScoreResult()
    gradient_result = _SgpuScoreGradientResult()
    code = lib.sgpu_score_coils_g3_gradient(
        coeffs_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coeffs_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        coeffs_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        currents_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(coeffs_x.shape[0]),
        ctypes.c_int(coeffs_x.shape[1]),
        ctypes.c_int(int(nfp)),
        ctypes.byref(config),
        ctypes.byref(score_result),
        gradient_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gradient_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gradient_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gradient_current.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(gradient_result),
    )
    _check_lib_code(lib, code)
    gradient_diagnostics = {
        name: getattr(gradient_result, name)
        for name, _ in _SgpuScoreGradientResult._fields_
        if name != "error_message"
    }
    gradient_diagnostics["error_message"] = bytes(gradient_result.error_message).split(b"\0", 1)[0].decode(
        "utf-8", "replace"
    )
    return {
        "score_result": _with_torsion(lib, _score_result_dict(score_result), coeffs_x, coeffs_y, coeffs_z),
        "gradient": {
            "x": gradient_x,
            "y": gradient_y,
            "z": gradient_z,
            "current": gradient_current,
        },
        "gradient_diagnostics": gradient_diagnostics,
    }


def native_score_config_snapshot(
    lib_path: str | Path,
    *,
    device_id: int = 0,
    target_helicity: tuple[int, int] = (1, 0),
    config_overrides: dict | None = None,
) -> dict:
    """Return the exact native score configuration used by score_coils_native."""
    path = str(Path(lib_path).resolve())
    lib = _NATIVE_SCORE_LIB_CACHE.get(path)
    if lib is None:
        lib = ctypes.CDLL(path)
        _bind_native_score(lib)
        _NATIVE_SCORE_LIB_CACHE[path] = lib
    config = _SgpuScoreConfig()
    _check_lib_code(lib, lib.sgpu_default_score_config(ctypes.byref(config)))
    config.device_id = int(device_id)
    config.target_M = int(target_helicity[0])
    config.target_N = int(target_helicity[1])
    _apply_score_config_overrides(config, config_overrides)
    snapshot = {}
    for name, _ in _SgpuScoreConfig._fields_:
        value = getattr(config, name)
        if isinstance(value, ctypes.Array):
            snapshot[name] = [item for item in value]
        else:
            snapshot[name] = value
    return snapshot


def _check_lib_code(lib: ctypes.CDLL, code: int):
    if code:
        msg = lib.sgpu_last_error()
        raise GpuError(msg.decode("utf-8", "replace") if msg else "unknown GPU backend error")


def _load_utility_lib(lib_path: str | Path) -> ctypes.CDLL:
    path = str(Path(lib_path))
    lib = _UTILITY_LIB_CACHE.get(path)
    if lib is not None:
        return lib
    lib = ctypes.CDLL(path)
    lib.sgpu_surface_points_from_level.restype = ctypes.c_int
    lib.sgpu_surface_points_from_level.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    lib.sgpu_last_error.restype = ctypes.c_char_p
    _UTILITY_LIB_CACHE[path] = lib
    return lib


def surface_points_from_level_gpu(
    lib_path: str | Path,
    coeffs,
    mode_a,
    mode_b,
    mode_m,
    mode_kind,
    *,
    nfp: int,
    a: float,
    poly_degree: int,
    m_tor: int,
    axis_R,
    axis_Z,
    order: int,
    psi_level: float,
    maxiter: int,
    tol: float,
    max_radius_scale: float,
    device_id: int = 0,
):
    lib = _load_utility_lib(lib_path)
    coeffs = np.ascontiguousarray(coeffs, dtype=np.float64).ravel()
    mode_a = np.ascontiguousarray(mode_a, dtype=np.int32).ravel()
    mode_b = np.ascontiguousarray(mode_b, dtype=np.int32).ravel()
    mode_m = np.ascontiguousarray(mode_m, dtype=np.int32).ravel()
    mode_kind = np.ascontiguousarray(mode_kind, dtype=np.int32).ravel()
    axis_R = np.ascontiguousarray(axis_R, dtype=np.float64).ravel()
    axis_Z = np.ascontiguousarray(axis_Z, dtype=np.float64).ravel()
    if not (mode_a.shape == mode_b.shape == mode_m.shape == mode_kind.shape):
        raise ValueError("mode arrays shape mismatch")
    if coeffs.size != mode_a.size:
        raise ValueError("coeff length must equal mode count")
    nphi = 2 * int(order) + 1
    ntheta = 2 * int(order) + 1
    xyz = np.empty((nphi, ntheta, 3), dtype=np.float64)
    radii = np.empty((nphi, ntheta), dtype=np.float64)
    stats = np.empty(5, dtype=np.float64)
    code = lib.sgpu_surface_points_from_level(
        coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mode_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        mode_b.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        mode_m.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        mode_kind.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(coeffs.size),
        ctypes.c_int(int(nfp)),
        ctypes.c_double(float(a)),
        ctypes.c_int(int(poly_degree)),
        ctypes.c_int(int(m_tor)),
        axis_R.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        axis_Z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(axis_R.size),
        ctypes.c_int(int(order)),
        ctypes.c_double(float(psi_level)),
        ctypes.c_int(int(maxiter)),
        ctypes.c_double(float(tol)),
        ctypes.c_double(float(max_radius_scale)),
        ctypes.c_int(int(device_id)),
        xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        radii.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(stats.size),
    )
    _check_lib_code(lib, code)
    return xyz, radii, {
        "copy_in_s": float(stats[0]),
        "coeff_build_s": float(stats[1]),
        "newton_s": float(stats[2]),
        "copy_out_s": float(stats[3]),
        "total_s": float(stats[4]),
    }


class CoilFieldGpu:
    def __init__(
        self,
        lib_path: str | Path,
        coeffs_x,
        coeffs_y,
        coeffs_z,
        currents_a,
        nfp: int,
        segments_per_coil: int = 256,
        device_id: int = 0,
    ):
        self.lib = ctypes.CDLL(str(lib_path))
        self._bind()
        self.handle = ctypes.c_void_p()
        self.coeffs_x = np.ascontiguousarray(np.atleast_2d(coeffs_x), dtype=np.float64)
        self.coeffs_y = np.ascontiguousarray(np.atleast_2d(coeffs_y), dtype=np.float64)
        self.coeffs_z = np.ascontiguousarray(np.atleast_2d(coeffs_z), dtype=np.float64)
        self.currents_a = np.ascontiguousarray(currents_a, dtype=np.float64).ravel()
        if not (self.coeffs_x.shape == self.coeffs_y.shape == self.coeffs_z.shape):
            raise ValueError("coeffs_x/y/z must have the same shape")
        if self.currents_a.size != self.coeffs_x.shape[0]:
            raise ValueError("currents size must equal n_base_coils")
        code = self.lib.sgpu_create_field(
            self.coeffs_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.coeffs_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.coeffs_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.currents_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(self.coeffs_x.shape[0]),
            ctypes.c_int(self.coeffs_x.shape[1]),
            ctypes.c_int(nfp),
            ctypes.c_int(segments_per_coil),
            ctypes.c_int(device_id),
            ctypes.byref(self.handle),
        )
        self._check(code)
        self.nfp = int(nfp)
        self.segments_per_coil = int(segments_per_coil)
        self.device_id = int(device_id)

    def _bind(self):
        self.lib.sgpu_create_field.restype = ctypes.c_int
        self.lib.sgpu_create_field.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.sgpu_destroy_field.restype = None
        self.lib.sgpu_destroy_field.argtypes = [ctypes.c_void_p]
        self.lib.sgpu_segment_count.restype = ctypes.c_int
        self.lib.sgpu_segment_count.argtypes = [ctypes.c_void_p]
        self.lib.sgpu_eval_B.restype = ctypes.c_int
        self.lib.sgpu_eval_B.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self.has_eval_B_f32 = hasattr(self.lib, "sgpu_eval_B_f32")
        if self.has_eval_B_f32:
            self.lib.sgpu_eval_B_f32.restype = ctypes.c_int
            self.lib.sgpu_eval_B_f32.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
        self.has_eval_B_grad = hasattr(self.lib, "sgpu_eval_B_grad") and hasattr(
            self.lib, "sgpu_eval_B_grad_f32"
        )
        if self.has_eval_B_grad:
            self.lib.sgpu_eval_B_grad.restype = ctypes.c_int
            self.lib.sgpu_eval_B_grad.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
            ]
            self.lib.sgpu_eval_B_grad_f32.restype = ctypes.c_int
            self.lib.sgpu_eval_B_grad_f32.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
        self.has_eval_B_grad_point_vjp = hasattr(
            self.lib, "sgpu_eval_B_grad_point_vjp_f32"
        )
        if self.has_eval_B_grad_point_vjp:
            self.lib.sgpu_eval_B_grad_point_vjp_f32.restype = ctypes.c_int
            self.lib.sgpu_eval_B_grad_point_vjp_f32.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
        self.lib.sgpu_normal_eq.restype = ctypes.c_int
        self.lib.sgpu_normal_eq.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.sgpu_normal_eq_f32.restype = ctypes.c_int
        self.lib.sgpu_normal_eq_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.sgpu_fit_psi_fullgpu.restype = ctypes.c_int
        self.lib.sgpu_fit_psi_fullgpu.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self.lib.sgpu_trace_period.restype = ctypes.c_int
        self.lib.sgpu_trace_period.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.sgpu_trace_period_blockline.restype = ctypes.c_int
        self.lib.sgpu_trace_period_blockline.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.sgpu_trace_period_blockline_mixed.restype = ctypes.c_int
        self.lib.sgpu_trace_period_blockline_mixed.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.sgpu_trace_axis_samples.restype = ctypes.c_int
        self.lib.sgpu_trace_axis_samples.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.sgpu_last_error.restype = ctypes.c_char_p

    def _check(self, code: int):
        if code:
            msg = self.lib.sgpu_last_error()
            raise GpuError(msg.decode("utf-8", "replace") if msg else "unknown GPU backend error")

    @property
    def segment_count(self) -> int:
        return int(self.lib.sgpu_segment_count(self.handle))

    def close(self):
        if getattr(self, "handle", None):
            self.lib.sgpu_destroy_field(self.handle)
            self.handle = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def eval_B(self, xyz, precision: str = "fp64"):
        precision = precision.lower()
        if precision not in {"fp32", "fp64"}:
            raise ValueError("eval_B precision must be 'fp32' or 'fp64'")
        if precision == "fp32" and not self.has_eval_B_f32:
            raise GpuError("this GPU backend library does not provide FP32 B evaluation")
        dtype = np.float32 if precision == "fp32" else np.float64
        pts = np.ascontiguousarray(xyz, dtype=dtype).reshape(-1, 3)
        out = np.empty_like(pts)
        pointer = ctypes.POINTER(ctypes.c_float) if precision == "fp32" else ctypes.POINTER(ctypes.c_double)
        function = self.lib.sgpu_eval_B_f32 if precision == "fp32" else self.lib.sgpu_eval_B
        code = function(
            self.handle,
            pts.ctypes.data_as(pointer),
            out.ctypes.data_as(pointer),
            ctypes.c_int(len(pts)),
        )
        self._check(code)
        return out

    def eval_B_grad(self, xyz, precision: str = "fp32"):
        if not self.has_eval_B_grad:
            raise GpuError("this GPU backend library does not provide B-gradient evaluation")
        precision = precision.lower()
        if precision not in {"fp32", "fp64"}:
            raise ValueError("eval_B_grad precision must be 'fp32' or 'fp64'")
        dtype = np.float32 if precision == "fp32" else np.float64
        points = np.ascontiguousarray(xyz, dtype=dtype).reshape(-1, 3)
        field = np.empty_like(points)
        gradient = np.empty((len(points), 3, 3), dtype=dtype)
        pointer = ctypes.POINTER(ctypes.c_float) if precision == "fp32" else ctypes.POINTER(ctypes.c_double)
        function = self.lib.sgpu_eval_B_grad_f32 if precision == "fp32" else self.lib.sgpu_eval_B_grad
        code = function(
            self.handle,
            points.ctypes.data_as(pointer),
            field.ctypes.data_as(pointer),
            gradient.ctypes.data_as(pointer),
            ctypes.c_int(len(points)),
        )
        self._check(code)
        return field, gradient

    def eval_B_grad_point_vjp(self, xyz, adj_B, adj_grad_B):
        if not self.has_eval_B_grad_point_vjp:
            raise GpuError("this GPU backend does not provide the B/grad(B) point VJP")
        points = np.ascontiguousarray(xyz, dtype=np.float32).reshape(-1, 3)
        field_adjoint = np.ascontiguousarray(adj_B, dtype=np.float32).reshape(-1, 3)
        gradient_adjoint = np.ascontiguousarray(adj_grad_B, dtype=np.float32).reshape(-1, 3, 3)
        if field_adjoint.shape != points.shape or gradient_adjoint.shape != (len(points), 3, 3):
            raise ValueError("point and B/grad(B) adjoint shapes must agree")
        point_adjoint = np.empty_like(points)
        code = self.lib.sgpu_eval_B_grad_point_vjp_f32(
            self.handle,
            points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            field_adjoint.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gradient_adjoint.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            point_adjoint.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(len(points)),
        )
        self._check(code)
        return point_adjoint

    def normal_eq(self, mat, rhs, precision: str = "fp64"):
        precision = precision.lower()
        if precision not in {"fp64", "fp32"}:
            raise ValueError("normal_eq precision must be 'fp64' or 'fp32'")
        dtype = np.float32 if precision == "fp32" else np.float64
        mat = np.ascontiguousarray(mat, dtype=dtype)
        if mat.ndim != 2:
            raise ValueError("mat must be a 2D array")
        rhs = np.ascontiguousarray(rhs, dtype=dtype).ravel()
        if rhs.size != mat.shape[0]:
            raise ValueError("rhs length must equal mat.shape[0]")
        ata = np.empty((mat.shape[1], mat.shape[1]), dtype=dtype)
        atb = np.empty(mat.shape[1], dtype=dtype)
        if precision == "fp32":
            code = self.lib.sgpu_normal_eq_f32(
                self.handle,
                mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                rhs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ata.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                atb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(mat.shape[0]),
                ctypes.c_int(mat.shape[1]),
            )
        else:
            code = self.lib.sgpu_normal_eq(
                self.handle,
                mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                rhs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ata.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                atb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(mat.shape[0]),
                ctypes.c_int(mat.shape[1]),
            )
        self._check(code)
        return ata.astype(np.float64, copy=False), atb.astype(np.float64, copy=False)

    def fit_psi_fullgpu(
        self,
        R,
        Z,
        phi,
        axis_R,
        axis_Z,
        axis_R_phi,
        axis_Z_phi,
        mode_a,
        mode_b,
        mode_m,
        mode_kind,
        *,
        a: float,
        poly_degree: int,
        m_tor: int,
        ridge: float,
        precision: str = "fp64",
        solver: str = "normal_eq",
    ):
        precision = precision.lower()
        if precision not in {"fp64", "fp32"}:
            raise ValueError("fit_psi_fullgpu precision must be 'fp64' or 'fp32'")
        solver = solver.lower()
        if solver not in {"normal_eq", "qr"}:
            raise ValueError("fit_psi_fullgpu solver must be 'normal_eq' or 'qr'")
        R = np.ascontiguousarray(R, dtype=np.float64).ravel()
        Z = np.ascontiguousarray(Z, dtype=np.float64).ravel()
        phi = np.ascontiguousarray(phi, dtype=np.float64).ravel()
        axis_R = np.ascontiguousarray(axis_R, dtype=np.float64).ravel()
        axis_Z = np.ascontiguousarray(axis_Z, dtype=np.float64).ravel()
        axis_R_phi = np.ascontiguousarray(axis_R_phi, dtype=np.float64).ravel()
        axis_Z_phi = np.ascontiguousarray(axis_Z_phi, dtype=np.float64).ravel()
        mode_a = np.ascontiguousarray(mode_a, dtype=np.int32).ravel()
        mode_b = np.ascontiguousarray(mode_b, dtype=np.int32).ravel()
        mode_m = np.ascontiguousarray(mode_m, dtype=np.int32).ravel()
        mode_kind = np.ascontiguousarray(mode_kind, dtype=np.int32).ravel()
        if not (R.shape == Z.shape == phi.shape):
            raise ValueError("R/Z/phi shape mismatch")
        if not (axis_R.shape == axis_Z.shape == axis_R_phi.shape == axis_Z_phi.shape):
            raise ValueError("axis arrays shape mismatch")
        if not (mode_a.shape == mode_b.shape == mode_m.shape == mode_kind.shape):
            raise ValueError("mode arrays shape mismatch")
        coeff = np.empty(mode_a.size, dtype=np.float64)
        train_rms = np.empty(1, dtype=np.float64)
        stats = np.empty(12, dtype=np.float64)
        code = self.lib.sgpu_fit_psi_fullgpu(
            self.handle,
            R.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            phi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(R.size),
            axis_R.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            axis_Z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            axis_R_phi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            axis_Z_phi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(axis_R.size),
            mode_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            mode_b.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            mode_m.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            mode_kind.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(mode_a.size),
            ctypes.c_int(self.nfp),
            ctypes.c_double(a),
            ctypes.c_int(poly_degree),
            ctypes.c_int(m_tor),
            ctypes.c_double(ridge),
            ctypes.c_int(1 if solver == "normal_eq" else 2),
            ctypes.c_int(1 if precision == "fp64" else 2),
            coeff.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            train_rms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(stats.size),
        )
        self._check(code)
        return coeff, float(train_rms[0]), {
            "copy_in_s": float(stats[0]),
            "assemble_s": float(stats[1]),
            "linear_prep_s": float(stats[2]),
            "solve_s": float(stats[3]),
            "residual_s": float(stats[4]),
            "copy_out_s": float(stats[5]),
            "total_s": float(stats[6]),
            "qr_transpose_s": float(stats[7]),
            "qr_scale_s": float(stats[8]),
            "qr_factor_s": float(stats[9]),
            "qr_apply_qtb_s": float(stats[10]),
            "qr_tri_s": float(stats[11]),
        }

    def trace_period(self, R0, Z0, steps: int, nfp: int | None = None):
        R0 = np.ascontiguousarray(R0, dtype=np.float64).ravel()
        Z0 = np.ascontiguousarray(Z0, dtype=np.float64).ravel()
        if R0.shape != Z0.shape:
            raise ValueError("R0/Z0 shape mismatch")
        R1 = np.empty_like(R0)
        Z1 = np.empty_like(Z0)
        code = self.lib.sgpu_trace_period(
            self.handle,
            R0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Z0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            R1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Z1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(R0.size),
            ctypes.c_int(self.nfp if nfp is None else nfp),
            ctypes.c_int(steps),
        )
        self._check(code)
        return R1, Z1

    def trace_period_blockline(self, R0, Z0, steps: int, threads_per_line: int = 256, nfp: int | None = None):
        R0 = np.ascontiguousarray(R0, dtype=np.float64).ravel()
        Z0 = np.ascontiguousarray(Z0, dtype=np.float64).ravel()
        if R0.shape != Z0.shape:
            raise ValueError("R0/Z0 shape mismatch")
        R1 = np.empty_like(R0)
        Z1 = np.empty_like(Z0)
        code = self.lib.sgpu_trace_period_blockline(
            self.handle,
            R0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Z0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            R1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Z1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(R0.size),
            ctypes.c_int(self.nfp if nfp is None else nfp),
            ctypes.c_int(steps),
            ctypes.c_int(threads_per_line),
        )
        self._check(code)
        return R1, Z1

    def trace_period_blockline_mixed(
        self,
        R0,
        Z0,
        steps: int,
        threads_per_line: int = 256,
        mode: str = "bf32_state64",
        nfp: int | None = None,
    ):
        mode_id = {"bf32_state64": 1, "f32": 2, "f32_state16": 3}[mode]
        R0 = np.ascontiguousarray(R0, dtype=np.float64).ravel()
        Z0 = np.ascontiguousarray(Z0, dtype=np.float64).ravel()
        if R0.shape != Z0.shape:
            raise ValueError("R0/Z0 shape mismatch")
        R1 = np.empty_like(R0)
        Z1 = np.empty_like(Z0)
        code = self.lib.sgpu_trace_period_blockline_mixed(
            self.handle,
            R0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Z0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            R1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Z1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(R0.size),
            ctypes.c_int(self.nfp if nfp is None else nfp),
            ctypes.c_int(steps),
            ctypes.c_int(threads_per_line),
            ctypes.c_int(mode_id),
        )
        self._check(code)
        return R1, Z1

    def trace_period_blockline_precision(
        self,
        R0,
        Z0,
        steps: int,
        precision: str = "mixed64",
        threads_per_line: int = 256,
        nfp: int | None = None,
    ):
        """Trace one field period with the block-per-line kernel.

        ``precision`` options:
        - ``mixed64``: fp32 Biot-Savart accumulation, fp64 RK state. This is the default.
        - ``fp64``: full fp64 block-per-line tracing.
        - ``fp32``: fp32 Biot-Savart accumulation and fp32 RK state. Use for coarse screening.
        """
        key = TRACE_PRECISION_ALIASES.get(precision)
        if key is None:
            choices = ", ".join(sorted(set(TRACE_PRECISION_ALIASES)))
            raise ValueError(f"unknown trace precision {precision!r}; choices: {choices}")
        if key == "mixed64":
            return self.trace_period_blockline_mixed(
                R0, Z0, steps, threads_per_line=threads_per_line, mode="bf32_state64", nfp=nfp
            )
        if key == "fp32":
            return self.trace_period_blockline_mixed(
                R0, Z0, steps, threads_per_line=threads_per_line, mode="f32", nfp=nfp
            )
        return self.trace_period_blockline(R0, Z0, steps, threads_per_line=threads_per_line, nfp=nfp)

    def trace_axis_samples(
        self,
        R0: float,
        Z0: float,
        *,
        integration_steps: int,
        sample_count: int,
        nfp: int | None = None,
    ):
        outputs = [np.empty(int(sample_count), dtype=np.float64) for _ in range(4)]
        code = self.lib.sgpu_trace_axis_samples(
            self.handle,
            ctypes.c_double(float(R0)),
            ctypes.c_double(float(Z0)),
            ctypes.c_int(self.nfp if nfp is None else int(nfp)),
            ctypes.c_int(int(integration_steps)),
            ctypes.c_int(int(sample_count)),
            *(array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for array in outputs),
        )
        self._check(code)
        return tuple(outputs)


class BatchCoilFieldGpu:
    """Experimental query-major FP32 field for local finite differences."""

    def __init__(
        self,
        lib_path: str | Path,
        coeffs_x,
        coeffs_y,
        coeffs_z,
        currents_a,
        nfp: int,
        segments_per_coil: int = 256,
        device_id: int = 0,
    ):
        self.lib = ctypes.CDLL(str(lib_path))
        self._bind()
        self.handle = ctypes.c_void_p()
        self.coeffs_x = np.ascontiguousarray(coeffs_x, dtype=np.float64)
        self.coeffs_y = np.ascontiguousarray(coeffs_y, dtype=np.float64)
        self.coeffs_z = np.ascontiguousarray(coeffs_z, dtype=np.float64)
        self.currents_a = np.ascontiguousarray(currents_a, dtype=np.float64)
        if self.coeffs_x.ndim != 3:
            raise ValueError("batch coefficients must have shape (query, coil, coefficient)")
        if not (self.coeffs_x.shape == self.coeffs_y.shape == self.coeffs_z.shape):
            raise ValueError("batch coeffs_x/y/z must have the same shape")
        if self.currents_a.shape != self.coeffs_x.shape[:2]:
            raise ValueError("batch currents must have shape (query, coil)")
        self.query_count, self.n_base_coils, self.n_coeff = self.coeffs_x.shape
        code = self.lib.sgpu_create_field_batch_f32(
            self.coeffs_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.coeffs_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.coeffs_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.currents_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(self.query_count),
            ctypes.c_int(self.n_base_coils),
            ctypes.c_int(self.n_coeff),
            ctypes.c_int(int(nfp)),
            ctypes.c_int(int(segments_per_coil)),
            ctypes.c_int(int(device_id)),
            ctypes.byref(self.handle),
        )
        self._check(code)
        self.nfp = int(nfp)
        self.segments_per_coil = int(segments_per_coil)
        self.device_id = int(device_id)

    def _bind(self):
        double_pointer = ctypes.POINTER(ctypes.c_double)
        float_pointer = ctypes.POINTER(ctypes.c_float)
        int_pointer = ctypes.POINTER(ctypes.c_int)
        _bind_native_score(self.lib)
        self.lib.sgpu_create_field_batch_f32.restype = ctypes.c_int
        self.lib.sgpu_create_field_batch_f32.argtypes = [
            double_pointer, double_pointer, double_pointer, double_pointer,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.sgpu_destroy_field_batch.restype = None
        self.lib.sgpu_destroy_field_batch.argtypes = [ctypes.c_void_p]
        self.lib.sgpu_batch_eval_B_f32.restype = ctypes.c_int
        self.lib.sgpu_batch_eval_B_f32.argtypes = [
            ctypes.c_void_p, float_pointer, float_pointer, ctypes.c_int,
        ]
        self.lib.sgpu_batch_eval_B_grad_f32.restype = ctypes.c_int
        self.lib.sgpu_batch_eval_B_grad_f32.argtypes = [
            ctypes.c_void_p, float_pointer, float_pointer, float_pointer, ctypes.c_int,
        ]
        self.lib.sgpu_batch_trace_period_mixed.restype = ctypes.c_int
        self.lib.sgpu_batch_trace_period_mixed.argtypes = [
            ctypes.c_void_p, double_pointer, double_pointer, double_pointer,
            double_pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.lib.sgpu_batch_trace_axis_samples.restype = ctypes.c_int
        self.lib.sgpu_batch_trace_axis_samples.argtypes = [
            ctypes.c_void_p, double_pointer, double_pointer, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, double_pointer, double_pointer,
            double_pointer, double_pointer,
        ]
        self.lib.sgpu_batch_refine_axis_hint.restype = ctypes.c_int
        self.lib.sgpu_batch_refine_axis_hint.argtypes = [
            ctypes.c_void_p, double_pointer, double_pointer,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            double_pointer, double_pointer, double_pointer, double_pointer,
            double_pointer, ctypes.POINTER(ctypes.c_ubyte),
        ]
        self.lib.sgpu_score_coils_capture_psi_center.restype = ctypes.c_int
        self.lib.sgpu_score_coils_capture_psi_center.argtypes = [
            double_pointer, double_pointer, double_pointer, double_pointer,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(_SgpuScoreConfig), ctypes.POINTER(_SgpuScoreResult),
            double_pointer, ctypes.c_int, int_pointer,
        ]
        self.lib.sgpu_fit_psi_batch_pcgls_f32.restype = ctypes.c_int
        self.lib.sgpu_fit_psi_batch_pcgls_f32.argtypes = [
            ctypes.c_void_p,
            double_pointer, double_pointer, double_pointer, double_pointer,
            ctypes.c_int,
            int_pointer, int_pointer, int_pointer, int_pointer,
            ctypes.c_int, ctypes.c_int, ctypes.c_double,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_int,
            double_pointer, double_pointer, double_pointer,
            double_pointer, ctypes.c_int,
        ]
        self.lib.sgpu_score_coils_local_batch.restype = ctypes.c_int
        self.lib.sgpu_score_coils_local_batch.argtypes = [
            ctypes.c_void_p, double_pointer, ctypes.c_int, ctypes.c_int,
            double_pointer, double_pointer, double_pointer, double_pointer,
            ctypes.c_int, double_pointer, double_pointer, double_pointer,
            double_pointer, ctypes.c_int, double_pointer, double_pointer,
            ctypes.POINTER(_SgpuScoreResult), ctypes.POINTER(_SgpuScoreConfig),
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(_SgpuScoreResult),
            double_pointer, ctypes.c_int,
        ]
        self.lib.sgpu_clear_psi_warm_preconditioner.restype = None
        self.lib.sgpu_clear_psi_warm_preconditioner.argtypes = []
        self.lib.sgpu_last_error.restype = ctypes.c_char_p

    def _check(self, code: int):
        if code:
            message = self.lib.sgpu_last_error()
            raise GpuError(
                message.decode("utf-8", "replace") if message else "unknown GPU backend error"
            )

    def close(self):
        if getattr(self, "handle", None):
            self.lib.sgpu_destroy_field_batch(self.handle)
            self.handle = ctypes.c_void_p()

    def captured_iota_profile(self):
        """Native volume polynomial in u=rho²; edge at u=1, after center capture."""
        getter = self.lib.sgpu_get_captured_iota_profile
        getter.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        getter.restype = ctypes.c_int
        coefficients = np.empty(64, dtype=np.float64)
        count = ctypes.c_int()
        self._check(getter(coefficients.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 64, ctypes.byref(count)))
        return coefficients[:count.value].copy()

    def capture_psi_center(
        self,
        coeffs_x,
        coeffs_y,
        coeffs_z,
        currents_a,
        *,
        target_helicity: tuple[int, int] = (1, 0),
        config_overrides: dict | None = None,
        coefficient_capacity: int = 8192,
        include_torsion: bool = True,
    ):
        """Score the center and retain its psi QR factor on this host thread."""
        center_x, center_y, center_z, center_current = _coerce_score_inputs(
            coeffs_x, coeffs_y, coeffs_z, currents_a
        )
        if center_x.shape != (self.n_base_coils, self.n_coeff):
            raise ValueError("center coil shape must match the batch field")
        config = _SgpuScoreConfig()
        self._check(self.lib.sgpu_default_score_config(ctypes.byref(config)))
        config.device_id = self.device_id
        config.target_M = int(target_helicity[0])
        config.target_N = int(target_helicity[1])
        _apply_score_config_overrides(config, config_overrides)
        coefficients = np.empty(int(coefficient_capacity), dtype=np.float64)
        coefficient_count = ctypes.c_int()
        result = _SgpuScoreResult()
        code = self.lib.sgpu_score_coils_capture_psi_center(
            center_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            center_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            center_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            center_current.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(self.n_base_coils),
            ctypes.c_int(self.n_coeff),
            ctypes.c_int(self.nfp),
            ctypes.byref(config),
            ctypes.byref(result),
            coefficients.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(coefficients.size),
            ctypes.byref(coefficient_count),
        )
        self._check(code)
        score_result = _with_boundary_iota(self.lib, _score_result_dict(result))
        if include_torsion:
            score_result = _with_torsion(self.lib, score_result, center_x, center_y, center_z)
        return {
            "score_result": score_result,
            "psi_coefficients": coefficients[: coefficient_count.value].copy(),
            "_result_struct": result,
            "_config_struct": config,
        }

    def fit_psi_pcgls(
        self,
        axis_R,
        axis_Z,
        axis_R_phi,
        axis_Z_phi,
        mode_a,
        mode_b,
        mode_m,
        mode_kind,
        center_coefficients,
        *,
        radius_scale: float,
        radial_grid: int = 48,
        vertical_grid: int = 48,
        phi_grid: int = 48,
        rho_min: float = 0.002,
        ridge: float = 1.0e-6,
        iterations: int = 4,
    ):
        """Fit all endpoint psi models with one matrix-free query-major PCGLS call."""
        axes = [
            np.ascontiguousarray(values, dtype=np.float64)
            for values in (axis_R, axis_Z, axis_R_phi, axis_Z_phi)
        ]
        if any(values.ndim != 2 or values.shape[0] != self.query_count for values in axes):
            raise ValueError("batch axis arrays must have shape (query, axis_sample)")
        if not all(values.shape == axes[0].shape for values in axes[1:]):
            raise ValueError("batch axis arrays must have matching shapes")
        modes = [
            np.ascontiguousarray(values, dtype=np.int32).ravel()
            for values in (mode_a, mode_b, mode_m, mode_kind)
        ]
        if not all(values.shape == modes[0].shape for values in modes[1:]):
            raise ValueError("psi mode arrays must have matching shapes")
        center = np.ascontiguousarray(center_coefficients, dtype=np.float64).ravel()
        if center.shape != modes[0].shape:
            raise ValueError("center psi coefficient count must match the mode count")
        coefficients = np.empty((self.query_count, center.size), dtype=np.float64)
        train_rms = np.empty(self.query_count, dtype=np.float64)
        stats = np.empty(6, dtype=np.float64)
        code = self.lib.sgpu_fit_psi_batch_pcgls_f32(
            self.handle,
            *(values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for values in axes),
            ctypes.c_int(axes[0].shape[1]),
            *(values.ctypes.data_as(ctypes.POINTER(ctypes.c_int)) for values in modes),
            ctypes.c_int(center.size),
            ctypes.c_int(self.nfp),
            ctypes.c_double(float(radius_scale)),
            ctypes.c_int(int(radial_grid)),
            ctypes.c_int(int(vertical_grid)),
            ctypes.c_int(int(phi_grid)),
            ctypes.c_double(float(rho_min)),
            ctypes.c_double(float(ridge)),
            ctypes.c_int(int(iterations)),
            center.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            coefficients.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            train_rms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(stats.size),
        )
        self._check(code)
        return coefficients, train_rms, {
            "point_count": int(stats[0]),
            "basis_s": float(stats[1]),
            "feature_s": float(stats[2]),
            "solve_s": float(stats[3]),
            "total_s": float(stats[4]),
            "basis_gib": float(stats[5]),
        }

    def clear_psi_preconditioner(self):
        self.lib.sgpu_clear_psi_warm_preconditioner()

    def score_local_batch(
        self,
        currents_a,
        axis_R,
        axis_Z,
        axis_R_phi,
        axis_Z_phi,
        axis_residual,
        axis_topology_trace,
        axis_topology_det,
        psi_coefficients,
        psi_train_rms,
        coil_components,
        center_capture,
        *,
        surface_theta_count: int = 64,
        alpha_iterations: int = 4,
    ):
        """Evaluate the gradient-only dynamic local score for all queries."""
        currents = np.ascontiguousarray(currents_a, dtype=np.float64)
        if currents.shape != (self.query_count, self.n_base_coils):
            raise ValueError("currents must have shape (query, coil)")
        axes = [
            np.ascontiguousarray(values, dtype=np.float64)
            for values in (axis_R, axis_Z, axis_R_phi, axis_Z_phi)
        ]
        if any(values.ndim != 2 or values.shape[0] != self.query_count for values in axes):
            raise ValueError("batch axis arrays must have shape (query, axis_sample)")
        if not all(values.shape == axes[0].shape for values in axes[1:]):
            raise ValueError("batch axis arrays must have matching shapes")
        diagnostics = [
            np.ascontiguousarray(values, dtype=np.float64).ravel()
            for values in (axis_residual, axis_topology_trace, axis_topology_det)
        ]
        if any(values.size != self.query_count for values in diagnostics):
            raise ValueError("axis diagnostics must have one value per query")
        psi = np.ascontiguousarray(psi_coefficients, dtype=np.float64)
        if psi.ndim != 2 or psi.shape[0] != self.query_count:
            raise ValueError("psi coefficients must have shape (query, coefficient)")
        psi_rms = np.ascontiguousarray(psi_train_rms, dtype=np.float64).ravel()
        coil = np.ascontiguousarray(coil_components, dtype=np.float64).ravel()
        if psi_rms.size != self.query_count or coil.size != self.query_count:
            raise ValueError("psi RMS and coil components must have one value per query")
        center_result = center_capture.get("_result_struct")
        config = center_capture.get("_config_struct")
        if center_result is None or config is None:
            raise ValueError("center_capture must come from capture_psi_center")
        if config.target_volume > 0:
            # The quality loss must use the same angular quantile grid at
            # center and endpoints; the legacy local-density argument remains
            # independent only for the no-target path.
            surface_theta_count = int(config.surface_theta_count)
        result_array = (_SgpuScoreResult * self.query_count)()
        stats = np.zeros(14, dtype=np.float64)
        code = self.lib.sgpu_score_coils_local_batch(
            self.handle,
            currents.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(self.n_base_coils), ctypes.c_int(self.nfp),
            *(values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for values in axes),
            ctypes.c_int(axes[0].shape[1]),
            *(values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for values in diagnostics),
            psi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(psi.shape[1]),
            psi_rms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            coil.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(center_result), ctypes.byref(config),
            ctypes.c_int(int(surface_theta_count)),
            ctypes.c_int(int(alpha_iterations)), result_array,
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(stats.size),
        )
        self._check(code)
        return [_with_torsion(self.lib, _with_boundary_iota(self.lib, _score_result_dict(value), i), self.coeffs_x[i], self.coeffs_y[i], self.coeffs_z[i])
                for i, value in enumerate(result_array)], {
            "query_count": int(stats[0]),
            "surface_s": float(stats[1]),
            "flux_s": float(stats[2]),
            "volume_s": float(stats[3]),
            "alpha_s": float(stats[4]),
            "qs_s": float(stats[5]),
            "total_s": float(stats[6]),
            "alpha_tile_size": int(stats[7]),
            "volume_points_per_query": int(stats[8]),
            "flux_points_per_query": int(stats[9]),
            "target_prediction_s": float(stats[10]),
            "surface_roots_s": float(stats[11]),
            "surface_trace_s": float(stats[12]),
            "surface_drift_s": float(stats[13]),
        }

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _points(self, xyz):
        points = np.ascontiguousarray(xyz, dtype=np.float32)
        if points.ndim != 3 or points.shape[0] != self.query_count or points.shape[2] != 3:
            raise ValueError("batch points must have shape (query, point, 3)")
        return points

    def eval_B(self, xyz):
        points = self._points(xyz)
        output = np.empty_like(points)
        code = self.lib.sgpu_batch_eval_B_f32(
            self.handle,
            points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(points.shape[1]),
        )
        self._check(code)
        return output

    def eval_B_grad(self, xyz):
        points = self._points(xyz)
        field = np.empty_like(points)
        gradient = np.empty((self.query_count, points.shape[1], 3, 3), dtype=np.float32)
        code = self.lib.sgpu_batch_eval_B_grad_f32(
            self.handle,
            points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            field.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(points.shape[1]),
        )
        self._check(code)
        return field, gradient

    def trace_period(self, R0, Z0, *, steps: int):
        R0 = np.ascontiguousarray(R0, dtype=np.float64)
        Z0 = np.ascontiguousarray(Z0, dtype=np.float64)
        if R0.ndim != 2 or R0.shape != Z0.shape or R0.shape[0] != self.query_count:
            raise ValueError("batch trace inputs must have shape (query, line)")
        R1 = np.empty_like(R0)
        Z1 = np.empty_like(Z0)
        code = self.lib.sgpu_batch_trace_period_mixed(
            self.handle,
            R0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Z0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            R1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Z1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(R0.shape[1]),
            ctypes.c_int(self.nfp),
            ctypes.c_int(int(steps)),
        )
        self._check(code)
        return R1, Z1

    def trace_axis_samples(self, R0, Z0, *, integration_steps: int, sample_count: int):
        R0 = np.ascontiguousarray(R0, dtype=np.float64).reshape(-1)
        Z0 = np.ascontiguousarray(Z0, dtype=np.float64).reshape(-1)
        if R0.shape != (self.query_count,) or Z0.shape != (self.query_count,):
            raise ValueError("batch axis starts must contain one point per query")
        shape = (self.query_count, int(sample_count))
        outputs = [np.empty(shape, dtype=np.float64) for _ in range(4)]
        code = self.lib.sgpu_batch_trace_axis_samples(
            self.handle,
            R0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Z0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(self.nfp),
            ctypes.c_int(int(integration_steps)),
            ctypes.c_int(int(sample_count)),
            *(array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for array in outputs),
        )
        self._check(code)
        return tuple(outputs)

    def refine_axis_hint(
        self,
        R0,
        Z0,
        *,
        trace_steps: int = 960,
        newton_iterations: int = 6,
        finite_difference_step: float = 2.0e-4,
        maximum_newton_step: float = 0.25,
        residual_tolerance: float = 1.0e-7,
        hint_max_distance: float = 0.08,
    ):
        R0 = np.ascontiguousarray(R0, dtype=np.float64).reshape(-1)
        Z0 = np.ascontiguousarray(Z0, dtype=np.float64).reshape(-1)
        if R0.shape != (self.query_count,) or Z0.shape != (self.query_count,):
            raise ValueError("batch axis hints must contain one point per query")
        outputs = [np.empty(self.query_count, dtype=np.float64) for _ in range(5)]
        valid = np.empty(self.query_count, dtype=np.uint8)
        code = self.lib.sgpu_batch_refine_axis_hint(
            self.handle,
            R0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Z0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(self.nfp),
            ctypes.c_int(int(trace_steps)),
            ctypes.c_int(int(newton_iterations)),
            ctypes.c_double(float(finite_difference_step)),
            ctypes.c_double(float(maximum_newton_step)),
            ctypes.c_double(float(residual_tolerance)),
            ctypes.c_double(float(hint_max_distance)),
            *(array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for array in outputs),
            valid.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        )
        self._check(code)
        return {
            "R": outputs[0],
            "Z": outputs[1],
            "residual": outputs[2],
            "topology_trace": outputs[3],
            "topology_det": outputs[4],
            "valid": valid.astype(bool),
        }


def load_case(path: str | Path, key: str = "raw"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    d = data[key]
    return (
        np.asarray(d["x"], dtype=np.float64),
        np.asarray(d["y"], dtype=np.float64),
        np.asarray(d["z"], dtype=np.float64),
        np.asarray(d["current"], dtype=np.float64) * 1e6,
        int(data["nfp"]),
    )


def eval_fourier_block(c, t):
    c = np.asarray(c, dtype=np.float64)
    order = (c.size - 1) // 2
    t = np.asarray(t, dtype=np.float64)
    x = np.full_like(t, c[0], dtype=np.float64)
    dxdt = np.zeros_like(t, dtype=np.float64)
    for m in range(1, order + 1):
        s = np.sin(2.0 * np.pi * m * t)
        co = np.cos(2.0 * np.pi * m * t)
        sin_c = c[2 * m - 1]
        cos_c = c[2 * m]
        x += sin_c * s + cos_c * co
        dxdt += 2.0 * np.pi * m * (sin_c * co - cos_c * s)
    return x, dxdt


def make_segments_cpu(coeffs_x, coeffs_y, coeffs_z, currents_a, nfp: int, segments_per_coil: int):
    xs = []
    wdl = []
    t = (np.arange(segments_per_coil, dtype=np.float64) + 0.5) / segments_per_coil
    for bx, by, bz, cur in zip(coeffs_x, coeffs_y, coeffs_z, currents_a):
        px, vx = eval_fourier_block(bx, t)
        py, vy = eval_fourier_block(by, t)
        pz, vz = eval_fourier_block(bz, t)
        vx = vx / segments_per_coil
        vy = vy / segments_per_coil
        vz = vz / segments_per_coil
        for k in range(nfp):
            ang = 2.0 * np.pi * k / nfp
            ca, sa = np.cos(ang), np.sin(ang)
            rx = ca * px - sa * py
            ry = sa * px + ca * py
            rvx = ca * vx - sa * vy
            rvy = sa * vx + ca * vy
            xs.append(np.column_stack([rx, ry, pz]))
            wdl.append(cur * np.column_stack([rvx, rvy, vz]))
            mx, my, mz = px, -py, -pz
            mvx, mvy, mvz = vx, -vy, -vz
            rx = ca * mx - sa * my
            ry = sa * mx + ca * my
            rvx = ca * mvx - sa * mvy
            rvy = sa * mvx + ca * mvy
            xs.append(np.column_stack([rx, ry, mz]))
            wdl.append((-cur) * np.column_stack([rvx, rvy, mvz]))
    return np.vstack(xs), np.vstack(wdl)


def eval_B_segments_cpu(points, seg_pos, seg_wdl):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = np.zeros_like(points)
    for i, p in enumerate(points):
        r = p[None, :] - seg_pos
        r2 = np.sum(r * r, axis=1)
        invr3 = 1.0 / np.maximum(r2, 1e-300) ** 1.5
        out[i] = 1e-7 * np.sum(np.cross(seg_wdl, r) * invr3[:, None], axis=0)
    return out


def eval_B_grad_segments_cpu(points, seg_pos, seg_wdl):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    seg_pos = np.asarray(seg_pos, dtype=np.float64).reshape(-1, 3)
    seg_wdl = np.asarray(seg_wdl, dtype=np.float64).reshape(-1, 3)
    field = np.zeros_like(points)
    gradient = np.zeros((len(points), 3, 3), dtype=np.float64)
    identity_cross = np.empty((len(seg_pos), 3, 3), dtype=np.float64)
    identity_cross[:, 0, :] = np.column_stack(
        [np.zeros(len(seg_pos)), -seg_wdl[:, 2], seg_wdl[:, 1]]
    )
    identity_cross[:, 1, :] = np.column_stack(
        [seg_wdl[:, 2], np.zeros(len(seg_pos)), -seg_wdl[:, 0]]
    )
    identity_cross[:, 2, :] = np.column_stack(
        [-seg_wdl[:, 1], seg_wdl[:, 0], np.zeros(len(seg_pos))]
    )
    for index, point in enumerate(points):
        displacement = point[None, :] - seg_pos
        radius2 = np.sum(displacement * displacement, axis=1)
        invr3 = 1.0 / np.maximum(radius2, 1e-300) ** 1.5
        invr5 = invr3 / np.maximum(radius2, 1e-300)
        cross = np.cross(seg_wdl, displacement)
        field[index] = 1e-7 * np.sum(cross * invr3[:, None], axis=0)
        gradient[index] = 1e-7 * np.sum(
            identity_cross * invr3[:, None, None]
            - 3.0 * cross[:, :, None] * displacement[:, None, :] * invr5[:, None, None],
            axis=0,
        )
    return field, gradient


def target_volume_config(volume: float, **overrides) -> dict:
    """ABI-12 calibrated preset; opt-in, leaving legacy config unchanged.

    Radius 0.08 m is the fitted-psi domain, not the target surface radius.
    Override it explicitly for domain studies; unreachable targets stay invalid.
    """
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("target volume must be finite and positive")
    config = dict(target_volume=float(volume), psi_a=0.08,
                  target_quality_threshold=0.01, target_quality_temperature=0.0025,
                  target_quality_weight=10000.0, target_volume_forward_tolerance=1e-4,
                  target_volume_neighbor_tolerance=0.01)
    config.update(overrides)
    return config

def coil_torsion_gradient_native(lib_path, coeffs_x, coeffs_y, coeffs_z, *, samples=1024, gradient=True):
    """Native arc-length mean absolute torsion; optional analytic coefficient gradient.

    Units are 1/m. Coefficients use interleaved sin/cos ordering. Currents and
    symmetry copies do not affect the metric. The zero-|tau| subgradient is zero.
    """
    lib = lib_path if isinstance(lib_path, ctypes.CDLL) else ctypes.CDLL(str(lib_path))
    fun = getattr(lib, 'sgpu_coil_torsion_gradient', None)
    if fun is None: raise GpuError('native library lacks the torsion extension')
    p = ctypes.POINTER(ctypes.c_double)
    fun.argtypes = [p,p,p,ctypes.c_int,ctypes.c_int,ctypes.c_int,p,p]
    fun.restype = ctypes.c_int
    arrays=[np.ascontiguousarray(v,dtype=np.float64) for v in (coeffs_x,coeffs_y,coeffs_z)]
    if arrays[0].ndim!=2 or any(v.shape!=arrays[0].shape or not np.isfinite(v).all() for v in arrays):
        raise ValueError('finite matching (coil, coefficient) arrays required')
    nc,nk=arrays[0].shape
    value=ctypes.c_double();derivative=np.zeros((3,nc,nk)) if gradient else None
    code=fun(*(v.ctypes.data_as(p) for v in arrays),nc,nk,int(samples),ctypes.byref(value),None if derivative is None else derivative.ctypes.data_as(p))
    if code:raise GpuError(f'torsion geometry invalid (code {code})')
    return {'value':value.value,'derivatives':derivative,'samples':int(samples)}


def coil_torsion_p95_gradient_native(lib_path, coeffs_x, coeffs_y, coeffs_z, *, gradient=True):
    """256-point arc-length-weighted P95 |tau| and selected-point derivative.

    The derivative is valid while quantile selection stays unchanged. At rank
    switches the discrete quantile is nonsmooth; local finite differences retain
    the actual rank selection rather than assuming a smooth surrogate.
    """
    lib=lib_path if isinstance(lib_path,ctypes.CDLL) else ctypes.CDLL(str(lib_path))
    fun=getattr(lib,'sgpu_coil_torsion_p95_gradient',None)
    if fun is None:raise GpuError('native library lacks P95 torsion extension')
    p=ctypes.POINTER(ctypes.c_double);fun.argtypes=[p,p,p,ctypes.c_int,ctypes.c_int,p,p];fun.restype=ctypes.c_int
    arrays=[np.ascontiguousarray(v,dtype=np.float64) for v in (coeffs_x,coeffs_y,coeffs_z)]
    if arrays[0].ndim!=2 or any(v.shape!=arrays[0].shape or not np.isfinite(v).all() for v in arrays):raise ValueError('finite matching coil arrays required')
    nc,nk=arrays[0].shape;value=ctypes.c_double();derivative=np.zeros((3,nc,nk)) if gradient else None
    code=fun(*(v.ctypes.data_as(p) for v in arrays),nc,nk,ctypes.byref(value),None if derivative is None else derivative.ctypes.data_as(p))
    if code:raise GpuError(f'P95 torsion geometry invalid (code {code})')
    return {'value':value.value,'derivatives':derivative,'samples':256}


def _with_torsion(lib,result,x,y,z):
    if getattr(lib,'sgpu_coil_torsion_p95_gradient',None) is not None:
        try:
            v=coil_torsion_p95_gradient_native(lib,x,y,z,gradient=False)
            result['diagnostics']['coil_torsion_abs_arclength_p95']=v['value']
            result['diagnostics']['coil_torsion_samples']=256
        except GpuError:
            result['diagnostics']['coil_torsion_abs_arclength_p95']=float('nan')
    return result
