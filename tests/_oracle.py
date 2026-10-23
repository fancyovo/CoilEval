"""An analytic test oracle that is independent of the evaluator implementation.

The components deliberately have unlike physical scales.  A nonlinear user loss
couples a positive quadratic QS proxy and two linear physical metrics, so errors
in endpoint ordering, basis reconstruction, chain rules, or unit conversion
cannot all disappear in a trivial linear test.
"""

from __future__ import annotations

import numpy as np


def make_coils():
    from coil_eval import CoilSet

    return CoilSet(
        coeffs_x=np.array([[1.0, 0.2, 0.0], [-0.9, 0.3, 0.05]]),
        coeffs_y=np.array([[0.0, 0.1, 0.2], [0.0, 0.05, 0.25]]),
        coeffs_z=np.array([[0.02, 0.01, 0.1], [-0.03, 0.01, 0.12]]),
        currents_a=np.array([1.5e5, -2.0e5]),
        nfp=2,
    )


class AnalyticOracle:
    def __init__(self, coils):
        p = coils.parameter_vector()
        self.scales = np.where(np.abs(p) > 1000.0, 2.0e5, 0.5)
        self.weights = np.linspace(0.3, 1.1, p.size)
        self.iota_coefficients = 0.015 * np.cos(np.arange(p.size) + 0.2)
        self.length_coefficients = 0.02 * np.sin(np.arange(p.size) + 0.4)

    def metrics(self, coils):
        z = coils.parameter_vector() / self.scales
        return {
            "qs_volume": 1.4 + 0.5 * np.dot(self.weights, z * z),
            "qs_outer": 1.8 + 0.2 * np.dot(self.weights, z * z),
            "iota_boundary": 0.8 + np.dot(self.iota_coefficients, z),
            "iota_min": 0.7 + np.dot(self.iota_coefficients, z),
            "iota_max": 0.9 + np.dot(self.iota_coefficients, z),
            "relative_normal_drift_p95": 0.003,
            "coil_length_mean": 2.0 + np.dot(self.length_coefficients, z),
            "coil_curvature_max": 5.0,
            "coil_gap_min": 0.2,
            "coil_radius_min": 0.5,
            "current_abs_max": float(np.max(np.abs(coils.currents_a))),
            "curvature_p95": 4.3,
            "high_mode_fraction": 0.01,
        }

    @staticmethod
    def loss(metrics):
        return float(
            np.log(metrics.qs_volume)
            + 0.7 * (metrics.iota_boundary - 1.1) ** 2
            + 0.03 * metrics.coil_length_mean ** 3
        )

    def physical_gradient(self, coils):
        z = coils.parameter_vector() / self.scales
        metrics = self.metrics(coils)
        g_z = (
            self.weights * z / metrics["qs_volume"]
            + 1.4 * (metrics["iota_boundary"] - 1.1) * self.iota_coefficients
            + 0.09 * metrics["coil_length_mean"] ** 2 * self.length_coefficients
        )
        return g_z / self.scales


class FakeBackend:
    """Return the public native-dictionary contract, without CUDA or a solver."""

    def __init__(self, coils):
        self.oracle = AnalyticOracle(coils)
        self.calls = []
        self.reject_nearby_index = None
        self.nonfinite_nearby_index = None
        self.invalidate_center = False
        self.last_raw = None

    def raw(self, coils, config):
        metric = self.oracle.metrics(coils)
        result = {
            "status": "ok",
            "score": -113.0,
            "components": {"coil": 81.0, "plasma": 27.0},
            "timing": {"field": 0.001, "psi": 0.002},
            "diagnostics": {
                "qs_target_global_error_per_helicity": metric["qs_volume"],
                "qs_target_edge_error_per_helicity": metric["qs_outer"],
                "iota_target_surface": metric["iota_boundary"],
                "iota_min": metric["iota_min"],
                "iota_max": metric["iota_max"],
                "target_quality_error": metric["relative_normal_drift_p95"],
                "surface_one_period_drift_relative_p95": 0.031,
                "surface_drift_relative_p95": 0.032,
                "coil_length_mean": metric["coil_length_mean"],
                "coil_curvature_max": metric["coil_curvature_max"],
                "coil_curvature_p95": metric["curvature_p95"],
                "coil_min_intercoil_distance": metric["coil_gap_min"],
                "coil_min_axis_distance": metric["coil_radius_min"],
                "coil_high_mode_energy_fraction": metric["high_mode_fraction"],
                "coil_current_abs_max_a": metric["current_abs_max"],
                "target_volume_requested": config.target_volume,
                "surface_volume": config.target_volume,
                "target_volume_relative_error": 0.0,
                "target_volume_cache_id": 42,
                "target_volume_verified": 1,
                "surface_strict": 1,
                "surface_stable": 1,
                "axis_R": 1.1,
                "axis_Z": 0.01,
                "alpha_relative_l2": 987.0,
                "psi_train_rms": 0.014,
                "score_qs_residual": 777.0,
                "target_quality_penalty": 888.0,
            },
        }
        return result

    def evaluate(self, coils, config):
        self.calls.append(("evaluate", coils.parameter_vector().copy()))
        result = self.raw(coils, config)
        if self.invalidate_center:
            result["status"] = "no_surface"
        self.last_raw = result
        return result

    def nearby(self, coils, candidates, center_raw, config):
        candidates = list(candidates)
        self.calls.append(
            ("nearby", [candidate.parameter_vector().copy() for candidate in candidates])
        )
        assert center_raw["status"] == "ok"
        results = [self.raw(candidate, config) for candidate in candidates]
        if self.reject_nearby_index is not None:
            results[self.reject_nearby_index]["status"] = "no_surface"
        if self.nonfinite_nearby_index is not None:
            results[self.nonfinite_nearby_index]["diagnostics"][
                "qs_target_global_error_per_helicity"
            ] = np.nan
        return results

    def close(self):
        self.calls.append(("close", None))
