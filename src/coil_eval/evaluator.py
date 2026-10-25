"""Metrics-first session and full-dimensional centered finite differences."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Callable, Sequence

import numpy as np

from .config import EvaluationConfig
from .types import CoilSet, EvaluationResult, GradientResult, LocalMetrics


class EvaluationError(RuntimeError):
    pass


class InvalidEvaluationError(EvaluationError):
    """A center or endpoint has no usable metrics; gradient is unavailable."""
    def __init__(self, message, *, results=()):
        super().__init__(message)
        self.results = tuple(results)


class CenterContextError(EvaluationError):
    """A center belongs to another session/thread, or has been superseded."""


_METRIC_FIELDS = {
    "qs_volume": "qs_target_global_error_per_helicity",
    "qs_outer": "qs_target_edge_error_per_helicity",
    "iota_boundary": "iota_target_surface",
    "iota_min": "iota_min", "iota_max": "iota_max",
    "relative_normal_drift_p95": "target_quality_error",
    "coil_length_mean": "coil_length_mean", "coil_curvature_max": "coil_curvature_max",
    "coil_gap_min": "coil_min_intercoil_distance", "coil_radius_min": "coil_min_axis_distance",
    "current_abs_max": "coil_current_abs_max_a", "coil_curvature_p95": "coil_curvature_p95",
    "high_mode_fraction": "coil_high_mode_energy_fraction",
    "psi_fit_rms": "psi_train_rms",
}
_OPTIONAL = {"coil_curvature_p95", "high_mode_fraction", "psi_fit_rms"}


def _decode_result(raw, config, *, mode="independent", context=None):
    """Convert a native record without creating or advancing a session."""
    status = str(raw.get("status", "internal_error"))
    d = {k: v for k, v in raw.get("diagnostics", {}).items()
         if not k.startswith("score") and k not in {"abi_version", "struct_size", "status", "target_volume_cache_id"}}
    for k in list(d):
        if "verified" in k or "strict" in k or "stable" in k:
            del d[k]
    metrics = None
    if status == "ok":
        try:
            metrics = LocalMetrics(**{name: d.get(key) if name in _OPTIONAL else d[key]
                                      for name, key in _METRIC_FIELDS.items()})
        except (KeyError, TypeError, ValueError) as exc:
            status = "invalid_metrics"
            d["metric_error"] = str(exc)
    d.update({"configuration": config.to_dict(), "trace_periods": 1, "full_physical_evaluation": "not_run",
              "geometry_model": "formal" if mode == "independent" else "center_linearized"})
    if metrics is not None:
        d["quality_threshold_pass"] = metrics.relative_normal_drift_p95 <= 0.01
    timing = {"native": raw.get("timing", {})}
    if "local_timing" in raw:
        timing["shared_local_batch"] = raw["local_timing"]
    return EvaluationResult(status, metrics, d, timing, mode, context)


@dataclass(frozen=True)
class _Center:
    owner: object
    generation: int
    coils: CoilSet
    raw: dict


class Evaluator:
    """One native session, one thread, one current independent center.

    A newer independent evaluation supersedes older centers. Nearby evaluations
    recapture their center and free native resources before returning. Losses see
    LocalMetrics only. Geometry metrics use the center's first-order local model.
    """
    def __init__(self, library: str | Path | None = None, *, config: EvaluationConfig,
                 device_id: int = 0, backend=None):
        if not isinstance(config, EvaluationConfig):
            raise TypeError("config must be an EvaluationConfig")
        self._config = config
        self.device_id = int(device_id)
        self._thread = threading.get_ident()
        self._owner = object()
        self._generation = 0
        self._closed = False
        if backend is None:
            from ._backend import NativeBackend
            from ._native_loader import resolve_library
            library = resolve_library(library)
            backend = NativeBackend(library, device_id=self.device_id)
        self._backend = backend

    @property
    def config(self) -> EvaluationConfig:
        return self._config

    def _check_session(self):
        if self._closed:
            raise CenterContextError("evaluator session is closed")
        if threading.get_ident() != self._thread:
            raise CenterContextError("evaluator and centers must be used on their creating thread")

    def _invoke(self, method, *args):
        try:
            return getattr(self._backend, method)(*args)
        except Exception as exc:
            from ._binding import GpuError
            if isinstance(exc, GpuError):
                raise EvaluationError(f"native {method} failed: {exc}") from exc
            raise

    def _result(self, raw, *, mode="independent", context=None):
        return _decode_result(raw, self.config, mode=mode, context=context)

    def evaluate(self, coils: CoilSet) -> EvaluationResult:
        self._check_session()
        self._generation += 1
        start = time.perf_counter()
        raw = self._invoke("evaluate", coils, self.config)
        context = _Center(self._owner, self._generation, coils, raw)
        result = self._result(raw, context=context)
        return EvaluationResult(result.status, result.metrics, result.diagnostics,
                                {**result.timings, "python_wall_s": time.perf_counter()-start}, result.mode, context)

    def evaluate_many(self, coils: Sequence[CoilSet]) -> tuple[EvaluationResult, ...]:
        """Independent evaluations in input order; only the last is a live center."""
        return tuple(self.evaluate(c) for c in coils)

    def evaluate_nearby(self, center: EvaluationResult,
                        candidates: Sequence[CoilSet]) -> tuple[EvaluationResult, ...]:
        self._check_session()
        state = center._context
        if not isinstance(state, _Center) or state.owner is not self._owner or state.generation != self._generation:
            raise CenterContextError("center is stale, deserialized, local-only, or belongs to another evaluator")
        if not center.valid:
            raise InvalidEvaluationError(f"invalid center: {center.status}", results=(center,))
        candidates = tuple(candidates)
        for candidate in candidates:
            if candidate.nfp != state.coils.nfp or candidate.coeffs_x.shape != state.coils.coeffs_x.shape:
                raise ValueError("nearby candidates must preserve the center's coil dimensions and nfp")
        if not candidates:
            return ()
        raw = self._invoke("nearby", state.coils, candidates, state.raw, self.config)
        if len(raw) != len(candidates):
            raise EvaluationError("native nearby batch returned the wrong number of endpoints")
        return tuple(self._result(r, mode="nearby") for r in raw)

    def loss_and_grad(self, coils: CoilSet, loss_fn: Callable[[LocalMetrics], float], *,
                      scales=None, h: float | None = None, basis=None) -> GradientResult:
        """Full orthonormal centered differences of the center-hinted local model.

        Endpoints are p +/- h * scales * basis[j]. Returned gradient is with
        respect to physical p, in parameter_vector order. Every direction is used.
        """
        self._check_session()
        start = time.perf_counter()
        h = self.config.h if h is None else float(h)
        if not np.isfinite(h) or h <= 0:
            raise ValueError("h must be finite and positive")
        n = coils.parameter_count
        if scales is None:
            raise ValueError("pass explicit physical scales (metres for geometry, amperes for currents)")
        scales = np.broadcast_to(np.asarray(scales, dtype=float), (n,)).copy()
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("scales must be finite positive physical parameter scales")
        q = np.eye(n) if basis is None else np.array(basis, dtype=float, copy=True)
        if q.shape != (n, n) or not np.all(np.isfinite(q)) or not np.allclose(q @ q.T, np.eye(n), rtol=1e-9, atol=1e-9):
            raise ValueError("basis must be a complete square orthonormal matrix with one direction per row")
        center = self.evaluate(coils)
        if not center.valid:
            raise InvalidEvaluationError(f"invalid center: {center.status}", results=(center,))
        value = _loss_value(loss_fn, center.metrics)
        p = coils.parameter_vector()
        candidates = tuple(coils.with_parameters(p + sign*h*scales*direction)
                           for direction in q for sign in (1, -1))
        endpoints = self.evaluate_nearby(center, candidates)
        invalid = [i for i, r in enumerate(endpoints) if not r.valid]
        if invalid:
            raise InvalidEvaluationError(f"gradient unavailable: invalid endpoint indices {invalid}", results=endpoints)
        values = np.asarray([_loss_value(loss_fn, r.metrics) for r in endpoints]).reshape(n, 2)
        slopes = (values[:, 0]-values[:, 1])/(2*h)
        gradient = (slopes @ q)/scales
        if not np.all(np.isfinite(gradient)):
            raise InvalidEvaluationError("finite differences produced a nonfinite gradient", results=endpoints)
        return GradientResult(value, gradient, center, endpoints, scales, h, q,
                              {"python_wall_s": time.perf_counter()-start, "endpoint_count": len(endpoints)})

    def close(self):
        self._check_session()
        if hasattr(self._backend, "close"):
            self._backend.close()
        self._generation += 1
        self._closed = True

    def __enter__(self):
        self._check_session()
        return self

    def __exit__(self, *exc):
        self.close()


def _loss_value(loss_fn, metrics):
    value = np.asarray(loss_fn(metrics))
    if value.shape != () or value.dtype.kind not in "fiu" or not np.isfinite(value.item()):
        raise ValueError("loss_fn must return one finite scalar from LocalMetrics")
    return float(value)
