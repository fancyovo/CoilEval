"""Inputs, loss-visible metrics, and serializable evaluation records."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def _readonly(value):
    # Immutable bytes prevent re-enabling writeability through an array view.
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        return _readonly(value)
    return value


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_json_ready(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True, eq=False)
class CoilSet:
    """Basic coils with stellarator symmetry. Metres and amperes.

    Each coordinate uses [constant, sin(1), cos(1), ..., sin(K), cos(K)].
    The native field expands each basic coil into 2*nfp physical coils.
    """
    coeffs_x: np.ndarray
    coeffs_y: np.ndarray
    coeffs_z: np.ndarray
    currents_a: np.ndarray
    nfp: int

    def __post_init__(self):
        xyz = [_readonly(v) for v in (self.coeffs_x, self.coeffs_y, self.coeffs_z)]
        currents = _readonly(self.currents_a)
        if any(v.ndim != 2 for v in xyz) or not (xyz[0].shape == xyz[1].shape == xyz[2].shape):
            raise ValueError("coordinate coefficients must have the same two-dimensional shape")
        nc, ncoeff = xyz[0].shape
        if nc == 0 or ncoeff < 3 or ncoeff % 2 != 1:
            raise ValueError("at least one coil and an odd coefficient count >= 3 are required")
        if currents.shape != (nc,):
            raise ValueError("currents_a must have shape (n_base_coils,)")
        if isinstance(self.nfp, bool) or int(self.nfp) != self.nfp or self.nfp <= 0:
            raise ValueError("nfp must be a positive integer")
        if any(not np.all(np.isfinite(v)) for v in (*xyz, currents)):
            raise ValueError("coil coefficients and currents must be finite")
        for name, value in zip(("coeffs_x", "coeffs_y", "coeffs_z"), xyz):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "currents_a", currents)
        object.__setattr__(self, "nfp", int(self.nfp))

    @property
    def n_base_coils(self) -> int:
        return self.coeffs_x.shape[0]

    @property
    def n_coefficients(self) -> int:
        return self.coeffs_x.shape[1]

    @property
    def parameter_count(self) -> int:
        return 3 * self.coeffs_x.size + self.currents_a.size

    def parameter_vector(self) -> np.ndarray:
        """Flatten x, then y, then z (coil-major), then physical currents."""
        return np.concatenate([v.ravel() for v in (self.coeffs_x, self.coeffs_y, self.coeffs_z, self.currents_a)])

    def with_parameters(self, vector) -> CoilSet:
        p = np.asarray(vector, dtype=np.float64)
        if p.shape != (self.parameter_count,):
            raise ValueError(f"expected a flat vector of {self.parameter_count} physical parameters")
        n = self.coeffs_x.size
        return CoilSet(*(p[i*n:(i+1)*n].reshape(self.coeffs_x.shape) for i in range(3)), p[3*n:], self.nfp)

    def to_dict(self) -> dict:
        return {"schema": "coil-eval-coils-v1", "coefficient_order": "constant,sin1,cos1,...",
                "stellarator_symmetry": True, "length_unit": "m", "current_unit": "A",
                "nfp": self.nfp, **{k: getattr(self, k).tolist() for k in ("coeffs_x", "coeffs_y", "coeffs_z", "currents_a")}}

    @classmethod
    def from_dict(cls, value: Mapping) -> CoilSet:
        if value.get("schema") != "coil-eval-coils-v1":
            raise ValueError("unsupported coil schema; convert external inputs explicitly")
        expected = {"coefficient_order": "constant,sin1,cos1,...", "stellarator_symmetry": True, "length_unit": "m", "current_unit": "A"}
        if any(value.get(k) != v for k, v in expected.items()):
            raise ValueError("coil convention or units do not match the package schema")
        return cls(*(value[k] for k in ("coeffs_x", "coeffs_y", "coeffs_z", "currents_a", "nfp")))

    def write_json(self, path):
        _write_json(path, self.to_dict())

    @classmethod
    def read_json(cls, path) -> CoilSet:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class LocalMetrics:
    """Only metrics updated by the local native path are visible to a loss."""
    qs_volume: float
    qs_outer: float
    iota_boundary: float
    iota_min: float
    iota_max: float
    relative_normal_drift_p95: float
    coil_length_mean: float
    coil_curvature_max: float
    coil_gap_min: float
    coil_radius_min: float
    current_abs_max: float
    coil_curvature_p95: float | None = None
    high_mode_fraction: float | None = None
    psi_fit_rms: float | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is None and name not in {"coil_curvature_p95", "high_mode_fraction", "psi_fit_rms"}:
                raise ValueError(f"metric {name} is required")
            if value is not None:
                if not np.isfinite(value):
                    raise ValueError(f"metric {name} must be finite")
                object.__setattr__(self, name, float(value))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, eq=False)
class EvaluationResult:
    status: str
    metrics: LocalMetrics | None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    timings: Mapping[str, Any] = field(default_factory=dict)
    mode: str = "independent"
    _context: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))
        object.__setattr__(self, "timings", _freeze(self.timings))

    @property
    def valid(self) -> bool:
        return self.status == "ok" and self.metrics is not None

    def to_dict(self) -> dict:
        return _json_ready({"schema": "coil-eval-result-v1", "status": self.status, "mode": self.mode,
                            "metrics": None if self.metrics is None else self.metrics.to_dict(),
                            "diagnostics": self.diagnostics, "timings": self.timings})

    def write_json(self, path):
        _write_json(path, self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping) -> EvaluationResult:
        if value.get("schema") != "coil-eval-result-v1":
            raise ValueError("unsupported result schema")
        metrics = value["metrics"]
        return cls(value["status"], None if metrics is None else LocalMetrics(**metrics),
                   value.get("diagnostics", {}), value.get("timings", {}), value.get("mode", "independent"))

    @classmethod
    def read_json(cls, path) -> EvaluationResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True, eq=False)
class GradientResult:
    value: float
    gradient: np.ndarray
    center: EvaluationResult
    endpoints: tuple[EvaluationResult, ...]
    scales: np.ndarray
    h: float
    basis: np.ndarray
    timings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("gradient", "scales", "basis"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, "timings", _freeze(self.timings))
