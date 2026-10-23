"""Explicit evaluator settings; numerical representation is frozen in v1.0."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class EvaluationConfig:
    target_volume: float
    psi_a: float = 0.15
    helicity: tuple[int, int] = (1, 4)
    h: float = 1e-4
    psi_iterations: int = 4
    alpha_iterations: int = 4
    segments_per_coil: int = 256
    fast: bool = False
    flux_method: str = "area"

    def __post_init__(self):
        if self.flux_method not in {"area", "stokes"}:
            raise ValueError("flux_method must be area or stokes")
        if self.flux_method == "stokes" and not self.fast:
            raise ValueError("stokes requires fast=True")
        if not isinstance(self.fast, bool):
            raise TypeError("fast must be a boolean")
        for name in ("target_volume", "psi_a", "h"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in ("psi_iterations", "alpha_iterations", "segments_per_coil"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if len(self.helicity) != 2 or any(isinstance(x, bool) or int(x) != x for x in self.helicity):
            raise ValueError("helicity must contain two integers (M, N)")
        if self.helicity[0] == 0:
            raise ValueError("helicity M must be nonzero for the per-helicity QS metric")
        object.__setattr__(self, "helicity", tuple(int(x) for x in self.helicity))

    def native_options(self) -> dict:
        return {
            **({"axis_trace_steps": 240, "axis_sample_count": 240} if self.fast else {}),
            "segments_per_coil": self.segments_per_coil,
            "iota_degree": 3,
            "surface_selection_mode": 1,
            "surface_confidence_periods": 1,
            "surface_theta_count": 128,
            "surface_trace_steps": 100 if self.fast else 400,
            "surface_flux_bisection_iters": 6,
            "target_volume": self.target_volume,
            "psi_a": self.psi_a,
            "target_quality_threshold": 0.01,
            "target_quality_temperature": 0.0025,
            "target_quality_weight": 10000.0,
            "target_volume_forward_tolerance": 1e-4,
            "target_volume_neighbor_tolerance": 0.01,
        }

    def to_dict(self) -> dict:
        return asdict(self)
