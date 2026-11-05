"""Explicit evaluator settings with compatible opt-in numerical paths."""
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
    psi_fit: str = "single_stage"
    psi_refit_points: int = 82368
    psi_nearby_method: str = "local"
    psi_fd_batch_size: int = 8
    psi_refit_sampling: str = "cartesian"
    alpha_radial_weighting: str = "histogram"
    alpha_weight_sigma: float = 0.05
    alpha_count_floor: float = 32.0
    alpha_radial_basis: str = "zernike"
    stokes_validation_rtol: float | None = 1e-3
    psi_refit_ridge: float | None = None
    psi_target_refit: bool = False
    psi_target_refit_factor: float = 1.2
    psi_target_refit_ridge: float = 1e-10

    def __post_init__(self):
        if not isinstance(self.psi_target_refit, bool):
            raise TypeError('psi_target_refit must be a boolean')
        if self.psi_target_refit and self.psi_fit != 'two_stage':
            raise ValueError("psi_target_refit requires psi_fit='two_stage'")
        for name, lower, inclusive in (('psi_target_refit_factor', 1., False),
                                       ('psi_target_refit_ridge', 0., True)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not math.isfinite(float(value)) or
                    (float(value) < lower if inclusive else float(value) <= lower)):
                raise ValueError(f'{name} must be finite and {">=" if inclusive else ">"} {lower}')
            object.__setattr__(self, name, float(value))
        if self.psi_refit_ridge is not None:
            value = self.psi_refit_ridge
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError("psi_refit_ridge must be finite and nonnegative, or None")
            if self.psi_fit != 'two_stage':
                raise ValueError("psi_refit_ridge requires psi_fit='two_stage'")
            object.__setattr__(self, "psi_refit_ridge", float(value))
        if self.stokes_validation_rtol is not None:
            tolerance = float(self.stokes_validation_rtol)
            if not math.isfinite(tolerance) or tolerance <= 0:
                raise ValueError("stokes_validation_rtol must be finite and positive, or None")
            object.__setattr__(self, "stokes_validation_rtol", tolerance)
        if self.psi_fit not in {"single_stage", "two_stage"}:
            raise ValueError("psi_fit must be single_stage or two_stage")
        if self.psi_nearby_method not in {"local", "formal"}:
            raise ValueError("psi_nearby_method must be local or formal")
        if self.psi_refit_sampling not in {"cartesian", "polar_quadrature"}:
            raise ValueError("psi_refit_sampling must be cartesian or polar_quadrature")
        if self.alpha_radial_weighting not in {"histogram", "smooth"}:
            raise ValueError("alpha_radial_weighting must be histogram or smooth")
        if self.alpha_radial_weighting == "smooth" and self.psi_fit != "two_stage":
            raise ValueError("smooth alpha radial weighting requires psi_fit='two_stage'")
        if self.alpha_radial_basis not in {"zernike", "flux_legendre"}:
            raise ValueError("alpha_radial_basis must be zernike or flux_legendre")
        if self.alpha_radial_basis == "flux_legendre" and self.psi_fit != "two_stage":
            raise ValueError("flux_legendre alpha radial basis requires psi_fit='two_stage'")
        for name in ("alpha_weight_sigma", "alpha_count_floor"):
            value=float(getattr(self,name))
            if not math.isfinite(value) or value<=0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self,name,value)
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
        for name in ("psi_iterations", "alpha_iterations", "segments_per_coil", "psi_refit_points", "psi_fd_batch_size"):
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
