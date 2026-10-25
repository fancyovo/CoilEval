"""Owned source and surface records for the psi -> alpha -> nu route."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

SOURCE_KIND = "source_psi"
INITIALIZER_KIND = "alpha_nu"
STANDARD_KIND = "alpha_nu_standard_ls_newton"
TWOPI = 2 * np.pi


def scalar(value):
    return np.asarray(value).reshape(()).item()


def require_kind(value, expected, stage):
    actual = str(scalar(value))
    if actual != expected:
        raise ValueError(f"{stage} requires kind={expected!r}; got {actual!r}")
    return actual


def _hermite(phi, values, derivatives, nfp):
    count = len(values)
    position = np.mod(np.asarray(phi), TWOPI / nfp) * (count * nfp / TWOPI)
    left = np.floor(position).astype(int) % count
    right = (left + 1) % count
    t = position - np.floor(position)
    h = TWOPI / (count * nfp)
    return ((2*t**3-3*t*t+1)*values[left] + (t**3-2*t*t+t)*h*derivatives[left]
            + (-2*t**3+3*t*t)*values[right] + (t**3-t*t)*h*derivatives[right])


@dataclass(frozen=True)
class SourcePsi:
    coefficients: np.ndarray
    mode_a: np.ndarray
    mode_b: np.ndarray
    mode_m: np.ndarray
    mode_kind: np.ndarray
    axis_samples: np.ndarray
    nfp: int
    a: float
    level: float
    kind: str = SOURCE_KIND
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        require_kind(self.kind, SOURCE_KIND, "source model")
        if self.nfp < 1 or self.a <= 0 or self.level <= 0:
            raise ValueError("source nfp, radius and selected level must be positive")
        for key in ("coefficients", "mode_a", "mode_b", "mode_m", "mode_kind", "axis_samples"):
            array = np.asarray(getattr(self, key)).copy()
            if not np.isfinite(array).all():
                raise ValueError(f"nonfinite source array {key}")
            array.setflags(write=False)
            object.__setattr__(self, key, array)
        size = self.coefficients.size
        if any(np.asarray(getattr(self, key)).shape != (size,) for key in
               ("mode_a", "mode_b", "mode_m", "mode_kind")):
            raise ValueError("source mode/solution sizes differ")
        if self.axis_samples.ndim != 2 or self.axis_samples.shape[1] != 4 or len(self.axis_samples) < 2:
            raise ValueError("axis_samples must have shape (samples>=2,4)")

    @classmethod
    def from_snapshot(cls, snapshot):
        config = snapshot.get("config", {})
        axis = snapshot.get("axis_samples")
        if axis is None:
            axis = np.column_stack([snapshot[key] for key in
                                    ("axis_R", "axis_Z", "axis_R_phi", "axis_Z_phi")])
        level = snapshot.get("target_level")
        if level is None:
            level = snapshot["score_result"]["diagnostics"]["surface_level"]
        return cls(snapshot["psi_coefficients"], *[snapshot["psi_mode_"+key]
                   for key in ("a", "b", "m", "kind")], axis,
                   int(scalar(snapshot["nfp"])), float(config["psi_a"]), float(scalar(level)),
                   provenance=dict(snapshot.get("source_provenance", {})))

    @classmethod
    def read(cls,path,*,level,provenance=None):
        """Load the source-psi NPZ layout used by the independent initializer."""
        with np.load(path,allow_pickle=False) as z:
            kinds=np.asarray(z['mode_kind'])
            if kinds.dtype.kind in ('U','S','O'):
                kinds=(kinds.astype(str)=='sin').astype(int)
            axis=np.column_stack([z[k] for k in ('R_axis','Z_axis','R_axis_phi','Z_axis_phi')])
            return cls(z['coeffs'],z['mode_a'],z['mode_b'],z['mode_m'],kinds,axis,
                       int(scalar(z['nfp'])),float(scalar(z['a'])),float(level),
                       provenance=dict(provenance or {}))

    def axis_at(self, phi):
        return (_hermite(phi, self.axis_samples[:,0], self.axis_samples[:,2], self.nfp),
                _hermite(phi, self.axis_samples[:,1], self.axis_samples[:,3], self.nfp))

    def surface_points(self, order=24, *, library=None, device_id=0, max_radius_scale=1.0,
                       maxiter=20, tol=1e-12):
        """Extract a geometric source-s level. Alpha+nu is applied separately."""
        if library is not None:
            from .._binding import surface_points_from_level_gpu
            from .._native_loader import resolve_library
            return surface_points_from_level_gpu(resolve_library(library), self.coefficients,
                self.mode_a, self.mode_b, self.mode_m, self.mode_kind, nfp=self.nfp, a=self.a,
                poly_degree=int(np.max(self.mode_a+self.mode_b)), m_tor=int(np.max(self.mode_m)),
                axis_R=self.axis_samples[:,0], axis_Z=self.axis_samples[:,1], order=order,
                psi_level=self.level, maxiter=maxiter, tol=tol,
                max_radius_scale=max_radius_scale, device_id=device_id)
        size = 2*int(order)+1
        theta, phi = np.meshgrid(np.arange(size)*TWOPI/size,
                                np.arange(size)*TWOPI/(size*self.nfp))
        co, si = np.cos(theta), np.sin(theta)
        degree = int(np.max(self.mode_a+self.mode_b))
        polynomial = np.zeros((degree+1, size, size))
        polynomial[2] = co*co
        phase = {}
        for c, a, b, m, kind in zip(self.coefficients, self.mode_a, self.mode_b,
                                   self.mode_m, self.mode_kind):
            key = int(m), int(kind)
            if key not in phase:
                phase[key] = (np.sin if kind else np.cos)(m*self.nfp*phi)
            polynomial[int(a+b)] += c*co**int(a)*si**int(b)*phase[key]
        limit = self.a*max_radius_scale
        radius = np.full_like(theta, min(self.a*np.sqrt(self.level), limit))
        for iteration in range(maxiter):
            x = radius/self.a
            value, derivative = polynomial[-1].copy(), np.zeros_like(x)
            for power in range(degree-1, -1, -1):
                derivative = derivative*x+value
                value = value*x+polynomial[power]
            residual = value-self.level
            if np.max(np.abs(residual)) <= tol:
                break
            slope = derivative/self.a
            slope = np.where(abs(slope)>1e-14, slope, np.where(slope>=0,1e-14,-1e-14))
            radius -= np.clip(residual/slope, -.4*radius, .4*radius)
            radius = np.clip(radius, self.a*1e-12, limit)
        if np.max(abs(residual)) > 1e-8:
            raise ValueError("source-s surface roots did not converge inside the extraction domain")
        ar, az = self.axis_at(phi)
        rr = ar+radius*co
        xyz = np.stack((rr*np.cos(phi), rr*np.sin(phi), az+radius*si), axis=-1)
        return xyz, radius, {"iterations":iteration+1,"maximum_s_residual":float(np.max(abs(residual)))}


@dataclass
class SurfaceSeed:
    dofs: np.ndarray
    nfp: int
    order: int
    iota: float
    G: float
    s_level: float
    radius_mean_m: float
    spectral_fit_rms_m: float
    stellsym: bool = True
    kind: str = INITIALIZER_KIND
    source_kind: str = SOURCE_KIND
    diagnostics: dict = field(default_factory=dict)
    source_axis_samples: np.ndarray | None = None

    def __post_init__(self):
        if self.source_axis_samples is not None:
            axis=np.array(self.source_axis_samples,dtype=np.float64,copy=True)
            if axis.ndim!=2 or axis.shape[1]!=4 or len(axis)<2 or not np.isfinite(axis).all():
                raise ValueError('source_axis_samples must be finite (samples>=2,4) data')
            axis.setflags(write=False)
            self.source_axis_samples=axis

    def save(self, path):
        values = {key:getattr(self,key) for key in self.__dataclass_fields__
                  if key!='diagnostics' and getattr(self,key) is not None}
        values.update(rho=1.0,s_edge=self.s_level)
        np.savez(Path(path), **values)

    @classmethod
    def read(cls, path):
        with np.load(path, allow_pickle=False) as z:
            require_kind(z['kind'], INITIALIZER_KIND, "alpha+nu input")
            if 'source_kind' in z:
                require_kind(z['source_kind'], SOURCE_KIND, "alpha+nu source")
            values = {key:(z[key].copy() if key in ('dofs','source_axis_samples') else scalar(z[key]))
                      for key in cls.__dataclass_fields__ if key not in ('diagnostics',) and key in z}
        return cls(**values)


@dataclass
class StandardSurface:
    seed: SurfaceSeed
    dofs: np.ndarray
    iota: float
    G: float
    accepted: bool
    quality: str
    diagnostics: dict
    kind: str = STANDARD_KIND

    def save(self, path):
        if not self.accepted:
            raise ValueError("a rejected surface cannot be saved as an accepted standard surface")
        require_kind(self.seed.kind, INITIALIZER_KIND, "standard surface source")
        source_axis=({} if self.seed.source_axis_samples is None else
                     {'source_axis_samples':self.seed.source_axis_samples})
        np.savez(path, dofs=self.dofs, nfp=self.seed.nfp, order=self.seed.order,
                 stellsym=self.seed.stellsym, iota=self.iota, G=self.G,
                 s_level=self.seed.s_level, s_edge=self.seed.s_level, rho=1.,
                 radius_mean_m=self.seed.radius_mean_m,
                 spectral_fit_rms_m=self.seed.spectral_fit_rms_m, kind=self.kind,
                 source_surface_kind=self.seed.kind, **source_axis)
