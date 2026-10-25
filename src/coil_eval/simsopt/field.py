"""Callable GPU magnetic field for source-psi/alpha+nu initialization."""
from __future__ import annotations

from ..types import CoilSet


class GpuCoilField:
    """Own a CUDA coil field and return Cartesian B in tesla.

    The coil inputs use metres and amperes. ``xyz`` has shape (N,3), in metres.
    Use this callable as AlphaNuInitializer's injected field callback.
    """
    def __init__(self, coils, *, library=None, device_id=0, segments_per_coil=256, precision="fp32"):
        if not isinstance(coils, CoilSet):
            raise TypeError("coils must be a CoilSet")
        if precision not in ("fp32", "fp64"):
            raise ValueError("precision must be fp32 or fp64")
        if isinstance(segments_per_coil, bool) or int(segments_per_coil) != segments_per_coil or segments_per_coil < 1:
            raise ValueError("segments_per_coil must be a positive integer")
        from .._native_loader import resolve_library
        from .._binding import CoilFieldGpu
        self.precision = precision
        self._field = CoilFieldGpu(resolve_library(library), coils.coeffs_x, coils.coeffs_y,
            coils.coeffs_z, coils.currents_a, coils.nfp,
            segments_per_coil=int(segments_per_coil), device_id=int(device_id))

    def __call__(self, xyz):
        if self._field is None:
            raise RuntimeError("GPU coil field is closed")
        return self._field.eval_B(xyz, precision=self.precision)

    def close(self):
        if self._field is not None:
            self._field.close()
            self._field = None

    def __enter__(self):
        if self._field is None:
            raise RuntimeError("GPU coil field is closed")
        return self

    def __exit__(self, *args):
        self.close()
