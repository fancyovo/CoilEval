"""Optional alpha+nu initialization and standard Simsopt reference surfaces.

Simsopt and PyTorch are imported when their corresponding operations run.
"""
from .state import SourcePsi, SurfaceSeed, StandardSurface
from .clebsch import AlphaFit, fit_alpha
from .initializer import AlphaNuInitializer, InitializerConfig
from .solver import StandardSimsoptSolver, SolverConfig, field_from_coils
from .reference import analyze_arrays, surface_fc, volume_average_fc
from .capture import SourceCapture, capture_sources
from .field import GpuCoilField

__all__ = ["SourcePsi", "SurfaceSeed", "StandardSurface", "AlphaFit", "fit_alpha",
           "AlphaNuInitializer", "InitializerConfig", "StandardSimsoptSolver",
           "SolverConfig", "field_from_coils", "analyze_arrays", "surface_fc",
           "volume_average_fc", "SourceCapture", "capture_sources", "GpuCoilField"]
