"""Standalone coil-to-metrics GPU evaluator for stellarator coils."""
from .config import EvaluationConfig
from .evaluator import CenterContextError, EvaluationError, Evaluator, InvalidEvaluationError
from .losses import quality_penalty
from .types import CoilSet, EvaluationResult, GradientResult, LocalMetrics

__version__ = "0.8.2"
__all__ = ["CoilSet", "EvaluationConfig", "EvaluationResult", "GradientResult", "LocalMetrics",
           "Evaluator", "EvaluationError", "InvalidEvaluationError", "CenterContextError", "quality_penalty"]
