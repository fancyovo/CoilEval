"""Composable scalar helpers; the evaluator supplies no aggregate score."""
import math
import numpy as np


def quality_penalty(error: float, *, threshold: float = 0.01,
                    temperature: float = 0.0025, weight: float = 10000.0) -> float:
    """Squared softplus of dimensionless one-period relative normal drift."""
    values = (error, threshold, temperature, weight)
    if any(not math.isfinite(float(v)) for v in values):
        raise ValueError("quality penalty arguments must be finite")
    if error < 0 or threshold < 0 or temperature <= 0 or weight < 0:
        raise ValueError("error/threshold/weight must be nonnegative and temperature positive")
    excess = temperature * np.logaddexp(0.0, (error - threshold) / temperature)
    return float(weight * excess * excess)
