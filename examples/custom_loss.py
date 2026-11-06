"""A NumPy objective with local finite-difference gradients."""
import math
from pathlib import Path
from time import perf_counter
import numpy as np
from coil_eval import CoilSet, EvaluationConfig, Evaluator, quality_penalty

COILS = Path(__file__).parent / "preview" / "example1.json"
VOLUME = 0.3
RADIUS = math.sqrt(0.06)
DIFFERENCE_STEP = 1e-4


def objective(metrics):
    return np.log(metrics.qs_volume + 1e-8) + quality_penalty(metrics.relative_normal_drift_p95)


def main():
    started = perf_counter()
    coils = CoilSet.read_json(COILS)
    config = EvaluationConfig(target_volume=VOLUME, psi_a=RADIUS, helicity=(1, 0))
    # Scale 1 m for coordinates and 100000 A for currents: each offset is step * scale.
    scales = np.r_[np.ones(3 * coils.coeffs_x.size), np.full(coils.n_base_coils, 100000.)]
    print("Computing the objective and local finite differences...", flush=True)
    with Evaluator(config=config) as evaluator:
        result = evaluator.loss_and_grad(coils, objective, scales=scales, h=DIFFERENCE_STEP)
    print(f"Objective: {result.value:.8f}; QS: {result.center.metrics.qs_volume:.8g}")
    print(f"Boundary iota: {result.center.metrics.iota_boundary:.8f}")
    print("Physical gradient: all x coefficients, then y, z, and currents (A).")
    print(result.gradient)
    print(f"Gradient norm: {np.linalg.norm(result.gradient):.6g}")
    print(f"Total elapsed: {perf_counter() - started:.3f} s")


if __name__ == "__main__":
    main()
