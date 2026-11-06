"""Compute Fourier and derivative-based QS metrics on the requested surface."""
import math
from pathlib import Path
from time import perf_counter
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.qs import SurfaceQSConfig, SurfaceQSEvaluator

COILS = Path(__file__).parent / "preview" / "example1.json"
VOLUME = 0.3
RADIUS = math.sqrt(0.06)
ACCURACY = "high"                  # "fast", "balanced", or "high"


def main():
    started = perf_counter()
    coils = CoilSet.read_json(COILS)
    config = EvaluationConfig(target_volume=VOLUME, psi_a=RADIUS, helicity=(1, 0))
    surface_config = SurfaceQSConfig.preset(ACCURACY, compute_f_T=True)
    print("Constructing the target surface and computing its QS metrics...", flush=True)
    with SurfaceQSEvaluator(config=config, surface_qs=surface_config) as evaluator:
        result = evaluator.evaluate(coils)
    print(f"Fourier QS f_B: {result.f_B:.8g}")
    print(f"Triple-product QS f_T: {result.f_T:.8g}")
    print(f"Total elapsed: {perf_counter() - started:.3f} s")


if __name__ == "__main__":
    main()
