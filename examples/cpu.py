"""Read coils, create an evaluator, and print a complete value result on CPU."""
import math
from pathlib import Path
from time import perf_counter
from coil_eval import CoilSet, EvaluationConfig, Evaluator

COILS = Path(__file__).parent / "preview" / "example1.json"
VOLUME = 0.3                         # m^3
RADIUS = math.sqrt(0.06)              # m
THREADS = 8
HELICITY = (1, 0)                    # Quasi-axisymmetry


def main():
    started = perf_counter()
    coils = CoilSet.read_json(COILS)
    config = EvaluationConfig(target_volume=VOLUME, psi_a=RADIUS, helicity=HELICITY)
    print("Evaluating on CPU...", flush=True)
    with Evaluator(config=config, device="cpu", threads=THREADS) as evaluator:
        result = evaluator.evaluate(coils)
    print(f"Status: {result.status}")
    if result.valid:
        m = result.metrics
        print(f"Volume QS residual: {m.qs_volume:.8g}; outer region: {m.qs_outer:.8g}")
        print(f"Boundary iota: {m.iota_boundary:.8f}")
        print(f"Mean length: {m.coil_length_mean:.5f} m; minimum gap: {m.coil_gap_min:.5f} m")
        print(f"Maximum curvature: {m.coil_curvature_max:.5f} 1/m")
        print(f"Normal displacement P95: {m.relative_normal_drift_p95:.6g}")
    else:
        print(result.diagnostics.get("failure_reason", "No usable magnetic surface"))
    print(f"Total elapsed: {perf_counter() - started:.3f} s")


if __name__ == "__main__":
    main()
