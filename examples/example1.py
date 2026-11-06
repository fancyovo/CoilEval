"""Evaluate one coil set. Edit the settings below, then run this file."""
import math
from pathlib import Path
from time import perf_counter

from coil_eval import CoilSet, EvaluationConfig, Evaluator

# Settings to try: another coil JSON, volume, fitting radius, or CPU/GPU.
COILS = Path(__file__).parent / "preview" / "example1.json"
DEVICE = "cpu"                  # "cpu" or "cuda"
THREADS = 8                     # Used only on CPU
TARGET_VOLUME = 0.3              # m^3
FITTING_RADIUS = math.sqrt(0.06) # m, around the magnetic axis
HELICITY = (1, 0)               # Quasi-axisymmetry
OUTPUT = Path("runs/example1/result.json")


def main():
    started = perf_counter()
    coils = CoilSet.read_json(COILS)
    config = EvaluationConfig(
        target_volume=TARGET_VOLUME, psi_a=FITTING_RADIUS, helicity=HELICITY,
    )
    print(f"Evaluating {coils.n_base_coils} basic coils on {DEVICE}...", flush=True)
    with Evaluator(config=config, device=DEVICE,
                   threads=THREADS if DEVICE == "cpu" else None) as evaluator:
        call_started = perf_counter()
        result = evaluator.evaluate(coils)
        call_seconds = perf_counter() - call_started

    print(f"Status: {result.status}")
    if result.valid:
        m = result.metrics
        print(f"Volume QS residual:        {m.qs_volume:.8g}")
        print(f"Outer-region QS residual:  {m.qs_outer:.8g}")
        print(f"Boundary iota:             {m.iota_boundary:.8f}")
        print(f"Normal displacement P95:   {m.relative_normal_drift_p95:.6g}")
        print(f"Mean coil length:          {m.coil_length_mean:.6f} m")
        print(f"Minimum coil gap:          {m.coil_gap_min:.6f} m")
        print(f"Maximum coil curvature:    {m.coil_curvature_max:.6f} 1/m")
        print(f"Maximum absolute current:  {m.current_abs_max:.6g} A")
    else:
        print("The evaluator could not score these coils; inspect the saved diagnostics.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.write_json(OUTPUT)
    print(f"Evaluation: {call_seconds:.3f} s; total: {perf_counter() - started:.3f} s")
    print(f"Saved result and diagnostics: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
