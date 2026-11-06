"""Use the default two scalar fits to evaluate several target volumes."""
import math
from pathlib import Path
from time import perf_counter
from coil_eval import CoilSet, EvaluationConfig, Evaluator

COILS = Path(__file__).parent / "preview" / "example1.json"
VOLUMES = [0.20, 0.25, 0.30]       # m^3, one independent query per target
RADIUS = math.sqrt(0.06)


def main():
    started = perf_counter()
    coils = CoilSet.read_json(COILS)
    configs = [EvaluationConfig(target_volume=v, psi_a=RADIUS, helicity=(1, 0)) for v in VOLUMES]
    print(f"Evaluating volumes {VOLUMES} m^3...", flush=True)
    with Evaluator(config=configs[0]) as evaluator:
        results = evaluator.evaluate_many([coils] * len(VOLUMES), configs=configs)
    elapsed = perf_counter() - started
    for volume, result in zip(VOLUMES, results):
        if result.valid:
            print(f"V={volume:.3f} m^3: QS={result.metrics.qs_volume:.8g}, "
                  f"iota={result.metrics.iota_boundary:.8f}")
        else:
            print(f"V={volume:.3f} m^3: {result.status}")
    print(f"Including setup: {elapsed / len(VOLUMES):.3f} s/sample; total {elapsed:.3f} s")


if __name__ == "__main__":
    main()
