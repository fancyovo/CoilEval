"""Evaluate two independent coil sets together, and differentiate both losses."""
import math
from pathlib import Path
from time import perf_counter
import torch
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.torch import TensorEvaluator, pack_coils

COILS = Path(__file__).parent / "preview" / "example1.json"
VOLUME = 0.3
RADIUS = math.sqrt(0.06)
CURRENT_FACTOR = 1.01               # Change the second set of coil currents


def main():
    started = perf_counter()
    coils = CoilSet.read_json(COILS)
    first = pack_coils(coils)
    second = first.clone()
    second[:, -1] *= CURRENT_FACTOR
    parameters = torch.stack([first, second])
    config = EvaluationConfig(target_volume=VOLUME, psi_a=RADIUS, helicity=(1, 0))
    print("Evaluating two independent queries...", flush=True)
    with TensorEvaluator(config=config, nfp=coils.nfp) as evaluator:
        call_started = perf_counter()
        result = evaluator.evaluate_batch(parameters)
        loss = torch.log(result.metrics.qs_volume + 1e-8)
        gradients, = torch.autograd.grad(loss.sum(), result.parameters)
        gradients = gradients.detach().cpu()
        elapsed = perf_counter() - call_started
        for index in range(len(parameters)):
            print(f"query {index + 1}: QS={float(result.metrics.qs_volume[index].detach()):.8g}, "
                  f"iota={float(result.metrics.iota_boundary[index].detach()):.8f}, "
                  f"gradient norm={float(torch.linalg.vector_norm(gradients[index])):.6g}")
    print(f"Batch value + gradient: {elapsed / len(parameters):.3f} s/sample")
    print(f"Whole call: {elapsed:.3f} s; total including setup: {perf_counter() - started:.3f} s")


if __name__ == "__main__":
    main()
