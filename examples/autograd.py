"""Use Torch tensors to differentiate a scalar objective through CoilEval."""
import math
from pathlib import Path
from time import perf_counter
import torch
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.torch import TensorEvaluator, pack_coils

COILS = Path(__file__).parent / "preview" / "example1.json"
DEVICE = "cuda"                     # "cpu" is also supported
VOLUME = 0.3                        # m^3
RADIUS = math.sqrt(0.06)             # m
TARGET_IOTA = 0.47


def main():
    started = perf_counter()
    coils = CoilSet.read_json(COILS)
    parameters = pack_coils(coils, device=DEVICE)
    config = EvaluationConfig(target_volume=VOLUME, psi_a=RADIUS, helicity=(1, 0))
    print("Evaluating and differentiating...", flush=True)
    with TensorEvaluator(config=config, device=DEVICE, nfp=coils.nfp) as evaluator:
        result = evaluator.evaluate(parameters)
        m = result.metrics
        # Replace this expression with your own differentiable objective.
        loss = torch.log(m.qs_volume + 1e-8) + ((m.iota_boundary - TARGET_IOTA) / 0.02).square()
        gradient, = torch.autograd.grad(loss, result.parameters)
        gradient = gradient.detach().cpu() # Waits for GPU work before timing ends
        print(f"Objective: {float(loss.detach()):.8f}")
        print(f"QS: {float(m.qs_volume.detach()):.8g}; iota: {float(m.iota_boundary.detach()):.8f}")
        print("Gradient columns: x coefficients, y coefficients, z coefficients, current (A).")
        print(gradient)
        print(f"Gradient norm: {float(torch.linalg.vector_norm(gradient)):.6g}")
    print(f"Value + gradient, including setup: {perf_counter() - started:.3f} s")


if __name__ == "__main__":
    main()
