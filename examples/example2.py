"""Perturb 32 starts, select one, and optimize with SciPy BFGS in one run."""
import csv
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
import torch
from torch.nn.functional import softplus

from coil_eval import CoilSet, EvaluationConfig, InvalidEvaluationError
from coil_eval.torch import TensorEvaluator, pack_coils

# Edit these settings directly. Coordinates are metres; currents are amperes.
COILS = Path(__file__).parent / "preview" / "example1.json"
DEVICE = "cuda"                 # Change to "cpu" for a slower CPU run
DEVICE_ID = 0                    # CUDA ordinal within the GPUs visible to this process
THREADS = 8
TARGET_VOLUME = 0.3
FITTING_RADIUS = math.sqrt(0.06)
HELICITY = (1, 0)               # Quasi-axisymmetry
N_STARTS = 32
NOISE = 0.01                    # c -> c * exp(log(1 + NOISE) * normal(0, 1))
SEED = 2026110502
MAX_STEPS = 100
GRADIENT_TOLERANCE = 1e-3
GEOMETRY_SCALE = 0.01           # Metres per optimizer coordinate
CURRENT_SCALE = 10000.0         # Amperes per optimizer coordinate
OUTPUT = Path("runs/example2")


def objective(m):
    """An editable compromise between QS, rotational transform and coil geometry."""
    return (
        torch.log(m.qs_volume + 1e-8)
        + ((m.iota_boundary - 0.47) / 0.02).square()
        + ((0.10 - m.coil_gap_min).clamp_min(0) / 0.01).square()
        + ((m.coil_curvature_max - 6.0).clamp_min(0) / 1.5).square()
        + ((m.coil_length_mean - 3.5) / 0.10).square()
        + 10000 * (0.0025 * softplus((m.relative_normal_drift_p95 - 0.01) / 0.0025)).square()
    )


def describe(m):
    return (f"QS={float(m.qs_volume.detach()):.6g}  iota={float(m.iota_boundary.detach()):.6f}  "
            f"gap={float(m.coil_gap_min.detach()):.4f} m  "
            f"curvature={float(m.coil_curvature_max.detach()):.3f} 1/m  "
            f"length={float(m.coil_length_mean.detach()):.4f} m")


def main():
    started = perf_counter()
    coils = CoilSet.read_json(COILS)
    tensor_device = f"cuda:{DEVICE_ID}" if DEVICE == "cuda" else DEVICE
    base = pack_coils(coils, device=tensor_device)
    config = EvaluationConfig(
        target_volume=TARGET_VOLUME, psi_a=FITTING_RADIUS, helicity=HELICITY,
    )
    rng = np.random.default_rng(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with TensorEvaluator(config=config, nfp=coils.nfp, device=DEVICE, device_id=DEVICE_ID,
                         threads=THREADS if DEVICE == "cpu" else None) as evaluator:
        # Screen one candidate at a time so each result is immediately visible.
        print(f"Screening {N_STARTS} starts with {NOISE:.1%} coefficient noise...", flush=True)
        best_loss, start, valid = np.inf, None, 0
        for index in range(N_STARTS):
            tick = perf_counter()
            candidate = base.clone()
            noise = torch.as_tensor(rng.standard_normal(base[:, :-1].shape), device=tensor_device)
            candidate[:, :-1] *= torch.exp(math.log1p(NOISE) * noise) # Currents unchanged
            try:
                metrics = evaluator.evaluate(candidate).metrics
                loss = float(objective(metrics).detach())
                valid += 1
                if loss < best_loss:
                    best_loss, start = loss, candidate.detach().clone()
                status = f"loss={loss:.6f}  {describe(metrics)}"
            except InvalidEvaluationError as error:
                status = f"unscorable: {error}"
            print(f"screen {index + 1:02d}/{N_STARTS}: {status}  "
                  f"time={perf_counter() - tick:.2f} s", flush=True)
        if start is None:
            print("No valid start. Reduce NOISE at the top of this file and run again.")
            print(f"Total: {perf_counter() - started:.2f} s")
            return
        print(f"Selected loss={best_loss:.6f} from {valid}/{N_STARTS} valid starts.\n", flush=True)

        # SciPy works with dimensionless coordinates; Torch supplies local gradients.
        scale = torch.full_like(start, GEOMETRY_SCALE)
        scale[:, -1] = CURRENT_SCALE

        def value_and_gradient(y):
            coordinates = torch.tensor(y, device=tensor_device, dtype=torch.float64, requires_grad=True)
            parameters = start + scale * coordinates.reshape(start.shape)
            try:
                loss = objective(evaluator.evaluate(parameters).metrics)
                gradient, = torch.autograd.grad(loss, coordinates)
                return float(loss.detach()), gradient.detach().cpu().numpy()
            except InvalidEvaluationError as error:
                print(f"  line-search trial unscorable: {error}", flush=True)
                # An invalid trial has no derivative; allow the line search to shorten it.
                return np.inf, np.full_like(y, np.nan)

        history = []
        optimization_started = previous_step = perf_counter()

        def show_step(y):
            nonlocal previous_step
            parameters = start + scale * torch.as_tensor(y, device=tensor_device).reshape(start.shape)
            m = evaluator.evaluate(parameters).metrics
            loss = float(objective(m).detach())
            now = perf_counter()
            history.append(dict(step=len(history) + 1, loss=loss,
                                qs=float(m.qs_volume.detach()), iota=float(m.iota_boundary.detach()),
                                seconds=now - previous_step))
            print(f"step {len(history):03d}: loss={loss:.6f}  {describe(m)}  "
                  f"step={now - previous_step:.2f} s  elapsed={now - optimization_started:.1f} s",
                  flush=True)
            previous_step = now

        solution = minimize(value_and_gradient, np.zeros(start.numel()), jac=True,
                            method="BFGS", callback=show_step,
                            options={"maxiter": MAX_STEPS, "gtol": GRADIENT_TOLERANCE})
        parameters = start + scale * torch.as_tensor(solution.x, device=tensor_device).reshape(start.shape)
        final = evaluator.evaluate(parameters).metrics
        values = parameters.detach().cpu().numpy()
        n = coils.n_coefficients
        optimized = CoilSet(values[:, :n], values[:, n:2*n], values[:, 2*n:3*n],
                            values[:, -1], coils.nfp)
        optimized.write_json(OUTPUT / "optimized_coils.json")
        with (OUTPUT / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["step", "loss", "qs", "iota", "seconds"])
            writer.writeheader()
            writer.writerows(history)
        print(f"\nOptimizer success: {solution.success}; {solution.message}")
        print(f"Iterations: {solution.nit}; objective calls: {solution.nfev}")
        print(f"Objective: {best_loss:.6f} -> {float(objective(final).detach()):.6f}")
        print(describe(final))
        print(f"Normal displacement P95: {float(final.relative_normal_drift_p95.detach()):.6g}")
        print(f"Optimization: {perf_counter() - optimization_started:.2f} s; "
              f"total: {perf_counter() - started:.2f} s")
        print(f"Saved optimized_coils.json and history.csv in {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
