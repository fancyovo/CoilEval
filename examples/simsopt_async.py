"""Prepare on GPU, then solve saved alpha+nu seeds in a CPU-only process pool."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import math
import multiprocessing as mp
import os
from pathlib import Path


def _threads_one():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"


def _cpu_environment():
    _threads_one()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"


def _json_ready(value):
    import numpy as np
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(_json_ready(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def prepare(args):
    from coil_eval import CoilSet, EvaluationConfig
    from coil_eval.simsopt import capture_sources, GpuCoilField, AlphaNuInitializer, InitializerConfig
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "tasks.json").exists():
        raise ValueError("Choose a new directory for another preparation run")
    if len(set(args.volumes)) != len(args.volumes):
        raise ValueError("Target volumes must be distinct")
    coils = CoilSet.read_json(args.coils)
    coils_path = directory / "coils.json"
    coils.write_json(coils_path)
    base = EvaluationConfig(target_volume=args.volumes[0], psi_a=args.radius,
        fast=True, flux_method="stokes", psi_fit="two_stage", psi_refit_sampling=args.sampling,
        alpha_radial_weighting=args.alpha_weighting, alpha_radial_basis=args.alpha_basis)
    configs = [replace(base, target_volume=volume) for volume in args.volumes]
    captures = capture_sources(coils, configs, library=args.library, device_id=args.device)
    tasks, failures = [], []
    with GpuCoilField(coils, library=args.library, device_id=args.device) as field, AlphaNuInitializer(
            field, library=args.library, device_id=args.device,
            config=InitializerConfig(device=f"cuda:{args.device}", alpha_fold_policy=args.alpha_fold_policy,
                                     alpha_min_jacobian=args.alpha_min_jacobian)) as initializer:
        for index, (config, capture) in enumerate(zip(configs, captures)):
            output = directory / f"volume_{index:03d}"
            output.mkdir(parents=True, exist_ok=True)
            capture.result.write_json(output / "native_result.json")
            if not capture.valid:
                failures.append({"index": index, "volume_m3": config.target_volume,
                                 "status": capture.result.status})
                continue
            try:
                seed = initializer.prepare(capture.source, capture.alpha)
                seed_path = output / "seed.npz"
                seed.save(seed_path)
                _write(output / "initializer.json", seed.diagnostics)
                tasks.append({"index": index, "target_volume_m3": config.target_volume,
                    "coils": str(coils_path), "seed": str(seed_path), "directory": str(output)})
            except Exception as exc:
                failure = {"index": index, "volume_m3": config.target_volume,
                           "status": "initializer_failed", "error": str(exc)}
                failures.append(failure)
                _write(output / "preparation_error.json", failure)
    _write(directory / "tasks.json", {"tasks": tasks, "failures": failures,
        "configuration": base.to_dict(), "volumes_m3": args.volumes, "alpha_origin": "native_capture"})
    print(f"Prepared {len(tasks)} CPU tasks; retained {len(failures)} failures in {directory}")


def _solve_one(task):
    from coil_eval import CoilSet
    from coil_eval.simsopt import SurfaceSeed, field_from_coils, StandardSimsoptSolver, surface_fc
    output = Path(task["directory"])
    try:
        coils = CoilSet.read_json(task["coils"])
        seed = SurfaceSeed.read(task["seed"])
        field = field_from_coils(coils)
        surface = StandardSimsoptSolver().solve(seed, field, target_volume=task["target_volume_m3"])
        result = {"index": task["index"], "target_volume_m3": task["target_volume_m3"],
                  "accepted": surface.accepted, "quality": surface.quality,
                  "diagnostics": surface.diagnostics}
        if surface.accepted:
            surface.save(output / "standard_surface.npz")
            result["reference_fc"] = surface_fc(surface, field, sizes=(64, 128))
        else:
            import numpy as np
            np.savez(output / "rejected_candidate.npz", dofs=surface.dofs, iota=surface.iota,
                     G=surface.G, kind="rejected_standard_candidate")
    except Exception as exc:
        result = {"index": task["index"], "accepted": False, "status": "solve_failed", "error": str(exc)}
    _write(output / "result.json", result)
    return result


def solve(args):
    _cpu_environment()
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if not 1 <= args.workers <= allocated:
        raise ValueError("Worker count must fit the assigned CPU cores")
    directory = args.directory.resolve()
    manifest = json.loads((directory / "tasks.json").read_text(encoding="utf-8"))
    # Spawn keeps CUDA state from a different process out of CPU workers.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=_cpu_environment) as pool:
        results = list(pool.map(_solve_one, manifest["tasks"]))
    _write(directory / "solve_summary.json", {"results": results, "preparation_failures": manifest["failures"]})
    print(f"Solved {len(results)} tasks; accepted {sum(row['accepted'] for row in results)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    gpu = phases.add_parser("prepare", help="Create source-psi/alpha+nu seeds on GPU and exit")
    gpu.add_argument("--directory", type=Path, required=True)
    gpu.add_argument("--coils", type=Path, default=Path(__file__).with_name("coils.json"))
    gpu.add_argument("--library", help="Optional custom native library")
    gpu.add_argument("--device", type=int, default=0)
    gpu.add_argument("--volumes", type=float, nargs="+", default=[.01, .03, .05], help="Target volumes in m^3")
    gpu.add_argument("--radius", type=float, default=.15, help="Shared fitting radius in metres")
    gpu.add_argument("--sampling", choices=("cartesian", "polar_quadrature"), default="polar_quadrature")
    gpu.add_argument("--alpha-weighting", choices=("histogram", "smooth"), default="smooth")
    gpu.add_argument("--alpha-basis", choices=("zernike", "flux_legendre"), default="flux_legendre")
    gpu.add_argument("--alpha-fold-policy", choices=("reject", "damp"), default="reject")
    gpu.add_argument("--alpha-min-jacobian", type=float, default=.05)
    cpu = phases.add_parser("solve", help="Consume saved seeds in a CPU-only process pool")
    cpu.add_argument("--directory", type=Path, required=True)
    cpu.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    _threads_one()
    prepare(args) if args.phase == "prepare" else solve(args)


if __name__ == "__main__":
    main()
