"""Prepare surface seeds on GPU and solve them concurrently on CPU in one run."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter

COILS = Path(__file__).parent / "preview" / "example1.json"
VOLUMES = [0.20, 0.30]           # m^3
RADIUS = math.sqrt(0.06)         # m
WORKERS = 2                    # CPU processes; allow at least this many CPU cores
OUTPUT = Path("runs/simsopt")


def cpu_environment():
    # Spawned workers use CPU libraries only, one numerical thread per process.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"


def solve_one(seed_path, volume):
    from coil_eval import CoilSet
    from coil_eval.simsopt import SurfaceSeed, field_from_coils, StandardSimsoptSolver

    started = perf_counter()
    coils = CoilSet.read_json(COILS)
    seed = SurfaceSeed.read(seed_path)
    field = field_from_coils(coils)
    surface = StandardSimsoptSolver().solve(seed, field, target_volume=volume)
    if surface.accepted:
        surface.save(seed_path.with_name("surface.npz"))
    return surface.accepted, surface.quality, surface.iota, perf_counter() - started


def main():
    # GPU imports stay in the parent process when CPU workers are spawned.
    from coil_eval import CoilSet, EvaluationConfig
    from coil_eval.simsopt import capture_sources, GpuCoilField, AlphaNuInitializer, InitializerConfig

    started = perf_counter()
    coils = CoilSet.read_json(COILS)
    configs = [EvaluationConfig(target_volume=v, psi_a=RADIUS, helicity=(1, 0)) for v in VOLUMES]
    print(f"Preparing {len(VOLUMES)} target surfaces...", flush=True)
    captures = capture_sources(coils, configs)
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("spawn"),
                             initializer=cpu_environment) as pool:
        futures = {}
        with GpuCoilField(coils) as field, AlphaNuInitializer(
                field, config=InitializerConfig(device="cuda:0")) as initializer:
            for index, (volume, capture) in enumerate(zip(VOLUMES, captures)):
                tick = perf_counter()
                if not capture.valid:
                    print(f"V={volume:.3f} m^3: {capture.result.status}", flush=True)
                    continue
                folder = OUTPUT / f"surface_{index}"
                folder.mkdir(parents=True, exist_ok=True)
                seed = initializer.prepare(capture.source, capture.alpha)
                seed_path = (folder / "seed.npz").resolve()
                seed.save(seed_path)
                futures[pool.submit(solve_one, seed_path, volume)] = volume
                print(f"V={volume:.3f} m^3: seed ready in {perf_counter() - tick:.2f} s; "
                      "CPU solve submitted", flush=True)
        for future in as_completed(futures):
            volume = futures[future]
            accepted, quality, iota, seconds = future.result()
            print(f"V={volume:.3f} m^3: accepted={accepted}, quality={quality}, "
                  f"iota={iota:.8f}, solve={seconds:.2f} s", flush=True)
    print(f"Total elapsed: {perf_counter() - started:.2f} s")
    print(f"Seeds and accepted surfaces: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
