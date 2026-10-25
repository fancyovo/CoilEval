"""Evaluate a coil set with the optional two-stage psi fit."""
import argparse
from pathlib import Path

from coil_eval import CoilSet, EvaluationConfig, Evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coils", type=Path, default=Path(__file__).with_name("coils.json"))
    parser.add_argument("--library", help="Optional custom native library")
    parser.add_argument("--volume", type=float, default=.05, help="Target volume in m^3")
    parser.add_argument("--radius", type=float, default=.15, help="Fitting radius in metres")
    parser.add_argument("--sampling", choices=("cartesian", "polar_quadrature"), default="cartesian")
    parser.add_argument("--alpha-weighting", choices=("histogram", "smooth"), default="histogram")
    parser.add_argument("--alpha-basis", choices=("zernike", "flux_legendre"), default="zernike")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--output", type=Path, help="Save EvaluationResult as JSON")
    args = parser.parse_args()
    config = EvaluationConfig(target_volume=args.volume, psi_a=args.radius,
        fast=True, flux_method="stokes", psi_fit="two_stage", psi_refit_sampling=args.sampling,
        alpha_radial_weighting=args.alpha_weighting, alpha_radial_basis=args.alpha_basis)
    with Evaluator(args.library, config=config, device_id=args.device) as evaluator:
        result = evaluator.evaluate(CoilSet.read_json(args.coils))
    print("Status:", result.status)
    if result.valid:
        print("Volume QS RMS:", result.metrics.qs_volume)
        print("Outer-bin QS RMS:", result.metrics.qs_outer)
        print("Boundary iota:", result.metrics.iota_boundary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.write_json(args.output)


if __name__ == "__main__":
    main()
