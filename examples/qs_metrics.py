"""Evaluate target-surface f_B/f_T and optionally differentiate one metric."""
import argparse
from pathlib import Path

from coil_eval import CoilSet, EvaluationConfig
from coil_eval.qs import SurfaceQSConfig, SurfaceQSEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coils", type=Path, default=Path(__file__).with_name("coils.json"))
    parser.add_argument("--library", help="Optional custom native backend")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--volume", type=float, default=.05, help="Target volume in m^3")
    parser.add_argument("--radius", type=float, default=.15, help="Psi fitting radius in metres")
    parser.add_argument("--accuracy", choices=("high", "balanced", "fast"), default="high")
    parser.add_argument("--nphi", type=int, help="Override the preset toroidal grid size")
    parser.add_argument("--ntheta", type=int, help="Override the preset poloidal grid size")
    parser.add_argument("--nu-order", type=int, help="Override the preset nu mode limit")
    parser.add_argument("--alpha-order", type=int, help="Override the target-surface alpha mode limit")
    parser.add_argument("--gradient", action="store_true", help="Differentiate the selected metric squared")
    parser.add_argument("--metric", choices=("f_B", "f_T"), default="f_B")
    args = parser.parse_args()

    coils = CoilSet.read_json(args.coils)
    config = EvaluationConfig(
        target_volume=args.volume, psi_a=args.radius,
        fast=True, flux_method="stokes", psi_fit="two_stage",
        psi_refit_sampling="polar_quadrature", alpha_radial_weighting="smooth",
        alpha_radial_basis="flux_legendre",
    )
    try:
        overrides = {name: getattr(args, name) for name in ('nphi', 'ntheta', 'nu_order', 'alpha_order')
                     if getattr(args, name) is not None}
        surface_qs = SurfaceQSConfig.preset(args.accuracy, compute_f_T=args.metric == 'f_T', **overrides)
    except ValueError as exc:
        parser.error(str(exc))
    if args.gradient:
        import torch
        from coil_eval.torch import TensorEvaluator, pack_coils
        parameters = pack_coils(coils, device=f"cuda:{args.device}")
        with TensorEvaluator(args.library, config=config, surface_qs=surface_qs,
                             device_id=args.device, field_dtype=torch.float64) as evaluator:
            result = evaluator.evaluate(parameters)
            qs = result.surface_qs
            objective = getattr(qs, args.metric).square()
            gradient, = torch.autograd.grad(objective, result.parameters)
            print("f_B:", float(qs.f_B.detach()))
            if qs.f_T is not None:
                print("f_T:", float(qs.f_T.detach()))
            print(f"{args.metric} squared:", float(objective.detach()))
            print("Physical parameter gradient:", gradient.cpu())
            print("Surface diagnostics:", qs.diagnostics)
    else:
        with SurfaceQSEvaluator(args.library, config=config, surface_qs=surface_qs,
                                device_id=args.device) as evaluator:
            qs = evaluator.evaluate(coils)
            print("f_B:", qs.f_B)
            if qs.f_T is not None:
                print("f_T:", qs.f_T)
            print("Surface diagnostics:", qs.diagnostics)


if __name__ == "__main__":
    main()
