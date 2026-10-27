# Versions

[简体中文](versions.zh-CN.md)

| Version | Added capability | Selection |
| --- | --- | --- |
| 0.0.0 | Independent evaluation and local finite-difference gradients | `Evaluator` |
| 0.1.0 | Faster integration settings | `EvaluationConfig(fast=True, ...)` |
| 0.2.0 | Stokes boundary flux in local evaluations | `flux_method='stokes'` with fast mode |
| 0.3.0a1 | PyTorch reverse gradients and independent-geometry batches | `coil_eval.torch.TensorEvaluator` |
| 0.3.0a2 | Linux CUDA 12.8 binary wheel and automatic backend loading | Install a wheel; omit `library` |
| 0.3.0a3 | Validated CUDA runtime dependency set | Install the updated wheel |
| 0.4.0a1 | Two-stage psi, higher Fourier orders and Simsopt integration | `psi_fit='two_stage'`; `coil_eval.simsopt` |
| 0.5.0a1 | Flux-polynomial alpha, shared tensor preparation and surface covering checks | `alpha_radial_basis='flux_legendre'`; batch and Simsopt APIs |
| 0.5.1 | Capacity guards, reference intersections and stable basis evaluation | Included in later versions |
| 0.6.0 | Surface f_B, optional f_T, reverse gradients and accuracy presets | `coil_eval.qs`; `TensorEvaluator(surface_qs=...)` |

The source version is 0.6.0 (unreleased), adding target-surface f_B, optional f_T,
their local reverse gradients and accuracy presets. It includes the 0.5.1
capacity, reference-surface and flux-Legendre fixes.

The default NumPy behavior stays compatible across these versions. New numerical
paths are selected explicitly. Use an exact version and save the configuration
alongside results when repeating an evaluation.

In 0.5.0a1, `psi_fit='single_stage'` remains the default. Two-stage fitting uses
`psi_refit_sampling='cartesian'` unless selected otherwise. Weighted
`'polar_quadrature'` is an optional experimental sampler. The Tensor API accepts
both fitting methods and preserves physical units for every Fourier order.
The Simsopt workflow has its own surface RMS diagnostics; choose comparable
averaging regions when combining them with volume QS metrics.

To build an earlier release from a clean checkout:

```sh
git switch --detach v0.0.0
python -m pip install .
cmake -S native -B build/v0.0.0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/v0.0.0 --parallel
```

Keep each native build paired with the Python version that built it. Release
tags identify complete source trees, including their documentation and examples.
