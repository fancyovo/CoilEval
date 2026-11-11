# Versions

[简体中文](versions.zh-CN.md)


This page records when features were introduced; defaults in historical entries belong to those versions. Psi/s is a scalar label whose level sets represent surfaces; alpha is a field-line angle correction. Two fits use an initial region and a selected closed interior; a third fit uses the full interior slightly beyond the target volume. Capacity is the largest representable volume and the first branch is the first outward monotone interval from the axis. A capture saves intermediate evaluation state.

QS means quasisymmetry; QA/QH are quasi-axisymmetry/quasi-helical symmetry. f_B/f_T measure asymmetric Fourier content and a field-derivative triple-product residual. nfp counts field periods and helicity selects a poloidal/toroidal angle combination. P90 is the 90th percentile. Stokes flux integration uses a boundary line integral; polar quadrature weights radial/angular samples; smooth weights correct uneven sample density; flux-Legendre uses Legendre polynomials of normalized flux. Simsopt is an independent surface-optimization library; a wheel is a built Python distribution. Regularization penalizes fit coefficients; angle inversion finds spatial positions for requested angles.

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
| 0.7.0 | CPU backend, thread controls and throughput improvements | `device="cpu"`; `threads=`; CPU `axis_grid=` |
| 0.8.0 | GPU capture overlap, batch surface sweeps and fewer synchronization barriers | GPU execution controls in the [Torch guide](torch.md) |
| 0.8.1 | Query and stage attribution for batch capture errors | Included in the GPU Tensor path |
| 0.8.2 | Early dependency checks and complete installation guidance | Install the wheel with the extra for the chosen backend |
| 0.9.0 | General field-period/coil counts and QA/QH modes | Explicit tensor nfp and configuration helicity |

Version 0.11.0 is the first preview candidate using two fits by default. Its public defaults are
`fast=True`, `flux_method='stokes'`, `psi_fit='two_stage'`, polar quadrature,
smooth volume-alpha weights and the flux-Legendre volume-alpha basis.
The third fit remains optional (`psi_target_refit=False`).

| Version | Added capability | Selection |
| --- | --- | --- |
| 0.9.1 | Angle inverse and first radial-branch repairs; the second fit ridge override | Included; `psi_refit_ridge=` optional |
| 0.10.0 | Optional third fit over the target interior and its local derivatives | `psi_target_refit=True` |
| 0.11.0 preview | Validated two-fit defaults, P90 calibration, optimization example selecting from 32 perturbations | Default configuration; example2 |
| 0.11.1 preview | Editable one-run examples, progress and timings | `examples/example1.py`, `examples/example2.py` |

Versions through 0.10.0 used a single fit by default. Reproduce historical runs with their
frozen version and full configuration. For an explicit legacy single-fit configuration
on 0.11.0 and 0.11.1, follow the [API guide](api.md#configuration). API argument order and
physical units were retained in 0.11.0, which changed numerical defaults; 0.11.1 retains those defaults.
The [validation record](preview_validation.md) identifies the scored cohort,
reference qualifications, errors, timing workload and examples.

To build an earlier release from a clean checkout:

```sh
git switch --detach v0.0.0
python -m pip install .
cmake -S native -B build/v0.0.0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/v0.0.0 --parallel
```

Keep each native build paired with the Python version that built it. Release
tags identify complete source trees, including their documentation and examples.

Version 0.11.1 revises tutorials with editable settings, one-run workflows, public APIs, progress and timings; the evaluator and two-fit defaults match 0.11.0.

Version 0.12.0 adds one fit in a region predicted from the near-axis magnetic
field. Select it with `psi_fit='near_axis_ellipse'`. The CPU and GPU interfaces
provide values and local gradients for the new fit. The default uses two fits.
