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
| 0.7.0 | CPU backend, thread controls and throughput improvements | `device="cpu"`; `threads=`; CPU `axis_grid=` |
| 0.8.0 | GPU capture overlap, batch surface sweeps and fewer synchronization barriers | GPU execution controls in the [Torch guide](torch.md) |
| 0.8.1 | Query and stage attribution for batch capture errors | Included in the GPU Tensor path |
| 0.8.2 | Early dependency checks and complete installation guidance | Install the wheel with the extra for the chosen backend |
| 0.9.0 | General field-period/coil counts and QA/QH modes | Explicit tensor nfp and configuration helicity |

Version 0.11.0 is the first S2 preview candidate. Its public defaults are
`fast=True`, `flux_method='stokes'`, `psi_fit='two_stage'`, polar quadrature,
smooth volume-alpha weights and the flux-Legendre volume-alpha basis.
S3 remains optional (`psi_target_refit=False`).

| Version | Added capability | Selection |
| --- | --- | --- |
| 0.9.1 | Angle inverse and first radial-branch repairs; S2 ridge override | Included; `psi_refit_ridge=` optional |
| 0.10.0 | Target-interior S3 fit and its local derivatives | `psi_target_refit=True` |
| 0.11.0 preview | Validated S2 defaults, P90 calibration, perturb32 optimization example | Default configuration; example2 |

Versions through 0.10.0 used S1 by default. Reproduce historical runs with their
frozen version and full configuration. For an explicit legacy S1 configuration
on 0.11.0, follow the [API guide](api.md#configuration). API argument order and
physical units are unchanged; numerical defaults changed deliberately.
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
