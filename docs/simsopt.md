# Simsopt integration

Version 0.9.0 is unreleased. The example below uses the supplied binary wheel.

[简体中文](simsopt.zh-CN.md)

`coil_eval.simsopt` provides source-psi and alpha records, alpha+nu surface
initialization, a standard Simsopt LS/Newton solve, and reference QS diagnostics.
Install Simsopt in the environment used for surface preparation and solves.
CUDA preparation also uses the optional PyTorch dependency. These dependencies
are imported when their operations run.

Install the surface dependencies from a local wheel with:

```sh
python -m pip install "./coil_eval-0.9.0-py3-none-manylinux_2_28_x86_64.whl[simsopt]"
```

The `simsopt` extra includes Simsopt and SciPy. Set up CUDA-enabled PyTorch
following [the installation guide](installation.md) for GPU preparation.

## Capture one geometry at several volumes

```python
from dataclasses import replace
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.simsopt import capture_sources

coils = CoilSet.read_json("examples/coils.json")
base = EvaluationConfig(
    target_volume=0.05, psi_a=0.15,
    fast=True, flux_method="stokes", psi_fit="two_stage",
)
captures = capture_sources(coils, [replace(base, target_volume=v) for v in (0.01, 0.03, 0.05)])
for capture in captures:
    print(capture.valid, capture.result.status)
```

`capture_sources(coils, configs, library=None, device_id=0, include_torsion=True)` accepts one
`EvaluationConfig` or a sequence and always returns a list in input order.
Configurations select `psi_fit='two_stage'` and differ only in `target_volume`.
One geometry shares its magnetic axis, first fit, training domain and second
fit across those volumes. Each volume receives its own downstream evaluation.
Set `include_torsion=False` when preparing sources without the optional torsion
diagnostic. Scoring metrics and source coefficients retain the same definitions.

Each `SourceCapture` contains `.result` (`EvaluationResult`), `.source`
(`SourcePsi`) and `.alpha` (`AlphaFit`). Check `.valid` before preparing a
surface. Invalid targets retain their result and have empty source/alpha fields.
The records own their host arrays and remain usable after capture returns.

`SourcePsi` exposes coefficients, mode arrays, axis samples, field periods,
fitting radius and selected level. `AlphaFit` exposes its modes, lambda and
iota coefficients. `from_snapshot()` supports advanced integrations with an
existing native capture. `SourcePsi.read(path, level=...)` and
`AlphaFit.read(path, nfp=...)` load saved fit records.
`AlphaFit.radial_basis` records the captured evaluator's radial basis, including
`'flux_legendre'`; initializer evaluation uses that same basis. Saved alpha NPZ
files may include a scalar `radial_basis`; older files use `'zernike'`.

## Prepare an alpha+nu initializer

```python
from coil_eval.simsopt import GpuCoilField, AlphaNuInitializer, InitializerConfig

capture = next(item for item in captures if item.valid)
with GpuCoilField(coils) as field, AlphaNuInitializer(
    field, config=InitializerConfig(device="cuda")
) as initializer:
    seed = initializer.prepare(capture.source, capture.alpha)
seed.save("surface_seed.npz")
```

`GpuCoilField(coils, library=None, device_id=0, segments_per_coil=256,
precision='fp32')` is a context-managed callable. Its `(N,3)` Cartesian inputs
use metres; its output is B in tesla. `precision='fp64'` is also available.
`close()` releases the field and is safe to call more than once.

`AlphaNuInitializer(field, library=None, config=None, device_id=None)` accepts a
Cartesian B callback. `prepare(source, alpha, rho=1.0)` returns `SurfaceSeed`.
With `InitializerConfig(device='cuda')`, `device_id` selects both native
extraction and PyTorch work. An explicit `device='cuda:N'` selects N when
`device_id` is omitted; conflicting explicit choices raise `ValueError`.
Omitting both choices retains GPU 0. CPU initialization keeps its selected
CPU implementation.
The source, alpha and field should describe the same coil geometry and volume.
Capture uses the evaluator's native alpha basis. A separately supplied dense
alpha fit can produce a different initializer and should be evaluated on the
same target volume when comparing choices.

| `InitializerConfig` option | Default |
| --- | --- |
| `surface_order`, `extract_order`, `nu_order` | `12`, `24`, `12` |
| `extract_maxiter`, `extract_tol` | `20`, `1e-12` |
| `fit_nphi`, `fit_ntheta`, `interpolation_size` | `65`, `67`, `97` |
| `regularization` | `0.0` |
| `device` | `'cpu'` |
| `nu_method`, `projection_method`, `inversion_method` | `'fft'`, `'cached_qr'`, `'resident'` |
| `geometry_method`, `geometry_cache_bytes` | `'reference'`, `536870912` |
| `reject_alpha_folds`, `max_radius_scale` | `True`, `1.0` |
| `alpha_fold_policy`, `alpha_min_jacobian` | `'reject'`, `0.05` |

Reuse an initializer to reuse its projection factors. `seed.diagnostics`
contains extraction, mapping, fit and timing information. `SurfaceSeed.save`
and `SurfaceSeed.read` preserve the alpha+nu and source-psi type markers used
by the solver.
Initializer-generated seeds also retain `source_axis_samples` with columns
`(R, Z, dR/dphi, dZ/dphi)` for angular-topology checks. This array is optional
when reading earlier seed files; accepted standard-surface files retain it.

`geometry_method='cached_basis'` optionally caches the surface geometry basis.
The reference geometry method is the default in 0.9.0.

`alpha_fold_policy='damp'` can regularize a folded alpha coordinate map during
initialization. It scales the lambda perturbation by one factor, preserves iota,
and recomputes nu. The map is checked on at least a `97 × 257` phi/theta grid and
on the extraction nodes; its Jacobian must meet `alpha_min_jacobian`.
The default `'reject'` policy keeps the existing fold check.

```python
config = InitializerConfig(
    device="cuda", geometry_method="cached_basis",
    alpha_fold_policy="damp", alpha_min_jacobian=0.05,
)
```

`seed.diagnostics["alpha_transform"]` records the damping factor, original and
final minimum Jacobians, and validation grid. This option changes the initial
coordinate map; native scores and gradients are unchanged. Pass the resulting
seed through the standard solve and reference diagnostics below.

## Solve on CPU

```python
from coil_eval.simsopt import (
    SurfaceSeed, field_from_coils, StandardSimsoptSolver, surface_fc,
)

seed = SurfaceSeed.read("surface_seed.npz")
field = field_from_coils(coils)
surface = StandardSimsoptSolver().solve(seed, field, target_volume=0.05)
print(surface.accepted, surface.quality, surface.diagnostics)
if surface.accepted:
    surface.save("standard_surface.npz")
    reference = surface_fc(surface, field, sizes=(64, 128))
    print(reference["surface_fC_flux_shell_rms"])
```

`field_from_coils(coils, quadpoints=None)` builds a Simsopt BiotSavart field
from the physical Fourier coefficients and ampere currents.
`StandardSimsoptSolver(config=SolverConfig(...))` performs least squares and
then Newton refinement. Positive `target_volume` is in m³; the solver preserves
the seed's signed surface orientation. `validation_field` optionally supplies
an independent B evaluation for the final geometry checks.

`SolverConfig` defaults are `ls_tol=1e-10`, `ls_maxiter=100`,
`newton_tol=1e-12`, `newton_maxiter=30`, `constraint_weight=1.0`,
`max_newton_residual_norm=1e-8`, `max_final_relative_l2=1e-4`,
`max_final_normal_p95=1e-4`, and `validation_grids=(49,97)`.
`StandardSurface` contains the final dofs, iota, G, acceptance, quality and
diagnostics. `save()` writes accepted surfaces; rejected candidates remain
available through the returned object for inspection.

With source-axis samples in the seed, the solver checks angular covering on
the solve grid and each validation grid. The clockwise poloidal angle must
wind once along the poloidal parameter and zero times along a toroidal field
period; the cylindrical toroidal angle must have the corresponding degrees
zero and one. `hard_downstream_checks['single_angular_covering']` records this
acceptance check, and each grid's `angular_covering` records the measured
degrees. Select `SolverConfig(validation_grids=(97,129))` for finer checks.
Legacy seeds without axis samples skip the axis-relative covering check and
record `source_axis_unavailable`. Dense residual quality is reported separately
as `strict_pass` or `accepted_with_quality_warning` for accepted surfaces.

The 0.9.0 solver also checks eight true geometric-toroidal sections with
256 poloidal samples for self-intersection. Polygon crossings nominate pairs
of parameters; a root on the original Fourier surface must confirm their
Cartesian coincidence within `1e-9` m before
`hard_downstream_checks['no_detected_self_intersection']` rejects the surface.
The search runs once on the final geometry and does not require source-axis
samples. The final validation grid's `self_intersection` record includes the
witness, sampling scope and elapsed time.

`not_detected_in_sampled_sections` records a finite search, not a global
embedding certificate. A nonmonotone geometric-toroidal map or unresolved
candidate remains inconclusive: the surface retains a quality warning through
`strict_quality_checks['self_intersection_audit_resolved']`, without a new hard
rejection solely for that inconclusive result. Non-star-shaped poloidal sections
are not themselves self-intersections. Earlier saved acceptance records keep
their original meaning and are not retroactively rewritten.

## Reference QS quantities

`surface_fc(surface, field, sizes=(64,128))` and `analyze_arrays(...)` implement
the QH reference for helicity `(1,nfp)`. Both functions have no helicity
argument in 0.9.0. For `nfp=4`, compare them with native results configured
with `helicity=(1,4)`; a QA f_C comparison requires an explicit-helicity
reference such as the parent validation campaign's matched-surface adapter.
The separate f_B/f_T reference `surface_qs_metrics` accepts QA `(1,0)`;
see the [surface QS guide](qs.md#compare-with-simsopt).
`surface_fc` evaluates an accepted standard surface on the requested grids
using that native-normalized differential residual. It returns:

- `surface_fC_flux_shell_rms`: RMS with magnetic-flux shell weights.
- `surface_fC_area_rms`, `surface_fC_angular_rms`: area and angular weightings.
- Actual enclosed volume, per-grid values, grid change and Boozer residual diagnostics.

These values are RMS quantities. `qs_volume` covers the evaluator's whole
selected volume, while `qs_outer` covers its outer radial bin. A solved
surface's flux-shell RMS has a different averaging region. Compare the same
definition, weighting and actual enclosed volume when calibrating results.

`volume_average_fc(volumes, surface_fc_values, inner_fc=None)` integrates
squared flux-shell RMS values in actual enclosed volume and takes the square
root. Supply increasing volumes on a common nested branch. Include a value at
zero volume, or provide an explicit `inner_fc` approximation for the unsampled
inner region. The returned record states that approximation.

## Separate GPU preparation and CPU workers

[simsopt_async.py](../examples/simsopt_async.py) saves initializers in one
process and solves them in a separate CPU process pool:

```sh
python examples/simsopt_async.py prepare --directory output/case1
python examples/simsopt_async.py solve --directory output/case1 --workers 4
```

The preparation command exits after writing its tasks. CPU jobs can consume
completed directories while other GPU jobs prepare more geometries. Each CPU
worker uses one BLAS/OpenMP thread and hides CUDA devices. Set `--workers` to
the CPU cores assigned to the job. Both phases preserve each target's success
or failure status and use explicit file paths through their CLI options.

See [target-surface f_B and f_T](qs.md) for optional metrics and accuracy presets.
