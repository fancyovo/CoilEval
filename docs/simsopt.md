# Simsopt integration

The example uses a wheel built from CoilEval 0.12.0. See the [installation guide](installation.md)
for building and installing the wheel.

[简体中文](simsopt.zh-CN.md)

Simsopt is an independent magnetic-field and surface-optimization library. LS/Newton means least-squares fitting followed by Newton refinement.

<details markdown="1">
<summary>Terms and coordinate conventions used on this page</summary>

On this page, **quasisymmetry (QS)** means that magnetic-field strength depends on one
helical angle in suitable surface coordinates. **QA** is quasi-axisymmetry and **QH**
is quasi-helical symmetry. `nfp` is the number of repeated field periods around the
device; `helicity=(M,N)` selects the angle `Mθ−Nφ`, with poloidal θ around a surface
cross-section and toroidal φ around the device. **Rotational transform ι (iota)** is
the average number of poloidal turns per toroidal turn of a field line. **RMS** is
the square root of a weighted mean of squares; **P90/P95** are the 90th/95th
percentiles. `f_B` measures field-strength Fourier content outside the requested
symmetry; `f_C` is a differential quasisymmetry residual; `f_T` is a triple-product
residual involving field derivatives. All have ideal value zero; comparisons
require matching averaging regions and normalization.

**Fits and coordinates.** The scalar surface label s approximates magnetic surfaces by its
level sets; fitting fields in the API retain the `psi_` prefix. Physical toroidal
flux ψ is calibrated separately by flux integration. The default two-fit procedure
fits an initial region, selects a closed region and fits that region again. An
optional third fit uses the entire interior of a boundary slightly beyond the requested volume; it is off by default.
Alpha (α) fitting constructs field-line angles, with lambda (λ) as a periodic
angular correction. Nu (ν) then constructs Boozer angles: coordinates in which
field lines are straight and covariant angular magnetic-field components depend
only on the surface. A volume-alpha fit spans the volume; a surface-alpha fit is
recomputed on one target surface, with orders referring to angular Fourier modes.
Rho (ρ) is a normalized radial label. `flux_legendre` uses Legendre polynomials of
normalized flux; `zernike` uses another radial polynomial basis. Polar quadrature uses weighted radial/angular samples; Cartesian sampling uses a rectangular grid. Smooth density weights estimate radial sample density; the positive pseudo-count added to the denominator prevents excessive weights in sparse regions. Ridge
regularization adds a coefficient penalty to stabilize a fit. Stellarator symmetry generates the full device by rotations and reflection of base coils. Stokes flux integration uses a boundary line integral; area integration directly samples the cross-section. QR solves least
squares using an orthogonal/upper-triangular matrix factorization.

**Saved state and precision.** The native backend is the packaged C++/CUDA library.
A capture saves the axis, fit coefficients and sample points from a complete
evaluation for later derivatives or surface construction. The center is that
evaluated parameter vector; endpoints are small perturbations. A local derivative
keeps the current discrete choices fixed while differentiating continuous changes.
Capacity is the largest volume represented by the fitted scalar's radially
monotone region. The first branch is the first outward monotone radial interval
from the magnetic axis. A fold is loss of coordinate monotonicity. A Jacobian is a
first-derivative matrix; the angular-map minimum Jacobian denotes its determinant.
FP32/float32 and FP64/float64 denote 32-bit single and 64-bit double precision.

</details>


`coil_eval.simsopt` provides source-psi and alpha records, alpha+nu surface
initialization, a standard Simsopt LS/Newton solve, and reference QS diagnostics.
Install Simsopt in the environment used for surface preparation and solves.
CUDA preparation also uses the optional PyTorch dependency. These dependencies
are imported when their operations run.

Install the surface dependencies from a local wheel with:

```sh
python -m pip install "./coil_eval-0.12.0-py3-none-manylinux_2_28_x86_64.whl[simsopt]"
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
Configurations select `psi_fit='two_stage'` or `psi_fit='near_axis_ellipse'` and
differ only in `target_volume`. The two-fit method shares its magnetic axis,
initial fit, training domain and second fit across volumes. Ellipse sampling
uses the requested volume to size the training region, so each volume gets its
own fit. Every volume receives its own downstream evaluation.
When `psi_target_refit=True`, each volume also receives its own third fit and
the returned `SourcePsi` contains the scalar from the third fit. The original volume-alpha
preparation remains part of the capture.
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

Orders in the table are retained angular-mode limits; nphi/ntheta are toroidal/poloidal sample counts. FFT means fast Fourier transform; `cached_qr` caches the least-squares QR factors; `resident` performs angle inversion on the device; `reference` uses direct geometry calculations and `cached_basis` caches geometry basis functions. The byte limit bounds cache memory.

Reuse an initializer to reuse its projection factors. `seed.diagnostics`
contains extraction, mapping, fit and timing information. `SurfaceSeed.save`
and `SurfaceSeed.read` preserve the alpha+nu and source-psi type markers used
by the solver.
Initializer-generated seeds also retain `source_axis_samples` with columns
`(R, Z, dR/dphi, dZ/dphi)` for angular-topology checks. This array is optional
when reading earlier seed files; accepted standard-surface files retain it.

`geometry_method='cached_basis'` optionally caches the surface geometry basis.
The reference geometry method is the default.

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

The solver also checks eight true geometric-toroidal sections with
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
the QH reference for the fixed helicity `(1,nfp)`. For `nfp=4`, compare them with
native results configured with `helicity=(1,4)`. A QA f_C comparison requires a
reference routine that accepts `helicity=(1,0)` on the same surface.
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

## GPU preparation and CPU workers in one run

[simsopt_async.py](../examples/simsopt_async.py) directly demonstrates seed construction and concurrent solves:

```sh
python examples/simsopt_async.py
```

Edit `VOLUMES`, `RADIUS`, `WORKERS` and `OUTPUT` at the top. The same run saves GPU seeds and submits CPU workers,
printing status, quality, rotational transform and timing for each target. Workers use one BLAS/OpenMP thread and hide CUDA;
keep `WORKERS` within the assigned CPU count. Seeds and accepted surfaces are saved under `OUTPUT`.
The library's geometry and provenance checks stay enabled.

See [target-surface f_B and f_T](qs.md) for optional metrics and accuracy presets.

A flux shell is the thin region between neighboring equal-flux surfaces; its weighted RMS differs from a whole-volume RMS. Relative L2 error is the Euclidean norm of the difference vector divided by that of the reference.
