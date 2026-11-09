# Python API

[简体中文](api.zh-CN.md)


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


## Coil inputs

`CoilSet(coeffs_x, coeffs_y, coeffs_z, currents_a, nfp)` describes basic coils
with stellarator symmetry. Each coordinate array has shape `(coils, 2*order+1)`
and coefficient order `[constant, sin(1), cos(1), ..., sin(order), cos(order)]`.
Coordinates use metres and currents use amperes. Symmetry produces `2*nfp`
physical coils for each basic coil.

Field periods `nfp` and the number of base coils may be any positive integers.
The geometry retains stellarator symmetry. QS helicity is explicit: QA uses
`(1,0)` and standard QH uses `(1,nfp)`, where N is the physical full-torus
helicity, not a field-period index. The compatibility default is still `(1,4)`;
changing `CoilSet.nfp` does not silently change the selected QS metric. M must
remain nonzero. Tensor inputs carry no nfp metadata: construct their evaluator
with `nfp=coils.nfp`. `evaluate_many` accepts mixed CoilSet layouts and preserves
input order; dense tensor batches share nfp, coil count and coefficient width.

`parameter_vector()` flattens all x coefficients, then y, z and currents.
`with_parameters(vector)` creates a new coil set in that order. `read_json(path)`
and `write_json(path)` preserve these units and conventions.

## Configuration

`EvaluationConfig(target_volume, ...)` is immutable. The target volume uses m³.
`to_dict()` returns a serializable configuration.

| Option | Default | Meaning |
| --- | --- | --- |
| `psi_a` | `0.15` | Fitting radius in metres |
| `helicity` | `(1, 4)` | QS helicity `(M, N)` |
| `h` | `1e-4` | Normalized centered-difference step |
| `segments_per_coil` | `256` | Coil discretization |
| `psi_iterations`, `alpha_iterations` | `4`, `4` | Local iterative fit settings |
| `fast` | `True` | Use 240 axis steps and 100 surface tracing steps |
| `flux_method` | `'stokes'` | `'area'` or fast local `'stokes'` boundary integration |
| `psi_fit` | `'two_stage'` | `'single_stage'` or `'two_stage'` |
| `psi_refit_points` | `82368` | Target training-row count for the second fit |
| `psi_refit_ridge` | `None` | Optional nonnegative column-scaled ridge for the second fit only; `None` retains the native setting |
| `psi_target_refit` | `False` | Optional third fit over the target interior after `psi_fit='two_stage'` |
| `psi_target_refit_factor` | `1.2` | Requested training volume for the third fit divided by target volume; must exceed one |
| `psi_target_refit_ridge` | `1e-10` | Nonnegative column-scaled ridge for the third fit only |
| `psi_refit_sampling` | `'polar_quadrature'` | Two-stage `'cartesian'` or `'polar_quadrature'` sampling |
| `psi_nearby_method` | `'local'` | Two-stage local response or `'formal'` endpoint reconstruction |
| `psi_fd_batch_size` | `8` | Two-stage local finite-difference endpoint batch size |
| `alpha_radial_weighting` | `'smooth'` | `'histogram'` or two-stage `'smooth'` radial density weighting |
| `alpha_weight_sigma` | `0.05` | Gaussian width in normalized rho for smooth weighting |
| `alpha_count_floor` | `32.0` | Positive additive pseudocount for smooth density denominators |
| `alpha_radial_basis` | `'flux_legendre'` | Original radial basis or two-stage `'flux_legendre'` |
| `stokes_validation_rtol` | `1e-3` | Tensor flux consistency tolerance; `None` disables the check |

Version 0.11.1 defaults to the validated two-fit configuration. The following explicit
configuration is equivalent to `EvaluationConfig(target_volume=0.05, psi_a=0.15)`:

```python
config = EvaluationConfig(
    target_volume=0.05, psi_a=0.15,
    fast=True, flux_method="stokes",
    psi_fit="two_stage", psi_refit_sampling="polar_quadrature",
)
```

Two-stage evaluation first fits psi in the initial domain, constructs an
axis-connected closed training domain, and fits psi again in that domain.
The target-volume surface and QS metrics use the second fit.
`psi_refit_ridge` can reduce regularization bias when resolving very small QS
errors. It applies to the second fit only, including local gradients; the first fit and its domain stay
fixed. The GPU option requires a native library exposing the 0.9.1 setter.
Smaller ridge does not guarantee a more accurate magnetic surface: compare
normal-field error, coordinate convergence and geometric capacity. Target-local
third-fit behavior is selected separately with `psi_target_refit`.

With `psi_target_refit=True`, the first fit and the second fit retain their configured objectives.
The second fit then supplies a closed first-root boundary of volume
`min(psi_target_refit_factor*V, (V+Vcapacity)/2)` for a third fit. The capacity
must exceed the requested physical volume V. The third fit uses weighted polar quadrature
over the entire interior and the same basis and backend precision. Its nominal
row count is `psi_refit_points`; `psi_target_refit_ridge` controls only the third fit.
The final surface uses the third fit at the original V. Unavailable capacity or a failed
continuation remains an explicit failure. Inspect `psi_fit_stages` and
`psi_target_refit_domain` in the native diagnostics. The third fit requires explicit
selection: shrinking the training domain together with weak regularization can
produce a nonmonotone scalar and lose usable capacity even on a previously
scorable input. Its fixed 4096-input validation does not support default promotion.
Changing the domain and changing ridge have separate effects and can interact;
restoring ridge alone does not establish accuracy for every target.

Multi-volume captures share the first and second fits; each target owns its coefficients from the third fit, QR
factors and downstream state. Local derivatives include the second fit's moving training
boundary, the third fit volume equation, capacity clipping and polar area weights on
the selected discrete branch. Surface-alpha and native volume-alpha settings
retain their independent meanings.
The target-volume capacity guard checks at least the psi fitting toroidal grid
at twice its resolution and at least `4*psi_m_tor+1` toroidal sections. Flux quadrature remains separately
configured. This is a sampled radial-branch check; an accepted result does not
certify a global magnetic surface or physical accuracy.
`psi_refit_sampling='polar_quadrature'` selects the default weighted polar
quadrature. Cartesian sampling remains explicitly available.
Both samplers retain complete sampling structures, so the actual row count can
differ from `psi_refit_points`; inspect `diagnostics['psi_refit_training_count']`.

`alpha_radial_weighting='smooth'` is the default for
`psi_fit='two_stage'`. Its positive sigma and pseudocount settings control the
smooth density weights. Histogram weighting remains explicitly available.

`alpha_radial_basis='flux_legendre'` fits the angular correction with
polynomials of normalized flux. Select it with `psi_fit='two_stage'`; the
choice applies to scalar evaluation, local gradients and captured alpha data.
The default is `'flux_legendre'`; the original `'zernike'` basis remains available.
For legacy the first fit select `psi_fit='single_stage'`, `alpha_radial_weighting='histogram'`
and `alpha_radial_basis='zernike'` together. To restore pre-0.11 standard tracing
also select `fast=False`, `flux_method='area'` and `psi_refit_sampling='cartesian'`.

`flux_method='stokes'` requires `fast=True` and selects boundary integration for
local neighborhoods and finite-difference endpoints. Independent center calls
retain their center integration rule. Results save the selected settings in
`diagnostics['configuration']`.

Tensor evaluation compares its Stokes flux and flux derivative with the captured
center's area-integral polynomial. If either relative error exceeds
`stokes_validation_rtol`, that query uses differentiable area integration.
`None` selects the original unchecked Tensor integration behavior. The setting
applies to the Tensor interface; the NumPy evaluator retains the integration
rules described above. See [tensor results](torch.md#results-and-local-derivatives)
for the selected method and error diagnostics.

## Evaluator sessions

Use `Evaluator(config=config, device_id=0)` as a context manager on its creating
thread. The wheel supplies the native backend; `library=...` selects a separately
built library.

| Method | Return value |
| --- | --- |
| `evaluate(coils)` | One `EvaluationResult` |
| `evaluate_many(coil_sets, configs=None)` | Independent results in input order, with batched axis tracing for compatible inputs |
| `evaluate_nearby(center, candidates)` | Results using the live center's local model |
| `loss_and_grad(coils, loss_fn, scales=..., h=..., basis=...)` | Scalar objective and physical parameter gradient |

Each independent call replaces the session's active center. Call
`evaluate_nearby` before the next independent evaluation, using the same session
and thread. `close()` releases the session and can be called repeatedly.
Saved results can be loaded with `EvaluationResult.read_json(path)`.

`evaluate_many` uses the session configuration unless `configs` supplies one
`EvaluationConfig` per input. This supports distinct target volumes and fitting
radii in one batch. Each result records its own configuration; only the final
result remains a live center, and its neighborhood uses that configuration.
The session's default configuration stays available for subsequent scalar calls.

```python
results = evaluator.evaluate_many(
    [coils_a, coils_b],
    configs=[EvaluationConfig(target_volume=0.03, psi_a=0.15),
             EvaluationConfig(target_volume=0.05, psi_a=0.18)],
)
```

GPU value batches require NumPy and the native backend; CPU evaluation also
uses the [CPU dependencies](cpu.md). Inputs sharing field periods, coefficient
shape and axis-search settings are grouped for magnetic-axis preparation, then
restored to input order. Each input retains its ordinary axis selection and
complete score. Singleton groups and older native libraries use the
ordinary sequential path. `timings['shared_independent_batch']` reports preparation,
per-query and total batch time; `python_wall_s` allocates the public batch wall
time across its results, including each query and its share of preparation.

## Results and metrics

`EvaluationResult` contains `status`, `metrics`, `diagnostics`, `timings` and
`mode`. Check `valid` before reading metrics. `write_json(path)` saves values,
configuration and diagnostics. `python_wall_s` measures the public call;
native stage timings can overlap or contain other stages.

| Metric | Meaning |
| --- | --- |
| `qs_volume` | Normalized QS residual RMS over the selected volume |
| `qs_outer` | Normalized QS residual RMS in the outer radial bin |
| `iota_boundary` | Fitted rotational transform at the target boundary |
| `iota_min`, `iota_max` | Minimum/maximum of the fitted profile on 257 uniformly spaced rho-squared samples |
| `relative_normal_drift_p95` | One-period normal displacement P95 divided by mean minor radius |
| `coil_length_mean` | Mean basic-coil length, m |
| `coil_curvature_max`, `coil_curvature_p95` | Sampled maximum and optional percentile curvature, m⁻¹ |
| `coil_gap_min` | Minimum sampled physical-coil separation, m |
| `coil_radius_min` | Minimum sampled cylindrical radius, m |
| `current_abs_max` | Maximum absolute basic-coil current, A |
| `high_mode_fraction` | Optional high Fourier mode energy fraction |
| `psi_fit_rms` | Optional psi training-equation residual RMS, T/m |

QS values use an RMS definition. Aggregate their squares with the appropriate
weights, then take the square root. A volume RMS, an outer radial-bin RMS and the
Simsopt reference's flux-shell RMS describe different regions; see
[Simsopt integration](simsopt.md).
Both native QS fields include division by `max(sqrt(M*M + N*N), 1)` for the
selected helicity. The outer bin uses the native point weights in the final
radial bin; iota profile samples cover the selected radial range.
The standalone `surface_fc`/`analyze_arrays` reference helpers use QH helicity
`(1,nfp)` and have no helicity argument in 0.11.1. The separate
`surface_qs_metrics` helper accepts explicit QA `helicity=(1,0)` for f_B/f_T.
Use matching native helicity when comparing either reference path.

`quality_penalty(error, threshold=0.01, temperature=0.0025, weight=10000)` is a
composable scalar helper. Fit and quality diagnostics are separate from any
user-defined objective. Coil geometry metrics use sampled curves.
`psi_fit_rms` has the same meaning as the Tensor metric: it measures the psi
training-equation residual in T/m. Older saved records can omit it.

## Finite-difference gradients

`loss_fn(LocalMetrics)` returns one finite scalar. Positive physical `scales`
define endpoints `p ± h*scales*direction`. Unit geometry scales, current scales
of `100000` and `h=1e-4` give 0.1 mm geometry and 10 A current perturbations.
The returned gradient follows the physical `parameter_vector()` order.

The default directions form the identity matrix. `basis` accepts a full square
orthonormal matrix. `GradientResult` includes the value, gradient, center,
endpoints, scales, h, basis and timings. The local response updates axis, psi
and surface metrics; coil geometry metrics use center linearization.

For two-stage evaluation, `psi_nearby_method='local'` retains the center's active
domain and sampling choices while updating their continuous response. It uses
the optional CUDA PyTorch path when available. `'formal'` reconstructs the
two-stage fit at each endpoint, following the center's axis branch. Finite-step
differences can change discrete choices and differ from a local derivative.
Configurations outside the Tensor API's supported settings, or environments
without CUDA-enabled PyTorch, use formal reconstruction. Inspect
`diagnostics['nearby_method']`, `['nearby_backend']` and `['geometry_model']`
for the executed path; `['nearby_fallback_reason']` explains a local fallback.
The [Tensor API](torch.md) provides local reverse gradients for PyTorch losses.

An unusable center or endpoint raises `InvalidEvaluationError`; `.results`
contains available results. An expired or cross-session center raises
`CenterContextError`. Native failures raise `EvaluationError`.

Flux-Legendre radial values use a three-term Legendre recurrence in the native, tensor and Simsopt initialization paths. This avoids cancellation from an expanded FP32 monomial polynomial; it does not increase the finite basis or remove alpha/psi fitting error. Regenerate saved evaluation states when changing the native library or basis implementation.

See [target-surface f_B and f_T](qs.md) for optional metrics and accuracy presets.

In 0.9.1, failure diagnostics distinguish a bounded axis search with no accepted
elliptic root from a requested volume above the fitted scalar's usable capacity.
`native_error_message` retains the original native text; `failure_reason` records
the classification. `capacity_scope` identifies the sampled angular rays used
for the radial certificate. The capacity check does not certify every angle
between those rays. Alpha inversion and radial sampling reject folds and select
the axis-connected first branch; a small residual at a later root is insufficient.
Native Bernstein certificates can conservatively reject a non-sign-changing
flat point that the polynomial root-isolation path can resolve. First-branch
agreement applies on their common usable domain.

Scalar-geometry diagnostics include `psi_fit_stage` and `failure_stage`.
`s_training_domain` means the first fit could not supply a closed region for
the second fit. `s_target_surface` means that selection of the requested volume
failed. The corresponding `failure_reason` distinguishes an outward decrease
near the axis (`s_near_axis_nonmonotone`), a radial turn that limits the volume
(`s_radial_fold_capacity`), a search radius limit (`s_radius_cap_capacity`),
and a volume solve that failed within capacity (`target_volume_inverse_failed_within_capacity`).
`s_radial_branch_uncertified` means the sampled check did not resolve a cause.
When available, `target_branch_diagnostic` records the limiting ray and its
capacity. These tests describe the fitted surface label on the sampled rays.

The legacy `flux_rejected` status is also used by native target-volume forward
validation. For that condition, `failure_reason` is
`target_volume_forward_precision_exceeded` and `failure_stage` is
`target_volume_forward_validation`; the numerical status remains unchanged.

## CPU backend

Select `Evaluator(config=..., device='cpu', threads=None)` for CPU execution.
The default device remains `'cuda'`. A positive `threads` selects the CPU
budget; `None` follows process affinity and the scheduler allocation.
The [CPU guide](cpu.md) covers installation, batch execution, finite differences
and `TensorEvaluator(..., device='cpu')` reverse gradients.

Here V is the requested volume and Vcapacity is the second fit’s representable capacity; `psi_n_phi` is the initial toroidal section count and `psi_m_tor` the scalar fit’s maximum toroidal Fourier mode.

A Bernstein certificate expresses the radial polynomial in a Bernstein basis and uses coefficient signs to test positivity of its derivative; root isolation instead brackets polynomial roots. An elliptic axis candidate is a periodic orbit around which nearby field lines rotate.
