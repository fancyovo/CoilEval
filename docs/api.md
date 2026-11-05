# Python API

[简体中文](api.zh-CN.md)

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
| `fast` | `False` | Use 240 axis steps and 100 surface tracing steps |
| `flux_method` | `'area'` | `'area'` or fast local `'stokes'` boundary integration |
| `psi_fit` | `'single_stage'` | `'single_stage'` or `'two_stage'` |
| `psi_refit_points` | `82368` | Target training-row count for the second fit |
| `psi_refit_ridge` | `None` | Optional nonnegative column-scaled ridge for S2 only; `None` retains the native setting |
| `psi_target_refit` | `False` | Candidate S3 target-interior refit after `psi_fit='two_stage'` |
| `psi_target_refit_factor` | `1.2` | Requested S3 training volume divided by target volume; must exceed one |
| `psi_target_refit_ridge` | `1e-10` | Nonnegative column-scaled ridge for S3 only |
| `psi_refit_sampling` | `'cartesian'` | Two-stage `'cartesian'` or `'polar_quadrature'` sampling |
| `psi_nearby_method` | `'local'` | Two-stage local response or `'formal'` endpoint reconstruction |
| `psi_fd_batch_size` | `8` | Two-stage local finite-difference endpoint batch size |
| `alpha_radial_weighting` | `'histogram'` | `'histogram'` or two-stage experimental `'smooth'` radial density weighting |
| `alpha_weight_sigma` | `0.05` | Gaussian width in normalized rho for smooth weighting |
| `alpha_count_floor` | `32.0` | Positive additive pseudocount for smooth density denominators |
| `alpha_radial_basis` | `'zernike'` | Original radial basis or two-stage `'flux_legendre'` |
| `stokes_validation_rtol` | `1e-3` | Tensor flux consistency tolerance; `None` disables the check |

The single-stage settings remain the default in 0.10.0. Select the
two-stage path explicitly:

```python
config = EvaluationConfig(
    target_volume=0.05, psi_a=0.15,
    fast=True, flux_method="stokes",
    psi_fit="two_stage", psi_refit_sampling="cartesian",
)
```

Two-stage evaluation first fits psi in the initial domain, constructs an
axis-connected closed training domain, and fits psi again in that domain.
The target-volume surface and QS metrics use the second fit.
`psi_refit_ridge` can reduce regularization bias when resolving very small QS
errors. It applies to S2 only, including local gradients; S1 and its domain stay
fixed. The GPU option requires a native library exposing the 0.9.1 setter.
Smaller ridge does not guarantee a more accurate magnetic surface: compare
normal-field error, coordinate convergence and geometric capacity. Target-local
third-fit behavior is selected separately with `psi_target_refit`.

With `psi_target_refit=True`, S1 and S2 retain their configured objectives.
S2 then supplies a closed first-root boundary of volume
`min(psi_target_refit_factor*V, (V+Vcapacity)/2)` for a third fit. The capacity
must exceed the requested physical volume V. S3 uses weighted polar quadrature
over the entire interior and the same basis and backend precision. Its nominal
row count is `psi_refit_points`; `psi_target_refit_ridge` controls only S3.
The final surface uses S3 at the original V. Unavailable capacity or a failed
continuation remains an explicit failure. Inspect `psi_fit_stages` and
`psi_target_refit_domain` in the native diagnostics. S3 requires explicit
selection: shrinking the training domain together with weak regularization can
produce a nonmonotone scalar and lose usable capacity even on a previously
scorable input. Its fixed4096 validation does not support default promotion.
Changing the domain and changing ridge have separate effects and can interact;
restoring ridge alone does not establish accuracy for every target.

Multi-volume captures share S1/S2; each target owns its S3 coefficients, QR
factors and downstream state. Local derivatives include S2's moving training
boundary, the S3 volume equation, capacity clipping and polar area weights on
the selected discrete branch. Surface-alpha and native volume-alpha settings
retain their independent meanings.
The target-volume capacity guard checks at least the psi fitting toroidal grid
and two samples per retained toroidal mode. Flux quadrature remains separately
configured. This is a sampled radial-branch check; an accepted result does not
certify a global magnetic surface or physical accuracy.
`psi_refit_sampling='polar_quadrature'` selects optional weighted polar
quadrature, an experimental numerical option in 0.9.0.
Both samplers retain complete sampling structures, so the actual row count can
differ from `psi_refit_points`; inspect `diagnostics['psi_refit_training_count']`.

`alpha_radial_weighting='smooth'` is an opt-in experimental option for
`psi_fit='two_stage'`. Its positive sigma and pseudocount settings control the
smooth density weights. Histogram weighting remains the default.

`alpha_radial_basis='flux_legendre'` fits the angular correction with
polynomials of normalized flux. Select it with `psi_fit='two_stage'`; the
choice applies to scalar evaluation, local gradients and captured alpha data.
The original `'zernike'` basis remains the default.

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
complete formal score. Singleton groups and older native libraries use the
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
weights, then take the square root. A volume RMS, an outer-bin RMS and the
Simsopt reference's flux-shell RMS describe different regions; see
[Simsopt integration](simsopt.md).
Both native QS fields include division by `max(sqrt(M*M + N*N), 1)` for the
selected helicity. The outer bin uses the native point weights in the final
radial bin; iota profile samples cover the selected radial range.
The standalone `surface_fc`/`analyze_arrays` reference helpers use QH helicity
`(1,nfp)` and have no helicity argument in 0.9.0. The separate
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

Flux-Legendre radial values use a three-term Legendre recurrence in the native, tensor and Simsopt initialization paths. This avoids cancellation from an expanded FP32 monomial polynomial; it does not increase the finite basis or remove alpha/psi fitting error. Regenerate source captures when changing the native library or basis implementation.

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
