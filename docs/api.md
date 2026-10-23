# Python API

[简体中文](api.zh-CN.md)

## CoilSet

`CoilSet(coeffs_x, coeffs_y, coeffs_z, currents_a, nfp)` describes basic coils
with stellarator symmetry. Coordinate arrays have shape `(coils, 2*order+1)`
and use `[constant, sin(1), cos(1), ..., sin(order), cos(order)]`. Lengths are
in metres and currents in amperes. Symmetry expands each basic coil into
`2*nfp` physical coils.

`parameter_vector()` flattens all x coefficients, then y, then z, then currents.
`with_parameters(vector)` creates a new coil set in that order.
Use `read_json(path)` and `write_json(path)` to load and save the coil schema.

## EvaluationConfig

`EvaluationConfig(target_volume, psi_a=0.15, helicity=(1,4), h=1e-4,
psi_iterations=4, alpha_iterations=4, segments_per_coil=256)` is immutable.
`target_volume` is in m³, `psi_a` in m, and helicity is the integer pair `(M,N)`.
`to_dict()` returns the settings for serialization.

## Evaluator

Create `Evaluator(library, config=config, device_id=0)` in the thread that will
use it. The context manager releases its native resources.

| Method | Result |
| --- | --- |
| `evaluate(coils)` | One independent `EvaluationResult` |
| `evaluate_many(coil_sets)` | Results in input order, evaluated sequentially |
| `evaluate_nearby(center, candidates)` | Local results using one live center |
| `loss_and_grad(coils, loss_fn, scales=..., h=..., basis=...)` | Scalar value and full physical gradient |

Each independent evaluation replaces the session's active center. Use a center
with the same session and thread, before the next independent evaluation.
Nearby results and JSON-loaded results contain values for inspection.

## Results and metrics

`EvaluationResult` contains `status`, `metrics`, `diagnostics`, `timings` and
`mode`. Check `valid` before using metrics. Save and load results with
`write_json` and `read_json`. Timing includes the public call's `python_wall_s`;
native stages can overlap or contain other stages. A local batch shares its
`shared_local_batch` timing across endpoints.

| Metric | Meaning |
| --- | --- |
| `qs_volume`, `qs_outer` | Normalized squared QS residual, volume-weighted and outer radial bin |
| `iota_boundary`, `iota_min`, `iota_max` | Boundary value and fitted profile extrema |
| `relative_normal_drift_p95` | One-period normal displacement P95 divided by mean minor radius |
| `coil_length_mean` | Mean basic-coil length, m |
| `coil_curvature_max`, `coil_curvature_p95` | Discrete curvature maximum and optional percentile, m⁻¹ |
| `coil_gap_min` | Minimum sampled physical-coil separation, m |
| `coil_radius_min` | Minimum sampled cylindrical radius, m |
| `current_abs_max` | Maximum absolute basic-coil current, A |
| `high_mode_fraction` | Optional high Fourier mode energy fraction |

`quality_penalty(error, threshold=0.01, temperature=0.0025, weight=10000)`
is a composable scalar helper. Diagnostics contain additional fit and quality
information. Coil geometry metrics use sampled curves; curvature P95 uses
uniform parameter samples.

## Gradients

`loss_fn` takes `LocalMetrics` and returns one finite scalar. Supply positive
physical `scales` for each parameter; endpoints are `p ± h*scales*direction`.
For example, unit geometry scales and current scales of `100000` with `h=1e-4`
give 0.1 mm geometry and 10 A current perturbations. The returned gradient is
with respect to physical parameters, in `parameter_vector()` order.

The default directions form the identity matrix. `basis` accepts a full square
orthonormal matrix. `GradientResult` includes the value, gradient, center,
endpoints, scales, h, basis and timings. The local model updates axis, psi and
surface metrics; coil geometry metrics use center linearization.

An unusable center or endpoint raises `InvalidEvaluationError`, with available
results in `.results`. Expired or cross-session centers raise `CenterContextError`.
Native failures raise `EvaluationError`. Check perturbation scales for your
geometry when interpreting finite-difference gradients.
