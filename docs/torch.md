# PyTorch interface

[简体中文](torch.zh-CN.md)

Install a CUDA-enabled PyTorch build compatible with your GPU and driver, then
run `python examples/autograd.py`. Installation options are described in
[the installation guide](installation.md).

## Physical parameter tensors

`pack_coils(coils, device='cuda:0', dtype=torch.float64)` converts a `CoilSet`
into a tensor of shape `(4, 3*(2*order+1)+1)`. Each row contains that coil's x,
y and z Fourier coefficients followed by its current. Order 4 gives `(4,28)`;
order 16 gives `(4,100)`. Coordinates use metres and currents use amperes.
The NumPy `parameter_vector()` retains its coordinate-major flat layout.

The Tensor API supports four basic coils, `nfp=4`, helicity `(1,4)`, 256 coil
segments and four psi/alpha local fit iterations. Select `fast=True` and
`flux_method='stokes'`. Both `psi_fit='single_stage'` and `'two_stage'` are
accepted. Psi calculations use float64; field calculations use float32 by
default, with `field_dtype=torch.float64` available.

## Evaluation and batches

Use `TensorEvaluator(library=None, field_library=None, config=...,
device_id=0, field_dtype=torch.float32, batch_axis=True, share_geometry=True)` as a context manager on its creating
thread. A wheel supplies both native libraries. Explicit paths select custom
builds; the field library otherwise comes from the same directory as `library`.

`batch_axis=True` batches ordinary magnetic-axis preparation for distinct
geometries. `share_geometry=True` shares captured two-stage fits for identical
physical parameters whose configurations differ only in target volume. Each
query retains its own downstream calculation and parameter gradient. Set either
option to `False` to select its independent preparation path. Libraries without
the batch-axis API use independent axis preparation.

GPU triangular solves run independently for each query, while field and feature
calculations remain batched. Temporary workspaces are bounded; the retained
gradient graph still grows with batch size. Select a batch size that fits your
GPU memory.

| Method | Physical input shape | Each metric's shape |
| --- | --- | --- |
| `evaluate(parameters)` | `(4, width)` | scalar |
| `evaluate_batch(parameters, configs=None)` | `(B,4,width)` | `(B,)` |

`configs` supplies one `EvaluationConfig` per independent geometry. Use a common
Fourier width, psi-fit method, `alpha_radial_basis`, `alpha_radial_weighting`, `alpha_weight_sigma`
and `alpha_count_floor` within a batch. Target volume and `psi_a` may vary by
query; two-stage sampling methods and row counts may also vary. Centers are captured in input
order; field, tracing and fit operations then use the batch operators. An
unusable captured center raises `InvalidEvaluationError` identifying the
affected geometry.

`diagnostics['capture']` records the actual geometry groups, axis preparation
and query count, including independent preparation when a cache is unavailable.

```python
import torch
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.torch import TensorEvaluator, pack_coils

x = pack_coils(CoilSet.read_json("examples/coils.json"))
config = EvaluationConfig(
    target_volume=0.05, fast=True, flux_method="stokes",
    psi_fit="two_stage", psi_refit_sampling="cartesian",
)
with TensorEvaluator(config=config) as evaluator:
    result = evaluator.evaluate(x)
    loss = torch.log(result.metrics.qs_volume + 1e-8)
    gradient, = torch.autograd.grad(loss, result.parameters)
```

For independent geometries, stack parameter tensors and call
`evaluate_batch`. Differentiating `loss.sum()` returns each geometry's gradient;
`loss.mean()` includes the averaging factor. The gradient has the input's layout
and physical units. Normalization and custom losses can use ordinary PyTorch
operations.
Inputs may use float32 or float64 on the evaluator's GPU. Match `pack_coils`'s
device with `TensorEvaluator(device_id=...)`. If an input does not request
gradients, the result contains a new leaf in `.parameters`; differentiate with
respect to that returned tensor, as in the example.

## Results and local derivatives

`TensorEvaluation` contains `parameters`, `metrics`, `diagnostics` and `timings`.
`TensorMetrics` exposes `qs_volume`, `qs_outer`, `iota_boundary`, `iota_min`,
`iota_max`, `relative_normal_drift_p95`, `coil_gap_min`, `coil_curvature_max`,
`coil_length_mean`, `surface_volume` and `psi_fit_rms`. QS quantities are RMS;
the whole-volume and outer-bin quantities have different weighting regions.
`psi_fit_rms` is the psi training-equation residual RMS in T/m, matching the
scalar `LocalMetrics` field.
`iota_min` and `iota_max` use the same 257 uniformly spaced rho-squared profile
samples as the scalar evaluator.

`diagnostics['flux_integration']` reports the actual flux quadrature: one
dictionary for scalar evaluation, or one dictionary per query in a batch.
`method` is `'stokes'` or `'area'`, `fallback` identifies a switch to area
integration, and `validation` is `'checked'` or `'disabled'`.
`edge_relative_error` compares boundary flux; `derivative_relative_error`
compares the flux derivative at 33 normalized positions, scaled by the largest
reference derivative. The reference is the captured formal center's area
polynomial. The default tolerance is `1e-3`; configure it with
`EvaluationConfig(stokes_validation_rtol=...)`, or use `None` to disable the
check. A batch recomputes area integration only for queries that fail the
check. The selected integration retains derivatives through the moving
geometry, field and surface level.
Two-stage local finite-difference neighborhoods choose this quadrature once at
their captured center and retain it for all endpoints. Their
`flux_integration.validation` is `'center_selected'`, with the original check
under `center_validation`. Physical changes in flux across endpoints therefore
do not trigger an integration-method change.

`timings` records forward stages, including center capture and geometry-metric
preparation. Objective construction and `torch.autograd.grad` run after
`evaluate` returns. Measure the whole call plus one reverse pass, with CUDA
synchronization at the endpoints, for value-and-gradient latency.

The reverse path differentiates the continuous response at a captured center,
including magnetic-axis movement, psi fitting, surface selection and integration
points. The two-stage response also includes movement of its training domain,
sampling coordinates and second fit. Optional polar quadrature includes the
continuous response of its row weights. The active axis branch, limiting ray,
lattice membership, histogram bins and percentile selections define the local
derivative. Finite perturbations that change those choices can give different
finite-difference directions.

Captured centers retain the formal evaluation's actual volume-point indices,
including valid-point compaction. The local response moves those selected
points continuously. The active nearest coil contact and maximum-curvature
sample similarly determine their engineering derivatives.

A small coil perturbation can change which rays are valid. The formal evaluator
then resamples the compacted volume points, introducing a discrete sampling
boundary even when the magnetic-axis branch stays fixed. Near such a boundary,
compare several finite-difference step sizes and inspect individual metrics.
The local derivative follows the center's selected points; a local endpoint that
loses a required surface root is reported as an invalid evaluation.

The interface supports first derivatives. Compare meaningful component
gradients and geometry/current parameter blocks when checking a new objective;
an aggregate loss can conceal a component discrepancy. The NumPy `Evaluator`
provides centered finite differences for scalar callbacks; see [the API](api.md).

See [target-surface f_B and f_T](qs.md) for optional metrics and accuracy presets.
