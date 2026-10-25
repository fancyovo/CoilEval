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
device_id=0, field_dtype=torch.float32)` as a context manager on its creating
thread. A wheel supplies both native libraries. Explicit paths select custom
builds; the field library otherwise comes from the same directory as `library`.

| Method | Physical input shape | Each metric's shape |
| --- | --- | --- |
| `evaluate(parameters)` | `(4, width)` | scalar |
| `evaluate_batch(parameters, configs=None)` | `(B,4,width)` | `(B,)` |

`configs` supplies one `EvaluationConfig` per independent geometry. Use a common
Fourier width and psi-fit method within a batch. Centers are captured in input
order; field, tracing and fit operations then use the batch operators. A batch
failure raises `InvalidEvaluationError` identifying the affected geometry.

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

## Results and local derivatives

`TensorEvaluation` contains `parameters`, `metrics`, `diagnostics` and `timings`.
`TensorMetrics` exposes `qs_volume`, `qs_outer`, `iota_boundary`, `iota_min`,
`iota_max`, `relative_normal_drift_p95`, `coil_gap_min`, `coil_curvature_max`,
`coil_length_mean`, `surface_volume` and `psi_fit_rms`. QS quantities are RMS;
the whole-volume and outer-bin quantities have different weighting regions.
`psi_fit_rms` is the psi training-equation residual RMS in T/m, matching the
scalar `LocalMetrics` field.

The reverse path differentiates the continuous response at a captured center,
including magnetic-axis movement, psi fitting, surface selection and integration
points. The two-stage response also includes movement of its training domain,
sampling coordinates and second fit. Optional polar quadrature includes the
continuous response of its row weights. The active axis branch, limiting ray,
lattice membership, histogram bins and percentile selections define the local
derivative. Finite perturbations that change those choices can give different
finite-difference directions.

The interface supports first derivatives. Compare meaningful component
gradients and geometry/current parameter blocks when checking a new objective;
an aggregate loss can conceal a component discrepancy. The NumPy `Evaluator`
provides centered finite differences for scalar callbacks; see [the API](api.md).
