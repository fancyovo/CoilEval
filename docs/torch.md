# PyTorch interface

[简体中文](torch.zh-CN.md)

Available in **0.3.0a1**. Install the optional dependency and build both native libraries:

```sh
python -m pip install '.[torch]'
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/native --parallel
python examples/autograd.py --library build/native/libcoil_eval_native.so
python examples/autograd.py --library build/native/libcoil_eval_native.so --batch
```

## Input and configuration

`pack_coils(coils, device='cuda:0', dtype=torch.float64)` converts a `CoilSet`
into a `(4,28)` tensor. Each row contains one coil's nine x coefficients, nine
y coefficients, nine z coefficients and current. Coefficients use metres and
currents use amperes. This row layout is convenient for tensors; the NumPy
`CoilSet.parameter_vector()` API retains its coordinate-major flat layout.

The prerelease supports four order-4 basic coils, `nfp=4`, helicity `(1,4)`,
256 segments per coil and four psi/alpha fit iterations. Use
`EvaluationConfig(target_volume=..., psi_a=..., fast=True, flux_method='stokes')`.
The psi calculation uses float64; field calculations use float32 by default.

## Evaluator

`TensorEvaluator(library, field_library=None, config=..., device_id=0,
field_dtype=torch.float32)` is a context manager. By default the companion
`libcoil_eval_autograd.so` is loaded from the scoring library's directory.
`field_dtype` also accepts `torch.float64`.

| Method | Input | Metric shape |
| --- | --- | --- |
| `evaluate(parameters)` | `(4,28)` CUDA tensor | scalar |
| `evaluate_batch(parameters, configs=None)` | `(B,4,28)` CUDA tensor | `(B,)` |

`configs` optionally supplies one `EvaluationConfig` per geometry. A batch
uses independent centers, then batches the field, tracing and fit operators.
Center captures are performed in input order. Use the evaluator on its creating
thread. An invalid geometry raises `InvalidEvaluationError` with its batch index.

## Results and gradients

`TensorEvaluation` exposes `parameters`, `metrics`, `diagnostics` and `timings`.
`TensorMetrics` contains `qs_volume`, `iota_boundary`,
`relative_normal_drift_p95`, `coil_gap_min`, `coil_curvature_max`,
`coil_length_mean`, `surface_volume` and `psi_fit_rms`.

```python
import torch
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.torch import TensorEvaluator, pack_coils

x = pack_coils(CoilSet.read_json('examples/coils.json'))
config = EvaluationConfig(target_volume=0.05, fast=True, flux_method='stokes')
with TensorEvaluator('build/native/libcoil_eval_native.so', config=config) as ev:
    result = ev.evaluate(x)
    loss = torch.log(result.metrics.qs_volume + 1e-8)
    gradient, = torch.autograd.grad(loss, result.parameters)
```

For a batch, differentiate `loss.sum()` for one gradient per geometry, or
`loss.mean()` to include the batch averaging factor. Gradients use the same
physical units and tensor layout as the input. Callers can compose differentiable
normalization or objectives with ordinary PyTorch operations.

The reverse path differentiates the local response around a captured center,
including movement of the magnetic axis, fitted surface and integration points.
Center solver factors and active discrete choices define that local model.
Histogram-bin and percentile changes can produce different finite-step differences.
First derivatives are supported; the 1.3 prerelease targets first-order optimization.

The NumPy `Evaluator` continues to accept ordinary scalar callbacks and uses
centered finite differences. Select the Tensor API for PyTorch objectives.
