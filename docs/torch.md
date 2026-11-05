# PyTorch interface

[简体中文](torch.zh-CN.md)

For GPU use, install a CUDA-enabled PyTorch build compatible with your GPU and
driver, then run `python examples/autograd.py`. CPU execution uses CPU PyTorch
and `TensorEvaluator(device='cpu', ...)`; see the [CPU guide](cpu.md).
Installation options are described in [the installation guide](installation.md).

## Physical parameter tensors

`pack_coils(coils, device='cuda:0', dtype=torch.float64)` converts a `CoilSet`
into a tensor of shape `(nc, 3*(2*order+1)+1)`, where `nc` is the positive number
of basic coils. Each row contains that coil's x, y and z Fourier coefficients
followed by its current. Order 4 gives `(nc,28)`; order 16 gives `(nc,100)`.
Coordinates use metres and currents use amperes.
The NumPy `parameter_vector()` retains its coordinate-major flat layout.

The Tensor API supports any positive integer `nfp`, any positive basic-coil
count and explicit helicity `(M,N)` with nonzero `M`. Stellarator symmetry is
required. Set `TensorEvaluator(nfp=coils.nfp, ...)`: tensor values do not encode
field periods, and `pack_coils` does not attach that metadata. The defaults
remain `nfp=4` and helicity `(1,4)`. Set `EvaluationConfig(helicity=(1,0), ...)`
for QA, or `(1,nfp)` for QH at the selected field period; helicity is never
inferred from `nfp`. The interface retains 256 coil segments and four
psi/alpha local fit iterations. Select `fast=True` and
`flux_method='stokes'`. S2 is the 0.11.0 default. Explicit legacy S1 requires histogram weights and
the Zernike basis as described in the API guide. Both fitting paths are
accepted. Psi calculations use float64; field calculations use float32 by
default, with `field_dtype=torch.float64` available.

## Evaluation and batches

Use `TensorEvaluator(library=None, field_library=None, config=...,
device_id=0, field_dtype=torch.float32, batch_axis=True, share_geometry=True,
device='cuda', threads=None, axis_grid=None, surface_qs=None,
capture_pipeline=True, batch_surface_qs=True, synchronize_stages=False,
engineering_overlap=True, capture_workers=1, nfp=4)` as a context manager on its creating
thread. A wheel supplies the GPU evaluation and gradient libraries and the CPU
library. Explicit paths select custom builds. On GPU, the field library otherwise
comes from the same directory as `library`; on CPU, both operations use the CPU library.

On GPU, `batch_axis=True` batches ordinary magnetic-axis preparation for distinct
geometries. `share_geometry=True` shares captured two-stage fits for identical
physical parameters whose configurations differ only in target volume. Each
query retains its own downstream calculation and parameter gradient. Set either
option to `False` to select its independent preparation path. Libraries without
the batch-axis API use independent axis preparation.
With `psi_target_refit=True`, sharing stops after S2. Each target has its own S3
training domain, coefficients and QR factors. Its local gradient includes the
moving training boundary and weights on the captured capacity-clipping branch.
CPU batches prepare queries independently; `threads` controls each query's CPU
budget and `axis_grid` optionally selects its primary magnetic-axis seed grid.
CPU query diagnostics and timings are available under `diagnostics['queries']`
and `timings['queries']`.

The following GPU execution controls are available. These
options change scheduling; numerical settings remain in `EvaluationConfig`
and `SurfaceQSConfig`.

| GPU option | Default | Execution behavior |
| --- | --- | --- |
| `capture_pipeline` | `True` | With `capture_workers=1`, overlap CPU closed-domain construction for distinct two-stage geometry groups with the next native first fit or a preceding second fit and formal continuation. |
| `capture_workers` | `1` | Use the validated value 1; native requests for 2–4 concurrent capture workers are currently rejected. |
| `batch_surface_qs` | `True` | Combine corresponding magnetic-field sweeps across queries during optional target-surface f_B/f_T sampling. |
| `synchronize_stages` | `False` | Record stage CUDA events and resolve them at the public-call boundary. `True` enables a device synchronization before and after each stage for diagnostic wall timing. |
| `engineering_overlap` | `True` | Prepare coil metrics and their active derivatives in one CPU worker while GPU response stages execute, then attach the returned derivatives on the caller. |

With `capture_workers=1`, the capture pipeline retains at most two native
contexts and keeps native capture calls on the creating thread. Target-surface geometry roots and angular
solves retain each query's independent numerical branch. These controls apply
within one evaluation call; schedule separate application batches explicitly.
Their effect on latency and throughput depends on batch size, host CPU resources
and GPU workload. Set `capture_pipeline`, `batch_surface_qs` or `engineering_overlap` to `False`
to compare its independent execution path; set `synchronize_stages=True` to
compare synchronized stage timing.
With `capture_workers=2`, `3` or `4`, each host worker would own a complete
geometry group and its native context. The public native path currently rejects
these values because the independent cuSOLVER handles used by two-stage capture
still need a dedicated isolation implementation. `capture_workers=1` is the
validated path; use it with `capture_pipeline=True` for CPU-domain/native
overlap.
The execution controls in the table are GPU-only; CPU evaluation retains its
existing thread budget and independent-query execution.

GPU triangular solves run independently for each query, while field and feature
calculations remain batched. Temporary workspaces are bounded; the retained
gradient graph still grows with batch size. Select a batch size that fits your
GPU memory.

| Method | Physical input shape | Each metric's shape |
| --- | --- | --- |
| `evaluate(parameters)` | `(nc, width)` | scalar |
| `evaluate_batch(parameters, configs=None)` | `(B,nc,width)` | `(B,)` |

`configs` supplies one `EvaluationConfig` per independent geometry. Use a common
`nfp`, basic-coil count, Fourier width, psi-fit method, `alpha_radial_basis`, `alpha_radial_weighting`, `alpha_weight_sigma`
and `alpha_count_floor` within a batch. Helicity, target volume and `psi_a` may vary by
query; two-stage sampling methods and row counts may also vary. Captured centers are returned in input
order; GPU field, tracing and fit operations then use the batch operators. An
unusable captured center raises `InvalidEvaluationError` identifying the
affected geometry. Since 0.8.1, pipeline exceptions also name the failing
stage. A shared preparation failure lists the affected queries; a continuation
failure identifies its individual target-volume query. The original exception
is retained in the exception chain.

`diagnostics['capture']` records the actual geometry groups, axis preparation
and query count, including independent preparation when a cache is unavailable.
Its `pipeline` entry, when present, records the selected execution mode,
effective workers and group spans. The one-worker pipeline also records
domain-worker waits. Use `pipeline['total_s']` for capture batch wall time;
group spans can overlap.
`diagnostics['engineering_overlap']` records whether the host worker was used.

```python
import torch
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.torch import TensorEvaluator, pack_coils

coils = CoilSet.read_json("examples/coils.json")
x = pack_coils(coils)
config = EvaluationConfig(
    target_volume=0.05, fast=True, flux_method="stokes",
    psi_fit="two_stage", psi_refit_sampling="polar_quadrature",
)
with TensorEvaluator(config=config, nfp=coils.nfp) as evaluator:
    result = evaluator.evaluate(x)
    loss = torch.log(result.metrics.qs_volume + 1e-8)
    gradient, = torch.autograd.grad(loss, result.parameters)
```

For independent geometries, stack parameter tensors and call
`evaluate_batch`. Differentiating `loss.sum()` returns each geometry's gradient;
`loss.mean()` includes the averaging factor. The gradient has the input's layout
and physical units. Normalization and custom losses can use ordinary PyTorch
operations.
Inputs may use float32 or float64 on the evaluator's device. Match `pack_coils`'s
device with `TensorEvaluator(device_id=...)` for GPU or use `device='cpu'`
in both calls. If an input does not request
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
preparation. With `synchronize_stages=False`, values are CUDA-event elapsed
spans on the caller's stream, resolved before return. Such spans can include
host launch gaps and waiting for work; they do not measure GPU active time.
The joining stage can include a remaining wait for its concurrent worker and
the result upload; work completed before that stage is already overlapped.
With `synchronize_stages=True`, values are synchronized host wall spans.
`diagnostics['stage_timing']` distinguishes `'deferred_cuda_events'` from
`'synchronized_wall'`. Overlapping stage and worker spans should not be added
to estimate complete latency.

Objective construction and `torch.autograd.grad` run after `evaluate` returns.
Measure the entire fresh call plus one reverse pass for value-and-gradient
latency, with CUDA synchronization at both endpoints. For example, inside the
GPU evaluator's context:

```python
import time

torch.cuda.synchronize(evaluator.device)
started = time.perf_counter()
result = evaluator.evaluate(x)
loss = torch.log(result.metrics.qs_volume + 1e-8)
gradient, = torch.autograd.grad(loss, result.parameters)
torch.cuda.synchronize(evaluator.device)
value_and_gradient_s = time.perf_counter() - started
```

Measure imports, compilation and warmup separately. CPU operations complete
synchronously and retain host wall timings.

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
