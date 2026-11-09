# PyTorch interface

[简体中文](torch.zh-CN.md)


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

To use the third GPU visible to the process, pass `device='cuda:2'` to
`pack_coils` and `device_id=2` to `TensorEvaluator`. If you launch with
`CUDA_VISIBLE_DEVICES=2`, that GPU appears inside the process as `cuda:0`.
Example2 uses its editable `DEVICE_ID` setting for both the evaluator and its tensors.

The Tensor API supports any positive integer `nfp`, any positive basic-coil
count and explicit helicity `(M,N)` with nonzero `M`. Stellarator symmetry is
required. Set `TensorEvaluator(nfp=coils.nfp, ...)`: tensor values do not encode
field periods, and `pack_coils` does not attach that metadata. The defaults
remain `nfp=4` and helicity `(1,4)`. Set `EvaluationConfig(helicity=(1,0), ...)`
for QA, or `(1,nfp)` for QH at the selected field period; helicity is never
inferred from `nfp`. The interface retains 256 coil segments and four
psi/alpha local fit iterations. Select `fast=True` and
`flux_method='stokes'`. The second fit is the 0.11.1 default. Explicit legacy single-fit evaluation requires histogram weights and
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
With `psi_target_refit=True`, sharing stops after the second fit. Each target has its own the third fit
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
these values because the independent cuSOLVER (NVIDIA’s matrix-solver library) handles used by two-stage capture
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
the whole-volume and outer radial-bin quantities have different weighting regions.
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
reference derivative. The reference is the captured complete-evaluation center's area
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

A CUDA stream is an ordered device execution queue; an event is a timing marker in that queue. An active branch is the currently selected nearest point, maximum point or limiting ray, held fixed for the local derivative.
