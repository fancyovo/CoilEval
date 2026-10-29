# CPU evaluation

[简体中文](cpu.zh-CN.md)

The CPU backend evaluates the same coil discretization, magnetic-axis search,
fitting bases and quadrature sizes as the GPU backend. It supports both psi
fitting methods, coil geometry metrics, finite differences and PyTorch local
reverse gradients. Optional target-surface `f_B` and `f_T` use the shared
Boozer-coordinate implementation.

## Installation

Install CPU PyTorch first, then the CPU dependencies and a CoilEval wheel:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "coil_eval-0.9.0-py3-none-manylinux_2_28_x86_64.whl[cpu]"
```

Replace the wheel filename with the supplied file. CPU evaluation loads
`libcoil_eval_cpu.so` and requires no NVIDIA driver or CUDA toolkit. See the
[installation guide](installation.md) for building a wheel from source.

## Values

```python
from coil_eval import CoilSet, EvaluationConfig, Evaluator

coils = CoilSet.read_json("examples/coils.json")
config = EvaluationConfig(target_volume=0.05, fast=True)
with Evaluator(config=config, device="cpu", threads=8) as evaluator:
    result = evaluator.evaluate(coils)
    print(result.status, result.metrics, result.timings)
```

`device` defaults to `'cuda'`. `threads` accepts a positive integer on CPU;
omitting it uses the available process affinity, processor count and
`SLURM_CPUS_PER_TASK` allocation. `library=` can select a custom CPU shared
library. The `COIL_EVAL_CPU_LIBRARY` environment variable supplies a default
custom path.

`axis_grid` accepts an integer of at least 4; omitting it retains the original
search settings. `axis_grid=24` or `axis_grid=32` reduces the initial magnetic-axis
seed grid from its default 48 points per direction. The existing 64-by-64
fallback policy and the original Newton and topology tests remain unchanged. Smaller
grids can select a different magnetic-axis branch, so check your sample set
before adopting them. Diagnostics include the effective grids, seed counts,
fallback use and axis-stage times. The option is available on all three CPU
evaluator classes.

CPU axis search also checks periodicity in double precision. If every candidate
fails acceptance and an elliptic candidate is already close to the required
tolerance, it performs up to two bounded double-precision Newton corrections
and repeats the original periodicity and topology checks. Accepted roots keep
their original selection. `axis_fp64_polish` diagnostics record this correction,
its displacement and any rejection reason.

`CoilSet.nfp` accepts any positive integer, and a coil set may contain any
positive number of basic coils with stellarator symmetry. `EvaluationConfig`
keeps helicity explicit: use `(1,0)` for QA or `(1,coils.nfp)` for QH. The
default remains `(1,4)`, even when `nfp` changes.

`evaluate_many()` accepts coil sets with different `nfp`, coil counts and
Fourier widths and preserves input order. CPU batches process samples in order,
with all selected threads available to each sample's field evaluations and
linear algebra. For multiple independent worker processes, divide the CPU
allocation among their `threads` settings.

CPU `loss_and_grad()` uses complete formal centered differences and follows
the center's magnetic-axis branch. Its endpoint diagnostics report
`nearby_method='formal'` and `nearby_backend='cpu'`. PyTorch reverse gradients
are preferable when a loss can be expressed using tensor metrics.

## Reverse gradients

```python
import torch
from coil_eval.torch import TensorEvaluator, pack_coils

config = EvaluationConfig(target_volume=0.05, fast=True, flux_method="stokes")
x = pack_coils(coils, device="cpu")
with TensorEvaluator(config=config, device="cpu", threads=8, nfp=coils.nfp) as evaluator:
    result = evaluator.evaluate(x)
    loss = torch.log(result.metrics.qs_volume + 1e-8)
    gradient, = torch.autograd.grad(loss, result.parameters)
```

The tensor interface accepts any positive `nfp` and basic-coil count. Set
`nfp=coils.nfp` explicitly: the tensor contains no field-period metadata, and
the evaluator defaults to 4. It retains stellarator symmetry, 256 segments
per coil and four psi/alpha local fit iterations. `evaluate_batch()` requires
a common `nfp`, coil count and Fourier width and returns stacked metrics and
independent gradients. Its configurations may specify a different helicity
for each query, including QA `(1,0)`; `M` must be nonzero.
`surface_qs=SurfaceQSConfig(...)` enables the optional
surface metrics. GPU preparation options `batch_axis` and `share_geometry`
apply to the GPU backend; CPU batches prepare each query independently.

For values alone, use `SurfaceQSEvaluator(config=config, device='cpu',
threads=8)`. Its `evaluate(coils)` and `evaluate_many(coils)` methods return
ordinary floating-point `SurfaceQSResult` values from one formal capture per
query. Its value batches accept mixed field periods and coil layouts and retain
input order.

Perform the forward calculation and backward pass inside the evaluator's
context manager so they share the requested thread budget. PyTorch and BLAS
thread limits are process-wide and are restored on exit. Use separate
processes for concurrent sessions with different thread budgets.

The CPU backend uses Householder QR for formal fits and the same local
four-step response for reverse gradients. CPU and GPU arithmetic can differ
in reduction order; compare metric errors and gradient direction and norm
rather than expecting bitwise equality.

An OpenMP-enabled CPU build parallelizes independent field points and
magnetic-field traces. AVX2 is selected automatically on supported systems,
with a scalar implementation available on other CPUs.

Run `python examples/cpu.py --threads 8` for a complete value example, adding
`--two-stage` for the second psi fit. Stage times are included in the result;
time `torch.autograd.grad` separately when measuring value-plus-gradient cost.
