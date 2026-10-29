# CoilEval

[简体中文](README.zh-CN.md)

CoilEval evaluates stellarator coils on NVIDIA GPUs and CPUs. Supply Fourier coefficients,
coil currents, field periods and a target volume to obtain quasi-symmetry,
rotational transform, surface quality and coil geometry metrics. Compose these
metrics into a scalar objective and compute its local parameter gradient.

The source package is 0.9.0 (unreleased). CPU and GPU evaluation support different
field-period and base-coil counts, with explicit QA or QH helicity. The tensor
constructor accepts `nfp=coils.nfp`; its default remains 4. For QA choose
`EvaluationConfig(..., helicity=(1,0))`; standard QH uses `(1,coils.nfp)`.
See the [tensor guide](docs/torch.md), [CPU guide](docs/cpu.md) and
[release history](CHANGELOG.md).

The examples accept `--helicity M N`; for a QA input use
`python examples/qs_metrics.py --coils qa.json --helicity 1 0 --volume V --radius A`.
Replace V and A with the target volume (m³) and fitting radius (m).

## Installation

Install the wheel with the extra matching your backend: `[cpu]` for CPU or
`[gpu]` for GPU. A bare wheel installs only the NumPy-based package interface.
CPU evaluation uses PyTorch for matrix solves even when gradients are not requested.

Install the supplied 0.9.0 binary wheel in a Python 3.10+ environment:

```sh
python -m pip install "./coil_eval-0.9.0-py3-none-manylinux_2_28_x86_64.whl[gpu]"
```

The wheel includes CPU and GPU backends. The `gpu` extra installs NumPy and a
matched CUDA 12.8 runtime set. Use Linux with glibc 2.28+ and an NVIDIA driver supporting CUDA 12.8
and your GPU; driver 570.169 supports the RTX 5090 configuration.
See [installation and GPU compatibility](docs/installation.md) for supported
architectures, PyTorch setup and source builds.

For a computer without an NVIDIA GPU, install CPU PyTorch and the CPU extra:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "./coil_eval-0.9.0-py3-none-manylinux_2_28_x86_64.whl[cpu]"
```

Select `Evaluator(..., device="cpu", threads=8)` or omit `threads` to use the
available CPU allocation. See the [CPU example](examples/cpu.py).

## Quick start

```python
from coil_eval import CoilSet, EvaluationConfig, Evaluator

coils = CoilSet.read_json("examples/coils.json")
config = EvaluationConfig(target_volume=0.05, psi_a=0.15, helicity=(1, 4))
with Evaluator(config=config) as evaluator:
    result = evaluator.evaluate(coils)
    if result.valid:
        print(result.metrics.qs_volume, result.metrics.iota_boundary)
    else:
        print(result.status)
```

The source archive includes example scripts and inputs. From its extracted
project directory, run:

```sh
python examples/evaluate.py
python examples/custom_loss.py
python examples/two_stage.py
```

After installing CUDA-enabled PyTorch, run `python examples/autograd.py` for
evaluation plus one objective gradient. `TensorEvaluator.evaluate_batch`
accepts independent parameter tensors; `capture_sources` shares two-stage fits
when preparing one geometry at several target volumes.
Run `python examples/qs_metrics.py --gradient --metric f_B` for the target-surface
Fourier metric; choose `--metric f_T` for the triple-product metric.

Use `psi_fit="two_stage"` to refit psi in a closed training domain. The optional
PyTorch interface computes local reverse gradients and accepts independent
batches, including order-16 coils. For standard surface reconstruction,
`coil_eval.simsopt` prepares alpha+nu seeds on GPU and runs Simsopt on CPU.
The [asynchronous example](examples/simsopt_async.py) separates those phases.
Install the `simsopt` extra for the surface workflow; GPU preparation also uses
the optional CUDA-enabled PyTorch installation described in the guide.

Select the two-stage path and flux-polynomial alpha fitting with:

```python
config = EvaluationConfig(
    target_volume=0.05, psi_a=0.15, fast=True, flux_method="stokes",
    psi_fit="two_stage", psi_refit_sampling="polar_quadrature",
    alpha_radial_weighting="smooth", alpha_radial_basis="flux_legendre",
)
```

These options are described in [the API guide](docs/api.md). A reconstructed
surface's acceptance and quality are available separately in the Simsopt result.

QS metrics are residual RMS quantities. `qs_volume` averages over the selected
volume; `qs_outer` averages over its outer radial bin. Simsopt reference helpers
also report the flux-shell RMS on a solved surface.
Those reference helpers default to QH helicity `(1,nfp)`; pass `helicity=(1,0)` for QA.

## Documentation

- [Installation and GPU compatibility](docs/installation.md)
- [API and parameter conventions](docs/api.md)
- [PyTorch gradients and batches](docs/torch.md)
- [Target-surface f_B and f_T](docs/qs.md)
- [Simsopt surfaces and reference QS](docs/simsopt.md)
- [Version selection](docs/versions.md)
- [Examples](examples/README.md)
- [Development and source layout](docs/development.md)
- [Release history](CHANGELOG.md)

CoilEval is available under the [MIT license](LICENSE).
