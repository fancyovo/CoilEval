# CoilEval

[简体中文](README.zh-CN.md)

CoilEval evaluates stellarator coils on NVIDIA GPUs. Supply Fourier coefficients,
coil currents, field periods and a target volume to obtain quasi-symmetry,
rotational transform, surface quality and coil geometry metrics. Compose these
metrics into a scalar objective and compute its local parameter gradient.

The 0.5.0a1 development release adds a flux-polynomial alpha basis, faster
geometry metrics and shared preparation for tensor batches. The Simsopt workflow
preserves the source magnetic axis and checks surface angular covering. Existing
calls retain the single-stage defaults; select numerical options explicitly with
`EvaluationConfig`. See the [release history](CHANGELOG.md) for changes.

## Installation

Install the Linux x86-64 wheel with Python 3.10+:

```sh
python -m pip install ./coil_eval-0.5.0a1-py3-none-manylinux_2_28_x86_64.whl
```

The wheel includes the GPU backend. Pip installs NumPy and the CUDA 12.8 runtime
libraries. Use Linux with glibc 2.28+ and an NVIDIA driver supporting CUDA 12.8
and your GPU; driver 570.169 supports the RTX 5090 configuration.
See [installation and GPU compatibility](docs/installation.md) for supported
architectures, PyTorch setup and source builds.

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

Run a complete example with:

```sh
python examples/evaluate.py
python examples/custom_loss.py
python examples/two_stage.py
```

After installing CUDA-enabled PyTorch, run `python examples/autograd.py` for
evaluation plus one objective gradient. `TensorEvaluator.evaluate_batch`
accepts independent parameter tensors; `capture_sources` shares two-stage fits
when preparing one geometry at several target volumes.

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
Those reference helpers use QH helicity `(1,nfp)`.

## Documentation

- [Installation and GPU compatibility](docs/installation.md)
- [API and parameter conventions](docs/api.md)
- [PyTorch gradients and batches](docs/torch.md)
- [Simsopt surfaces and reference QS](docs/simsopt.md)
- [Version selection](docs/versions.md)
- [Examples](examples/README.md)
- [Development and source layout](docs/development.md)
- [Release history](CHANGELOG.md)

CoilEval is available under the [MIT license](LICENSE).
