# CoilEval

[简体中文](README.zh-CN.md)

CoilEval evaluates stellarator coils on NVIDIA GPUs. Supply Fourier coefficients,
coil currents, field periods and a target volume to obtain quasi-symmetry,
rotational transform, surface quality and coil geometry metrics. Compose these
metrics into a scalar objective and compute its local parameter gradient.

## Installation

Install the Linux x86-64 wheel with Python 3.10+:

```sh
python -m pip install ./coil_eval-0.3.0a2-py3-none-manylinux_2_28_x86_64.whl
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
```

## Documentation

- [Installation and GPU compatibility](docs/installation.md)
- [API and parameter conventions](docs/api.md)
- [PyTorch gradients and batches](docs/torch.md)
- [Version selection](docs/versions.md)
- [Examples](examples/README.md)
- [Development and source layout](docs/development.md)
- [Release history](CHANGELOG.md)

CoilEval is available under the [MIT license](LICENSE).
