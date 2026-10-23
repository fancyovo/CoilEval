# CoilEval

[简体中文](README.zh-CN.md)

CoilEval evaluates stellarator coils on NVIDIA GPUs. Supply Fourier coefficients,
coil currents, field periods and a target volume to obtain quasi-symmetry,
rotational transform, surface quality and coil geometry metrics. Compose these
metrics into a scalar objective and compute its local parameter gradient.

## Installation

Python 3.10+ and NumPy 1.24+ are required. The GPU backend builds on Linux with
CUDA, CMake 3.22+ and a C++17 compiler. CUDA supplies cuBLAS and cuSOLVER.

```sh
python -m pip install .
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/native --parallel
```

Choose the CUDA architecture for your GPU; `120` targets RTX 5090.
The Python wheel contains the Python interface. Build the native library from
the source checkout or source distribution and pass its path to `Evaluator`.

## Quick start

```python
from coil_eval import CoilSet, EvaluationConfig, Evaluator

coils = CoilSet.read_json("examples/coils.json")
config = EvaluationConfig(target_volume=0.05, psi_a=0.15, helicity=(1, 4))
with Evaluator("build/native/libcoil_eval_native.so", config=config) as evaluator:
    result = evaluator.evaluate(coils)
    if result.valid:
        print(result.metrics.qs_volume, result.metrics.iota_boundary)
    else:
        print(result.status)
```

Run a complete example with:

```sh
python examples/evaluate.py --library build/native/libcoil_eval_native.so
python examples/custom_loss.py --library build/native/libcoil_eval_native.so
```

## Documentation

- [API and parameter conventions](docs/api.md)
- [Examples](examples/README.md)
- [Development and source layout](docs/development.md)
- [Release history](CHANGELOG.md)

CoilEval is available under the [MIT license](LICENSE).
