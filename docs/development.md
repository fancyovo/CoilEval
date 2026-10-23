# Development

[简体中文](development.zh-CN.md)

For Python-only development and CPU tests:

```sh
python -m pip install numpy pytest build scikit-build-core
python -m pip install -e . --no-deps --config-settings=wheel.cmake=false
python -m pytest -q
python tools/check_layout.py
python -m build --sdist
```

For a release wheel, create and activate the environment from
`tools/build-environment.yml` with micromamba, then run:

```sh
micromamba create -f tools/build-environment.yml
micromamba activate coileval-build
bash tools/build_linux_wheel.sh
```

The build uses CUDA 12.8, GCC 11 and a glibc 2.28 sysroot. `auditwheel` checks
the platform baseline; NVIDIA runtime libraries are supplied by Python package
dependencies. The manually triggered GitHub Actions wheel workflow saves build
artifacts for review. Verify a wheel in a fresh GPU environment with nvcc absent
from PATH, including NumPy gradients and the optional Tensor API:

```sh
python -I tools/check_installation.py examples/coils.json --require-no-nvcc --torch
```

| Directory | Responsibility |
| --- | --- |
| `src/coil_eval` | Public Python API, records, configuration and native binding |
| `native/src/field` | Coil discretization, field evaluation, tracing and psi fitting |
| `native/src/pipeline` | Axis, target surface, flux, alpha, QS and local evaluation |
| `native/include` | Shared records and module interfaces |
| `tests` | API, numerical oracle and source layout tests |
| `examples` | Runnable evaluation and custom objective examples |

Native implementation files compile independently. Headers carry interfaces,
shared types and small inline helpers. Field state is owned by `field/state.cpp`;
evaluation session state is owned by `pipeline/session_state.cu`.

Use additive configuration and public methods for compatible changes. Record
new behavior in CHANGELOG.md and update both documentation languages. Release
tags use `v` followed by the package version. A prerelease uses a Python version
such as `0.3.0a1`. Earlier tags remain available for reproducible installations.

The optional `src/coil_eval/_autograd` package groups field, trace, front and downstream operators; `native/src/autograd` contains their CUDA primitives. `src/coil_eval/torch.py` owns the public Tensor API. Install PyTorch to run the mathematical operator tests. Set `COIL_EVAL_TEST_INSTALLED=1` when running the suite against an installed wheel.
