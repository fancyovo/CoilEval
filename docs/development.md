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
| `src/coil_eval/_two_stage` | Closed-domain geometry, sampling and native two-stage orchestration |
| `src/coil_eval/_autograd` | Local field, tracing, fitting and downstream reverse operators |
| `src/coil_eval/simsopt` | Source records, initialization, standard solves and reference QS |
| `native/src/field` | Coil discretization, field evaluation, tracing and psi fitting |
| `native/src/pipeline` | Axis, target surface, flux, alpha, QS and local evaluation |
| `native/include` | Shared records and module interfaces |
| `native/src/autograd` | CUDA primitives for the optional Tensor API |
| `tests` | API, numerical oracle and source layout tests |
| `examples` | Runnable evaluation and custom objective examples |

Native implementation files compile independently. Headers carry interfaces,
shared types and small inline helpers. Field state is owned by `field/state.cpp`;
evaluation session state is owned by `pipeline/session_state.cu`.

The optional `src/coil_eval/_autograd` package groups field, trace, front and downstream operators; `native/src/autograd` contains their CUDA primitives. `src/coil_eval/torch.py` owns the public Tensor API. Install PyTorch to run the mathematical operator tests. Set `COIL_EVAL_TEST_INSTALLED=1` when running the suite against an installed wheel.

Use additive configuration and public methods for compatible changes. Record
new behavior in CHANGELOG.md and update both documentation languages.

## Version numbering for future releases

Use `major.minor.patch` without an `a` suffix:

- New features increment the second (minor) component and reset patch to zero,
  for example `0.5.1` to `0.6.0`.
- Bug fixes increment the third (patch) component, including small patches and
  obvious corrections, for example `0.5.1` to `0.5.2`.

When releasing the next change after `0.5.0a1`, a fix uses `0.5.1` and a new
feature uses `0.6.0`. These examples describe future numbering, not existing
releases. Keep historical version numbers, artifacts and tags unchanged,
including `0.5.0a1` and earlier `a` versions. Release tags use `v` followed by
the package version; retain earlier tags for reproducible installations.

## Numerical changes

Keep `single_stage` as the compatible default. Add new fitting or sampling
choices through `EvaluationConfig`, and preserve units and parameter order.
Test both order-4 and higher-order inputs when changing Fourier handling.

Two-stage samplers return coordinates, sampling choices and any least-squares
row weights. Keep the native fit and the local reverse operator consistent with
those records. Use focused CPU reference tests for geometry and derivatives,
then GPU tests for values, per-metric gradient directions and norm changes.
Check coordinate and current blocks separately when validating composed losses.

Simsopt tests separate initializer geometry, standard LS/Newton acceptance and
reference f_C. Preserve the source/initializer type markers when serializing
surfaces. Measure single calls and independent batches for performance changes;
state the selected fitting, sampling and precision settings with results.
