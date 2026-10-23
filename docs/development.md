# Development

[简体中文](development.zh-CN.md)

```sh
python -m pip install -e '.[test]'
python -m pytest -q
python tools/check_layout.py
python -m build
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

The optional `src/coil_eval/_autograd` package groups field, trace, front and downstream operators; `native/src/autograd` contains their CUDA primitives. `src/coil_eval/torch.py` owns the public Tensor API. Install `.[test,torch]` to run the mathematical operator tests.
