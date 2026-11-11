# Installation

Version 0.12.0 is a development version. The wheel commands use a wheel built
from the matching source. See [CPU source build](#cpu-source-build) or
[GPU source build](#gpu-source-build).
The accompanying source archive includes the examples and development files.

[简体中文](installation.zh-CN.md)


A wheel (`.whl`) is a built Python distribution. An sdist (`.tar.gz`) contains source.
Extras are optional dependency groups such as `[cpu]`. Select the PyTorch build
with the commands below. CUDA is NVIDIA’s GPU computing platform. Running a
prebuilt GPU wheel requires the corresponding runtime libraries and driver.

## Binary wheel

Create or activate a Python 3.10+ environment, then install the wheel:

```sh
python -m pip install "./coil_eval-0.12.0-py3-none-manylinux_2_28_x86_64.whl[gpu]"
```

The `gpu` extra installs a matched, version-pinned CUDA 12.8 runtime set.
The wheel contains the CPU and GPU native libraries and loads the selected backend automatically. The host provides
an NVIDIA driver and access to a supported GPU. CMake and nvcc are needed only
when building CoilEval from source.

| Component | Binary wheel requirement |
| --- | --- |
| OS and CPU | Linux x86-64, glibc 2.28+ |
| Python | 3.10+ |
| CUDA runtime | 12.8, installed by pip |
| Driver | Supports CUDA 12.8 and the selected GPU; 570.169 meets this for RTX 5090 |
| GPU code | sm_80, sm_86, sm_89, sm_90, sm_100, sm_120, plus compute_120 PTX |

The architecture list covers A100, RTX 30/40/50 series, H100 and compatible
devices. Runtime regression is checked on RTX 5090; the other architectures
are included as build targets. Each device uses its matching compiled code.
The wheel also retains compute_120 PTX. Support for additional GPU generations
is validated per release, including the NVIDIA runtime libraries. Source builds
can select additional architectures supported by their toolkit.

## CPU installation

Use both commands in the same virtual environment. The first selects the
CPU PyTorch distribution; the second installs CoilEval and all remaining
CPU requirements automatically.

```sh
python -m pip install "torch==2.7.1+cpu" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "./coil_eval-0.12.0-py3-none-manylinux_2_28_x86_64.whl[cpu]"
python examples/cpu.py
```

The `cpu` extra supplies SciPy, threadpoolctl and PyTorch. Installing CPU PyTorch
first selects its CPU distribution. CPU evaluation loads the CPU native library
and runs without an NVIDIA driver. `device="cpu"` selects this backend in
`Evaluator`, `TensorEvaluator` and `SurfaceQSEvaluator`. Omit `threads` for an
adaptive thread count; see the [CPU guide](cpu.md).

Fresh GPU installations use the `gpu` extra; CPU installations use `cpu`.
An environment may have both extras installed. The default API device remains
`"cuda"`.

## Dependencies by workflow

| Workflow | Extra / additional packages |
| --- | --- |
| Read inputs, configurations and saved results | NumPy (base dependency) |
| CPU values, finite differences or reverse gradients | `[cpu]`: SciPy, threadpoolctl and PyTorch |
| GPU scalar evaluation | `[gpu]`: the matched NVIDIA runtime set |
| GPU reverse gradients or surface f_B / f_T | `[gpu,torch]`, with CUDA-enabled PyTorch |
| Simsopt surface solves | `[simsopt]`: Simsopt and SciPy; GPU initialization also needs CUDA-enabled PyTorch and `[gpu]` |

These packages are declared in wheel metadata and installed by pip when the
corresponding extra is requested. Installing the bare wheel selects only the
base dependency. CPU evaluation uses PyTorch's linear algebra for ordinary
values as well as gradients. The binary wheel includes its native CPU/GPU
libraries and OpenMP runtime; installing it requires no C++, CMake or nvcc.

For an existing installation missing CPU dependencies, run:

```sh
python -m pip install "torch==2.7.1+cpu" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "scipy>=1.10" "threadpoolctl>=3"
python -m pip check
```

This also repairs an existing 0.7.0 or 0.8.1 CPU environment. From 0.8.2,
CPU entry points list absent dependencies and installation commands before
loading the numerical backend. Run the commands with the same `python` used
for the example. Third-party binary wheels must support that Python version.

## PyTorch

The validated CPU and CUDA 12.8 PyTorch builds are available for Python 3.10–3.13.
For GPU tensors and gradients, select the CUDA 12.8 build in the same environment.

```sh
python -m pip install "torch==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128
python examples/autograd.py
python examples/batch.py
```

The suffix in `2.7.1+cpu` or `2.7.1+cu128` selects a specific build. A requirement
of `torch==2.7.1` also accepts an existing build with either suffix. For other
Python versions, select a compatible build from the corresponding PyTorch index
and check its runtime requirements. The examples are in the source archive and
checkout. Ordinary `coil_eval` imports use NumPy. `coil_eval.torch` is optional.

Check the PyTorch version and CUDA build in the environment used to run the example.

```sh
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip check
```

For the CPU build, `torch.version.cuda` reports `None`. For the GPU setup,
expect `12.8` and `True` when a supported GPU is visible and its driver works.

## Simsopt

For the optional surface workflow, install the wheel with its `simsopt` extra:

```sh
python -m pip install "./coil_eval-0.12.0-py3-none-manylinux_2_28_x86_64.whl[gpu,simsopt]"
```

This installs Simsopt and SciPy. The [Simsopt guide](simsopt.md) shows GPU
initialization and CPU-only surface solves.

## CPU source build

With a C++17 compiler and CMake 3.22+, build a CPU-only wheel:

```sh
python -m pip install "torch==2.7.1+cpu" --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[cpu]" --config-settings=cmake.define.COIL_EVAL_BUILD_CUDA=OFF
```

An OpenMP-capable compiler enables parallel field and trace evaluation. This
build does not invoke nvcc. The [development guide](development.md) describes
source layout and validation.

## GPU source build

To create a wheel, install CUDA Toolkit 12.8, a C++17 compiler and CMake 3.22+:

```sh
export CUDACXX=/path/to/cuda-12.8/bin/nvcc
python -m pip install ".[gpu]"
```

The source build compiles the native backend and installs it with the Python
package. To target one GPU architecture, add
`--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=120` for RTX 5090.
The release build targets all architectures listed above.

For an independently managed native library, retain the separate build:

```sh
python -m pip install . --config-settings=wheel.cmake=false
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/native --parallel

```

`Evaluator(library=..., config=...)` and `TensorEvaluator(library=..., config=...)`
continue to accept an explicit library. This path uses the selected library's
runtime environment. The standalone CMake build also supports separately
configured toolkits, including the CUDA 13 build used in earlier releases.

## Troubleshooting

- **nvcc was not found during installation:** pip is building from source.
  Install the `.whl` file directly for the binary installation path, or configure
  the CUDA Toolkit compiler for a source build.
- **Missing CUDA runtime dependency:** reinstall the wheel with the `gpu` extra and dependencies enabled. An offline installation needs the NVIDIA dependency wheels too.
- **Driver or kernel compatibility error:** compare the driver and GPU with the
  table above. `nvidia-smi` reports driver capability; `nvcc --version` reports
  the separately installed compiler.
- **PyTorch fails to initialize CUDA:** check `torch.version.cuda` in the example's
  Python environment. If the installed build requires a newer driver, select the
  CUDA 12.8 build above and check that the existing driver supports CUDA 12.8.

In the architecture table, sm_XX identifies a GPU architecture target and PTX is intermediate code compiled by the driver. OpenMP supplies CPU parallel execution. f_B/f_T are quasisymmetry-error metrics (field strength depending on one helical angle), using Fourier content and a field-derivative triple product respectively.

Example2 uses SciPy BFGS: `python -m pip install scipy`. Edit example parameters at the top of each script.
