# Installation

Version 0.7.0 is unreleased. The commands below install the supplied binary wheel.
The accompanying source archive includes the examples and development files.

[简体中文](installation.zh-CN.md)

## Binary wheel

Create or activate a Python 3.10+ environment, then install the supplied wheel:

```sh
python -m pip install "./coil_eval-0.7.0-py3-none-manylinux_2_28_x86_64.whl[gpu]"
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

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "./coil_eval-0.7.0-py3-none-manylinux_2_28_x86_64.whl[cpu]"
python examples/cpu.py --threads 8
```

The `cpu` extra supplies SciPy, threadpoolctl and PyTorch. Installing CPU PyTorch
first selects its CPU distribution. CPU evaluation loads the CPU native library
and runs without an NVIDIA driver. `device="cpu"` selects this backend in
`Evaluator`, `TensorEvaluator` and `SurfaceQSEvaluator`. Omit `threads` for an
adaptive thread count; see the [CPU guide](cpu.md).

Fresh GPU installations use the `gpu` extra; CPU installations use `cpu`.
An environment may have both extras installed. The default API device remains
`"cuda"`.

## PyTorch

Install a CUDA 12.8 PyTorch build in the same environment. For example:

```sh
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python examples/autograd.py
python examples/autograd.py --batch
```

The examples are included in the source archive and checkout. `coil_eval.torch` is optional;
ordinary `coil_eval` imports use NumPy. A newer PyTorch release can also be
used when its CUDA build supports the host driver and GPU.

## Simsopt

For the optional surface workflow, install the wheel with its `simsopt` extra:

```sh
python -m pip install "./coil_eval-0.7.0-py3-none-manylinux_2_28_x86_64.whl[gpu,simsopt]"
```

This installs Simsopt and SciPy. The [Simsopt guide](simsopt.md) shows GPU
initialization and CPU-only surface solves.

## CPU source build

With a C++17 compiler and CMake 3.22+, build a CPU-only wheel:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
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
python examples/evaluate.py --library build/native/libcoil_eval_native.so
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
