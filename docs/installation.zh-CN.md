# 安装

版本 0.7.0 尚未发布。以下命令安装随附的预编译 wheel。
配套源码压缩包包含示例和开发文件。

[English](installation.md)

## 预编译安装包

创建或激活 Python 3.10+ 环境，然后安装：

```sh
python -m pip install "./coil_eval-0.7.0-py3-none-manylinux_2_28_x86_64.whl[gpu]"
```

`gpu` 依赖组安装固定版本、配套验证过的 CUDA 12.8 运行库。
安装包内含 CPU 和 GPU 原生库，接口会按所选后端自动加载。
宿主机需要 NVIDIA 驱动，并能访问支持的 GPU。只有从源码构建时才需要 CMake 和 nvcc。

| 组件 | 安装包要求 |
| --- | --- |
| 系统与 CPU | Linux x86-64，glibc 2.28+ |
| Python | 3.10+ |
| CUDA 运行库 | 12.8，由 pip 安装 |
| 驱动 | 支持 CUDA 12.8 及所用 GPU；RTX 5090 的 570.169 驱动满足要求 |
| GPU 代码 | sm_80、sm_86、sm_89、sm_90、sm_100、sm_120，以及 compute_120 PTX |

这些架构覆盖 A100、RTX 30/40/50 系列、H100 及兼容设备。运行回归在 RTX 5090 上检查，
其他架构作为编译目标包含在包中，运行时会选取对应的机器码。包内还保留 compute_120 PTX。
新增 GPU 世代的支持会随版本验证，包括 NVIDIA 运行库的兼容性。
源码构建允许自行选择其他受工具链支持的架构。

## CPU 安装

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "./coil_eval-0.7.0-py3-none-manylinux_2_28_x86_64.whl[cpu]"
python examples/cpu.py --threads 8
```

`cpu` 依赖组提供 SciPy、threadpoolctl 和 PyTorch；先安装 CPU 版 PyTorch 可选定其 CPU 发行包。
CPU 求值加载 CPU 原生库，无需 NVIDIA 驱动。
在 `Evaluator`、`TensorEvaluator` 或 `SurfaceQSEvaluator` 中指定 `device="cpu"` 即可使用，
省略 `threads` 时自动确定可用线程数。详见 [CPU 指南](cpu.zh-CN.md)。

新的 GPU 环境安装 `gpu` 依赖组，CPU 环境安装 `cpu`；同一环境也可同时安装两组。
API 默认设备仍为 `"cuda"`。

## PyTorch

在同一环境中安装 CUDA 12.8 版 PyTorch，例如：

```sh
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python examples/autograd.py
python examples/autograd.py --batch
```

示例包含在源码压缩包和仓库中。`coil_eval.torch` 是可选接口，普通 `coil_eval` 导入使用 NumPy。
较新的 PyTorch 版本也可以使用，前提是其 CUDA 构建支持宿主机驱动和 GPU。

## Simsopt

磁面工作流可通过安装包的 `simsopt` 可选依赖组安装：

```sh
python -m pip install "./coil_eval-0.7.0-py3-none-manylinux_2_28_x86_64.whl[gpu,simsopt]"
```

该依赖组安装 Simsopt 与 SciPy。
[Simsopt 文档](simsopt.zh-CN.md) 提供 GPU 初始值和 CPU 磁面求解的使用方式。

## 从源码构建 CPU 版本

准备 C++17 编译器和 CMake 3.22+，然后运行：

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[cpu]" --config-settings=cmake.define.COIL_EVAL_BUILD_CUDA=OFF
```

支持 OpenMP 的编译器可启用磁场和追踪计算的多线程。此构建流程不调用 nvcc。
代码结构与验证方式见[开发文档](development.zh-CN.md)。

## 从源码构建 GPU 版本

制作 wheel 需要 CUDA Toolkit 12.8、C++17 编译器和 CMake 3.22+：

```sh
export CUDACXX=/path/to/cuda-12.8/bin/nvcc
python -m pip install ".[gpu]"
```

源码安装会编译后端并与 Python 接口一起安装。仅针对 RTX 5090 构建时，可增加
`--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=120`。
发行包默认编译上表列出的全部架构。

需要自行管理原生库时，也可以分别构建：

```sh
python -m pip install . --config-settings=wheel.cmake=false
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/native --parallel
python examples/evaluate.py --library build/native/libcoil_eval_native.so
```

`Evaluator(library=..., config=...)` 和 `TensorEvaluator(library=..., config=...)`
继续接受显式库路径，并使用该库对应的运行环境。
单独的 CMake 构建也允许配置其他工具链，包括先前版本使用的 CUDA 13。

## 常见安装问题

- **安装时找不到 nvcc：** pip 正在从源码构建。直接安装 `.whl` 文件可使用预编译路径；
  源码构建则需要配置 CUDA Toolkit 编译器。
- **缺少 CUDA 运行库依赖：** 重新安装 wheel 的 `gpu` 依赖组，并启用依赖安装。离线安装还需要准备 NVIDIA 依赖包。
- **驱动或 GPU 内核不兼容：** 对照上表检查驱动和显卡。`nvidia-smi` 显示驱动支持能力，
  `nvcc --version` 显示单独安装的编译器版本。
