# 安装

[English](installation.md)

## 预编译安装包

创建或激活 Python 3.10+ 环境，然后安装：

```sh
python -m pip install ./coil_eval-0.4.0a1-py3-none-manylinux_2_28_x86_64.whl
```

pip 会安装固定版本、配套验证过的 CUDA 12.8 运行库依赖。
安装包内含两个 CoilEval 原生库，接口会自动加载。
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

## PyTorch

在同一环境中安装 CUDA 12.8 版 PyTorch，例如：

```sh
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python examples/autograd.py
python examples/autograd.py --batch
```

示例位于源码仓库。`coil_eval.torch` 是可选接口，普通 `coil_eval` 导入使用 NumPy。
较新的 PyTorch 版本也可以使用，前提是其 CUDA 构建支持宿主机驱动和 GPU。

## Simsopt

磁面工作流可通过安装包的 `simsopt` 可选依赖组安装：

```sh
python -m pip install "./coil_eval-0.4.0a1-py3-none-manylinux_2_28_x86_64.whl[simsopt]"
```

该依赖组安装 Simsopt 与 SciPy。
[Simsopt 文档](simsopt.zh-CN.md) 提供 GPU 初始值和 CPU 磁面求解的使用方式。

## 从源码构建

制作 wheel 需要 CUDA Toolkit 12.8、C++17 编译器和 CMake 3.22+：

```sh
export CUDACXX=/path/to/cuda-12.8/bin/nvcc
python -m pip install .
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
- **缺少 CUDA 运行库依赖：** 重新安装 wheel 并启用依赖安装。离线安装还需要准备 NVIDIA 依赖包。
- **驱动或 GPU 内核不兼容：** 对照上表检查驱动和显卡。`nvidia-smi` 显示驱动支持能力，
  `nvcc --version` 显示单独安装的编译器版本。
