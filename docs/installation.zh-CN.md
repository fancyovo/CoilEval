# 安装

版本 0.11.0 尚未发布。以下命令安装随附的预编译 wheel。
配套源码压缩包包含示例和开发文件。

[English](installation.md)


wheel（`.whl`）是已构建的 Python 安装包，sdist（`.tar.gz`）是源码包。extras 是方括号中的可选依赖组，如 `[cpu]`；它不会把 PyTorch 自动切换为 CPU 构建。CUDA 是 NVIDIA 的 GPU 计算平台，nvcc 是它的编译器，运行预编译包通常只需相应运行库和驱动。

## 预编译安装包

创建或激活 Python 3.10+ 环境，然后安装：

```sh
python -m pip install "./coil_eval-0.11.0-py3-none-manylinux_2_28_x86_64.whl[gpu]"
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

在同一虚拟环境中执行下面两条安装命令。第一条选定 CPU 版 PyTorch，
第二条自动安装 CoilEval 及其余 CPU 依赖。

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "./coil_eval-0.11.0-py3-none-manylinux_2_28_x86_64.whl[cpu]"
python examples/cpu.py --threads 8
```

`cpu` 依赖组提供 SciPy、threadpoolctl 和 PyTorch；先安装 CPU 版 PyTorch 可选定其 CPU 发行包。
CPU 求值加载 CPU 原生库，无需 NVIDIA 驱动。
在 `Evaluator`、`TensorEvaluator` 或 `SurfaceQSEvaluator` 中指定 `device="cpu"` 即可使用，
省略 `threads` 时自动确定可用线程数。详见 [CPU 指南](cpu.zh-CN.md)。

新的 GPU 环境安装 `gpu` 依赖组，CPU 环境安装 `cpu`；同一环境也可同时安装两组。
API 默认设备仍为 `"cuda"`。

## 各功能的依赖

| 用途 | 依赖组／额外安装包 |
| --- | --- |
| 读取输入、配置和保存结果 | NumPy，作为基础依赖安装 |
| CPU 求值、差分或反传 | `[cpu]`：SciPy、threadpoolctl、PyTorch |
| GPU 普通求值 | `[gpu]`：配套 NVIDIA 运行库 |
| GPU 反传或磁面 f_B / f_T | `[gpu,torch]`，使用 CUDA 版 PyTorch |
| Simsopt 磁面求解 | `[simsopt]`：Simsopt、SciPy；GPU 初始值准备还需要 CUDA 版 PyTorch 和 `[gpu]` |

wheel 元数据已声明这些依赖，指定对应依赖组时由 pip 自动安装；安装裸 wheel
只选择基础依赖。CPU 普通求值也使用 PyTorch 的线性代数功能。
预编译 wheel 包含原生 CPU/GPU 库和 OpenMP 运行库，安装时无需 C++、CMake 或 nvcc。

已安装的环境如果缺少 CPU 依赖，可以补齐：

```sh
python -m pip install "torch>=2.5" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "scipy>=1.10" "threadpoolctl>=3"
python -m pip check
```

这同样适用于已有的 0.7.0 或 0.8.1 CPU 环境。从 0.8.2 起，CPU 入口会在加载
计算后端前列出缺少的依赖和安装命令。请使用运行 example 的同一个 `python`，
第三方依赖的二进制发行包也需支持所用 Python 版本。

## PyTorch

已验证的 GPU 配置使用 Python 3.10–3.13，在同一环境中安装 CUDA 12.8 版 PyTorch：

```sh
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python examples/autograd.py
python examples/autograd.py --batch
```

示例包含在源码压缩包和仓库中。`coil_eval.torch` 是可选接口，普通 `coil_eval` 导入使用 NumPy。
较新版本需同时满足 Python、CUDA 构建和运行库版本的兼容要求。
CPU 安装从 CPU 索引选择兼容版本，无需固定为这里较早的 GPU 构建版本。

## Simsopt

磁面工作流可通过安装包的 `simsopt` 可选依赖组安装：

```sh
python -m pip install "./coil_eval-0.11.0-py3-none-manylinux_2_28_x86_64.whl[gpu,simsopt]"
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

表中的 sm_XX 是 GPU 架构目标，PTX 是供驱动编译的中间指令。OpenMP 提供 CPU 并行运行支持。f_B/f_T 是准对称性（磁场强度只依赖一个螺旋角）误差指标，分别使用傅里叶分量和磁场导数三重积。
