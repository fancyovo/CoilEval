# 开发

[English](development.md)

仅开发 Python 接口及运行 CPU 测试时：

```sh
python -m pip install numpy pytest build scikit-build-core
python -m pip install -e . --no-deps --config-settings=wheel.cmake=false
python -m pytest -q
python tools/check_layout.py
python -m build --sdist
```

制作发行 wheel 时，用 micromamba 创建并激活 `tools/build-environment.yml` 中的环境：

```sh
micromamba create -f tools/build-environment.yml
micromamba activate coileval-build
bash tools/build_linux_wheel.sh
```

构建使用 CUDA 12.8、GCC 11 和 glibc 2.28 sysroot；`auditwheel` 检查系统兼容基线，
NVIDIA 运行库由 Python 包依赖提供。GitHub Actions 中的 wheel 流程可手动触发，保存构建产物供审阅。
在全新的 GPU 环境中验证安装包，确认 PATH 中没有 nvcc，并检查 NumPy 梯度及可选 Tensor 接口：

```sh
python -I tools/check_installation.py examples/coils.json --require-no-nvcc --torch
```

| 目录 | 职责 |
| --- | --- |
| `src/coil_eval` | Python API、结果、配置和原生接口 |
| `native/src/field` | 线圈离散、磁场、追踪和 psi 拟合 |
| `native/src/pipeline` | 磁轴、目标磁面、磁通、alpha、QS 和近邻求值 |
| `native/include` | 共享类型与模块接口 |
| `tests` | API、数值 oracle 和源码结构测试 |
| `examples` | 求值与自定义目标示例 |

实现文件独立编译；头文件声明接口、共享类型和小型内联函数。
磁场状态由 `field/state.cpp` 管理，求值会话状态由 `pipeline/session_state.cu` 管理。
采用新增配置项和方法保持兼容；更新 CHANGELOG 和中英文文档。
版本标签为 `v` 加包版本号，预发布版本使用 `0.3.0a1` 等 Python 版本形式。
旧标签保留用于复现安装。

可选包 `src/coil_eval/_autograd` 按磁场、追踪、前端拟合和下游指标划分算子，`native/src/autograd` 提供 CUDA 核。`src/coil_eval/torch.py` 提供公开 Tensor API。安装 PyTorch 可运行数学算子测试。测试已安装的 wheel 时设置 `COIL_EVAL_TEST_INSTALLED=1`。
