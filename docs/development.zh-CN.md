# 开发

[English](development.md)

开发 Python 接口，并按已安装的可选依赖运行测试时：

```sh
python -m pip install numpy pytest build scikit-build-core
python -m pip install -e . --no-deps --config-settings=wheel.cmake=false
python -m pytest -q
python tools/check_layout.py
python -m build --sdist
```

CPU 后端测试使用 `cpu` 依赖和编译的 CPU 库，安装方式见 [CPU 源码构建](installation.zh-CN.md#从源码构建-cpu-版本)。
CUDA 测试还需要 GPU 后端和可用设备。

制作发行 wheel 时，用 micromamba 创建并激活 `tools/build-environment.yml` 中的环境：

```sh
micromamba create -f tools/build-environment.yml
micromamba activate coileval-build
bash tools/build_linux_wheel.sh
```

构建使用 CUDA 12.8、GCC 11 和 glibc 2.28 sysroot；`auditwheel` 检查系统兼容基线，
NVIDIA 运行库由 `gpu` 安装依赖组提供。GitHub Actions 中的 wheel 流程可手动触发，保存构建产物供审阅。
在全新的 GPU 环境中验证安装包，确认 PATH 中没有 nvcc，并检查 NumPy 梯度及可选 Tensor 接口：

```sh
python -I tools/check_installation.py examples/coils.json --require-no-nvcc --torch
```

| 目录 | 职责 |
| --- | --- |
| `src/coil_eval` | Python API、结果、配置和原生接口 |
| `src/coil_eval/_two_stage` | 闭合区域几何、采样和两阶段原生求值编排 |
| `src/coil_eval/_autograd` | 局部磁场、追踪、拟合及下游反传算子 |
| `src/coil_eval/_cpu` | CPU 源捕获、拟合、磁场算子和线程控制 |
| `src/coil_eval/simsopt` | 源记录、初始值、标准求解与参考 QS |
| `native/src/field` | 线圈离散、磁场、追踪和 psi 拟合 |
| `native/src/pipeline` | 磁轴、目标磁面、磁通、alpha、QS 和近邻求值 |
| `native/include` | 共享类型与模块接口 |
| `native/src/autograd` | 可选 Tensor 接口的 CUDA 算子 |
| `native/cpu` | CPU 磁场、追踪、几何与主机搜索策略适配 |
| `tests` | API、数值 oracle 和源码结构测试 |
| `examples` | 求值与自定义目标示例 |

实现文件独立编译；头文件声明接口、共享类型和小型内联函数。
磁场状态由 `field/state.cpp` 管理，求值会话状态由 `pipeline/session_state.cu` 管理。

可选包 `src/coil_eval/_autograd` 按磁场、追踪、前端拟合和下游指标划分算子，`native/src/autograd` 提供 CUDA 核。`src/coil_eval/torch.py` 提供公开 Tensor API。安装 PyTorch 可运行数学算子测试。测试已安装的 wheel 时设置 `COIL_EVAL_TEST_INSTALLED=1`。

采用新增配置项和方法保持兼容；更新 CHANGELOG 和中英文文档。

## 后续版本的编号规则

后续版本使用 `主版本.次版本.修订版本`（`major.minor.patch`），不再添加 `a` 后缀：

- 新增功能时增加第二位 minor，并将第三位重置为零，例如 `0.5.1` → `0.6.0`。
- 修复问题时增加第三位 patch；小补丁和明显错误的修复也按此规则编号，例如 `0.7.0` → `0.7.1`。

源码包版本为 `0.11.0`，包含显式开启的第三次标量拟合、训练域局部导数，
并保留 `0.9.1` 的反解修复；
变更记录在实际发布前标记为未发布。交付实现改动时同步更新包版本元数据和 CHANGELOG，
不必等到打标签或构建发行二进制。仅修改文档或诊断报告不需要递增包版本。
下一次修复使用 `0.11.1`，新增功能使用 `0.12.0`。
历史版本号、发行产物和标签保持原样，包括 `0.5.0a1` 及更早带 `a` 的版本。
版本标签为 `v` 加包版本号，旧标签保留用于复现安装。

## 数值修改

保留 0.11.0 已验证的 S2 默认配方，S3 为可选项。通过 `EvaluationConfig` 增加拟合或采样选项，
保持单位和参数顺序。修改 Fourier 参数处理时，同时验证 4 阶与更高阶输入。

两阶段采样器提供坐标、采样选择和最小二乘行权重；原生拟合与局部反传算子应使用一致的
记录。先用有针对性的 CPU 参考测试检查几何与导数，再用 GPU 检查数值、各指标梯度方向
与范数变化。验证组合目标时，分别检查坐标参数块与电流参数块。

Simsopt 测试分别检查初始磁面几何、标准 LS/Newton 的接受状态与参考 f_C。
序列化时保留源与初始值的类型标记。性能改动需要测量单次调用和独立批量场景，
并随结果说明拟合、采样与精度设置。
