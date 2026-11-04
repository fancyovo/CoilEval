# CoilEval

[English](README.md)

CoilEval 使用 NVIDIA GPU 或 CPU 评估仿星器线圈。输入 Fourier 系数、电流、场周期数和目标体积，
获得准对称性、旋转变换、磁面质量及线圈几何指标。用户可以组合指标定义标量目标，并计算局部参数梯度。

源码包版本为 0.9.1（尚未发布）。CPU 与 GPU 评估支持不同场周期数、基本线圈数，
以及显式指定的 QA/QH 螺旋度。张量构造器传入 `nfp=coils.nfp`，兼容默认值仍为 4。
QA 设置 `EvaluationConfig(..., helicity=(1,0))`，标准 QH 设置 `(1,coils.nfp)`。
参见 [张量接口](docs/torch.zh-CN.md)、[CPU 指南](docs/cpu.zh-CN.md)与[版本历史](CHANGELOG.md)。

0.9.1 修复角度反解和轴连通径向首支采样，并提供显式的第二次拟合正则化设置。
下方冻结的 0.9.0 标定 wheel 早于这些修复；使用 0.9.1 改动需要从源码构建。

示例支持 `--helicity M N`。对 QA 输入可运行
`python examples/qs_metrics.py --coils qa.json --helicity 1 0 --volume V --radius A`，
其中 V、A 分别替换为目标体积（m³）和拟合半径（m）。

## 安装

请按后端选择依赖组：CPU 用 `[cpu]`，GPU 用 `[gpu]`。裸 wheel 仅安装依赖 NumPy
的基础接口。CPU 普通求值也使用 PyTorch 的矩阵求解，因此不求梯度时同样需要 PyTorch。

在 Python 3.10+ 环境中安装随附的冻结 0.9.0 预编译 wheel：

```sh
python -m pip install "./coil_eval-0.9.0-py3-none-manylinux_2_28_x86_64.whl[gpu]"
```

安装包包含 CPU 和 GPU 后端；`gpu` 依赖组会安装 NumPy 和配套的 CUDA 12.8 运行库。
需要 glibc 2.28+ 的 Linux，以及支持 CUDA 12.8 和所用显卡的 NVIDIA 驱动；
570.169 驱动满足 RTX 5090 的这一要求。
支持的 GPU 架构、PyTorch 配置和源码构建方式见[安装与兼容性](docs/installation.zh-CN.md)。

没有 NVIDIA GPU 的电脑可以安装 CPU 版 PyTorch 和 CPU 依赖组：

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "./coil_eval-0.9.0-py3-none-manylinux_2_28_x86_64.whl[cpu]"
```

通过 `Evaluator(..., device="cpu", threads=8)` 选择 CPU；省略 `threads` 时使用当前可用的 CPU 配额。
完整用法见 [CPU 示例](examples/cpu.py)。

## 快速开始

```python
from coil_eval import CoilSet, EvaluationConfig, Evaluator

coils = CoilSet.read_json("examples/coils.json")
config = EvaluationConfig(target_volume=0.05, psi_a=0.15, helicity=(1, 4))
with Evaluator(config=config) as evaluator:
    result = evaluator.evaluate(coils)
    if result.valid:
        print(result.metrics.qs_volume, result.metrics.iota_boundary)
    else:
        print(result.status)
```

源码压缩包包含示例脚本和输入。在解压后的项目目录运行：

```sh
python examples/evaluate.py
python examples/custom_loss.py
python examples/two_stage.py
```

配置 CUDA 版 PyTorch 后，可运行 `python examples/autograd.py` 完成求值和一次目标反传。
`TensorEvaluator.evaluate_batch` 接收独立参数张量；`capture_sources` 在同一几何的多个目标体积之间
共享两阶段拟合。
运行 `python examples/qs_metrics.py --gradient --metric f_B` 可计算目标磁面的傅里叶指标及梯度；
选择 `--metric f_T` 则计算三重积指标。

`psi_fit="two_stage"` 在闭合训练区域重新拟合 psi。
可选的 PyTorch 接口提供局部反传梯度和独立批量求值，支持 16 阶线圈。
`coil_eval.simsopt` 可在 GPU 上准备 alpha+nu 初始值，再在 CPU 上运行标准磁面重建；
[异步示例](examples/simsopt_async.py) 将两个阶段分开运行。
磁面工作流使用 `simsopt` 可选依赖组；GPU 准备阶段还需要按安装文档配置 CUDA 版 PyTorch。

通过以下配置选择两阶段路径与磁通多项式 alpha 拟合：

```python
config = EvaluationConfig(
    target_volume=0.05, psi_a=0.15, fast=True, flux_method="stokes",
    psi_fit="two_stage", psi_refit_sampling="polar_quadrature",
    alpha_radial_weighting="smooth", alpha_radial_basis="flux_legendre",
)
```

各选项见 [API 文档](docs/api.zh-CN.md)。Simsopt 返回结果分别提供重建磁面的接受状态与质量状态。

微分 f_C 指标采用残差 RMS 定义：`qs_volume` 对所选体积求平均，`qs_outer` 对外侧径向 bin
求平均。Simsopt 参考函数 `surface_fc` 和 `analyze_arrays` 在 0.9.0 中固定使用
QH `(1,nfp)`，没有 helicity 参数。另一个 f_B/f_T 参考接口 `surface_qs_metrics`
接受 `helicity=(1,0)` 以计算 QA，详见[参考定义](docs/simsopt.zh-CN.md#参考-qs-指标)。

## Preview 验证

尚未发布的 0.9.0 验证实验已完成固定随机 QUASR 样本的基础计算：1024 个配置、
每个四档噪声，共 4096 个任务（含失败记录）。完成复核的结果包含 1981 个完成的 Simsopt 参考，
以及 1336 对可比较结果。全部配对的 f_B/f_C/f_T 对数尺度 R² 为
0.928267/0.977176/0.995077；零噪声档 385 对的 f_B R² 为 0.744453。
145 个异常样本均有复核结果：144 个完成视觉检查，一个无可用几何。两类主统计继续
保留大误差，另列 612 对的几何证据筛选辅助统计。
[精度表与散点图](docs/preview_validation.zh-CN.md#固定精度样本完成复核的结果)列出分母、参考资格与限制。

[密集庞加莱示例](docs/preview_validation.zh-CN.md#庞加莱几何检查)展示几何尚不能解释的大误差、岛链邻近及细长截面。

CoilEval 0.9.0 的冻结基准使用四个原始 16 阶 QUASR 线圈。标量求值/求值加梯度
的中位耗时，8 线程 CPU 为 22.483/42.531 秒，RTX 5090 为 1.553/1.959 秒，
A100 为 1.903/2.473 秒。批量 8 的整批耗时分别为 245.300/385.934、9.013/9.420
和 11.282/12.657 秒。[完整表格与测量口径](docs/preview_validation.zh-CN.md)
列出批量 1、2、4、8、冷调用、硬件与固定工作量。
这些数字对应常规体积 f_C 求值和含工程指标的损失梯度；目标磁面 f_B/f_T 为单独请求。

选中的 QA BFGS 结果 `n3-QA-i0-l5-s2-v5` 已完成完整独立参考流程：体积等效
Ap 为 8.4326、Simsopt 目标磁面 iota 为 0.469986、Simsopt 磁通壳层 RMS f_C
为 0.00816269、DESC 边界 f_C 为 0.00722812。DESC f_C 使用表面加权的绝对残差均值
及其自身归一化。最终 DESC 磁面嵌套；Simsopt 磁面保留两项致密网格质量警告。
另一实验在三个体积下分别筛选同一组 32 个简单圆环种子，均未得到有效初始值，BFGS 因而未启动。
[验证记录](docs/preview_validation.zh-CN.md)给出定义、参数和限制。
原始示例证据保存在上层验证工作区
`evaluation/coileval_preview_20261102/examples/results_20261103/`。

## 文档

- [安装与 GPU 兼容性](docs/installation.zh-CN.md)
- [API 与参数约定](docs/api.zh-CN.md)
- [PyTorch 梯度和批量求值](docs/torch.zh-CN.md)
- [目标磁面的 f_B 与 f_T](docs/qs.zh-CN.md)
- [Simsopt 磁面与参考 QS](docs/simsopt.zh-CN.md)
- [版本选择](docs/versions.zh-CN.md)
- [示例](examples/README.zh-CN.md)
- [开发与源码结构](docs/development.zh-CN.md)
- [版本记录](docs/changelog.zh-CN.md)

CoilEval 使用 [MIT 许可证](LICENSE)。
