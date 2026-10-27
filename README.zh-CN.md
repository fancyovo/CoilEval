# CoilEval

[English](README.md)

CoilEval 使用 NVIDIA GPU 评估仿星器线圈。输入 Fourier 系数、电流、场周期数和目标体积，
获得准对称性、旋转变换、磁面质量及线圈几何指标。用户可以组合指标定义标量目标，并计算局部参数梯度。

源码包版本为 0.5.1（尚未发布），包含目标体积容量保护、Simsopt 参考面自交拒绝
和稳定 flux-Legendre 递推修复。已有调用保留单阶段默认设置，
可通过 `EvaluationConfig` 显式选择数值路径。改动见[版本记录](docs/changelog.zh-CN.md)。

## 安装

下例使用 Python 3.10+ 安装历史 0.5.0a1 wheel，不包含 0.5.1 修复。
0.5.1 发行 wheel 尚未发布；使用更新后的代码请按安装文档从源码构建。

```sh
python -m pip install ./coil_eval-0.5.0a1-py3-none-manylinux_2_28_x86_64.whl
```

安装包包含 GPU 后端，pip 会安装 NumPy 和 CUDA 12.8 运行库。
需要 glibc 2.28+ 的 Linux，以及支持 CUDA 12.8 和所用显卡的 NVIDIA 驱动；
570.169 驱动满足 RTX 5090 的这一要求。
支持的 GPU 架构、PyTorch 配置和源码构建方式见[安装与兼容性](docs/installation.zh-CN.md)。

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

完整命令示例：

```sh
python examples/evaluate.py
python examples/custom_loss.py
python examples/two_stage.py
```

配置 CUDA 版 PyTorch 后，可运行 `python examples/autograd.py` 完成求值和一次目标反传。
`TensorEvaluator.evaluate_batch` 接收独立参数张量；`capture_sources` 在同一几何的多个目标体积之间
共享两阶段拟合。

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

QS 指标采用残差 RMS 定义：`qs_volume` 对所选体积求平均，`qs_outer` 对外侧径向 bin
求平均；Simsopt 参考接口还提供求解磁面上的磁通壳层 RMS。
该参考接口使用 QH 螺旋度 `(1,nfp)`。

## 文档

- [安装与 GPU 兼容性](docs/installation.zh-CN.md)
- [API 与参数约定](docs/api.zh-CN.md)
- [PyTorch 梯度和批量求值](docs/torch.zh-CN.md)
- [Simsopt 磁面与参考 QS](docs/simsopt.zh-CN.md)
- [版本选择](docs/versions.zh-CN.md)
- [示例](examples/README.zh-CN.md)
- [开发与源码结构](docs/development.zh-CN.md)
- [版本记录](docs/changelog.zh-CN.md)

CoilEval 使用 [MIT 许可证](LICENSE)。
