# CoilEval

[English](README.md)

CoilEval 使用 NVIDIA GPU 评估仿星器线圈。输入 Fourier 系数、电流、场周期数和目标体积，
获得准对称性、旋转变换、磁面质量及线圈几何指标。用户可以组合指标定义标量目标，并计算局部参数梯度。

0.4.0a1 开发版新增可选的两阶段 psi 拟合、高阶 Fourier 张量与 Simsopt 磁面工作流。
已有调用保留单阶段默认设置，可通过 `EvaluationConfig` 显式选择数值路径。

## 安装

使用 Python 3.10+ 安装 Linux x86-64 安装包：

```sh
python -m pip install ./coil_eval-0.4.0a1-py3-none-manylinux_2_28_x86_64.whl
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

`psi_fit="two_stage"` 在闭合训练区域重新拟合 psi。
可选的 PyTorch 接口提供局部反传梯度和独立批量求值，支持 16 阶线圈。
`coil_eval.simsopt` 可在 GPU 上准备 alpha+nu 初始值，再在 CPU 上运行标准磁面重建；
[异步示例](examples/simsopt_async.py) 将两个阶段分开运行。
磁面工作流使用 `simsopt` 可选依赖组；GPU 准备阶段还需要按安装文档配置 CUDA 版 PyTorch。

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
