# CoilEval

[English](README.md)

CoilEval 使用 NVIDIA GPU 评估仿星器线圈。输入 Fourier 系数、电流、场周期数和目标体积，
获得准对称性、旋转变换、磁面质量及线圈几何指标。用户可以组合指标定义标量目标，并计算局部参数梯度。

## 安装

使用 Python 3.10+ 安装 Linux x86-64 安装包：

```sh
python -m pip install ./coil_eval-0.3.0a2-py3-none-manylinux_2_28_x86_64.whl
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
```

## 文档

- [安装与 GPU 兼容性](docs/installation.zh-CN.md)
- [API 与参数约定](docs/api.zh-CN.md)
- [PyTorch 梯度和批量求值](docs/torch.zh-CN.md)
- [版本选择](docs/versions.zh-CN.md)
- [示例](examples/README.zh-CN.md)
- [开发与源码结构](docs/development.zh-CN.md)
- [版本记录](docs/changelog.zh-CN.md)

CoilEval 使用 [MIT 许可证](LICENSE)。
