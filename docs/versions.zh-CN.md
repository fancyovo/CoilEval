# 版本

[English](versions.md)

| 版本 | 新增能力 | 选择方式 |
| --- | --- | --- |
| 0.0.0 | 独立求值与局部差分梯度 | `Evaluator` |
| 0.1.0 | 加速积分设置 | `EvaluationConfig(fast=True, ...)` |
| 0.2.0 | 近邻求值的 Stokes 边界磁通积分 | fast 模式下指定 `flux_method='stokes'` |
| 0.3.0a1 | PyTorch 反传梯度与独立几何批次 | `coil_eval.torch.TensorEvaluator` |
| 0.3.0a2 | Linux CUDA 12.8 预编译包与自动加载后端 | 安装 wheel，省略 `library` |
| 0.3.0a3 | 已验证的 CUDA 运行库配套版本 | 安装更新后的 wheel |

这些版本保持 NumPy 接口的默认行为兼容，通过显式选项使用新的数值路径。
复现时固定准确版本，并随结果保存配置。

在干净的工作区中构建旧版本：

```sh
git switch --detach v0.0.0
python -m pip install .
cmake -S native -B build/v0.0.0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/v0.0.0 --parallel
```

原生构建应与对应 Python 版本配套使用。每个版本标签都包含完整源码、文档和示例。
