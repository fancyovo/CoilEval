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
| 0.4.0a1 | 两阶段 psi、高阶 Fourier 张量与 Simsopt 接口 | `psi_fit='two_stage'`；`coil_eval.simsopt` |
| 0.5.0a1 | 磁通多项式 alpha、共享张量准备和磁面覆盖检查 | `alpha_radial_basis='flux_legendre'`；批量与 Simsopt 接口 |
| 0.5.1 | 容量保护、自交检查和稳定基函数 | 已包含在后续版本中 |
| 0.6.0 | 目标磁面 f_B、可选 f_T、反传梯度与精度档位 | `coil_eval.qs`；`TensorEvaluator(surface_qs=...)` |
| 0.7.0 | CPU 求值和梯度、线程及初筛网格配置、吞吐改进 | `device="cpu"`、`threads=`、CPU `axis_grid=` |
| 0.8.0 | GPU 捕获重叠、表面批量场评估与同步优化 | [Torch 指南](torch.zh-CN.md)中的 GPU 执行选项 |
| 0.8.1 | 批量捕获错误标明查询编号和失败阶段 | 已包含在 GPU Tensor 路径中 |

源码版本为 0.8.1（尚未发布），改进 GPU 批量调度，保留 CPU 求值和梯度、线程及磁轴搜索选项，
并提高求值吞吐。该版本包含 0.6.0 引入的磁面 QS 功能。

这些版本保持 NumPy 接口的默认行为兼容，通过显式选项使用新的数值路径。
复现时固定准确版本，并随结果保存配置。

0.5.0a1 默认仍为 `psi_fit='single_stage'`。
两阶段拟合默认使用 `psi_refit_sampling='cartesian'`；
加权 `'polar_quadrature'` 是可选的实验性采样方式。
Tensor 接口支持两种拟合方式，并为各阶 Fourier 参数保留物理单位。
Simsopt 工作流提供独立的磁面 RMS 诊断，与体积 QS 联合分析时应选择可比的平均区域。

在干净的工作区中构建旧版本：

```sh
git switch --detach v0.0.0
python -m pip install .
cmake -S native -B build/v0.0.0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/v0.0.0 --parallel
```

原生构建应与对应 Python 版本配套使用。每个版本标签都包含完整源码、文档和示例。
