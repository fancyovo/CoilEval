# 版本

[English](versions.md)


本页按版本说明功能引入时间，历史条目的默认值属于对应版本。术语约定：psi/s 是用等值面表示磁面的标量标签；alpha 是磁力线角修正。两次拟合先拟合初始区域，再重拟合选定闭合区域；第三次拟合覆盖略大于目标体积的整个内部。容量是拟合能表示的最大体积，首支是从磁轴向外的第一段单调区间。capture（捕获）指保存一次求值的中间状态。

QS 是准对称性，QA/QH 为准轴对称/准螺旋对称；f_B/f_T 分别衡量傅里叶非对称分量和磁场导数三重积残差。nfp 是场周期数，helicity 指所需对称性的极向/环向角组合。P90 是第 90 百分位数。Stokes 磁通积分用边界线积分；极坐标求积按半径和角度加权采样；平滑权重减小采样密度不均的影响；flux-Legendre 基使用归一化磁通的 Legendre 多项式。Simsopt 是独立的磁面优化库；wheel 是预编译 Python 安装包。正则是拟合中的系数惩罚，角反解是由指定角度求空间位置。

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
| 0.8.2 | 提前检查依赖并完善安装指引 | 按所选后端安装 wheel 依赖组 |
| 0.9.0 | 不同场周期数、线圈数与 QA/QH 模态 | 张量评估器显式传入 nfp，配置显式指定 helicity |

0.11.0 是首个默认采用两次拟合的预发布候选版。公开默认采用 `fast=True`、`flux_method='stokes'`、
`psi_fit='two_stage'`、极坐标求积、平滑体 α 权重及 flux-Legendre 体 α 基。
第三次拟合保留为可选项，`psi_target_refit=False`。

| 版本 | 新增能力 | 选择方式 |
| --- | --- | --- |
| 0.9.1 | 角反解、径向首支求根修复；第二次拟合正则配置 | 已包含；可选 `psi_refit_ridge=` |
| 0.10.0 | 目标内部第三次拟合与局部导数 | `psi_target_refit=True` |
| 0.11.0 preview | 两次拟合默认、P90 标定、扰动 32 选 1 优化示例 | 默认配置与 example2 |
| 0.11.1 preview | 简明、可编辑、一次运行的示例；逐步输出与计时 | `examples/example1.py`、`examples/example2.py` |

直到 0.10.0 的历史版本默认采用单次拟合。复现历史实验应加载冻结版本及完整配置；
在 0.11.0 显式使用旧单次拟合 的方式见 [API 指南](api.zh-CN.md#配置)。
0.11.0 保持接口参数顺序与物理单位，并更新了数值默认值；0.11.1 沿用这些默认值。
[验证记录](preview_validation.zh-CN.md)列明样本、参考质量条件、误差、计时任务和示例。

在干净的工作区中构建旧版本：

```sh
git switch --detach v0.0.0
python -m pip install .
cmake -S native -B build/v0.0.0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/v0.0.0 --parallel
```

原生构建应与对应 Python 版本配套使用。每个版本标签都包含完整源码、文档和示例。

0.11.1 修订使用示例：顶部参数、单次运行、公开接口、逐步输出和计时；评分器及两次拟合默认与 0.11.0 一致。
