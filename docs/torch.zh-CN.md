# PyTorch 接口

[English](torch.md)

安装与显卡和驱动兼容的 CUDA 版 PyTorch，再运行 `python examples/autograd.py`。
具体环境设置见[安装文档](installation.zh-CN.md)。

## 物理参数张量

`pack_coils(coils, device='cuda:0', dtype=torch.float64)` 将 `CoilSet` 转为形状
`(4, 3*(2*阶数+1)+1)` 的张量。每行依次包含该线圈的 x、y、z Fourier 系数和电流。
4 阶对应 `(4,28)`，16 阶对应 `(4,100)`。坐标使用米，电流使用安培。
NumPy 的 `parameter_vector()` 仍使用按坐标分组的展平布局。

Tensor 接口支持四条基本线圈、`nfp=4`、螺旋度 `(1,4)`、每条线圈 256 段及四次
psi/alpha 局部拟合迭代。配置使用 `fast=True`、`flux_method='stokes'`，
`psi_fit` 可以选择 `'single_stage'` 或 `'two_stage'`。
psi 计算使用 float64，磁场计算默认使用 float32，也可指定 `field_dtype=torch.float64`。

## 求值与批量

在创建它的线程上，通过上下文管理器使用
`TensorEvaluator(library=None, field_library=None, config=..., device_id=0,
field_dtype=torch.float32, batch_axis=True, share_geometry=True)`。安装包提供两个原生库。
显式路径可选择自定义构建；省略 `field_library` 时，从 `library` 所在目录加载。

`batch_axis=True` 将不同几何的常规磁轴准备合批。
`share_geometry=True` 在物理参数相同、配置仅目标体积不同的查询之间共享已捕获的两阶段拟合。
每个查询保留独立的下游计算和参数梯度。可分别设为 `False` 选择独立准备路径；
原生库缺少批量磁轴 API 时，采用逐个磁轴准备。

GPU 三角求解按查询独立执行，磁场与特征计算仍保持批量。临时工作区有明确上限，
保留的梯度图仍随批量大小增长；请根据 GPU 显存选择批量大小。

| 方法 | 物理参数形状 | 每个指标的形状 |
| --- | --- | --- |
| `evaluate(parameters)` | `(4, width)` | 标量 |
| `evaluate_batch(parameters, configs=None)` | `(B,4,width)` | `(B,)` |

`configs` 可为每个独立几何提供一份 `EvaluationConfig`。同一批次使用一致的 Fourier
宽度、psi 拟合方式以及 `alpha_radial_basis`、`alpha_radial_weighting`、`alpha_weight_sigma` 和
`alpha_count_floor`。目标体积、`psi_a` 可以逐样本变化，两阶段采样方式和行数也可不同。
接口按输入顺序捕获中心，再用批量算子执行磁场、追踪和拟合。
捕获到不可用中心时抛出 `InvalidEvaluationError`，并标明相应几何。

`diagnostics['capture']` 记录实际几何分组、磁轴准备方式和查询数量，
也会记录缓存不可用时采用的独立准备方式。

```python
import torch
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.torch import TensorEvaluator, pack_coils

x = pack_coils(CoilSet.read_json("examples/coils.json"))
config = EvaluationConfig(
    target_volume=0.05, fast=True, flux_method="stokes",
    psi_fit="two_stage", psi_refit_sampling="cartesian",
)
with TensorEvaluator(config=config) as evaluator:
    result = evaluator.evaluate(x)
    loss = torch.log(result.metrics.qs_volume + 1e-8)
    gradient, = torch.autograd.grad(loss, result.parameters)
```

将独立几何的参数张量堆叠后调用 `evaluate_batch`。
对 `loss.sum()` 求导得到各几何的梯度，`loss.mean()` 则包含批次平均因子。
梯度与输入使用相同布局和物理单位；归一化与自定义目标可以使用普通 PyTorch 运算。
输入可使用评估器 GPU 上的 float32 或 float64；`pack_coils` 的 device 应与
`TensorEvaluator(device_id=...)` 一致。输入未请求梯度时，返回的 `.parameters`
是新的叶张量；请像示例一样对这个返回张量求导。

## 结果与局部导数

`TensorEvaluation` 包含 `parameters`、`metrics`、`diagnostics` 和 `timings`。
`TensorMetrics` 提供 `qs_volume`、`qs_outer`、`iota_boundary`、`iota_min`、`iota_max`、
`relative_normal_drift_p95`、`coil_gap_min`、`coil_curvature_max`、`coil_length_mean`、
`surface_volume` 和 `psi_fit_rms`。QS 指标采用 RMS 定义，整个体积与外侧径向 bin
使用不同的加权区域。
`psi_fit_rms` 表示 ψ 训练方程残差 RMS，单位 T/m，与标量 `LocalMetrics` 字段同义。
`iota_min`、`iota_max` 与标量接口一致，在 rho 平方上均匀取 257 个剖面采样点。

`diagnostics['flux_integration']` 记录实际磁通积分方法：单点返回一个字典，
批量返回逐查询的字典列表。`method` 为 `'stokes'` 或 `'area'`，
`fallback` 表示是否补算了面积积分，`validation` 为 `'checked'` 或 `'disabled'`。
`edge_relative_error` 比较边界磁通；`derivative_relative_error` 在 33 个归一化位置
比较磁通导数，并用参考导数的最大绝对值归一化。参考值来自正式中心的面积积分多项式。
默认容差为 `1e-3`，可通过 `EvaluationConfig(stokes_validation_rtol=...)` 设置，
或用 `None` 关闭检查。批量中只为未通过检查的查询补算面积积分。
所选积分过程完整保留几何、磁场与磁面层级变化的导数。
两阶段局部差分近邻只在捕获中心选择一次积分方法，所有端点沿用该选择。
端点的 `flux_integration.validation` 为 `'center_selected'`，
`center_validation` 保存中心检查状态，因此端点磁通的真实物理变化不会触发积分方法切换。

`timings` 记录前向阶段，包括中心捕获和线圈几何指标准备。
目标构造和 `torch.autograd.grad` 发生在 `evaluate` 返回之后。
统计一次求值加一次梯度耗时时，应覆盖完整调用与反传，并在两端同步 CUDA。

反传计算捕获中心处的连续响应，包括磁轴移动、psi 拟合、磁面选择和积分点。
两阶段路径还包含训练区域、采样坐标和第二次拟合的变化；可选极坐标求积还包含行权重的
连续变化。局部导数以所选磁轴分支、限制射线、格点成员、直方图 bin 和百分位选择为条件。
有限扰动若改变这些选择，其差分方向可能与局部导数不同。

捕获的中心保留正式求值实际使用的体积点索引，包括有效点压缩后的选择；
局部响应继续计算这些所选点坐标的连续变化。
最近线圈接触点和最大曲率采样点的当前选择，也分别决定对应几何指标的局部导数。

线圈的小幅扰动可能改变有效射线的集合。正式求值会在压缩后的体积点中重新采样，
因此即使磁轴分支保持固定，也可能跨过离散采样边界。靠近此类边界时，应比较多个差分步长，
并逐项检查指标。局部导数沿用中心选定的点；局部端点若失去必需的磁面根，会返回无效求值状态。

接口支持一阶导数。验证新目标时，应同时比较有意义的各分量梯度以及几何参数、电流参数
各块的梯度；只比较总目标可能掩盖分量差异。NumPy `Evaluator` 为标量回调提供中心差分，
详见 [API](api.zh-CN.md)。
