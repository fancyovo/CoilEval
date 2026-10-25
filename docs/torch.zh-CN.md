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
field_dtype=torch.float32)`。安装包提供两个原生库。
显式路径可选择自定义构建；省略 `field_library` 时，从 `library` 所在目录加载。

| 方法 | 物理参数形状 | 每个指标的形状 |
| --- | --- | --- |
| `evaluate(parameters)` | `(4, width)` | 标量 |
| `evaluate_batch(parameters, configs=None)` | `(B,4,width)` | `(B,)` |

`configs` 可为每个独立几何提供一份 `EvaluationConfig`。同一批次使用一致的 Fourier
宽度和 psi 拟合方式。接口按输入顺序捕获中心，再用批量算子执行磁场、追踪和拟合。
批次失败时抛出 `InvalidEvaluationError`，并标明相应几何。

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

## 结果与局部导数

`TensorEvaluation` 包含 `parameters`、`metrics`、`diagnostics` 和 `timings`。
`TensorMetrics` 提供 `qs_volume`、`qs_outer`、`iota_boundary`、`iota_min`、`iota_max`、
`relative_normal_drift_p95`、`coil_gap_min`、`coil_curvature_max`、`coil_length_mean`、
`surface_volume` 和 `psi_fit_rms`。QS 指标采用 RMS 定义，整个体积与外侧径向 bin
使用不同的加权区域。
`psi_fit_rms` 表示 ψ 训练方程残差 RMS，单位 T/m，与标量 `LocalMetrics` 字段同义。

反传计算捕获中心处的连续响应，包括磁轴移动、psi 拟合、磁面选择和积分点。
两阶段路径还包含训练区域、采样坐标和第二次拟合的变化；可选极坐标求积还包含行权重的
连续变化。局部导数以所选磁轴分支、限制射线、格点成员、直方图 bin 和百分位选择为条件。
有限扰动若改变这些选择，其差分方向可能与局部导数不同。

接口支持一阶导数。验证新目标时，应同时比较有意义的各分量梯度以及几何参数、电流参数
各块的梯度；只比较总目标可能掩盖分量差异。NumPy `Evaluator` 为标量回调提供中心差分，
详见 [API](api.zh-CN.md)。
