# PyTorch 接口

[English](torch.md)

从 **0.3.0a1** 开始提供。安装可选依赖并构建两个原生库：

```sh
python -m pip install '.[torch]'
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/native --parallel
python examples/autograd.py --library build/native/libcoil_eval_native.so
python examples/autograd.py --library build/native/libcoil_eval_native.so --batch
```

## 输入与配置

`pack_coils(coils, device='cuda:0', dtype=torch.float64)` 将 `CoilSet` 转成 `(4,28)` 张量。
每行依次包含一条线圈的 9 个 x 系数、9 个 y 系数、9 个 z 系数和电流，单位分别为米和安培。
这个按线圈排列的张量布局便于计算；NumPy 接口 `CoilSet.parameter_vector()` 继续采用按坐标排列的一维布局。

预发布接口支持 4 条四阶基本线圈、`nfp=4`、螺旋度 `(1,4)`、每条线圈 256 个离散段和各 4 次 psi/alpha 拟合迭代。
使用 `EvaluationConfig(target_volume=..., psi_a=..., fast=True, flux_method='stokes')`。
psi 使用 float64，磁场默认使用 float32。

## 求值

`TensorEvaluator(library, field_library=None, config=..., device_id=0,
field_dtype=torch.float32)` 是上下文管理器。默认从评估库的同目录加载 `libcoil_eval_autograd.so`。
`field_dtype` 也可设为 `torch.float64`。

| 方法 | 输入 | 指标形状 |
| --- | --- | --- |
| `evaluate(parameters)` | `(4,28)` CUDA 张量 | 标量 |
| `evaluate_batch(parameters, configs=None)` | `(B,4,28)` CUDA 张量 | `(B,)` |

`configs` 可为每个几何指定配置。批次使用独立中心，再批量计算磁场、追踪和拟合算子；中心按输入顺序捕获。
在创建会话的线程调用它。无效几何会抛出带批内序号的 `InvalidEvaluationError`。

## 结果与梯度

`TensorEvaluation` 包含 `parameters`、`metrics`、`diagnostics` 和 `timings`。
`TensorMetrics` 包含 `qs_volume`、`iota_boundary`、`relative_normal_drift_p95`、
`coil_gap_min`、`coil_curvature_max`、`coil_length_mean`、`surface_volume` 和 `psi_fit_rms`。

```python
import torch
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.torch import TensorEvaluator, pack_coils

x = pack_coils(CoilSet.read_json('examples/coils.json'))
config = EvaluationConfig(target_volume=0.05, fast=True, flux_method='stokes')
with TensorEvaluator('build/native/libcoil_eval_native.so', config=config) as ev:
    result = ev.evaluate(x)
    loss = torch.log(result.metrics.qs_volume + 1e-8)
    gradient, = torch.autograd.grad(loss, result.parameters)
```

批量输入对 `loss.sum()` 求导可得到每个样本的梯度；`loss.mean()` 会再乘批平均因子。
梯度遵循输入的物理单位和张量布局，调用方可以使用 PyTorch 操作组合可微归一化与自定义目标。

反传计算捕获中心附近的局部响应，包括磁轴、拟合磁面和积分点的移动。
中心求解器因子与离散选择共同定义该局部模型；跨越直方图 bin 或百分位选择变化时，有限步长差分可能不同。
1.3 预发布接口支持一阶导数，面向一阶优化。

NumPy `Evaluator` 继续支持普通标量回调和中心差分梯度；PyTorch 目标使用 Tensor 接口。
