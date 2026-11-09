# PyTorch 接口

[English](torch.md)


<details markdown="1">
<summary>本页术语与坐标约定（首次阅读可展开）</summary>

本页用语：**准对称性（QS）**指磁场强度在合适的磁面角坐标中只依赖一个螺旋角；
**准轴对称（QA）**和**准螺旋对称（QH）**分别是其中的轴对称型和螺旋型。
`nfp` 是绕装置一周重复的场周期数，`helicity=(M,N)` 指所要求的角组合 `Mθ−Nφ`，
θ 为极向角（绕磁面截面），φ 为环向角（绕整个装置）。**ι（iota，旋转变换）**是磁力线
每绕环向一圈平均绕极向的圈数。**RMS**是先求平方的加权平均、再开平方的均方根；
**P90/P95**分别是第 90/95 百分位数。`f_B` 衡量不符合指定对称性的磁场强度傅里叶分量，
`f_C` 是微分形式的准对称性残差，`f_T` 是包含磁场导数的三重积残差；它们的理想值均为零，
比较时必须使用相同的积分区域和归一化。

**拟合与坐标约定。** 磁面标签 s 是用等值面近似磁面的标量函数；程序的 `psi_` 拟合字段沿用这一命名，
物理磁通 ψ 则由后续磁通积分标定。默认的“两次拟合”先在初始区域拟合，再在选出的闭合区域内重拟合。
“第三次拟合”在略大于目标体积的边界包围的整个内部重新拟合，默认关闭。
α（alpha）拟合构造磁力线角坐标，λ（lambda）是其中的周期角修正；ν（nu）进一步构造
Boozer 角坐标，即把磁力线表示为直线并使磁场协变角分量只随磁面变化的坐标。
“体 α”在整个体积内拟合，“面 α”在单个目标磁面上重新拟合；后者的阶数指角向傅里叶阶数。
ρ（rho）表示归一化径向标签。`flux_legendre` 是以归一化磁通为变量的 Legendre 多项式基；
`zernike` 是另一种径向多项式基。极坐标求积按半径和角度加权采样；Cartesian 使用直角坐标网格。平滑密度权重估计径向采样密度，伪计数是加到分母的正数，防止稀疏区域权重过大。正则化（ridge）在拟合目标中加入系数惩罚，以抑制不稳定解；
仿星器对称性通过旋转和反射从基本线圈生成完整装置。Stokes 磁通积分用边界线积分计算穿过截面的磁通，area 则直接在截面上积分。QR 是用正交矩阵和上三角矩阵分解来求解最小二乘问题的方法。

**计算记录与精度。** “原生后端”指随包提供的 C++/CUDA 计算库；“捕获（capture）”指保存一次
完整求值的磁轴、拟合系数和采样点，以便后续求导或构造磁面。“中心”是这次求值的线圈参数，
“端点”是施加小扰动后的参数；局部梯度保持当前离散选择不变，计算连续变化。
“容量”是当前拟合函数在径向单调区域内能表示的最大体积；“首支”是从磁轴向外遇到的第一段
单调径向区间。“折叠”指坐标映射失去单调性、不同坐标可能对应同一位置。
Jacobian（雅可比）是映射的一阶导数矩阵；用于角映射的最小 Jacobian 指其行列式。
FP32/float32 与 FP64/float64 分别是 32 位单精度和 64 位双精度浮点数。

</details>


使用 GPU 时，安装与显卡和驱动兼容的 CUDA 版 PyTorch，再运行 `python examples/autograd.py`。
CPU 执行使用 CPU 版 PyTorch 和 `TensorEvaluator(device='cpu', ...)`，详见 [CPU 指南](cpu.zh-CN.md)。
具体环境设置见[安装文档](installation.zh-CN.md)。

## 物理参数张量

`pack_coils(coils, device='cuda:0', dtype=torch.float64)` 将 `CoilSet` 转为形状
`(nc, 3*(2*阶数+1)+1)` 的张量，`nc` 是正整数基本线圈数量。
每行依次包含该线圈的 x、y、z Fourier 系数和电流。
4 阶对应 `(nc,28)`，16 阶对应 `(nc,100)`。坐标使用米，电流使用安培。
NumPy 的 `parameter_vector()` 仍使用按坐标分组的展平布局。

要用进程可见的第三张 GPU，给 `pack_coils` 传 `device='cuda:2'`，
给 `TensorEvaluator` 传 `device_id=2`。如果用 `CUDA_VISIBLE_DEVICES=2`
启动程序，进程内这张卡的编号为 `cuda:0`。Example2 用同一个 `DEVICE_ID`
参数选择求值器和张量的设备。

Tensor 接口支持任意正整数 `nfp`、任意正整数基本线圈数量，以及 `M` 非零的显式
螺旋度 `(M,N)`，要求保留 仿星器对称性。
请设置 `TensorEvaluator(nfp=coils.nfp, ...)`：张量数值不包含场周期信息，
`pack_coils` 也不附加这项元数据。默认仍为 `nfp=4`、螺旋度 `(1,4)`。
QA 使用 `EvaluationConfig(helicity=(1,0), ...)`，对应场周期的 QH 使用 `(1,nfp)`；
螺旋度不会由 `nfp` 自动推断。每条线圈仍使用 256 段及四次 psi/alpha 局部拟合迭代。
配置使用 `fast=True`、`flux_method='stokes'`，
0.11.1 默认采用两次拟合流程。显式选择历史单次拟合时，还需同时设置 histogram 权重与 Zernike 基，见 API 指南。
psi 计算使用 float64，磁场计算默认使用 float32，也可指定 `field_dtype=torch.float64`。

## 求值与批量

在创建它的线程上，通过上下文管理器使用
`TensorEvaluator(library=None, field_library=None, config=..., device_id=0,
field_dtype=torch.float32, batch_axis=True, share_geometry=True, device='cuda',
threads=None, axis_grid=None, surface_qs=None, capture_pipeline=True,
batch_surface_qs=True, synchronize_stages=False, engineering_overlap=True,
capture_workers=1, nfp=4)`。
安装包提供 GPU 求值库、GPU 梯度库和 CPU 库。
显式路径可选择自定义构建；GPU 省略 `field_library` 时，从 `library` 所在目录加载，
CPU 的两类操作则共用 CPU 库。

GPU 的 `batch_axis=True` 将不同几何的常规磁轴准备合批。
`share_geometry=True` 在物理参数相同、配置仅目标体积不同的查询之间共享已捕获的两阶段拟合。
每个查询保留独立的下游计算和参数梯度。可分别设为 `False` 选择独立准备路径；
原生库缺少批量磁轴 API 时，采用逐个磁轴准备。
启用 `psi_target_refit=True` 后，共享止于第二次拟合；每个目标拥有独立的第三次拟合训练域、
系数和 QR 因子。局部梯度包含训练边界与权重的变化，容量裁剪分支固定于捕获状态。
CPU 批量独立准备每个查询；`threads` 指定单个查询的 CPU 预算，`axis_grid` 可选配磁轴初筛网格。
CPU 的逐查询诊断和计时分别保存在 `diagnostics['queries']` 和 `timings['queries']` 中。

GPU 提供以下执行选项。选项控制调度方式；数值设置仍由
`EvaluationConfig` 和 `SurfaceQSConfig` 指定。

| GPU 选项 | 默认值 | 执行方式 |
| --- | --- | --- |
| `capture_pipeline` | `True` | `capture_workers=1` 时，对不同的两阶段几何分组，将 CPU 闭域构建与下一组的原生第一次拟合，或前一组的第二次拟合及正式后续计算重叠。 |
| `capture_workers` | `1` | 使用已验证的取值 1；请求 2–4 个并发捕获线程的原生路径目前会被拒绝。 |
| `batch_surface_qs` | `True` | 可选目标磁面 f_B/f_T 采样中，将不同查询对应的磁场评估合批。 |
| `synchronize_stages` | `False` | 记录各阶段的 CUDA event，在公开调用返回前统一读取；设为 `True` 时在每个阶段前后同步设备，供诊断实际经过时间计时使用。 |
| `engineering_overlap` | `True` | GPU 响应阶段执行期间，由一个 CPU 工作线程准备线圈几何指标及其活动分支导数，再由调用线程接入梯度图。 |

`capture_workers=1` 时，捕获流水线最多保留两个原生上下文，
原生捕获调用始终由创建评估器的线程执行。
目标磁面的几何求根和角坐标求解继续保留各查询独立的数值分支。
这些控制作用于单次求值调用；应用程序中的不同批次需要另行调度。
耗时和吞吐变化取决于批量大小、可用 CPU 资源与 GPU 工作负载。
可将相应并行选项设为 `False`，比较其独立执行路径；设 `synchronize_stages=True`
可比较带同步的阶段计时。
`capture_workers=2`、`3` 或 `4` 的设计让各主机工作线程独立拥有一个完整几何分组及其原生上下文。
目前公开的原生路径会拒绝这些取值，因为两阶段捕获内部的独立 cuSOLVER（NVIDIA 矩阵求解库）
句柄仍需要专门的隔离实现。`capture_workers=1` 已完成验证；请配合
`capture_pipeline=True` 使用 CPU 闭域与原生 GPU 阶段的重叠。
表中执行选项仅适用于 GPU；CPU 继续使用原有线程预算和逐查询执行方式。

GPU 三角求解按查询独立执行，磁场与特征计算仍保持批量。临时工作区有明确上限，
保留的梯度图仍随批量大小增长；请根据 GPU 显存选择批量大小。

| 方法 | 物理参数形状 | 每个指标的形状 |
| --- | --- | --- |
| `evaluate(parameters)` | `(nc, width)` | 标量 |
| `evaluate_batch(parameters, configs=None)` | `(B,nc,width)` | `(B,)` |

`configs` 可为每个独立几何提供一份 `EvaluationConfig`。同一批次使用一致的 `nfp`、
基本线圈数量、傅里叶系数数组宽度、psi 拟合方式以及 `alpha_radial_basis`、`alpha_radial_weighting`、
`alpha_weight_sigma` 和 `alpha_count_floor`。螺旋度、目标体积、`psi_a` 可以逐样本变化，
两阶段采样方式和行数也可不同。
接口按输入顺序返回捕获中心，GPU 再用批量算子执行磁场、追踪和拟合。
捕获到不可用中心时抛出 `InvalidEvaluationError`，并标明相应几何。
自 0.8.1 起，流水线异常还标明失败阶段；共享准备失败列出受影响的所有查询，
某个目标体积的后续计算失败则只列出该查询。原始异常保留在异常链中。

`diagnostics['capture']` 记录实际几何分组、磁轴准备方式和查询数量，
也会记录缓存不可用时采用的独立准备方式。
存在 `pipeline` 条目时，其中记录实际执行方式、工作线程数和分组时间跨度；
单工作线程流水线还记录 CPU 域构建等待。捕获批次的实际耗时应使用
`pipeline['total_s']`，各分组时间跨度可能互相重叠。
`diagnostics['engineering_overlap']` 表示是否使用了几何指标工作线程。

```python
import torch
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.torch import TensorEvaluator, pack_coils

coils = CoilSet.read_json("examples/coils.json")
x = pack_coils(coils)
config = EvaluationConfig(
    target_volume=0.05, fast=True, flux_method="stokes",
    psi_fit="two_stage", psi_refit_sampling="polar_quadrature",
)
with TensorEvaluator(config=config, nfp=coils.nfp) as evaluator:
    result = evaluator.evaluate(x)
    loss = torch.log(result.metrics.qs_volume + 1e-8)
    gradient, = torch.autograd.grad(loss, result.parameters)
```

将独立几何的参数张量堆叠后调用 `evaluate_batch`。
对 `loss.sum()` 求导得到各几何的梯度，`loss.mean()` 则包含批次平均因子。
梯度与输入使用相同布局和物理单位；归一化与自定义目标可以使用普通 PyTorch 运算。
输入可使用评估器设备上的 float32 或 float64；GPU 上 `pack_coils` 的 device 应与
`TensorEvaluator(device_id=...)` 一致，CPU 上两次调用均指定 `device='cpu'`。输入未请求梯度时，返回的 `.parameters`
是新的叶张量；请像示例一样对这个返回张量求导。

## 结果与局部导数

`TensorEvaluation` 包含 `parameters`、`metrics`、`diagnostics` 和 `timings`。
`TensorMetrics` 提供 `qs_volume`、`qs_outer`、`iota_boundary`、`iota_min`、`iota_max`、
`relative_normal_drift_p95`、`coil_gap_min`、`coil_curvature_max`、`coil_length_mean`、
`surface_volume` 和 `psi_fit_rms`。QS 指标采用 RMS 定义，整个体积与最外径向分组
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
`synchronize_stages=False` 时，数值来自调用线程 CUDA stream 上的 event 时间跨度，
并在返回前统一读取。时间跨度可能包含主机发射间隙和等待，不能解读为 GPU 活跃时间。
汇合阶段可能包含等待工作线程剩余计算及结果上传的时间；
在该阶段前已完成的计算已经与其他工作重叠。
`synchronize_stages=True` 时，记录每个阶段同步后的主机实际经过时间。
`diagnostics['stage_timing']` 分别为 `'deferred_cuda_events'` 或 `'synchronized_wall'`。
相互重叠的阶段和工作线程时间不宜求和后作为完整耗时。

目标构造和 `torch.autograd.grad` 发生在 `evaluate` 返回之后。
统计一次全新求值加一次梯度耗时时，应覆盖完整调用与反传，并在两端同步 CUDA。
例如，在 GPU 评估器的上下文中：

```python
import time

torch.cuda.synchronize(evaluator.device)
started = time.perf_counter()
result = evaluator.evaluate(x)
loss = torch.log(result.metrics.qs_volume + 1e-8)
gradient, = torch.autograd.grad(loss, result.parameters)
torch.cuda.synchronize(evaluator.device)
value_and_gradient_s = time.perf_counter() - started
```

导入、编译和预热应单独统计。CPU 操作同步完成，继续使用主机实际经过时间计时。

反传计算捕获中心处的连续响应，包括磁轴移动、psi 拟合、磁面选择和积分点。
两阶段路径还包含训练区域、采样坐标和第二次拟合的变化；可选极坐标求积还包含行权重的
连续变化。局部导数以所选磁轴分支、限制射线、格点成员、直方图分组 和百分位选择为条件。
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

可选指标与精度档位见[目标磁面的 f_B 和 f_T](qs.zh-CN.md)。

CUDA stream 是设备上的有序执行队列，event 是队列中的计时标记。“活动分支”指当前选中的最近点、最大值点或限制射线；这些离散选择固定后才计算局部导数。
