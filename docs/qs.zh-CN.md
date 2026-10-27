# 磁面准对称性指标

[English](qs.md)

CoilEval 0.6 新增傅里叶指标 `f_B` 和可选三重积指标 `f_T`，作用于目标体积对应的磁面。两者共用可导的 alpha+nu 坐标构造，包括目标磁面的 alpha 拟合与基于真空磁场势的 nu 修正。线圈求值接口需要 CUDA 版 PyTorch；Simsopt 用于单独计算参考磁面。

## 计算线圈指标

```python
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.qs import SurfaceQSConfig, SurfaceQSEvaluator

coils = CoilSet.read_json("examples/coils.json")
config = EvaluationConfig(
    target_volume=0.05, psi_a=0.15,
    fast=True, flux_method="stokes", psi_fit="two_stage",
    psi_refit_sampling="polar_quadrature",
    alpha_radial_weighting="smooth", alpha_radial_basis="flux_legendre",
)
with SurfaceQSEvaluator(config=config, surface_qs=SurfaceQSConfig()) as evaluator:
    result = evaluator.evaluate(coils)
    print(result.f_B)
    print(result.diagnostics)
```

纯求值接口接收 `CoilSet`，返回 Python 标量指标及诊断信息，磁场计算使用 float64。`library`、`field_library`、`device_id` 可选择自定义后端或 GPU。线圈接口支持 4 个基本线圈、`nfp=4`、螺旋度 `(1,4)`、每条线圈 256 段，以及四次 psi/alpha 局部拟合迭代。配置使用 `fast=True`、`flux_method='stokes'`，参数约定见 [PyTorch 接口](torch.zh-CN.md)。

`f_T` 默认为 `None`。选择 `SurfaceQSConfig(compute_f_T=True)` 后，利用解析空间磁场雅可比同时计算两项指标。

多项请求可使用 `evaluate_many(coil_sets, configs=None)`，按输入顺序返回元组，并允许为每个线圈输入指定一个 `EvaluationConfig`。例如，对同一个几何计算两个体积：

```python
from dataclasses import replace

configs = [replace(config, target_volume=v) for v in (0.025, 0.05)]
with SurfaceQSEvaluator(config=config) as evaluator:
    results = evaluator.evaluate_many([coils, coils], configs=configs)
    print([result.f_B for result in results])
```

兼容输入采用批量磁轴预处理；相同两阶段几何且配置仅目标体积不同时，共享源拟合，再逐个采样目标磁面。构造参数 `batch_axis=False`、`share_geometry=False` 分别关闭这两项优化。每项结果的 `diagnostics["batch"]` 记录分组、执行方式与均摊耗时。源捕获无效时，与单点接口一样抛出 `InvalidEvaluationError`。

运行示例：

```sh
python examples/qs_metrics.py
python examples/qs_metrics.py --accuracy balanced
python examples/qs_metrics.py --gradient --metric f_B
python examples/qs_metrics.py --gradient --metric f_T
```

## 对目标函数求梯度

```python
import torch
from coil_eval.qs import SurfaceQSConfig
from coil_eval.torch import TensorEvaluator, pack_coils

parameters = pack_coils(coils, device="cuda:0")
with TensorEvaluator(
    config=config, surface_qs=SurfaceQSConfig(), field_dtype=torch.float64,
) as evaluator:
    result = evaluator.evaluate(parameters)
    objective = result.surface_qs.f_B.square()
    gradient, = torch.autograd.grad(objective, result.parameters)
```

设置 `compute_f_T=True` 并选择 `result.surface_qs.f_T`，即可对三重积指标求导。梯度遵循 `result.parameters` 的物理参数排列。`evaluate_batch` 为每个输入返回已启用的指标；对各独立目标之和求导，可得到对应输入的梯度。

`TensorEvaluator` 的默认值为 `surface_qs=None`，已有调用保留原有输出与计算路径。新增指标位于 `result.surface_qs`；`result.metrics.qs_volume` 和 `qs_outer` 继续描述各自体积与最外径向层中的微分残差。

局部反向导数包含磁轴、psi 拟合、目标层、alpha、iota、nu、采样位置和磁场的变化。角坐标反演要求正 Jacobian。与差分核对时，应分别检查各项指标以及几何、电流参数块。

磁面拟合采用的旋转变换位于 `diagnostics["surfaces"][i]["iota_surface"]`。它与既有评分器的 `result.metrics.iota_boundary` 分别报告；后者保留原有拟合结果。

## 分辨率与诊断

| `SurfaceQSConfig` 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `nphi` | `129` | 一个环向场周期内的点数 |
| `ntheta` | `131` | 一个完整极向周期内的点数 |
| `nu_order` | `24` | nu 保留的最高极向模数与每场周期环向模数 |
| `compute_f_T` | `False` | 同时计算使用解析磁场导数的三重积指标 |
| `alpha_order` | `32` | 目标磁面 alpha 模数上限；`0` 使用原 alpha 拟合 |
| `alpha_ridge` | `1e-12` | 目标磁面 alpha 拟合的正则化系数 |
| `alpha_coordinates` | `'geometric'` | 在几何角上拟合，或选择原有 `'straight'` 直线角映射 |
| `alpha_fit_nphi`、`alpha_fit_ntheta` | `None` | 可单独指定 alpha 拟合网格，否则由输出网格与阶数决定 |
| `nu_method` | `'potential'` | 真空磁场势拟合；`'transport'` 使用输运方程形式 |
| `max_iterations` | `30` | 每次角坐标反演的最大迭代数 |
| `tolerance` | `1e-11` | 角坐标反演容差，单位 rad |
| `min_jacobian` | `1e-6` | 角坐标映射允许的最小 Jacobian |
| `resonance_tolerance` | `1e-10` | nu 输运方程保留模式的最小绝对除数 |

两个网格维度均须为不小于 5 的奇数，并大于 `2*nu_order`。奇数网格使所有保留傅里叶模式的导数都有明确含义。网格均不重复采样周期端点。

`SurfaceQSConfig.preset(name, **overrides)` 提供以下分辨率设置：

| 预设 | `nphi × ntheta` | `alpha_order` | `nu_order` | alpha 坐标 |
| --- | --- | --- | --- | --- |
| `'high'`（默认） | `129 × 131` | `32` | `24` | geometric |
| `'balanced'` | `97 × 99` | `24` | `16` | geometric |
| `'fast'` | `65 × 67` | `8` | `12` | straight |

这些是分辨率选项，具体磁场仍需检查收敛。分别增大输出网格、alpha 阶数与 nu 阶数，可区分采样误差和坐标表示误差。未指定 alpha 拟合网格时，环向至少取 `4*alpha_order+1` 点、极向至少取 `4*alpha_order+3` 点，并且不低于输出网格分辨率。可选的 `f_T` 使用解析磁场雅可比，以及 `B dot grad(B)` 的一阶角向导数；应单独检查它的网格收敛。

Tensor 结果在 `result.surface_qs.diagnostics["surfaces"]` 中按输入保存诊断；纯求值结果通过 `result.diagnostics["surfaces"]` 提供相同记录。诊断包括角坐标反演误差、最小 Jacobian、alpha 与 nu 拟合残差、磁场切向性、各协变分量与 Boozer 方程残差、归一化半径，以及 `spectral_tail_fraction`。后者给出任一频率方向最外三分之一频带占总傅里叶能量的比例，可帮助判断是否需要加密网格；比例很小本身不构成收敛结论。

启用目标磁面 alpha 拟合时，`surface_alpha_solver` 记录线性求解方法、相对残差、迭代次数和是否回退到 FP64 直接求解。Tensor 梯度调用会追加对应的反向求解记录。

## 指标定义

以下角度均用弧度，傅里叶展开为

\[
B(\theta_B,\phi_B)=\sum_{m,n}B_{mn}e^{i(m\theta_B-n\phi_B)}.
\]

对于螺旋度 `(M,N)`，`N` 是完整环向周期对应的模数。无量纲傅里叶指标定义为

\[
f_B=\sqrt{\frac{\sum_{mN-nM\ne0}|B_{mn}|^2}
                       {\sum_{m,n}|B_{mn}|^2}}.
\]

分母为均匀 Boozer 角度下的傅里叶能量。三重积中，\(\psi\) 表示环向磁通除以 \(2\pi\)，\(B=|\mathbf B|\)，定义

\[
T=(\nabla\psi\times\nabla B)\cdot\nabla(\mathbf B\cdot\nabla B).
\]

\[
f_T=\sqrt{\left\langle\left(\frac{R^2T}{B^4}\right)^2\right\rangle_\psi}.
\]

磁通壳平均采用权重 \(|\mathcal J|=|G+\iota I|/B^2\)。线圈求值采用真空场约定 `I=0`，`R` 为磁轴平均柱坐标半径。与 Simsopt 比较时，应使用同一个半径。这里明确规定了标量 RMS 的归一化方式；其他软件可能采用不同的平均和归一化。

## 输入已有 Boozer 网格

`coil_eval.qs_metrics` 中的 `boozer_spectrum_metrics` 用一次 FFT 返回 `f_B`、`f_B_squared`、`spectrum` 与 `spectral_tail_fraction`，只需输入均匀角网格上的磁场模长。

```python
from coil_eval.qs_metrics import boozer_spectrum_metrics

metrics = boozer_spectrum_metrics(B, nfp=4, helicity=(1, 4))
```

`coil_eval.qs_metrics.boozer_qs_metrics(B, nfp=..., helicity=..., iota=..., G=..., major_radius=..., I=0.0)` 接收形状为 `(..., nphi, ntheta)` 的 NumPy 数组或 PyTorch 张量。输入是在均匀 Boozer 角度上的磁场模长；phi 覆盖一个场周期，theta 覆盖完整极向周期。`G`、`I` 使用弧度角度的协变分量，单位 T·m。磁面参数可向前导维度广播。

返回字典包含 `f_B`、`f_T`、两者平方、逐点数组 `f_T_raw` 和 `f_T_normalized`、`jacobian`、复数傅里叶谱 `spectrum`，以及 `spectral_tail_fraction`。归一化长度 `major_radius` 须显式给出。NumPy 路径不依赖 PyTorch 或 native GPU 后端；张量路径保留输入的梯度。

`boozer_qs_metrics_field(B_vector, dBdx, x_theta, x_phi, ...)` 则根据笛卡尔磁场值、雅可比和 Boozer 切向量计算三重积。向量形状为 `(..., nphi, ntheta, 3)`，雅可比再增加一个长度为 3 的末维，约定 `dBdx[..., i, j] = d B_i / d x_j`；切向量对弧度制 Boozer 角度求导。这是线圈接口在 `compute_f_T=True` 时使用的路径。

## 与 Simsopt 参考比较

通过 [Simsopt 流程](simsopt.zh-CN.md) 得到已接受的 `StandardSurface` 后，可调用：

```python
from coil_eval.simsopt.qs_reference import surface_qs_metrics

reference = surface_qs_metrics(
    surface, field, major_radius=axis_mean_radius,
    helicity=(1, coils.nfp), sizes=(65, 97, 129),
)
print(reference["f_B"], reference["f_T"])
print(reference["quality"], reference["convergence"])
```

参考实现用解析笛卡尔磁场导数和 `B dot grad(B)` 的角向导数计算三重积，同时复核磁面自交并报告固定磁面的网格收敛。确认自交的磁面会被拒绝；无法确定的检查保留质量提示。

同一磁面上的公式核对，与 native 目标磁面相对于求解后参考磁面的比较，应分别进行。后一种比较还包含 psi 与坐标拟合差异。磁岛和严重扭曲的磁面可能妨碍可靠的嵌套磁面表示，评估一致性时应保留对应失败与质量诊断。固定拟合磁面的网格收敛与长时间磁力线约束分别回答不同问题。
