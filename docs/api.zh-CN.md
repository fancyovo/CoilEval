# Python API

[English](api.md)

## 线圈输入

`CoilSet(coeffs_x, coeffs_y, coeffs_z, currents_a, nfp)` 描述具有仿星器对称性的基本线圈。
每个坐标数组形状为 `(线圈数, 2*阶数+1)`，系数顺序为
`[常数项, sin1, cos1, ..., sinK, cosK]`。坐标单位为米，电流单位为安培。
对称展开后，每条基本线圈对应 `2*nfp` 条物理线圈。

`parameter_vector()` 依次展平全部 x、y、z 系数，最后排列电流；
`with_parameters(vector)` 按此顺序建立新对象。
`read_json(path)` 和 `write_json(path)` 保存和读取相同的单位与约定。

## 配置

`EvaluationConfig(target_volume, ...)` 是只读配置，目标体积单位为 m³。
`to_dict()` 返回可序列化的配置。

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `psi_a` | `0.15` | 拟合半径，m |
| `helicity` | `(1, 4)` | QS 螺旋度 `(M, N)` |
| `h` | `1e-4` | 归一化中心差分步长 |
| `segments_per_coil` | `256` | 每条线圈的离散段数 |
| `psi_iterations`、`alpha_iterations` | `4`、`4` | 局部迭代拟合设置 |
| `fast` | `False` | 使用 240 步磁轴积分和 100 步磁面追踪 |
| `flux_method` | `'area'` | `'area'` 或 fast 近邻路径的 `'stokes'` 边界积分 |
| `psi_fit` | `'single_stage'` | `'single_stage'` 或 `'two_stage'` |
| `psi_refit_points` | `82368` | 第二次拟合的目标训练行数 |
| `psi_refit_sampling` | `'cartesian'` | 两阶段拟合的 `'cartesian'` 或 `'polar_quadrature'` 采样 |
| `psi_nearby_method` | `'local'` | 两阶段局部响应，或 `'formal'` 端点重建 |
| `psi_fd_batch_size` | `8` | 两阶段局部差分端点的批量大小 |
| `alpha_radial_weighting` | `'histogram'` | `'histogram'` 或两阶段实验性的 `'smooth'` 径向密度权重 |
| `alpha_weight_sigma` | `0.05` | 平滑权重在归一化 rho 上的 Gaussian 宽度 |
| `alpha_count_floor` | `32.0` | 平滑密度分母中的正加性伪计数 |

0.4.0a1 默认保留单阶段设置。显式选择两阶段路径：

```python
config = EvaluationConfig(
    target_volume=0.05, psi_a=0.15,
    fast=True, flux_method="stokes",
    psi_fit="two_stage", psi_refit_sampling="cartesian",
)
```

两阶段求值先在初始区域拟合 psi，再构造与磁轴相连的闭合训练区域，并在该区域重新拟合。
目标体积磁面与 QS 指标使用第二次拟合结果。
`psi_refit_sampling='polar_quadrature'` 选择可选的加权极坐标求积，
在此开发版中属于实验性数值选项。两种采样都保留完整采样结构，因此实际训练行数可能
与 `psi_refit_points` 不同，可查看 `diagnostics['psi_refit_training_count']`。

`alpha_radial_weighting='smooth'` 是 `psi_fit='two_stage'` 的可选实验性设置，
使用正的 sigma 与伪计数参数控制平滑密度权重。默认仍采用 histogram 权重。

`flux_method='stokes'` 要求 `fast=True`，为局部近邻和差分端点选择边界积分。
独立中心调用保留其中心积分规则。结果的 `diagnostics['configuration']` 保存所选设置。

## 求值会话

在创建它的线程上，通过上下文管理器使用 `Evaluator(config=config, device_id=0)`。
安装包提供默认原生后端，`library=...` 可选择独立构建的库。

| 方法 | 返回内容 |
| --- | --- |
| `evaluate(coils)` | 单个 `EvaluationResult` |
| `evaluate_many(coil_sets)` | 按输入顺序逐个独立求值 |
| `evaluate_nearby(center, candidates)` | 使用有效中心局部模型的结果 |
| `loss_and_grad(coils, loss_fn, scales=..., h=..., basis=...)` | 标量目标与物理参数梯度 |

每次独立求值会替换会话的有效中心。应在下一次独立求值前，使用相同会话和线程调用
`evaluate_nearby`。保存的结果可通过 `EvaluationResult.read_json(path)` 加载查看。

## 结果与指标

`EvaluationResult` 包含 `status`、`metrics`、`diagnostics`、`timings` 和 `mode`。
读取指标前检查 `valid`。`write_json(path)` 保存配置、数值和诊断。
`python_wall_s` 表示公开调用耗时；原生阶段计时可能相互包含或重叠。

| 指标 | 含义 |
| --- | --- |
| `qs_volume` | 所选体积内归一化 QS 残差的 RMS |
| `qs_outer` | 外侧径向 bin 内归一化 QS 残差的 RMS |
| `iota_boundary`、`iota_min`、`iota_max` | 边界旋转变换与拟合剖面极值 |
| `relative_normal_drift_p95` | 一场周期法向位移 P95 除以平均小半径 |
| `coil_length_mean` | 基本线圈平均长度，m |
| `coil_curvature_max`、`coil_curvature_p95` | 采样最大曲率和可选百分位曲率，m⁻¹ |
| `coil_gap_min` | 物理线圈采样点间最小距离，m |
| `coil_radius_min` | 采样点最小柱坐标半径，m |
| `current_abs_max` | 基本线圈最大绝对电流，A |
| `high_mode_fraction` | 可选高 Fourier 模能量占比 |
| `psi_fit_rms` | 可选的 ψ 训练方程残差 RMS，T/m |

QS 数值采用 RMS 定义。汇总时应先对其平方按相应权重求平均，再开平方。
整个体积、外侧径向 bin 和 Simsopt 参考结果的磁通壳层 RMS 对应不同区域，
详见 [Simsopt 接口](simsopt.zh-CN.md)。
参考接口实现 QH 螺旋度 `(1,nfp)`，与原生结果比较时应选择相同螺旋度。

`quality_penalty(error, threshold=0.01, temperature=0.0025, weight=10000)` 提供可组合的标量惩罚。
拟合与质量诊断独立于用户定义的目标函数。线圈几何指标基于采样曲线。
`psi_fit_rms` 与 Tensor 指标同义，表示单位为 T/m 的 ψ 训练方程残差。
旧版保存记录可以省略该字段。

## 有限差分梯度

`loss_fn(LocalMetrics)` 返回有限标量。正的物理 `scales` 定义端点
`p ± h*scales*direction`。几何尺度为 1、电流尺度为 `100000`、`h=1e-4` 时，
几何扰动为 0.1 mm，电流扰动为 10 A。
返回梯度相对于物理参数，顺序与 `parameter_vector()` 一致。

默认方向为单位阵，`basis` 可指定完整方阵正交基。`GradientResult` 包含目标值、梯度、
中心、端点、尺度、步长、方向和计时。局部响应更新磁轴、psi 和磁面指标；
线圈几何指标使用中心线性化。

两阶段求值的 `psi_nearby_method='local'` 保留中心的有效区域和采样选择，
同时更新其连续响应；可用时采用可选的 CUDA PyTorch 路径。
`'formal'` 在每个端点沿中心磁轴分支重新构建两阶段拟合。
有限步长可能改变离散选择，因此与局部导数存在差异。
[Tensor 接口](torch.zh-CN.md) 为 PyTorch 目标提供局部反传梯度。

中心或端点不可用时抛出 `InvalidEvaluationError`，其 `.results` 包含可用结果。
失效或跨会话中心抛出 `CenterContextError`；原生调用失败抛出 `EvaluationError`。
