# Python API

[English](api.md)

## 输入与配置

`CoilSet(coeffs_x, coeffs_y, coeffs_z, currents_a, nfp)` 描述具有仿星器对称性的基本线圈。
坐标数组形状为 `(线圈数, 2*阶数+1)`，系数顺序为常数项、sin1、cos1、…。
长度单位为米，电流为安培；每条基本线圈展开为 `2*nfp` 条物理线圈。
`parameter_vector()` 依次展平全部 x、y、z 系数，最后排列电流；
`with_parameters(vector)` 按此顺序建立新对象。`read_json` 和 `write_json` 用于保存与读取。

`EvaluationConfig(target_volume, psi_a=0.15, helicity=(1,4), h=1e-4,
psi_iterations=4, alpha_iterations=4, segments_per_coil=256)` 是只读配置。
目标体积单位为 m³，`psi_a` 为 m，螺旋度为整数 `(M,N)`；`to_dict()` 可用于序列化。

## 求值接口

使用 `Evaluator(library, config=config, device_id=0)` 创建会话，使用上下文管理器释放资源。

| 方法 | 返回内容 |
| --- | --- |
| `evaluate(coils)` | 单个独立求值结果 |
| `evaluate_many(coil_sets)` | 按输入顺序逐个独立求值 |
| `evaluate_nearby(center, candidates)` | 同一中心附近的局部批量结果 |
| `loss_and_grad(coils, loss_fn, scales=..., h=..., basis=...)` | 标量值与完整物理参数梯度 |

会话和中心应在创建它们的线程使用。每次独立求值会替换有效中心；近邻调用应在下次独立求值前完成。
从 JSON 加载的结果和近邻结果用于读取数值。

## 输出指标

`EvaluationResult` 包含 `status`、`metrics`、`diagnostics`、`timings` 和 `mode`。
读取指标前检查 `valid`。通过 `write_json` 与 `read_json` 保存和加载。
`python_wall_s` 为公开调用耗时；原生阶段可嵌套，近邻批次共享 `shared_local_batch` 计时。

| 指标 | 含义 |
| --- | --- |
| `qs_volume`、`qs_outer` | 归一化 QS 残差平方的体积加权值、外侧径向 bin 值 |
| `iota_boundary`、`iota_min`、`iota_max` | 边界旋转变换与拟合剖面极值 |
| `relative_normal_drift_p95` | 一场周期法向位移 P95 除以平均小半径 |
| `coil_length_mean` | 基本线圈平均长度，m |
| `coil_curvature_max`、`coil_curvature_p95` | 离散最大曲率和可选百分位曲率，m⁻¹ |
| `coil_gap_min` | 物理线圈采样点间最小距离，m |
| `coil_radius_min` | 采样点最小柱坐标半径，m |
| `current_abs_max` | 基本线圈最大绝对电流，A |
| `high_mode_fraction` | 可选高 Fourier 模能量占比 |

`quality_penalty(error, threshold=0.01, temperature=0.0025, weight=10000)` 提供可组合的质量惩罚。
诊断中提供额外的拟合和质量信息。几何指标基于采样曲线，曲率 P95 使用均匀参数采样。

## 梯度

`loss_fn` 接收 `LocalMetrics`，返回有限标量。`scales` 明确每个参数的正物理尺度；
扰动为 `p ± h*scales*direction`。例如几何尺度为 1、电流尺度为 100000、`h=1e-4` 时，
几何扰动为 0.1 mm，电流扰动为 10 A。返回梯度相对于物理参数，顺序与 `parameter_vector()` 一致。

默认使用单位阵方向；`basis` 可指定完整方阵正交基。`GradientResult` 保存值、梯度、中心、端点、
尺度、步长、方向和计时。局部模型更新磁轴、psi 和磁面指标，线圈几何指标使用中心线性化。

中心或端点不可用时抛出 `InvalidEvaluationError`，其 `.results` 保存可用的诊断结果；
失效或跨会话中心抛出 `CenterContextError`；原生调用失败抛出 `EvaluationError`。
解释差分梯度时应检查扰动尺度与线圈几何的匹配程度。

## Fast 模式

从 0.1.0 起，`EvaluationConfig(..., fast=True)` 使用 240 步磁轴积分和 100 步磁面追踪。默认 `fast=False` 保留原设置。结果的 `diagnostics["configuration"]` 保存所选配置。

## 磁通积分

从 0.2.0 起，`EvaluationConfig(..., fast=True, flux_method="stokes")` 为近邻和差分端点选择 Stokes 边界积分。`flux_method="area"` 保留面积求积。独立中心求值沿用原积分方法。
