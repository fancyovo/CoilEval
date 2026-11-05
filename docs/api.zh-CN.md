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

`nfp` 和基本线圈数可为任意正整数，仍要求 stellarator symmetry。
QS 螺旋度需显式选择：QA 使用 `(1,0)`，标准 QH 使用 `(1,nfp)`。
其中 N 是完整环面的物理螺旋度，不是单个场周期的模式编号。兼容默认值仍为 `(1,4)`，
修改 `CoilSet.nfp` 不会自动更换 QS 指标；M 必须非零。
张量本身不携带 nfp，构造评估器时传入 `nfp=coils.nfp`。
`evaluate_many` 支持不同 CoilSet 布局并恢复输入顺序；稠密张量批次共享 nfp、线圈数和系数宽度。

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
| `psi_refit_ridge` | `None` | 可选的 S2 列归一化正则系数，须非负；`None` 保留原生设置 |
| `psi_target_refit` | `False` | `psi_fit='two_stage'` 后增加目标内部 S3 拟合的候选选项 |
| `psi_target_refit_factor` | `1.2` | S3 请求训练体积与目标体积的比值，须大于 1 |
| `psi_target_refit_ridge` | `1e-10` | 仅用于 S3 的非负列归一化正则系数 |
| `psi_refit_sampling` | `'cartesian'` | 两阶段拟合的 `'cartesian'` 或 `'polar_quadrature'` 采样 |
| `psi_nearby_method` | `'local'` | 两阶段局部响应，或 `'formal'` 端点重建 |
| `psi_fd_batch_size` | `8` | 两阶段局部差分端点的批量大小 |
| `alpha_radial_weighting` | `'histogram'` | `'histogram'` 或两阶段实验性的 `'smooth'` 径向密度权重 |
| `alpha_weight_sigma` | `0.05` | 平滑权重在归一化 rho 上的 Gaussian 宽度 |
| `alpha_count_floor` | `32.0` | 平滑密度分母中的正加性伪计数 |
| `alpha_radial_basis` | `'zernike'` | 原始径向基，或两阶段路径的 `'flux_legendre'` |
| `stokes_validation_rtol` | `1e-3` | Tensor 磁通一致性容差；`None` 关闭检查 |

0.10.0 默认保留单阶段设置。显式选择两阶段路径：

```python
config = EvaluationConfig(
    target_volume=0.05, psi_a=0.15,
    fast=True, flux_method="stokes",
    psi_fit="two_stage", psi_refit_sampling="cartesian",
)
```

两阶段求值先在初始区域拟合 psi，再构造与磁轴相连的闭合训练区域，并在该区域重新拟合。
目标体积磁面与 QS 指标使用第二次拟合结果。
`psi_refit_ridge` 可用于减小很低 QS 误差下的正则化偏差。它只作用于 S2，
包括对应的局部梯度；S1 及其训练区域保持原设置。GPU 路径需要提供 0.9.1
设置函数的原生库。减小正则并不保证磁面更准确，应同时检查法向磁场误差、
坐标收敛与几何容量。第三次拟合由独立选项 `psi_target_refit` 控制。

启用 `psi_target_refit=True` 后，S1、S2 保持各自配置的拟合目标，再从 S2 选择体积为
`min(psi_target_refit_factor*V, (V+Vcapacity)/2)` 的闭合首根边界。容量必须大于原请求体积 V。
S3 在该边界包围的整个内部进行加权极坐标求积拟合，沿用原基函数和后端精度；
目标行数为 `psi_refit_points`，正则系数仅使用 `psi_target_refit_ridge`。
最终磁面由 S3 在原体积 V 上求得。容量不足或续算失败会明确报错。
原生诊断中的 `psi_fit_stages`、`psi_target_refit_domain` 记录实际流程。
该方法保留显式启用：缩小训练域与弱正则共同作用时，标量可能失去径向单调性，
使原先可评分的输入丢失可用容量。固定 4096 验证不支持将该选项推广为默认。
训练域与正则各有影响，也存在交互；仅恢复正则不能保证所有目标的准确性。

多体积捕获共享 S1、S2，每个目标独立保存 S3 系数、QR 因子和下游状态。
局部梯度包含 S2 训练边界运动、S3 体积方程、容量裁剪及极坐标面积权重的响应，
适用于已选定的离散分支。面 α 与原生体 α 保留各自独立的配置含义。
目标体积容量检查至少覆盖 psi 拟合的环向网格，并为保留的环向模提供每周期至少两个采样点。
磁通积分仍使用其独立配置的网格。这是采样截面上的径向分支检查；通过检查不代表全局磁面或物理精度得到认证。
`psi_refit_sampling='polar_quadrature'` 选择可选的加权极坐标求积，
在 0.9.0 中属于实验性数值选项。两种采样都保留完整采样结构，因此实际训练行数可能
与 `psi_refit_points` 不同，可查看 `diagnostics['psi_refit_training_count']`。

`alpha_radial_weighting='smooth'` 是 `psi_fit='two_stage'` 的可选实验性设置，
使用正的 sigma 与伪计数参数控制平滑密度权重。默认仍采用 histogram 权重。

`alpha_radial_basis='flux_legendre'` 使用归一化磁通的多项式拟合角度修正。
此选项要求 `psi_fit='two_stage'`，同时作用于标量求值、局部梯度与捕获的 alpha 数据。
默认仍使用原始的 `'zernike'` 基。

`flux_method='stokes'` 要求 `fast=True`，为局部近邻和差分端点选择边界积分。
独立中心调用保留其中心积分规则。结果的 `diagnostics['configuration']` 保存所选设置。

Tensor 求值将 Stokes 磁通及其导数与捕获中心的面积积分多项式比较。
任一相对误差超过 `stokes_validation_rtol` 时，该查询改用可反传的面积积分。
设为 `None` 可选择原始的不检查 Tensor 积分行为。此设置用于 Tensor 接口；
NumPy 评估器沿用上面说明的积分规则。实际方法与误差诊断见
[Tensor 结果文档](torch.zh-CN.md#结果与局部导数)。

## 求值会话

在创建它的线程上，通过上下文管理器使用 `Evaluator(config=config, device_id=0)`。
安装包提供默认原生后端，`library=...` 可选择独立构建的库。

| 方法 | 返回内容 |
| --- | --- |
| `evaluate(coils)` | 单个 `EvaluationResult` |
| `evaluate_many(coil_sets, configs=None)` | 按输入顺序返回独立结果，兼容的输入共享批量磁轴追踪 |
| `evaluate_nearby(center, candidates)` | 使用有效中心局部模型的结果 |
| `loss_and_grad(coils, loss_fn, scales=..., h=..., basis=...)` | 标量目标与物理参数梯度 |

每次独立求值会替换会话的有效中心。应在下一次独立求值前，使用相同会话和线程调用
`evaluate_nearby`。`close()` 释放会话，可重复调用。
保存的结果可通过 `EvaluationResult.read_json(path)` 加载查看。

`evaluate_many` 默认使用会话配置，也可以通过 `configs` 为每个输入提供一个
`EvaluationConfig`，在同一批中使用不同的目标体积和拟合半径。
每个结果记录自己的配置；只有最后一个结果保留有效中心，其邻域求值也使用该结果的配置。
后续单点调用继续使用会话原来的默认配置。

```python
results = evaluator.evaluate_many(
    [coils_a, coils_b],
    configs=[EvaluationConfig(target_volume=0.03, psi_a=0.15),
             EvaluationConfig(target_volume=0.05, psi_a=0.18)],
)
```

GPU 批量评分需要 NumPy 和原生后端；CPU 求值还需 [CPU 依赖](cpu.zh-CN.md)。
场周期、系数形状和磁轴搜索设置相同的输入分组共享磁轴准备，随后恢复输入顺序。
每个输入仍执行原有的磁轴选别和完整正式评分；只有一个输入的组及旧版原生库使用普通顺序路径。
`timings['shared_independent_batch']` 给出准备、各输入及整批耗时；
`python_wall_s` 将公开批量调用的总耗时分配到各结果，包含该输入的求值及其分摊的准备时间。

## 结果与指标

`EvaluationResult` 包含 `status`、`metrics`、`diagnostics`、`timings` 和 `mode`。
读取指标前检查 `valid`。`write_json(path)` 保存配置、数值和诊断。
`python_wall_s` 表示公开调用耗时；原生阶段计时可能相互包含或重叠。

| 指标 | 含义 |
| --- | --- |
| `qs_volume` | 所选体积内归一化 QS 残差的 RMS |
| `qs_outer` | 外侧径向 bin 内归一化 QS 残差的 RMS |
| `iota_boundary` | 目标边界处的拟合旋转变换 |
| `iota_min`、`iota_max` | 拟合剖面在 rho 平方上均匀取 257 点后的最小值和最大值 |
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
两个原生 QS 字段都除以所选螺旋度的 `max(sqrt(M*M + N*N), 1)`。
外侧 bin 保留最后一个径向 bin 内的原生点权重；iota 剖面采样覆盖所选径向范围。
独立的 `surface_fc`/`analyze_arrays` 参考函数固定使用 QH 螺旋度 `(1,nfp)`，
在 0.9.0 中没有 helicity 参数。另一个 `surface_qs_metrics` 函数为 f_B/f_T 接受显式
QA `helicity=(1,0)`。与原生结果比较时，应为对应参考路径选择相同螺旋度。

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
配置超出 Tensor 接口支持范围，或环境中没有 CUDA 版 PyTorch 时，采用正式重建。
`diagnostics['nearby_method']`、`['nearby_backend']` 和 `['geometry_model']`
记录实际执行路径，`['nearby_fallback_reason']` 说明回退原因。
[Tensor 接口](torch.zh-CN.md) 为 PyTorch 目标提供局部反传梯度。

中心或端点不可用时抛出 `InvalidEvaluationError`，其 `.results` 包含可用结果。
失效或跨会话中心抛出 `CenterContextError`；原生调用失败抛出 `EvaluationError`。

原生求值、Tensor 反向与 Simsopt 初始化中的 flux-Legendre 径向函数均采用 Legendre 三项递推，避免 FP32 展开单项式的消减误差。该修复不增加有限基空间，也不消除 alpha/psi 拟合误差。切换原生库或基函数实现时，应重新生成 source capture。

可选指标与精度档位见[目标磁面的 f_B 和 f_T](qs.zh-CN.md)。

0.9.1 的失败诊断区分搜索范围内未找到可接受椭圆磁轴，以及目标体积超出拟合标量的
可用容量。`native_error_message` 保留原生错误文本，`failure_reason` 记录分类。
`capacity_scope` 说明径向证书采用的角向采样射线；容量检查不保证这些射线之间的所有角度。
角度反解与径向采样保留折叠拒绝，并选择轴连通首支；后支上的小残差不足以接受该根。
原生 Bernstein 证书会保守拒绝部分不翻号的平坦点，多项式根隔离路径可能解析这些点。
两条路径在共同可用域上保持首支一致。
原生目标体积的正向精度校验也沿用 `flux_rejected` 状态。此时 `failure_reason`
为 `target_volume_forward_precision_exceeded`，`failure_stage` 为
`target_volume_forward_validation`；数值状态保持兼容。

## CPU 后端

使用 `Evaluator(config=..., device='cpu', threads=None)` 选择 CPU，默认设备仍为
`'cuda'`。`threads` 为正整数时指定 CPU 预算，省略时根据进程亲和性和调度器配额
自动选择。[CPU 说明](cpu.zh-CN.md) 包含安装、批量、有限差分及
`TensorEvaluator(..., device='cpu')` 反传梯度用法。
