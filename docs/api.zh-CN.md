# Python API

[English](api.md)


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


## 线圈输入

`CoilSet(coeffs_x, coeffs_y, coeffs_z, currents_a, nfp)` 描述具有仿星器对称性的基本线圈。
每个坐标数组形状为 `(线圈数, 2*阶数+1)`，系数顺序为
`[常数项, sin1, cos1, ..., sinK, cosK]`。坐标单位为米，电流单位为安培。
对称展开后，每条基本线圈对应 `2*nfp` 条物理线圈。

`parameter_vector()` 依次展平全部 x、y、z 系数，最后排列电流；
`with_parameters(vector)` 按此顺序建立新对象。
`read_json(path)` 和 `write_json(path)` 保存和读取相同的单位与约定。

`nfp` 和基本线圈数可为任意正整数，仍要求仿星器对称性。
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
| `fast` | `True` | 使用 240 步磁轴积分和 100 步磁面追踪 |
| `flux_method` | `'stokes'` | `'area'` 或 fast 近邻路径的 `'stokes'` 边界积分 |
| `psi_fit` | `'two_stage'` | 两次拟合、近轴椭圆域内拟合一次（`'near_axis_ellipse'`），或历史圆域单次拟合（`'single_stage'`） |
| `psi_refit_points` | `82368` | 第二次拟合的目标训练行数 |
| `psi_refit_ridge` | `None` | 可选的第二次拟合列归一化正则系数，须非负；`None` 保留原生设置 |
| `psi_target_refit` | `False` | `psi_fit='two_stage'` 后增加目标内部第三次拟合的候选选项 |
| `psi_target_refit_factor` | `1.2` | 第三次拟合请求训练体积与目标体积的比值，须大于 1 |
| `psi_target_refit_ridge` | `1e-10` | 仅用于第三次拟合的非负列归一化正则系数 |
| `psi_refit_sampling` | `'polar_quadrature'` | 两阶段拟合的 `'cartesian'` 或 `'polar_quadrature'` 采样 |
| `psi_nearby_method` | `'local'` | 两次拟合和椭圆域方法的局部响应，或 `'formal'` 端点重建 |
| `psi_fd_batch_size` | `8` | 局部差分端点的批量大小 |
| `alpha_radial_weighting` | `'smooth'` | `'histogram'` 或两次拟合和椭圆域方法的 `'smooth'` 径向密度权重 |
| `alpha_weight_sigma` | `0.05` | 平滑权重在归一化 rho 上的 Gaussian 宽度 |
| `alpha_count_floor` | `32.0` | 平滑密度分母中的正加性伪计数 |
| `alpha_radial_basis` | `'flux_legendre'` | 原始径向基，或两次拟合和椭圆域方法的 `'flux_legendre'` |
| `stokes_validation_rtol` | `1e-3` | Tensor 磁通一致性容差；`None` 关闭检查 |

0.12.0 默认采用两次拟合配置。下面的显式配置与
`EvaluationConfig(target_volume=0.05, psi_a=0.15)` 等价：

```python
config = EvaluationConfig(
    target_volume=0.05, psi_a=0.15,
    fast=True, flux_method="stokes",
    psi_fit="two_stage", psi_refit_sampling="polar_quadrature",
)
```

两阶段求值先在初始区域拟合 psi，再构造与磁轴相连的闭合训练区域，并在该区域重新拟合。
目标体积磁面与 QS 指标使用第二次拟合结果。

启用 `psi_fit='near_axis_ellipse'` 后，程序根据磁轴附近磁场的变化，预测各环向角处的截面椭圆。
预测区域的体积是 `target_volume` 的 1.2 倍。一次最小二乘拟合使用该区域整个内部的取样点，
并沿用物理坐标下的多项式基和归一化。最终磁面按原请求体积求解。训练区域依赖目标体积，
每个目标体积各自拟合。CPU 和 GPU 都支持该选项，GPU 路径需要 0.12.0 原生库提供的磁轴求解入口。

椭圆预测要求轴附近的磁力线能稳定旋转，且预测区域在物理半径上限内。预测失败时返回
`ellipse_domain_failed`，具体原因见 `diagnostics['failure_reason']`。如果拟合出的标量向外折返，
无法在首段单调区域内达到目标体积，`no_surface` 会报告可用体积和拟合阶段。
`psi_refit_*` 和 `psi_target_refit*` 设置用于两次拟合及可选的第三次拟合。椭圆域方法使用固定取样格点和原生首次拟合正则。

`psi_refit_ridge` 可用于减小很低 QS 误差下的正则化偏差。它只作用于第二次拟合，
包括对应的局部梯度；首次拟合 及其训练区域保持原设置。GPU 路径需要提供 0.9.1
设置函数的原生库。减小正则并不保证磁面更准确，应同时检查法向磁场误差、
坐标收敛与几何容量。第三次拟合由独立选项 `psi_target_refit` 控制。

启用 `psi_target_refit=True` 后，首次拟合、第二次拟合保持各自配置的拟合目标，再从第二次拟合选择体积为
`min(psi_target_refit_factor*V, (V+Vcapacity)/2)` 的闭合首根边界。容量必须大于原请求体积 V。
第三次拟合在该边界包围的整个内部进行加权极坐标求积拟合，沿用原基函数和后端精度；
目标行数为 `psi_refit_points`，正则系数仅使用 `psi_target_refit_ridge`。
最终磁面由第三次拟合在原体积 V 上求得。容量不足或续算失败会明确报错。
原生诊断中的 `psi_fit_stages`、`psi_target_refit_domain` 记录实际流程。
该方法保留显式启用：缩小训练域与弱正则共同作用时，标量可能失去径向单调性，
使原先可评分的输入丢失可用容量。固定 4096 验证不支持将该选项推广为默认。
训练域与正则各有影响，也存在交互；仅恢复正则不能保证所有目标的准确性。

多体积捕获共享首次拟合、第二次拟合，每个目标独立保存第三次拟合系数、QR 因子和下游状态。
局部梯度包含第二次拟合训练边界运动、第三次拟合体积方程、容量裁剪及极坐标面积权重的响应，
适用于已选定的离散分支。面 α 与原生体 α 保留各自独立的配置含义。
这里 V 是目标体积，Vcapacity 是第二次拟合可表示的最大体积；`psi_n_phi` 是初始环向截面数，`psi_m_tor` 是标量拟合的最大环向傅里叶模数。
目标体积容量检查使用 `max(2*psi_n_phi, 4*psi_m_tor+1)` 个环向截面。
磁通积分仍使用其独立配置的网格。这是采样截面上的径向分支检查；通过检查不代表全局磁面或物理精度得到认证。
`psi_refit_sampling='polar_quadrature'` 选择默认的加权极坐标求积；Cartesian 采样仍可显式选择。两种采样都保留完整采样结构，因此实际训练行数可能
与 `psi_refit_points` 不同，可查看 `diagnostics['psi_refit_training_count']`。

`alpha_radial_weighting='smooth'` 是两次拟合和椭圆域方法的默认设置，
使用正的 sigma 与伪计数参数控制平滑密度权重。histogram 权重仍可显式选择。

`alpha_radial_basis='flux_legendre'` 使用归一化磁通的多项式拟合角度修正。
此选项要求 `psi_fit='two_stage'`，同时作用于标量求值、局部梯度与捕获的 alpha 数据。
默认采用 `'flux_legendre'`，原始的 `'zernike'` 基仍可选择。
使用历史单次拟合时同时指定 `psi_fit='single_stage'`、`alpha_radial_weighting='histogram'`
和 `alpha_radial_basis='zernike'`。若需恢复 0.11 之前的标准追踪配置，另设
`fast=False`、`flux_method='area'`、`psi_refit_sampling='cartesian'`。

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
| `qs_outer` |最外径向分组内归一化 QS 残差的 RMS |
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
整个体积、最外径向分组和 Simsopt 参考结果的磁通壳层 RMS 对应不同区域，
详见 [Simsopt 接口](simsopt.zh-CN.md)。
两个原生 QS 字段都除以所选螺旋度的 `max(sqrt(M*M + N*N), 1)`。
最外径向分组 保留最后一个径向分组内的原生点权重；iota 剖面采样覆盖所选径向范围。
独立的 `surface_fc`/`analyze_arrays` 参考函数固定使用 QH 螺旋度 `(1,nfp)`，
这些参考函数的螺旋度由接口固定。另一个 `surface_qs_metrics` 函数为 f_B/f_T 接受显式
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

原生求值、Tensor 反向与 Simsopt 初始化中的 flux-Legendre 径向函数均采用 Legendre 三项递推，避免 FP32 展开单项式的消减误差。该修复不增加有限基空间，也不消除 alpha/psi 拟合误差。切换原生库或基函数实现时，应重新生成保存的完整求值记录。

可选指标与精度档位见[目标磁面的 f_B 和 f_T](qs.zh-CN.md)。

0.9.1 的失败诊断区分搜索范围内未找到可接受椭圆磁轴，以及目标体积超出拟合标量的
可用容量。`native_error_message` 保留原生错误文本，`failure_reason` 记录分类。
`capacity_scope` 说明径向证书采用的角向采样射线；容量检查不保证这些射线之间的所有角度。
角度反解与径向采样保留折叠拒绝，并选择轴连通首支；后支上的小残差不足以接受该根。
原生 Bernstein 证书会保守拒绝部分不翻号的平坦点，多项式根隔离路径可能解析这些点。
两条路径在共同可用域上保持首支一致。
标量几何诊断包含拟合次数 `psi_fit_stage` 和失败阶段 `failure_stage`。
`s_training_domain` 表示第一次拟合无法为第二次拟合提供闭合区域。
`s_target_surface` 表示未能选出所需体积的面。`failure_reason` 区分近轴向外下降
（`s_near_axis_nonmonotone`）、径向折返限制容量（`s_radial_fold_capacity`）、
达到搜索半径上限（`s_radius_cap_capacity`）和容量足够但体积反解失败
（`target_volume_inverse_failed_within_capacity`）。若取样检查无法确定具体原因，
则报告 `s_radial_branch_uncertified`。可计算时，`target_branch_diagnostic`
记录限制容量的射线及体积。这些检查描述取样射线上的拟合标量。

原生目标体积的正向精度校验也沿用 `flux_rejected` 状态。此时 `failure_reason`
为 `target_volume_forward_precision_exceeded`，`failure_stage` 为
`target_volume_forward_validation`；数值状态保持兼容。

## CPU 后端

使用 `Evaluator(config=..., device='cpu', threads=None)` 选择 CPU，默认设备仍为
`'cuda'`。`threads` 为正整数时指定 CPU 预算，省略时根据进程亲和性和调度器配额
自动选择。[CPU 说明](cpu.zh-CN.md) 包含安装、批量、有限差分及
`TensorEvaluator(..., device='cpu')` 反传梯度用法。

Bernstein 证书把径向多项式转换为 Bernstein 基，用系数符号检查导数是否为正；根隔离则直接划分多项式根所在区间。椭圆磁轴指附近磁力线绕轴旋转的稳定周期轨道候选。
