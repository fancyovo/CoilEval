# Simsopt 接口

[English](simsopt.md)

`coil_eval.simsopt` 提供 source-psi 与 alpha 拟合记录、alpha+nu 磁面初始值、
标准 Simsopt LS/Newton 求解和参考 QS 诊断。磁面准备与求解环境需要安装 Simsopt；
CUDA 准备阶段还使用可选的 PyTorch 依赖。相应依赖会在运行具体操作时导入。

使用本地安装包安装磁面相关依赖：

```sh
python -m pip install "./coil_eval-0.5.0a1-py3-none-manylinux_2_28_x86_64.whl[simsopt]"
```

`simsopt` 可选依赖组包含 Simsopt 与 SciPy。
GPU 准备所需的 CUDA 版 PyTorch 按[安装文档](installation.zh-CN.md)配置。

## 同一几何的多个体积

```python
from dataclasses import replace
from coil_eval import CoilSet, EvaluationConfig
from coil_eval.simsopt import capture_sources

coils = CoilSet.read_json("examples/coils.json")
base = EvaluationConfig(
    target_volume=0.05, psi_a=0.15,
    fast=True, flux_method="stokes", psi_fit="two_stage",
)
captures = capture_sources(coils, [replace(base, target_volume=v) for v in (0.01, 0.03, 0.05)])
for capture in captures:
    print(capture.valid, capture.result.status)
```

`capture_sources(coils, configs, library=None, device_id=0, include_torsion=True)` 接收单份或一组
`EvaluationConfig`，始终按输入顺序返回列表。配置选择 `psi_fit='two_stage'`，
各份配置只允许目标体积不同。同一几何的多个体积共享磁轴、第一次拟合、训练区域与
第二次拟合，各目标体积分别完成下游求值。
只准备源数据而不需要可选扭率诊断时，可设置 `include_torsion=False`。
评分指标与源系数的定义保持一致。

每个 `SourceCapture` 包含 `.result`（`EvaluationResult`）、`.source`（`SourcePsi`）
和 `.alpha`（`AlphaFit`）。准备磁面前检查 `.valid`。不可用目标保留结果，source/alpha
为空。记录持有独立的主机数组，在捕获调用结束后仍然有效。

`SourcePsi` 提供系数、模式数组、磁轴样本、场周期、拟合半径和所选层级；
`AlphaFit` 提供模式、lambda 与 iota 系数。`from_snapshot()` 用于对接已有原生捕获。
`SourcePsi.read(path, level=...)` 和 `AlphaFit.read(path, nfp=...)` 可读取保存的拟合记录。
`AlphaFit.radial_basis` 记录求值时使用的径向基，包括 `'flux_legendre'`，初始值计算
使用相同的基。保存的 alpha NPZ 文件可包含标量 `radial_basis`；旧文件采用 `'zernike'`。

## 准备 alpha+nu 初始值

```python
from coil_eval.simsopt import GpuCoilField, AlphaNuInitializer, InitializerConfig

capture = next(item for item in captures if item.valid)
with GpuCoilField(coils) as field, AlphaNuInitializer(
    field, config=InitializerConfig(device="cuda")
) as initializer:
    seed = initializer.prepare(capture.source, capture.alpha)
seed.save("surface_seed.npz")
```

`GpuCoilField(coils, library=None, device_id=0, segments_per_coil=256,
precision='fp32')` 是支持上下文管理器的磁场回调。输入为 `(N,3)` 的 Cartesian 坐标，
单位米；输出 B，单位特斯拉。也可选择 `precision='fp64'`。
`close()` 释放磁场资源，可重复调用。

`AlphaNuInitializer(field, library=None, config=None, device_id=None)` 接收 Cartesian B 回调，
`prepare(source, alpha, rho=1.0)` 返回 `SurfaceSeed`。
使用 `InitializerConfig(device='cuda')` 时，`device_id` 同时选择原生提取与 PyTorch
工作的 GPU。显式的 `device='cuda:N'` 在省略 `device_id` 时选择 N；两个显式设置
冲突时抛出 `ValueError`。两者均省略时仍使用 GPU 0；CPU 初始化保留所选 CPU 实现。
source、alpha 和磁场应对应相同的线圈几何与体积。捕获接口使用求值器原生 alpha 基；
单独提供的高阶 alpha 拟合可能产生不同初始值，比较时应使用相同目标体积。

| `InitializerConfig` 选项 | 默认值 |
| --- | --- |
| `surface_order`、`extract_order`、`nu_order` | `12`、`24`、`12` |
| `extract_maxiter`、`extract_tol` | `20`、`1e-12` |
| `fit_nphi`、`fit_ntheta`、`interpolation_size` | `65`、`67`、`97` |
| `regularization` | `0.0` |
| `device` | `'cpu'` |
| `nu_method`、`projection_method`、`inversion_method` | `'fft'`、`'cached_qr'`、`'resident'` |
| `geometry_method`、`geometry_cache_bytes` | `'reference'`、`536870912` |
| `reject_alpha_folds`、`max_radius_scale` | `True`、`1.0` |
| `alpha_fold_policy`、`alpha_min_jacobian` | `'reject'`、`0.05` |

重复使用初始值生成器可复用投影因子。`seed.diagnostics` 提供提取、映射、拟合和计时信息。
`SurfaceSeed.save` 和 `SurfaceSeed.read` 保留求解器使用的 alpha+nu 与 source-psi 类型标记。
初始化器生成的 seed 还保留 `source_axis_samples`，各列依次为
`(R, Z, dR/dphi, dZ/dphi)`，用于角坐标拓扑检查。读取早期文件时该数组可省略；
被接受的标准磁面文件也保留该数组。

`geometry_method='cached_basis'` 可选择缓存磁面几何基。
此开发版默认保留 reference 几何方法。

`alpha_fold_policy='damp'` 可在准备初始值时修正发生折叠的 alpha 坐标映射。
它用同一个系数缩小 lambda 扰动，保留 iota，并重新计算 nu。
映射会在至少 `97 × 257` 的 phi/theta 网格和实际提取节点上检查，
Jacobian 必须达到 `alpha_min_jacobian`。默认 `'reject'` 保留原有折叠检查。

```python
config = InitializerConfig(
    device="cuda", geometry_method="cached_basis",
    alpha_fold_policy="damp", alpha_min_jacobian=0.05,
)
```

`seed.diagnostics["alpha_transform"]` 记录缩放系数、原始和最终最小 Jacobian
以及验证网格。该选项调整初始坐标映射，原生评分和梯度不变。
生成的初始值仍应交给下文的标准求解与参考诊断。

## 在 CPU 上求解

```python
from coil_eval.simsopt import (
    SurfaceSeed, field_from_coils, StandardSimsoptSolver, surface_fc,
)

seed = SurfaceSeed.read("surface_seed.npz")
field = field_from_coils(coils)
surface = StandardSimsoptSolver().solve(seed, field, target_volume=0.05)
print(surface.accepted, surface.quality, surface.diagnostics)
if surface.accepted:
    surface.save("standard_surface.npz")
    reference = surface_fc(surface, field, sizes=(64, 128))
    print(reference["surface_fC_flux_shell_rms"])
```

`field_from_coils(coils, quadpoints=None)` 使用物理 Fourier 系数和安培电流构建 Simsopt
BiotSavart 磁场。`StandardSimsoptSolver(config=SolverConfig(...))` 先进行最小二乘，
再进行 Newton 修正。正的 `target_volume` 单位为 m³，求解器保留初始磁面的有向体积方向。
`validation_field` 可提供独立的 B 求值，用于最终几何检查。

`SolverConfig` 默认值为 `ls_tol=1e-10`、`ls_maxiter=100`、`newton_tol=1e-12`、
`newton_maxiter=30`、`constraint_weight=1.0`、`max_newton_residual_norm=1e-8`、
`max_final_relative_l2=1e-4`、`max_final_normal_p95=1e-4` 和 `validation_grids=(49,97)`。
`StandardSurface` 包含最终 dofs、iota、G、接受状态、质量和诊断。
`save()` 保存被接受的磁面；未被接受的候选几何仍保留在返回对象中，供用户检查。

初始值带有源磁轴采样时，求解器会在求解网格及各验证网格上检查角度覆盖。
相对源磁轴的顺时针极向角沿极向参数应绕行一次，沿一个环向场周期应绕行零次；
柱坐标环向角对应的绕行次数应分别为零和一次。
`hard_downstream_checks['single_angular_covering']` 记录这项接受检查，
各网格的 `angular_covering` 记录实测绕数。
可用 `SolverConfig(validation_grids=(97,129))` 加密检查。
缺少磁轴采样的早期初始值保留原有几何检查，并记录 `source_axis_unavailable`。
通过接受检查的磁面另以 `strict_pass` 或 `accepted_with_quality_warning` 报告稠密残差质量。

## 参考 QS 指标

`surface_fc(surface, field, sizes=(64,128))` 与 `analyze_arrays(...)` 实现螺旋度
`(1,nfp)` 的 QH 参考指标。`nfp=4` 时，对应原生求值的 `helicity=(1,4)`。
`surface_fc` 在指定网格上计算被接受的标准磁面，使用该螺旋度下与原生求值器一致的
归一化微分残差。返回内容包括：

- `surface_fC_flux_shell_rms`：按磁通壳层权重计算的 RMS。
- `surface_fC_area_rms`、`surface_fC_angular_rms`：面积权重与角度权重的 RMS。
- 实际包围体积、各网格数值、网格变化和 Boozer 残差诊断。

这些数值采用 RMS 定义。`qs_volume` 覆盖求值器所选的整个体积，`qs_outer` 覆盖其外侧
径向 bin，而求解磁面的磁通壳层 RMS 使用另一个平均区域。标定时应保持定义、权重和
实际包围体积一致。

`volume_average_fc(volumes, surface_fc_values, inner_fc=None)` 按实际包围体积对壳层 RMS
的平方积分，再开平方。体积需要递增，磁面需要属于同一嵌套分支。
应包含零体积处的值，或通过 `inner_fc` 显式指定未采样内层的近似值；返回记录会保留该近似。

## 分开运行 GPU 准备与 CPU 进程池

[simsopt_async.py](../examples/simsopt_async.py) 在一个进程中保存初始值，
再由独立 CPU 进程池求解：

```sh
python examples/simsopt_async.py prepare --directory output/case1
python examples/simsopt_async.py solve --directory output/case1 --workers 4
```

准备命令写完任务后退出。CPU 作业可以消费已完成目录，同时其他 GPU 作业继续准备新几何。
每个 CPU 进程使用一个 BLAS/OpenMP 线程并隐藏 CUDA 设备。
`--workers` 应与作业分配的 CPU 核数匹配。两阶段都保留各目标的成功或失败状态，
输入输出路径可通过命令行参数指定。
