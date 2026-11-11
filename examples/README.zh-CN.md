# 运行和修改示例

[English](README.md)

先按[安装说明](../docs/installation.zh-CN.md)安装库，再从解压的源码包根目录运行下面的脚本。
每个示例都自包含：**直接修改文件顶部的参数，再运行 `python examples/文件名.py`**。
脚本不接受 `--` 参数，也不要求先运行另一个脚本。线圈文件默认是随包的 [example1.json](preview/example1.json)。

| 脚本 | 展示的主要对象与用途 |
| --- | --- |
| [example1.py](example1.py) | `CoilSet`、`EvaluationConfig`、`Evaluator`：读入、求值、打印指标与计时、保存结果；默认 CPU |
| [example2.py](example2.py) | `TensorEvaluator` + SciPy BFGS：32 次筛选接优化，每候选、每步实时输出；默认 GPU |
| [evaluate.py](evaluate.py) | 最短的 GPU 单点求值 |
| [cpu.py](cpu.py) | CPU 单点求值，顶部 `THREADS` 控制线程数 |
| [autograd.py](autograd.py) | `pack_coils`、Torch 目标函数及自动求导 |
| [batch.py](batch.py) | `evaluate_batch`：两个独立输入和各自梯度，汇报秒/样本 |
| [custom_loss.py](custom_loss.py) | NumPy 目标函数与 `loss_and_grad` 的局部有限差分 |
| [qs_metrics.py](qs_metrics.py) | `SurfaceQSConfig`、`SurfaceQSEvaluator`：目标磁面的 f_B、f_T |
| [two_stage.py](two_stage.py) | `evaluate_many`：在默认两次拟合流程下评估三个目标体积 |
| [simsopt_async.py](simsopt_async.py) | 一次运行 GPU 初值准备和 CPU 进程池的独立磁面求解 |

例如：

```sh
python examples/example1.py
python examples/example2.py
```

这些是两个独立入口；Example2 直接读取原始线圈，不需要 Example1 的输出。
全部示例打印数值和耗时。首次运行含初始化；计时范围在输出中注明，不应直接当作预热后的测速结果。
重新运行会更新同一输出文件；修改 `OUTPUT` 可保留旧结果。

如需试用椭圆域单次拟合，将任一脚本的 `PSI_FIT` 设为 `"near_axis_ellipse"`，
并将 `TARGET_VOLUME` 设为 `0.15`。该方法根据磁轴附近的磁场预测取样区域。
随附线圈用单次拟合时，标量在达到 0.3 m³ 前折返。可用拟合方式见 [API 指南](../docs/api.zh-CN.md)。

## 修改 Example2

`N_STARTS=32`、`NOISE=0.01`、`SEED=2026110502` 控制筛选。扰动是 `c*exp(log1p(NOISE)*z)`，
z 为标准高斯数；系数的符号和精确零值保持，筛选时电流保持不变。`MAX_STEPS=100` 是迭代上限。
在 `objective(metrics)` 中直接修改目标和权重：默认 ι=0.47、最小间距 0.10 m、最大曲率 6 m⁻¹、平均长度 3.5 m。
体积由 `EvaluationConfig` 固定为 0.3 m³。`GEOMETRY_SCALE`、`CURRENT_SCALE` 将优化坐标换算为米、安培。
优化时系数和电流均可改变；如果筛选完全失败，可减小 `NOISE` 再运行。

QS 是准对称性残差，理想值为零；ι（旋转变换）是磁力线每绕环向一圈平均绕极向的圈数。
法向漂移描述磁力线偏离拟合磁面的程度。`f_B` 用磁场强度的傅里叶分量，`f_T` 用磁场导数三重积衡量目标磁面的准对称性。
BFGS 是利用梯度更新曲率近似的优化算法；见 [SciPy 接口说明](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html)。
`history.csv` 保存每个接受迭代的目标值、QS、ι 和耗时；`optimized_coils.json` 可再次用 `CoilSet.read_json` 读取。

## CoilEval 0.12.0 的 Example2 实跑结果

两次运行均在 RTX 5090 上完成筛选并保存 `optimized_coils.json` 和 `history.csv`。
SciPy 的 `success` 表示是否达到收敛条件。默认配置因精度损失停止，椭圆域配置达到 100 步迭代上限。

| 拟合方式 | 目标体积 | BFGS 步数 | 初始 → 最终目标值 | 最终体积 QS | 最终 ι | 总耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 两次拟合 | 0.30 m³ | 92 | −2.763355 → −5.471306 | 0.00409435 | 0.469690 | 449.83 秒 |
| 椭圆域单次拟合 | 0.15 m³ | 100 | 1.112674 → −6.261718 | 0.00184542 | 0.469527 | 915.24 秒 |

总耗时包含筛选、线搜索试步和输出进度。两组示例使用不同目标体积。同一批输入的评估器耗时见[预热测速](../README.zh-CN.md#validation)。

## 依赖和独立磁面求解

CPU 示例需要 CPU 版 Torch；GPU 梯度/优化示例需要 CUDA 版 Torch，Example2 还需要 SciPy。
`simsopt_async.py` 另外需要 Simsopt（独立磁面优化库），顶部 `WORKERS` 指 CPU 工作进程数。
脚本在 GPU 上用 `AlphaNuInitializer` 构造角坐标初值，再由 `StandardSimsoptSolver` 在 CPU 上求解；
输出接受状态、质量等级、旋转变换和时间，保存初值及被接受的磁面。详见 [Simsopt 说明](../docs/simsopt.zh-CN.md)。
这里没有执行 DESC 平衡计算。旧预发布中的完整物理报告属于历史复现实验，详见[验证记录](../docs/preview_validation.zh-CN.md)。
