# CPU 求值

[English](cpu.md)

本页介绍 CPU 安装、线程设置和求导。**QS** 是准对称性，即磁场强度在磁面角坐标中只依赖一个螺旋角；`f_B`、`f_T` 分别是其傅里叶分量残差和磁场导数三重积残差。**QA/QH** 分别指准轴对称/准螺旋对称。`nfp` 是场周期数；仿星器对称性通过旋转和反射由基本线圈生成完整装置。

默认先拟合磁面标签 s，再在选出的闭合区域内重拟合；接口沿用 `psi` 名称。α（alpha）拟合构造沿磁力线的角坐标，Boozer 坐标进一步使磁力线表示为直线并满足磁场协变角分量的约束。“捕获”是保存一次完整求值的中间状态；“中心”是原参数，“端点”是有限差分的小扰动参数。QR 是求解最小二乘的正交/三角矩阵分解。BLAS 是线性代数库，OpenMP 是 CPU 并行框架，AVX2 是 CPU 向量指令集。

CPU 后端沿用 GPU 后端的线圈离散、磁轴搜索策略、拟合基函数和积分采样规模，
支持默认两次拟合、根据近轴磁场预测取样区域的 `psi_fit='near_axis_ellipse'` 和历史圆域单次拟合，
并提供线圈几何指标、有限差分与 PyTorch 局部反传梯度。
可选的目标磁面 `f_B` 和 `f_T` 复用 Boozer 坐标实现。

## 安装

先安装 CPU 版 PyTorch，再安装 CoilEval wheel（免本地编译的 Python 安装包）和 CPU 依赖：

```sh
python -m pip install "torch==2.7.1+cpu" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "coil_eval-0.12.0-py3-none-manylinux_2_28_x86_64.whl[cpu]"
```

请将 wheel 文件名换成实际收到的文件。CPU 求值加载 `libcoil_eval_cpu.so`，
不需要 NVIDIA 驱动或 CUDA toolkit。源码构建方式见[安装说明](installation.zh-CN.md)。

## 求值

```python
from coil_eval import CoilSet, EvaluationConfig, Evaluator

coils = CoilSet.read_json("examples/coils.json")
config = EvaluationConfig(target_volume=0.05, fast=True)
with Evaluator(config=config, device="cpu", threads=8) as evaluator:
    result = evaluator.evaluate(coils)
    print(result.status, result.metrics, result.timings)
```

`device` 默认仍为 `'cuda'`。CPU 的 `threads` 接受正整数；省略时根据当前进程
可用 CPU、处理器数量及 `SLURM_CPUS_PER_TASK` 自动选择。`library=` 可指定
自定义 CPU 动态库，环境变量 `COIL_EVAL_CPU_LIBRARY` 提供默认自定义路径。

`axis_grid` 接受不小于 4 的整数；省略时保留原搜索设置。
例如 `axis_grid=24` 或 `axis_grid=32` 将磁轴初筛网格从默认每方向 48 个点缩小。
既有的 64×64 回退策略、Newton 迭代和拓扑检验保持不变。
较稀网格可能选择不同磁轴分支，采用前应检查目标样本集。
诊断包含实际网格、种子数量、是否使用回退和磁轴各阶段计时。
三个 CPU evaluator 均支持此选项。

Newton 迭代利用导数求解磁轴方程；椭圆型磁轴候选是附近磁力线绕其旋转的周期轨道。
CPU 磁轴搜索使用双精度检验周期闭合误差。如果所有候选都未通过验收，但某个
椭圆型候选已经接近要求的容差，会再做至多两步位移受限的双精度 Newton 修正，
然后重新执行原有的周期闭合与拓扑检验。已有合格磁轴时保留原选择。
`axis_fp64_polish` 诊断记录是否修正、位移以及未通过的原因。

`CoilSet.nfp` 接受任意正整数，基本线圈数量也可为任意正整数，并要求保留
仿星器对称性。`EvaluationConfig` 的螺旋度仍需显式指定：QA 用 `(1,0)`，
对应场周期的 QH 用 `(1,coils.nfp)`。即使改变 `nfp`，默认螺旋度仍为 `(1,4)`。
整数对 `(M,N)` 指定角组合 `Mθ−Nφ`，θ 为极向角，φ 为环向角，N 对应完整环向一圈。

`evaluate_many()` 接受不同 `nfp`、基本线圈数量和 傅里叶系数数组宽度的线圈集合，
按输入顺序返回结果。CPU 批量模式依次处理样本，每个样本的
磁场计算和线性代数使用所指定的线程数。若使用多个独立工作进程，应把 CPU
配额分配给各进程的 `threads`。

CPU 的 `loss_and_grad()` 在中心磁轴分支上重新计算完整的正式拟合，执行中心
有限差分。端点诊断记录 `nearby_method='formal'`、`nearby_backend='cpu'`。
目标函数可用 Tensor 指标表达时，优先使用 PyTorch 反传。

## 反传梯度

```python
import torch
from coil_eval.torch import TensorEvaluator, pack_coils

config = EvaluationConfig(target_volume=0.05, fast=True, flux_method="stokes")
x = pack_coils(coils, device="cpu")
with TensorEvaluator(config=config, device="cpu", threads=8, nfp=coils.nfp) as evaluator:
    result = evaluator.evaluate(x)
    loss = torch.log(result.metrics.qs_volume + 1e-8)
    gradient, = torch.autograd.grad(loss, result.parameters)
```

Tensor 接口接受任意正整数 `nfp` 和基本线圈数量。请显式传入 `nfp=coils.nfp`：
张量不包含场周期元数据，评估器默认仍为 4。接口保留仿星器对称性、
每条线圈 256 段及四次 psi/alpha 局部拟合迭代。
`evaluate_batch()` 要求统一的 `nfp`、基本线圈数量和 傅里叶系数数组宽度，
返回堆叠指标及相互独立的梯度。各查询的配置可指定不同螺旋度，包括 QA `(1,0)`；
`M` 必须非零。
`surface_qs=SurfaceQSConfig(...)` 启用可选磁面指标。
`batch_axis` 和 `share_geometry` 是 GPU 准备阶段选项；CPU 批量逐个准备查询。

仅需数值时可用 `SurfaceQSEvaluator(config=config, device='cpu', threads=8)`。
其 `evaluate(coils)` 和 `evaluate_many(coils)` 为每个查询进行一次正式拟合，
返回包含普通浮点数的 `SurfaceQSResult`；数值批次可混用不同场周期和线圈布局，
并保留输入顺序。

请将前向和反传都放在 evaluator 的上下文管理器内，以使用同一线程预算。
PyTorch 和 BLAS 的线程上限对整个进程生效，退出上下文后恢复。
需要不同线程预算的并发会话时，请使用独立进程。

CPU 的正式拟合使用 Householder QR，反传使用相同的四步局部响应。
CPU 与 GPU 的归约次序可能不同；比较数值误差、梯度方向和范数即可，不要求逐位一致。

启用 OpenMP 的 CPU 构建会并行计算相互独立的磁场点和磁力线。
在支持的系统上自动使用 AVX2，其他 CPU 使用标量实现。

运行 `python examples/cpu.py`，在脚本顶部修改 `THREADS`、体积和半径。脚本打印主要指标与总耗时；梯度示例见 `examples/autograd.py`，将其 `DEVICE` 改为 `"cpu"`。

CPU 8、16、32 逻辑线程的同型号实测比较见 [README 速度表](../README.zh-CN.md#validation)，单位为秒/样本。
