# CoilEval

[English](README.md)

CoilEval 根据仿星器线圈的 Fourier 系数、电流、场周期数和目标体积，在 NVIDIA GPU 或 CPU
上计算准对称性、旋转变换、磁面质量与线圈几何指标，并支持自定义目标的局部参数梯度。

**0.11.0 预发布候选版**统一采用已验证的 S2 默认配方：两次 s 拟合、fast/Stokes、
极坐标求积、平滑体 α 权重及 flux-Legendre 体 α 基。保留已有角反解和径向首支求根修复。
S3 通过 `psi_target_refit=True` 显式开启，默认关闭。

## 安装与运行

Linux x86-64、glibc 2.28+，Python 3.10+。从候选发行包安装：

```sh
python -m pip install "./coil_eval-0.11.0-py3-none-manylinux_2_28_x86_64.whl[gpu]"
```

GPU 依赖组使用已验证的 CUDA 12.8 运行库；驱动须支持 GPU 和 CUDA 12.8。
CPU 用户先安装 CPU PyTorch，再安装 wheel 的 `[cpu]` 依赖组。
详细依赖、GPU 架构及源码构建见[安装指南](docs/installation.zh-CN.md)。

```python
from coil_eval import CoilSet, EvaluationConfig, Evaluator

coils = CoilSet.read_json("examples/coils.json")
config = EvaluationConfig(target_volume=0.05, psi_a=0.15, helicity=(1, 4))
with Evaluator(config=config) as evaluator:
    result = evaluator.evaluate(coils)
    if result.valid:
        print(result.metrics.qs_volume, result.metrics.iota_boundary)
    else:
        print(result.status)
```

CPU 使用 `Evaluator(config=config, device="cpu", threads=8)`。
QA 显式指定 `helicity=(1,0)`，常见 QH 使用 `(1,coils.nfp)`；兼容默认值仍为 `(1,4)`。
张量输入不携带场周期元数据，`TensorEvaluator` 需传入 `nfp=coils.nfp`。

运行 autograd 和目标面 QS 示例前，还需安装 CUDA 版 PyTorch。源码包中的入口：

```sh
python examples/evaluate.py
python examples/autograd.py --batch
python examples/qs_metrics.py --gradient --metric f_B
```

体积 `qs_volume` 与最外径向层 `qs_outer` 是各自区域的微分 f_C RMS。
目标面 Fourier f_B、三重积 f_T、Simsopt 参考及 DESC 结果使用各自明确的平均和归一化，
见[QS 指南](docs/qs.zh-CN.md)和[参考定义](docs/simsopt.zh-CN.md)。

## S2 精度与速度

固定 1024 个 QUASR 输入、四个噪声尺度，共 **4096** 个任务。保留已有修复的 S2
得到 **2127** 个完整评分；与冻结 Simsopt 参考可配对 **1944** 个，其中 **817** 个满足
原定参考质量门槛。所有大误差点均保留。1981 个完成的参考不重算；无参考样本只计成功率。

![CoilEval S2 与 Simsopt 的 QS、ι 散点及各样本集实测 P90 条带](docs/assets/preview_s2_20261105/calibration_p90.png)

图中 QS 条带为 `exp(±P90(|ln(预测/参考)|))`，ι 条带为 `±P90(|预测−参考|)`。
它描述本样本集的误差覆盖范围。QS 全配对的实测倍数依次为 1.986、1.368、1.157；
参考质量合格集为 1.926、1.304、1.122。表格另列易读的相对误差 P90，两者定义不同。

| 样本集 | 数量 | f_B P90 相对误差 | f_C P90 相对误差 | f_T P90 相对误差 | ι P90 绝对误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全部配对 | 1944 | 93.86% | 34.67% | 15.20% | 0.01270 |
| 参考质量合格 | 817 | 87.04% | 30.17% | 12.17% | 0.01015 |

以下全部单位为 **秒/样本**：一次预热后计时三次完整调用，取中位数，batch 再除以批大小。
“求值＋梯度”包含完整目标的一次反传。batch 均摊耗时反映吞吐，单样本响应延迟见 scalar。

| 接口 / 指标 | 模式 | CPU 8 线程 | RTX 5090 | A100 |
| --- | --- | ---: | ---: | ---: |
| scalar / volume f_C | 求值 | 22.922 | 1.775 | 1.911 |
| scalar / volume f_C | 求值＋梯度 | 40.000 | 2.168 | 2.397 |
| batch 1 / volume f_C | 求值 | 22.843 | 1.785 | 1.915 |
| batch 1 / volume f_C | 求值＋梯度 | 42.364 | 2.159 | 2.410 |
| batch 2 / volume f_C | 求值 | 31.186 | 1.502 | 1.606 |
| batch 2 / volume f_C | 求值＋梯度 | 50.412 | 1.572 | 1.823 |
| batch 4 / volume f_C | 求值 | 30.449 | 1.347 | 1.453 |
| batch 4 / volume f_C | 求值＋梯度 | 47.093 | 1.418 | 1.633 |
| batch 8 / volume f_C | 求值 | 30.669 | 1.296 | 1.396 |
| batch 8 / volume f_C | 求值＋梯度 | 47.288 | 1.375 | 1.547 |
| scalar / surface f_T | 求值 | 36.701 | 2.024 | 2.057 |
| scalar / surface f_T | 求值＋梯度 | 56.714 | 2.585 | 2.584 |

计时复用 2026-11-05 的同一 S2 配方和原生库实测，四个原始 order-16 线圈组构成固定八查询任务，
每个硬件分配八个逻辑 CPU。普通体积 f_C 和附加目标面 f_T 分开列出；
冷启动、整批总时间、硬件及哈希保留在[计时原始证据](docs/assets/preview_s2_20261105/timings.json)。
样本定义、失败分母、参考质量与数值限制见[完整验证记录](docs/preview_validation.zh-CN.md)。

## 示例与使用边界

Example1 在原 QA 线圈、V=0.3 m³、a=√0.06 m 下，用 S2 重新跑通了完整 source→α→ν→Simsopt→DESC
流程。DESC 优化器达到 gtol，最终面保持嵌套；残余力仍有可见误差，平衡解认证为 false。
Example2 从 example1 的系数取 `c·exp(ln(1.01)z)` 独立扰动，筛选时固定电流，抽样 32 个，
选完整目标最优的有效样本进入解析梯度 BFGS。1% 下 25/32 可评分，接受 116 次更新后目标
从 −3.6904 降到 −5.4891，并完成最终完整参考流程。DESC 残余力警告及停止原因见
[验证记录](docs/preview_validation.zh-CN.md)。

S3 的固定 4096 对照得到 1821 个评分，相对 S2 丢失 314 个、恢复 8 个，因此保留可选。
S2 的剩余误差包括全域 s 拟合目标和正则偏差、角方程离散误差、共振附近误差放大，以及坐标表示限制。
光滑的庞加莱图不足以保证低 QS 误差；有限诊断通过也不构成全局磁面认证。

## 文档

- [API 与默认配置](docs/api.zh-CN.md)
- [PyTorch 梯度与批次](docs/torch.zh-CN.md)
- [CPU](docs/cpu.zh-CN.md)及 [Simsopt](docs/simsopt.zh-CN.md)
- [运行示例](examples/README.zh-CN.md)、[版本](docs/versions.zh-CN.md)、[更新日志](docs/changelog.zh-CN.md)
- [开发说明](docs/development.zh-CN.md)

许可证：[MIT](LICENSE)。
