# 运行示例

[English](README.md)

以下命令在解压的源码包根目录运行，先按[安装说明](../docs/installation.zh-CN.md)安装 CoilEval。
wheel 安装包提供库；源码包另外提供这里的脚本和线圈输入文件。

## 从哪里开始

- **示例 1：评估固定线圈。** [README 的完整命令](../README.zh-CN.md#example1)读取 `preview/example1.json`，在目标体积 0.3 m³ 上评分，可用 CPU 或 GPU。
- **示例 2：扰动后优化。** [筛选和优化命令](preview/README.zh-CN.md)先对示例 1 生成 32 组约 1% 的乘性扰动，再优化评分最优的一组；需要 GPU 和 CUDA 版 PyTorch。

## 其他接口示例

QS（准对称性）描述磁场强度在磁面角坐标中只依赖一个螺旋角的性质。
`f_B` 衡量偏离该对称性的傅里叶分量，`f_C` 是微分残差，`f_T` 是磁场导数三重积残差；三者理想值都为零。
默认先在初始区域拟合磁面标签，再在选出的闭合区域重新拟合。各示例采用这一流程。

| 脚本 | 用途与运行命令 |
| --- | --- |
| [evaluate.py](evaluate.py) | 读取 [coils.json](coils.json) 并评分：`python examples/evaluate.py` |
| [cpu.py](cpu.py) | CPU 评分：`python examples/cpu.py --threads 8` |
| [custom_loss.py](custom_loss.py) | 自定义标量目标及参数梯度：`python examples/custom_loss.py` |
| [autograd.py](autograd.py) | PyTorch 自动求导：`python examples/autograd.py`；加 `--batch` 评估独立批量 |
| [qs_metrics.py](qs_metrics.py) | 目标磁面的 f_B：`python examples/qs_metrics.py`；加 `--gradient --metric f_T` 计算 f_T 与梯度 |
| [two_stage.py](two_stage.py) | 显式演示两次拟合：`python examples/two_stage.py --output runs/metrics.json` |

GPU 脚本加载安装包中的计算库；`--library /path/to/libcoil_eval_native.so` 可选择自建库。
CPU 脚本的依赖和线程设置见 [CPU 指南](../docs/cpu.zh-CN.md)。PyTorch 脚本还需对应设备的 PyTorch。
各脚本的 `--help` 列出输入参数。`--two-stage` 是兼容旧命令的选项，0.11.0 已默认启用两次拟合。
`autograd.py --batch` 汇报每样本耗时，包含一次完整反向求导；首次调用包含初始化。

## 独立磁面验证

[simsopt_async.py](simsopt_async.py) 用 GPU 构造磁面及角坐标初值，再交给独立的 Simsopt 库在 CPU 上优化磁面。
准备阶段需要 CUDA PyTorch 和 Simsopt；求解阶段每进程使用一个线性代数/并行线程（BLAS/OpenMP）。

```sh
python examples/simsopt_async.py prepare --directory runs/case1 --volumes 0.01 0.03 0.05
python examples/simsopt_async.py solve --directory runs/case1 --workers 4
```

准备阶段保存已完成任务和失败记录；CPU 求解可以与其他几何的准备并行。`--workers` 不应超过分配的 CPU 数。
结果包含磁通薄壳上的 f_C 均方根（先平均平方、再开方），与整个体积内的值分开记录。
准备时可加 `--alpha-fold-policy damp --alpha-min-jacobian 0.05`，缩小角度修正以恢复单调映射；默认拒绝折叠映射。
该操作只改变初值，最终磁面仍需独立求解验收，详见 [Simsopt 指南](../docs/simsopt.zh-CN.md)。
