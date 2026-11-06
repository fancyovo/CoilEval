# 示例 1 与示例 2

[English](README.md)

两个示例都从 [example1.json](example1.json) 的固定线圈开始。该输入有 4 条基本线圈，每条曲线的傅里叶展开保留到 4 阶；
场周期数 `nfp=3`，利用旋转和仿星器反射对称性展开完整装置。
目标是准轴对称（QA）：磁场强度在合适的磁面角坐标中与环向角无关，设置 `helicity=(1,0)`。
目标体积为 0.3 m³，初始拟合半径为 √0.06 m。0.11.0 默认先拟合磁面标签，再在选出的闭合区域重新拟合。

## 示例 1：评估固定线圈

先按[安装指南](../../docs/installation.zh-CN.md)安装 CPU 或 GPU 依赖。
在解压后的源码包根目录，使用 [README 中的完整 Python 命令](../../README.zh-CN.md#example1)评估这组线圈。
数值、独立参考结果和误差限制见[验证报告](../../docs/preview_validation.zh-CN.md)。

## 示例 2：32 次扰动筛选后优化

脚本需要 GPU、CUDA 版 PyTorch 和 CoilEval 的 GPU 依赖。在源码包根目录运行：

```sh
python examples/preview/perturb_optimize.py screen --output runs/example2-screen
python examples/preview/perturb_optimize.py optimize --screens runs/example2-screen/screen.json --output runs/example2-optimize
```

第一条命令生成并评分 32 个候选；第二条读取全部结果，选择有效候选中目标函数最小的一组，开始优化。
可选 `--library /path/to/libcoil_eval_native.so` 指定自建库。输出目录须为新目录，以保留此前运行。

每个几何傅里叶系数 c 独立变为 `c * exp(log1p(0.01) * z)`，z 是标准高斯随机数。
这相当于对数扰动的标准差为 ln(1.01)，保留系数的符号与精确零值。固定随机种子为 2026110502。
32 个候选都从原示例 1 产生；筛选阶段保持电流、体积和拟合半径固定。
优化阶段允许改变 108 个几何系数和 4 个电流，共 112 个参数；随包统计文件提供固定的参数归一化尺度。

目标函数合并体积准对称性微分残差 f_C 的对数，以及旋转变换、线圈间距、曲率、长度、体积、法向漂移和磁轴半径代价。
旋转变换 ι 是磁力线每绕环向一圈平均绕极向的圈数；法向漂移衡量磁力线偏离拟合磁面的程度。
五个条件目标依次为 ι=0.47、最小间距 0.1 m、最大曲率 6 m⁻¹、平均线圈长度 3.5 m、体积 0.3 m³。

BFGS 是利用梯度逐步更新曲率近似的准 Newton 优化法；本例保存完整的逆曲率矩阵。
Powell 阻尼稳定矩阵更新，Armijo 回溯逐步缩短试探步直到目标充分下降。梯度由一次反向求导给出，试探点重新完整评分。
默认最多 400 次外迭代，连续拒绝试探点也会停止；达到停止条件不表示获得全局最优或严格物理认证。

`screen.json` 保存 32 个候选的状态；`history.json`、`evaluations/` 保存优化历史和试探结果，
`result.json` 给出停止原因，`start.npz` 和 `selected.npz` 保存起点与所选最优参数。
本次验证在 1% 扰动下有 25/32 个有效候选，优化接受 116 次更新后因连续拒绝停止，无须降低扰动。

若有效环境中的数值运行不能完成，可把 `screen --fraction` 依次改为 0.005、0.0025 等，在新目录重新运行全部 32 个候选，
保留相同随机种子和随机数；体积、半径不变。环境或输入格式错误应先修正。运行结果仍需按其实际状态判断。

有多张已分配的 GPU 时，可对每张运行 `screen --shards 4 --shard I`（I=0,1,2,3），每次指定独立输出目录和 GPU。
优化命令的 `--screens` 接受全部四个 `screen.json` 路径，必须收齐 32 个不同编号。优化本身是前后依赖的一条迭代链。

发行验证还用 Simsopt 独立优化磁面，再用 DESC 求解磁流体平衡；它们是独立的物理计算库。
评分/优化跑通与这些参考解的残差、收敛和认证状态分别汇报。
原始物理数组 SHA256 为 `aebf705719b77ad064624362f5d90601dc4b8a2cfbaf3f8240b427faf8e1eaf7`，用于核对输入一致性。

减小噪声后，`screen` 和 `optimize` 两条命令都必须传入相同的 `--fraction`，例如 `--fraction 0.005`。
