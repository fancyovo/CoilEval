# 固定 QA 预发布示例

[English](README.md)

`example1.json` 是原始四条基本线圈（Fourier四阶、nfp3、stellarator symmetry），
已转换为公开 CoilEval JSON 格式。物理数组SHA256为
`aebf705719b77ad064624362f5d90601dc4b8a2cfbaf3f8240b427faf8e1eaf7`。
使用QA `(1,0)`、V=0.3m³、拟合半径a=√0.06m，统一采用0.11.0的S2默认配方。
完整参考流程结果及限制见[验证记录](../../docs/preview_validation.zh-CN.md)。

Example2对108个Fourier系数独立取`c*exp(log1p(fraction)*z)`，初始fraction=0.01、
seed2026110502。32个候选均从原example1出发，符号和精确零不变，筛选时电流不扰动。
后续优化沿用原BFGS的112个物理参数，包括电流；随附数据集统计给出固定的参数归一化。

在解压后的源码目录中，安装GPU依赖组和CUDA PyTorch后运行：

```sh
python examples/preview/perturb_optimize.py screen --output runs/example2-screen
python examples/preview/perturb_optimize.py optimize --screens runs/example2-screen/screen.json --output runs/example2-optimize
```

可选`--library /path/to/libcoil_eval_native.so`指定自定义库。
有多张已分配GPU时，分别运行`screen --shards 4 --shard I`（I=0,1,2,3），
各进程分配一张卡和不同输出目录，再把四份`screen.json`传给`optimize --screens`。
程序要求恰好32个互不重复的候选编号后才能选点。后续优化是一条有依赖的单链。

按完整目标选有效候选的最优者：log体积f_C、五个条件惩罚、法向漂移质量和可导磁轴平均半径代价。
目标依次为`[0.47,0.1,6.0,3.5,0.3]`，对应ι、间距(m)、最大曲率(1/m)、平均长度(m)、体积(m³)。
K不进入目标。一次解析反传给出完整梯度；稠密逆BFGS使用Powell阻尼和仅求值的Armijo回溯。
默认最多400外迭代并有连续拒绝停止条件，这些预算停止条件不构成收敛认证。

`screen.json`保留全部候选状态；`evaluations/`、`history.json`、`result.json`、
`start.npz`和`selected.npz`保留试探及完整目标最优线圈。输出必须使用新目录，保留之前的失败。

如果物理尝试跑不通，逐档减半`--fraction`（0.005、0.0025……）并保留旧输出。
每档仍抽取同一32份高斯噪声，只改变幅度，目标体积和拟合半径保持不变。
输入格式或环境错误应先纠正，不能把它们算成噪声过大的失败。

发行验证另将所选线圈通过source psi→α→ν→标准Simsopt LS/Newton→诊断→DESC。
原生优化成功与独立物理/参考验收分别报告。
