# 示例

[evaluate.py](evaluate.py) 加载 [coils.json](coils.json) 并求值。
[custom_loss.py](custom_loss.py) 组合标量目标并计算完整物理梯度。
脚本默认加载安装包内的后端。通过 `--library /path/to/libcoil_eval_native.so` 可指定自定义原生库，
`--help` 显示参数。

[autograd.py](autograd.py) 计算 PyTorch 目标与梯度，添加 `--batch` 使用独立几何批次。运行前安装 `torch` 可选依赖。

[qs_metrics.py](qs_metrics.py) 计算目标磁面的 f_B。添加 `--gradient` 计算局部参数梯度，
`--metric f_T` 同时计算三重积指标，`--accuracy high|balanced|fast` 选择角向分辨率。
CUDA 版 PyTorch 的配置见[安装文档](../docs/installation.zh-CN.md)。

[two_stage.py](two_stage.py) 选择可选的两阶段 psi 拟合。
使用 `--coils`、`--volume`、`--radius` 和 `--output` 指定输入与输出。
`--sampling polar_quadrature` 选择实验性的加权采样。
`--alpha-weighting smooth` 选择平滑的径向 alpha 权重；两阶段、autograd 和异步准备
三个示例均提供这两个选项。`autograd.py` 还支持 `--volume`、`--radius` 和 `--device`。
三个示例还支持 `--alpha-basis flux_legendre`，选择开发源码中的磁通多项式 alpha 基函数；
默认仍为 `zernike`。autograd 示例应同时使用 `--two-stage`。
它打印的一次求值加梯度时间覆盖完整调用与一次目标反传，首次运行可能包含冷启动成本。

```sh
python examples/two_stage.py --output output/metrics.json
python examples/autograd.py --two-stage --batch
python examples/autograd.py --two-stage --sampling polar_quadrature --alpha-weighting smooth
python examples/autograd.py --two-stage --sampling polar_quadrature --alpha-weighting smooth --alpha-basis flux_legendre
```

[simsopt_async.py](simsopt_async.py) 将 GPU 源捕获和 alpha+nu 准备，与 CPU 标准 Simsopt
求解分开运行。准备阶段需要 Simsopt 和 CUDA 版 PyTorch。
求解阶段通过 spawn 创建进程，每个进程使用一个 BLAS/OpenMP 线程并隐藏 GPU。

```sh
python examples/simsopt_async.py prepare --directory output/case1 --volumes 0.01 0.03 0.05
python examples/simsopt_async.py solve --directory output/case1 --workers 4
```

准备阶段保存有效任务和失败记录后退出。CPU 作业可以消费已完成目录，其他 GPU 作业则继续
准备新几何。`--workers` 应与分配的 CPU 核数匹配。
被接受的磁面提供磁通壳层 f_C RMS 诊断，保存的原生结果还包含体积和外侧 bin 的 QS RMS。
平均区域的约定见 [Simsopt API](../docs/simsopt.zh-CN.md)。

在 `prepare` 中添加 `--alpha-fold-policy damp --alpha-min-jacobian 0.05`，
可对发生折叠的 alpha 映射启用坐标阻尼。默认策略为 `reject`；
求解阶段仍独立检查每个最终标准磁面。
