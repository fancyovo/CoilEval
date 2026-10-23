# 示例

[evaluate.py](evaluate.py) 加载 [coils.json](coils.json) 并求值。
[custom_loss.py](custom_loss.py) 组合标量目标并计算完整物理梯度。
脚本默认加载安装包内的后端。通过 `--library /path/to/libcoil_eval_native.so` 可指定自定义原生库，
`--help` 显示参数。

[autograd.py](autograd.py) 计算 PyTorch 目标与梯度，添加 `--batch` 使用独立几何批次。运行前安装 `torch` 可选依赖。
