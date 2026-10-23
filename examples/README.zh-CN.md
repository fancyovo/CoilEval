# 示例

[evaluate.py](evaluate.py) 加载 [coils.json](coils.json) 并求值。
[custom_loss.py](custom_loss.py) 组合标量目标并计算完整物理梯度。
两个脚本均通过 `--library /path/to/libcoil_eval_native.so` 指定原生库，`--help` 显示参数。
