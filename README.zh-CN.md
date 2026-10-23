# CoilEval

[English](README.md)

CoilEval 使用 NVIDIA GPU 评估仿星器线圈。输入 Fourier 系数、电流、场周期数和目标体积，
获得准对称性、旋转变换、磁面质量及线圈几何指标。用户可以组合指标定义标量目标，并计算局部参数梯度。

## 安装

需要 Python 3.10+、NumPy 1.24+。GPU 后端在 Linux 上使用 CUDA、CMake 3.22+ 和 C++17 编译器构建；
cuBLAS 与 cuSOLVER 由 CUDA 提供。

```sh
python -m pip install .
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/native --parallel
```

按 GPU 型号选择 CUDA architecture；`120` 对应 RTX 5090。Python wheel 提供 Python 接口，
原生库从源码仓库或源码发行包构建，并将其路径传给 `Evaluator`。

## 快速开始

```python
from coil_eval import CoilSet, EvaluationConfig, Evaluator

coils = CoilSet.read_json("examples/coils.json")
config = EvaluationConfig(target_volume=0.05, psi_a=0.15, helicity=(1, 4))
with Evaluator("build/native/libcoil_eval_native.so", config=config) as evaluator:
    result = evaluator.evaluate(coils)
    if result.valid:
        print(result.metrics.qs_volume, result.metrics.iota_boundary)
    else:
        print(result.status)
```

完整命令示例：

```sh
python examples/evaluate.py --library build/native/libcoil_eval_native.so
python examples/custom_loss.py --library build/native/libcoil_eval_native.so
```

## 文档

- [API 与参数约定](docs/api.zh-CN.md)
- [示例](examples/README.zh-CN.md)
- [开发与源码结构](docs/development.zh-CN.md)
- [版本记录](docs/changelog.zh-CN.md)

CoilEval 使用 [MIT 许可证](LICENSE)。
