# 开发

[English](development.md)

```sh
python -m pip install -e '.[test]'
python -m pytest -q
python tools/check_layout.py
python -m build
```

| 目录 | 职责 |
| --- | --- |
| `src/coil_eval` | Python API、结果、配置和原生接口 |
| `native/src/field` | 线圈离散、磁场、追踪和 psi 拟合 |
| `native/src/pipeline` | 磁轴、目标磁面、磁通、alpha、QS 和近邻求值 |
| `native/include` | 共享类型与模块接口 |
| `tests` | API、数值 oracle 和源码结构测试 |
| `examples` | 求值与自定义目标示例 |

实现文件独立编译；头文件声明接口、共享类型和小型内联函数。
磁场状态由 `field/state.cpp` 管理，求值会话状态由 `pipeline/session_state.cu` 管理。
采用新增配置项和方法保持兼容；更新 CHANGELOG 和中英文文档。
版本标签为 `v` 加包版本号，预发布版本使用 `0.3.0a1` 等 Python 版本形式。
旧标签保留用于复现安装。
