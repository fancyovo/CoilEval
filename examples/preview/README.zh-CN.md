# 示例输入与历史验证

[English](README.md)

[example1.json](example1.json) 是教学示例使用的固定线圈：4 条基本线圈、傅里叶阶数 4、场周期数 nfp=3。
nfp 表示绕装置一周重复的场周期数；傅里叶系数以米为单位，电流以安培为单位。

两个独立入口是 [example1.py](../example1.py) 和 [example2.py](../example2.py)：

```sh
python examples/example1.py
python examples/example2.py
```

直接编辑脚本顶部参数和 Example2 的目标函数；完整使用说明见[示例指南](../README.zh-CN.md)。
0.11.1 已将旧的分片筛选/优化复现程序换成单次运行的教学示例。
旧报告的 116 次接受更新来自另一套完整目标和优化策略，不是当前教学脚本的预期轨迹。
旧计算与完整物理报告保留为[历史验证](../../docs/preview_validation.zh-CN.md)，原输入数值没有改变。
