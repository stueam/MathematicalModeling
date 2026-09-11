# 问题三

## Method 2：本地模拟器与 PPO

[method2 使用说明](method2/README.md) · [首轮实测报告](method2/reports/本地运行报告.md)

包含独立于 method1 的批量模拟器、PPO 训练/评估代码、测试、配置、首轮训练模型与日志。

- [最终模型](method2/runs/full_pilot/final.pt)
- [训练曲线](method2/reports/training_report.png)
- [路线示例](method2/reports/trajectory.png)

本地环境需要按使用说明自行安装。Python 虚拟环境、Numba/Python 缓存、临时检查点和后续训练目录不纳入版本控制。

当前结果来自本地模拟器，尚未接入官方 HTTP 接口。
