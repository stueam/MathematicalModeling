# 问题三

## 粒子信念与 POMCP-lite（第三问-算法2）

[方法二原理与结果说明](第三问-算法2/方法二原理与结果说明.md) · [运行说明](第三问-算法2/README.md)

使用 16 个进程进行完整配对世界模拟，组合保守几何、粒子信念、共享测量与开放路线。包含本地配对结果、既有官方演练成绩和三倍参数退步记录；后续改进仅做本地验证。相邻 `第三问-算法1/q3` 为必要公共内核，此版本与下方 PPO 实现分别保留。

## Method 2：本地模拟器与 PPO

[method2 使用说明](method2/README.md) · [首轮实测报告](method2/reports/本地运行报告.md)

包含独立于 method1 的批量模拟器、PPO 训练/评估代码、测试、配置、首轮训练模型与日志。

- [最终模型](method2/runs/full_pilot/final.pt)
- [训练曲线](method2/reports/training_report.png)
- [路线示例](method2/reports/trajectory.png)

本地环境需要按使用说明自行安装。Python 虚拟环境、Numba/Python 缓存、临时检查点和后续训练目录不纳入版本控制。

当前结果来自本地模拟器，尚未接入官方 HTTP 接口。
