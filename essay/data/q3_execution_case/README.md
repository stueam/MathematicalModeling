# Q3 本地案例轨迹与耗时构成

运行 `python generate.py`，从冻结数据生成唯一图件 `q3_execution_case.png`。需要 Python 3.10+、NumPy、Matplotlib；中文字体优先 Microsoft YaHei / SimHei。脚本会逐动作重新计算移动、检测、切频和清除费用，并核对所有时间戳及汇总费用。

该案例来自 2026-09-12 的历史本地验证（当时说明保存在 `src/Q3/VALIDATION.md`，可从 Git 提交 `cba9db1` 查看）：`bayes-fast`、seed 0、uniform 位置、iid 测角误差。选择既有验证案例，没有筛选最佳运行。15/15 个源成功清除且算法确认完成，虚拟时间 2730.530849 s，移动距离 10142.654245 m，121 次操作，最后一次清除后原地排查 30 s。它不是官方测试成绩，也不是多局平均性能。

## 数据

- `actions.csv`：原始本地运行的动作顺序、坐标、频道、反馈、虚拟时间和规划时间；仅移除不参与分析的请求编号和墙钟时间戳。
- `summary.json`：原运行汇总，包含完整参数、分项费用和完成状态。`real_time_s` 是该历史运行的墙钟时间，未在图中用作跨机器性能比较。
- `sources.csv`：按原场景配置在运行后重建的源坐标和接收半径，仅供审计；未传给策略。图中绘制成功清除位置，不将源真值误标为实际到访点。
- `provenance.json`：原日志哈希、24 个源码文件的哈希、选例说明和重放审计结果。
- `prepare_case.py`：兼容当前及历史 runner 的配置和文件名，先核对源码哈希，再逐动作重放。默认只审计；`--output` 导出到新目录，已有目录会被拒绝，不覆盖当前论文的冻结数据。

冻结该图时，原日志的 24 个源码文件与当时工作区一致，121 次操作逐条重放通过。`provenance.json` 保留的是该历史工作区的哈希，不是当前 `src/Q3` 的哈希。现有 Git 提交 `cba9db1` 中有 23 个文件匹配，另一个 `shared.py` 为路径迁移差异，因此不能把该提交声称为逐字一致的原工作区。当前程序重跑的动作和耗时应作为新案例，不能改写旧图的审计结论。

在仓库根目录运行：

```bash
python essay/data/q3_execution_case/prepare_case.py --run-dir <当前运行目录> --check
python essay/data/q3_execution_case/prepare_case.py --run-dir <当前运行目录> --output <新数据目录>
# 历史日志须提供与日志清单逐项匹配的原 src 目录；不允许绕过哈希检查。
python essay/data/q3_execution_case/prepare_case.py --run-dir <历史运行目录> --source-root <原src目录> --check
```

## 绘图约定

左图按所有已执行动作的坐标顺序连接路线，原地多次扫描只显示一个检测位置；数字 1—15 是成功清除的先后顺序，不是频道编号。箭头仅为方向提示。右图以线性横轴展示四类虚拟时间；清除合计 78 s，其中成功 75 s、失败 3 s。计时遵循模拟器规则：检测才更新接收频道，清除不额外收取切频费用。
