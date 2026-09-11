# B5：分区推进与弹性检测、清除

B5 的策略、实验入口、测试、说明和实验记录统一放在 `Q3/B2/B5/`。公共环境、Q2 几何与选点模块、B4 服务成本函数仍从父目录复用。

- [算法与参数说明](B5算法说明.md)
- [实验结果与消融](B5实验结果.md)
- [发布与验证说明](B5发布说明.md)
- [策略代码](b5_strategy.py)
- [历史实验记录](evidence/)

在 `Q3/B2/` 中运行（需先安装父目录依赖）：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B -X utf8 -m unittest B5.tests.test_b5 -v
.\.venv\Scripts\python.exe -B -X utf8 B5/benchmark_b5.py --methods b2 q2 b4 b5 b5_global b5_single --seed 2026091122 --stratified35 --output outputs/b5_reproduce --plot
```

也可以在此目录用 `..\.venv\Scripts\python.exe benchmark_b5.py ...`，或在父目录用 `python -m B5.benchmark_b5 ...`。相对输入、输出路径以运行时工作目录为准。

统一策略入口为父目录的 `build_strategy(action, "b5")`。最终 35 场景实验中，B5 比原 B2 快、比 B4 慢；当前仍作为实验候选，默认策略保持 `latest`。
