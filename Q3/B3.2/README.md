# B3.2：问题三干扰源搜索与清除

本目录是团队保存的 **B3.2 原版**，默认方法名为 **`current`**，对应 `candidate.OptimizedB3`。`latest` 是保留的较早基线名称；运行 B3.2 时请明确使用 `--methods current`。

全部原始 Python 源码、依赖文件、测试和 20 局场景均按保存的 SHA-256 清单逐字节复制。`source_manifest.json` 可用于核验；原来的说明保留为 `README_original.md`，其中的本机路径仅是历史记录。本次发布没有修改算法或调参。

## 安装与离线运行

在本目录打开终端，建议使用 Python 3.12：

```sh
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
python benchmark.py --methods current --output outputs/b32_check
```

最后一条命令使用随附的 `scenarios/screening20.json` 跑 20 局离线检查，将逐局指标和汇总写入输出目录的 `cases.csv` 与 `summary.json`。这些场景属于开发检查集，成绩不能当作独立泛化评估。

上述命令使用本地环境回调，不连接官方模拟器，不使用正式测试次数。目录不包含队号、密码或自动连接官方模拟器的启动器。

## 调用入口

```python
from solver import build_strategy

# action 是环境提供的测量/清除回调，协议见 offline_environment.py。
strategy = build_strategy(action, method="current")
strategy.run()
```

算法只通过回调取得观测结果；离线环境持有源坐标等真值。对接其他环境时应实现同一回调协议。

## 方法概览

- 原点扫描频道，按示向度维护各源的几何可行区域。
- 使用 150 米谱方向试探及已有观测缩小定位区域。
- 结合覆盖扫描点与已有源的位置估计安排搜索顺序，复用沿途观测。
- 使用问题二的几何定位与局部观测选择，在信息足够时清除源。
- 在可行区域能够被清除半径覆盖时，优化较近的清除位置，减少行走。

关键实现是 `candidate.py`，工厂与基础配置在 `solver.py`。定位与路径的继承实现分布在 `q2_lookahead.py`、`peripheral_prior.py` 及其基础模块中，因此应保留本目录的完整依赖文件。

## 版本与评价口径

本目录单独保存 B3.2；仓库 `Q3/B2` 的方法和附属 B5 等目录有各自的版本含义，不能仅凭文件夹名判定算法相同。

评价时首先检查全部源是否清除及是否错误排除频道，再计算每局总虚拟时间除以源数，并对各局等权平均。应同时报告慢局与失败，不能只挑低于 200 秒/个的场景。本目录保留的算法并不保证每局低于 200 秒/个。

`candidate_provenance.json` 是原始开发记录；发布时的原版文件一致性以 `source_manifest.json` 为准。
