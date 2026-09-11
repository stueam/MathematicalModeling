# B2：问题三几何定位与联合调度优化

本目录按项目约定命名为 **B2**，包含原始 B2 基线和最新优化候选。统一入口为 `solver.build_strategy`；最新候选内部类名为 `PeripheralProposal`，继承 `FlexibleB3`。这些内部名称表示迭代关系，不是另一道题。

## Q2 期望选点增强（2026-09-11）

新增可选方法 `mec`（最小包围圆）和 `q2`（最小包围圆 + 期望时间补测选点）；默认仍为 `latest`。在 140 个新种子场景中，三者均完成全部 1820 个源，`q2` 的平均每源虚拟耗时由 239.21 s 降至 235.74 s，改善约 1.45%；98 局更快，42 局更慢。单独 `mec` 基本持平。本地模拟结果不代表官方评测保证。

```powershell
python -B -X utf8 benchmark.py --methods latest mec q2 --generate-seed 2026091101 --output outputs/q2_reproduce
python -B -X utf8 analyze_q2.py outputs/q2_reproduce
```

完整方法、假设、敏感性和运行命令见 [Q2 融合实验说明](Q2融合实验说明.md)。[逐场结果](evidence/q2_fresh140_20260911/cases.csv)、[配置与摘要](evidence/q2_fresh140_20260911/summary.json)、[配对统计](evidence/q2_fresh140_20260911/paired_analysis.json)及[对比图](evidence/q2_fresh140_20260911/comparison.png)已归档。策略接口使用 `build_strategy(action, method="q2")`；仍只接收观测反馈，未给未知接收半径设定概率分布。

## 快速复现

建议 Python 3.12。在本目录打开终端：

```powershell
python -m pip install -r requirements-lock.txt
python -B -X utf8 benchmark.py
python -B -X utf8 -m unittest discover -s tests -v
```

默认比较 `latest` 与 `b2`，输出到 `outputs/benchmark/`。也可以从任意目录执行脚本的完整路径。全部运行均为本地合成环境，不包含联网入口，不调用官方模拟器，不使用正式测试次数。

```powershell
python -B -X utf8 benchmark.py --methods latest b2 validated --output outputs/three_methods
python -B -X utf8 benchmark.py --methods latest --scenarios scenarios/screening20.json
```

`requirements-lock.txt` 记录本次复现版本；`requirements.txt` 给出较宽安装范围。不同 NumPy/SciPy 版本或平台可能改变数值优化轨迹，逐位一致应使用记录的环境。

## 本次发布复现结果

在同一份 `scenarios/screening20.json` 上重新运行完整代码包：

| 方法 | 全清场景 | 清除总数 | 场景均值（秒/个） | P90（秒/个） | 最慢场景（秒/个） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始 B2 | 20/20 | 257/257 | 281.57 | 377.92 | 395.03 |
| 最新候选 | 20/20 | 257/257 | **235.94** | **296.73** | **339.90** |

最新候选的均值降低约 **16.21%**。20 个场景中有 5 个低于 200 秒/个。

**这是用于选参数的开发筛选集，不是独立验收集。尚不能宣称达到“独立测试集均值低于 200 秒/个”，也不能保证每局低于 200 秒。** 本目录的 `PASS` 仅表示给定场景全部清除且没有错误的不存在判定，不表示性能目标通过或官方评测通过。

原始数据与证据：

- [场景数据](scenarios/screening20.json)：固定种子 20261001 生成 140 个开发场景后按 `[::7]` 取 20 个。
- [逐场结果](evidence/screening20/cases.csv)：包含移动距离、测量次数、清除失败次数、计算耗时和慢局。
- [运行摘要](evidence/screening20/summary.json)：配置、源码 SHA-256、场景 SHA-256、依赖版本、执行时间。
- [源码来源](source_provenance.json)：记录最初发布时复制的 12 个算法模块及其历史哈希；Q2 扩展后的运行源码哈希见对应实验摘要。

统计口径：先计算每个场景的“总虚拟时间 / 成功清除数”，再对场景等权平均。计算机实际运行秒数另外记录为 `runtime_s`。若任意场景失败，聚合速度指标置空，失败场景不会被静默剔除。

## 最新方法在做什么

1. **维护可行位置多边形。** 每次示向度反馈形成角度误差扇区，并与历史可行区域相交；多边形外接圆用于判断能否一次清除。这里用的是保守包围圆，不是最小覆盖圆。
2. **利用未收到信号的信息。** `negative_strategy.py` 排除肯定能接收到信号的内部区域。`radius_coupling.py` 把同一源的有信号与无信号位置配对，利用其接收半径固定但未知的条件进一步裁剪。条件不一致会报错，不凭空补坐标。
3. **把搜索和清除一起排路线。** `joint_strategy.py` 提供 B2 基线；多起点贪心与 2-opt 改进访问顺序，每次行动后重新规划。不是先遍历所有扫描点再逐一清除。
4. **移动扫描站、复用清除位置。** 最新候选将外环划为 7 个扇区，在保持扇区覆盖条件的前提下，用 SLSQP 调整扫描站以缩短前后路程。若清除点已经能覆盖一个待扫描扇区，就在那里完成扫描。
5. **减少定位末段的移动。** 可行区域已经较小时，尝试直接清除；失败成本完整计入。必要时侧向移动再测量，最后保留有限覆盖搜索作为兜底。
6. **按观测触发外围候选点。** 仅当起点全频扫描没有发现源时，对单次示向度目标尝试半径 1750 m 上的射线交点，用于路线估价或进一步观测。它是启发式位置建议，不是已知真值，不会因此删除其他可行位置。这个机制可能偏好外围分布，需要独立样本继续验证。

实际参数全部集中于 [solver.py](solver.py) 的 `LATEST_CONFIG`。`validated` 是之前的 8 扇区配置，供回归对照；这一名称不代表它已经通过官方测试或 200 秒目标。

## 场景与接口约定

当前模型：全向源位于半径 1800 m 圆盘内，共 10–16 个源，分占 1–20 的不同频道；每个源的固定接收半径为 1000–1500 m；示向误差不超过 1°，反馈保留两位小数；近距离提示阈值 5 m，清除距离 20 m。

离线计时：移动速度 5 m/s；测量 5 s，切换测量频道另加 1 s；清除成功 5 s，失败 3 s。进入和退出不增加虚拟时间。合成误差包括固定正负极值、空间平滑误差和位置哈希误差；它们不是对官方随机分布的估计。

合成源真值仅在 `offline_environment.py` 的评价环境中使用。策略通过回调获得反馈，接口如下：

```python
from solver import build_strategy

# action(path, point=None, channel=None) -> dict
# point 是 [x, y]，单位 m；channel 是整数。
strategy = build_strategy(action, method="latest")
result = strategy.run()
```

| path | 输入 | 反馈必需字段（除 accepted=True、virtual_time_s 外） |
| --- | --- | --- |
| `/enter` | 无 | 可选 remaining_real_duration_s，默认 1200 |
| `/measure` | point、channel | measure_result：no_signal / near / direction；direction 还需 svd_deg |
| `/clear` | point、channel | clear_result：success / no_target_in_range |
| `/exit` | 无 | 无额外必需字段 |

这里只定义回调协议，没有 HTTP 请求实现。若接入团队自己的演练客户端，应由该客户端负责确认演练类型与反馈字段映射。算法不会自行开启测试，也不包含队号、密码或个人配置。

## 文件关系

- `solver.py`：统一构造入口与固定配置。
- `peripheral_prior.py`、`flexible_b3.py`：最新候选、扫描站调整和任务执行。
- `b3_strategy.py`、`route_tuning.py`、`joint_strategy.py`：观测共享、联合调度与基线。
- `route_search_probe.py`：确定性路线优化；文件名保留开发阶段名称，实际由正式入口导入。
- `localization_probe.py`、`radius_coupling.py`、`negative_strategy.py`：末段定位和约束裁剪。
- `q3_strategy.py`、`sweep_strategy.py`、`fast_strategy.py`：几何基础、基础策略与兜底。
- `offline_environment.py`、`benchmark.py`：合成环境和可复现评价。

本目录与 `Q3/method2` 的 PPO 实现分别运行；两者没有在同一环境与场景上重新对比，因此不据此声称优于 PPO。
