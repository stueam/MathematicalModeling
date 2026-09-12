# D1：第四问几何搜索、定位与清除联合调度

本目录发布当前第四版 **`q4_service`**，入口为 `service_strategy.ServiceB2(action)`。它在第三版 `q4_compact` 上增加连续扫描点优化、可靠清除点优化，以及至多12个固定任务点的精确开放路径。

这是几何可行域与滚动调度方法，未实现蒙特卡洛 Rollout 或贝叶斯后验控制。发布的是已经离线验证的原策略；整理目录没有重新调参。**本目录没有官方模拟器客户端，不会启动演练或正式测试。**

## 成绩怎么理解

| 同场离线对照 | 混合案例 | 第三版 秒/源 | 第四版 秒/源 | 平均下降 |
|---|---:|---:|---:|---:|
| 旧种子回归 | 20 | 490.89 | 488.75 | 0.44% |
| 新种子留出 | 20 | 497.49 | 491.34 | 1.23% |
| 合并 | 40 | 494.19 | **490.05** | **0.84%** |

平均秒/源是逐场虚拟时间除以该场源数，再对场景等权平均。40个混合场景有32场更快、8场更慢；不是每场都能低于500秒/源。加入16个全向/全定向诊断案例后，第四版56/56场、728/728个源全清，没有误判不存在或评估器发现的真源可行域剔除。

本机第四版策略运行平均约4.49秒/场，和490秒/源的虚拟耗时含义不同。此前约476秒/源是**第二版的一次模拟器演练成绩**，不属于本目录第四版的官方实测结果。

## 安装与运行

推荐 Python 3.12，在仓库根目录运行：

```bash
cd Q4/D1
python -m pip install -r requirements-lock.txt
python -B -X utf8 -m unittest discover -s tests -v
```

小规模离线运行（同一场景比较第三版、第四版，输出目录必须尚不存在）：

```bash
python -B -X utf8 benchmark_service.py --seed 2026091183 --count 7 --output local_runs/demo
```

复现28场留出组：

```bash
python -B -X utf8 benchmark_service.py --seed 2026091183 --count 28 --output local_runs/holdout
```

运行结果保存到 `tables/cases.csv`、`metrics/scenarios.json` 和 `run_summary.json`，保留失败案例。上述命令全部调用本地 `OfflineEnvironment`。

发布包完整性和代表案例复现检查：

```bash
python -B -X utf8 scripts/verify_release.py
```

该检查校验原算法文件哈希、运行29项测试，再重放两个场景的两种策略；核对除运行时间外的逐场指标，结果写入 `local_runs/publication_check.json`。

## 与已有程序连接

```python
from service_strategy import ServiceB2

# action 是调用方提供的公开反馈回调，不能向策略传入源真值。
strategy = ServiceB2(action)
result = strategy.run()
```

回调签名为 `action(path, point=None, channel=None)`，path 为 `/enter`、`/measure`、`/clear`、`/exit`，返回约定的反馈字典。完整本地协议示例见 [offline.py](offline.py) 的 `OfflineEnvironment.action`。真实客户端需负责认证、协议转换和场次管理，本包不提供这些操作。不要将 `vendor_b2/solver.py` 当作第四问入口，它属于历史全向源依赖快照。

## 模型和执行逻辑

1. 源位置位于1800米圆盘内，接收半径1000—1500米。定向源使用固定闭半平面与接收圆盘的交；清除只要求20米距离，与朝向无关。
2. 初始22个扫描点：原点、999米内环9点、1875米外环12点。外侧检测按机器狗可离开源分布圆盘处理。凸单元距离与扫描点凸包条件验证连续位置和全部朝向；不是只检查采样点。
3. 正示向通过误差楔形与距离外包络更新完整位置多边形。无信号只有在距离保证和成对几何反证成立时才能裁剪；不做95%概率截断。
4. 搜索、侧向补测和清除组成动态任务列表，每执行一个任务重新规划。已能保证清除时，在与全部多边形顶点距离不超过19.8米的区域内选择更顺路的停靠点。
5. 普通定位困难时进入成对补测，最终使用有限光学网格覆盖。动作或时间限额不足时报告失败，不冒充已完成。
6. 只有实际扫描历史完成连续覆盖，或已识别题目上限16个不同源，才允许判断剩余频道不存在；所有已发现源清除后才退出。

局部选点和多数调度规则是启发式；精确路径的最优性仅限当前固定任务点，没有第四问全局最优证明。

## 文件与证据

| 文件 | 用途 |
|---|---|
| `service_strategy.py` | 第四版入口 |
| `service_geometry.py` | 连续扫描点和可靠清除点优化 |
| `service_routes.py` | 小任务集精确开放路径 |
| `compact_strategy.py` | 第三版同场基线 |
| `adaptive_strategy.py` / `adaptive_coverage.py` | 定向负观测约束、连续覆盖与定位回退 |
| `strategy.py` / `coverage.py` | Q4基础策略与早期覆盖工具 |
| `vendor_b2/` | 原B2依赖快照，保持原始文件 |
| `offline.py` / `benchmark_service.py` | 离线环境和同场对照 |
| `tests/` | 29项几何、规则和路线检查 |
| `results/Q4/experiments/round18_service_regression/` | 旧种子完整离线证据 |
| `results/Q4/experiments/round19_service_holdout/` | 新种子完整离线证据 |

- [第四版详细方法](methods/Q4/q4_service_plan.md)
- [第四版完整测试报告](Q4_B2第四版_Q3迁移与离线测试报告.md)
- [原工作目录代码审核](code/Q4/reviews/qx_python_review_v4.json)
- [来源与迁移记录](methods/Q4/probes/q3_v4_reference.json)
- [本发布包核验结果](publication_check.json)

参考 Q3 提交 `242b8786d71cfb4ef49f2d52590ff388b292386e` 中“第三问-算法3”的服务点优化与开放路径思想；原B2快照来源为 `02d2cd36455475ea0be72235b5e147bc2c19a818`。具体文件哈希见来源记录。

原始结果JSON保留当时完整工作目录的源码哈希，其中包含未发布的旧演练入口 `practice_adaptive.py`；它未参与离线策略运行。发布检查逐个验证本包保留的全部原Python文件与历史哈希一致。历史审核、开发轮次与本次发布核验的范围分别记录，不改写历史实验数据。
