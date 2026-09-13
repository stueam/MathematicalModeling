# 实验代码与数据提交包

本目录可以整体复制到仓库之外运行。以打包时 main 的 `essay/essay.tex`、`src/` 和 `正式实验日志/` 为依据，按论文复现需要补入论文已引用的 Q3 历史统计与全知者代码。确切基准提交、原路径及 SHA-256 见 `MANIFEST.json`。

## 内容与论文对应

| 内容 | 路径 | 用途 |
| --- | --- | --- |
| 问题一 | `essay/data/q1_two_cases/`、`q1_wedge_intersection/`、`q1_cover_cases/` | 半平面交、直径与直径圆覆盖；构造观测、顶点和结果 |
| 问题二 | `src/Q2/`、`essay/data/q2_strategy_heatmaps/`、`q2_intersection_comparison/` | 贝叶斯选点、冻结网格、三种选点交会数据与绘图 |
| 问题三 | `src/Q3/`、`essay/data/q3_*/` | bayes-fast、本地模拟器、实际接口、冻结案例轨迹与路线示意 |
| 问题四 | `src/Q4/`、`essay/data/q4_coverage_geometry/` | S21＋probes、本地模拟器、实际接口、覆盖证书与布局坐标 |
| 全知者参照 | `src/oracle/` | 十万张随机地图的精确开放 TSP 实验，可按种子重建地图和路线 |
| 论文历史表格 | `evidence/q3_history/` | 40＋24 局原批次汇总、配置和十万图汇总；仅筛选 bayes-fast 行 |
| 当前算法历史验证 | `src/evidence/` | 现有 64＋30 局验证记录及后续修复检查，保留原版本哈希 |
| 正式测试原件 | `正式实验日志/3/`、`4/` | 每题三份原始加密 `.jlog`，文件名及字节不变 |

Q2–Q4 的运行代码、参数、协议保护及已有测试逐字保留。绘图目录的数值模块是论文冻结版本，保留用于复现历史图；不可用它们替换正式算法。同名 vendor 模块在各题内各自加载，不能随意合并。

## 安装与验证

建议 Python 3.12。进入本目录，创建环境并安装：

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python verify_submission.py
python src/verify.py
```

`verify_submission.py` 只用标准库，核对清单、六份日志、Q2 冻结网格与 Q3 历史统计。`src/verify.py` 运行原有算法检查，默认写入新的 `src/results/validation-*`，不会连接官方模拟器。实测范围见 `VALIDATION.md`。

## 运行与重现

以下命令均从本目录执行：

```bash
# Q1：重算两组固定构造算例，并输出 JSON/CSV/PNG
python essay/data/q1_two_cases/generate.py
# Q2：两组完整搜索
python src/Q2/run.py --scenario symmetric
python src/Q2/run.py --scenario asymmetric
# Q3/Q4：单局与论文地图批次
python src/Q3/run.py local --seed 0
python src/Q4/run.py local --seed 800
python src/Q3/run.py reproduce --workers 4
python src/Q4/run.py reproduce --workers 4
# 全知者：重建十万图（耗时较长）；只计移动
python src/oracle/run.py --samples 100000 --workers 4
# 原历史统计复核
python evidence/q3_history/verify.py
```

Q3/Q4 和全知者运行会新建结果目录，包含配置、地图参数、动作或路线、哈希和汇总。Q2 重复运行会覆盖同名结果，可用 `--output` 指定新目录。论文的平均每源指标采用总时间除以总源数，全知者含清除参照在纯移动值上加 5 秒/源。

每个 `essay/data/<图名>/generate.py` 可独立运行。Q1 与交会构造图重新计算数值，其余图读取配套冻结数据重绘；重绘不等于重跑完整任务。PNG 是可再生结果，未重复放进本代码数据包；中文图使用 Microsoft YaHei/SimHei，其他平台可安装相应中文字体。绘图会在各自目录生成 PNG，部分构造脚本还重写 CSV/JSON；需要保持原始哈希时在副本中运行。

正式模拟器的接入程序是各题 `start_formal.py`，详见 `src/FORMAL_TESTING.md`。本次整理不启动新正式测试。

## 取舍与数据边界

- 不包含旧策略、消融/对照策略代码、神经网络权重、备份、论文编译产物、编辑审查记录、UI 截图、账号配置或开发缓存。历史统计文件中存在原批次的其他策略行，是原始证据；核验脚本显式只使用 bayes-fast，保留原文件以核对原哈希。
- 论文源文件与最终插图属于论文交付，不重复纳入本实验包。三张仅有 PNG、标记 `SOURCE_PENDING.tex` 的 Q2 说明图没有生成源码，未伪造或补写脚本。已具备源码的九个图目录完整保留代码与数据。
- Q3 的论文表格是历史版本 64 局记录；冻结示例为 121 个动作、2730.530849 秒。当前代码 seed 0 的结果约 2738.940927 秒，二者不能替换。提交包保留历史汇总和当前可运行算法，不复制整套旧策略。
- Q4 论文引用批次 `20260913-013804-paper-q4-recheck`，表中均值 5694.39 秒；指定来源及当前仓库中未找到该批次完整原始记录。现有 `src/evidence/paper_validation.json` 的 30 局为另一版本验证，均值 5694.319195 秒，明确不冒充原稿批次。当前 `reproduce` 可生成新批次。
- 十万图原始 NPZ 未在当前仓库归档；保留原汇总、配置、历史核验及确切路线求解器，并提供按原种子重建的程序。历史验证文件仅陈述当时已完成的核验，不表示本次重新核验了十万条路线。
- 六份 `.jlog` 是原始加密文件，本次只核对存在性、文件名、长度及 SHA-256，未解密、修改或据此推断成绩。论文的正式成绩表仍待填，且仅凭这些文件不能核实正式运行的源码哈希。日志清单见 `正式实验日志/README.md`。
