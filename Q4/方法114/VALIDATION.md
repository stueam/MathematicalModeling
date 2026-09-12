# 发布核验

2026-09-12，本次只整理发布包、执行离线测试并验证迁移等价性，没有运行官方演练或正式测试。

## 发布包回归

- `python -m pytest -q`：**95 passed，108.17 s**。
- 测试在本目录执行，使用本包 `vendor/q3`，不依赖原工作区的第三问目录。
- 包含定向反馈与几何、完整覆盖、计费与幂等、定位、路线、假想世界采样、后备及官方演练假服务检查。
- 73个复制文件与来源字节一致；另外2个只调整依赖路径：`q4/shared.py` 和 `diagnose_official_logs.py`。逐文件来源散列与发布散列见 [publication_source_manifest.json](publication_source_manifest.json)。新增发布说明及本次验证结果不属于原文件复制清单。

## 当前源码与发布包的同环境逐动作比较

在同一Python和数值库环境中，分别运行当前原工作区及独立发布包的 `probes / seed=800`：

| 指标 | 原工作区 | 发布包 |
|---|---:|---:|
| 清除／总数 | 10/10 | 10/10 |
| 公开完成证明 | 成立 | 成立 |
| 虚拟总时间 | 5667.875379 s | 5667.875379 s |
| 移动距离 | 19794.376872 m | 19794.376872 m |
| 测量次数 | 279 | 279 |
| 清除失败次数 | 1 | 1 |
| 完整动作序列长度 | 290 | 290 |

**290个动作、全部公开反馈（除实际时间戳）、虚拟时间、世界及计费均逐项相同。** 本次实际运行时间分别为4.19秒和4.00秒，不要求墙钟一致。[机器可读结果](evidence/publication_smoke.json)。

这验证的是本次目录迁移前后等价性。它不是旧16图批次的完整重跑，也不能用这一个案例更新历史平均成绩。旧批次使用Python3.14.6／NumPy2.4.6／Shapely2.1.2；当前验证使用下列环境。同一种子的历史动作轨迹没有与本次完全一致，因此历史摘要保持原记录，不宣称本次精确复现历史16图成绩。

## 本次环境

- Python 3.10.19
- NumPy 2.2.6
- SciPy 1.15.3
- Shapely 2.1.2
- Requests 2.34.2
- pytest 8.4.2
- 数值库线程数：`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`

常规安装使用 `requirements.txt`；若需要固定本次直接依赖版本，可使用 [requirements-tested.txt](requirements-tested.txt)，它不是所有传递依赖的完整锁文件。

## 已有证据范围

- `evidence/local_holdout_800_815.json`：历史16张图、两策略，共32次完整任务。
- `evidence/local_stress_42.json`：历史42个压力配置、两策略，共84次完整任务。
- `evidence/official_mobile_15_summary.json`：历史旧版15局官方演练归因汇总。
- `evidence/official_mobile_diagnostic_summary.json`：既有官方日志的离线诊断汇总。

摘要保留原始记录，完整动作、候选、官方HTTP日志及赛题附件不在本包内。各摘要中的不同策略、不同批次不混算，历史官方结果不记到当前 `probes` 上。
