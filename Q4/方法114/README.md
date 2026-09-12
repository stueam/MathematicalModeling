# 方法114：Q4 双环覆盖、贝叶斯定位与联合路线

本目录是我们自主实现的 Q4 策略独立发布包。默认策略 **`probes`**，实现版本 **`q4_v4_ring22_route_probes`**：22 点双环提供完整覆盖后备，结合固定接收半径／未知发射朝向的贝叶斯积分、顺路第二测点和搜索／清除联合路线。此前的对照与未胜出实验分支也保留在代码中。

本包自带所需基础模块，单独下载本目录即可运行，不依赖仓库其他方法文件夹。发布整理没有调参或改变决策逻辑；仅将共享依赖路径改为本包的 `vendor/q3`，并同步离线诊断脚本的依赖散列路径。

## 安装与运行

建议 Python 3.10 或更新版本。以下命令在本目录执行：

```bash
python -m venv .venv
# Linux / WSL
source .venv/bin/activate
# Windows PowerShell 改用：.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 完整回归
python -m pytest -q

# 单局本地仿真；仅使用自建模拟器
python run.py local --policy probes --seed 0

# 同地图比较旧版 mobile 与当前 probes
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run.py benchmark \
  --seed 800 --rounds 16 --compare mobile probes --workers 16
```

最后一条采用 Bash 环境变量语法；Windows PowerShell 可先设置 `$env:OPENBLAS_NUM_THREADS="1"` 和 `$env:OMP_NUM_THREADS="1"`，再运行 `python run.py benchmark ...`。

每次运行在 `results/时间戳/` 保存配置、动作、决策、世界和结果。世界真值只供本地评价器使用，不传入在线决策器。

## 已有结果与适用范围

| 既有验证 | 旧版 mobile | 当前 probes |
|---|---:|---:|
| 本地独立种子800～815，平均逐局秒/源 | 647.78 | **471.47** |
| 同批完整完成 | 16/16 | 16/16 |
| 同批平均整局虚拟时间 | 7919.20 s | **5792.08 s** |
| 同批平均移动距离 | 25.45 km | **21.28 km** |
| 42个压力配置，平均逐局秒/源 | 580.62 | **444.92** |
| 压力配置完整完成 | 42/42 | 42/42 |

这些是本地同图对照。历史官方15局演练跑的是旧版 `mobile`：194源全部清除，平均7814.03虚拟秒、25.72km、618.26秒/源。**现有官方日志没有当前 `probes` 的成绩，350秒/源目标尚未达到。**

秒/源指机器狗的虚拟任务时间，移动仍按5m/s计费；不是Python运行时间。不得将本表与其他方法不同地图的均值当作配对比较。原始批次摘要在 [evidence/](evidence/)，方法细节见 [METHOD.md](METHOD.md)，本次发布核验见 [VALIDATION.md](VALIDATION.md)。

## 官方演练适配器

`practice_windows.py` 是已有的 **WSL Python＋Windows官方模拟器** 演练适配器，依赖Windows UIAutomation及PowerShell互操作。本地仿真不需要Windows。该适配器只识别“问题4演练测试”，没有正式测试入口。

在明确要运行一次演练、模拟器已准备好后，使用：

```bash
python practice_windows.py --connect --rounds 1 --policy probes
```

必须显式选择 `probes`：为保留原实现，适配器默认仍是旧版 `mobile`。不带 `--connect` 只读取界面，不进入测试。发布核验仅调用假服务和本地世界，未连接官方模拟器。

## 文件结构

| 文件／目录 | 用途 |
|---|---|
| `q4/` | 覆盖证明、后验、定位、联合路线、默认策略与实验分支 |
| `vendor/q3/` | 本项目自写Q3代码中复用的五个基础文件；Q4用自己的观测与覆盖规则 |
| `run.py` | 本地仿真、同图对照、压力验证 |
| `practice_windows.py` | 官方演练界面核验与串行HTTP适配 |
| `tests/` | 几何、计费、采样、路线、后备与演练假服务回归 |
| `audit*.py` / `analyze_practice.py` / `diagnose_official_logs.py` | 已有日志的离线审计与诊断 |
| `optimize_*.py` / `experiment_clear_first.py` | 保留的布局与调度实验 |
| `evidence/` | 精简的既有完整批次摘要，非全部实验日志 |
| `docs/history/` | 原开发报告原文；内部路径指原工作区，阅读说明见该目录README |
| `publication_source_manifest.json` | 发布文件与原文件散列、迁移差异 |

`plot_practice.py` 需要可选依赖 `matplotlib`。历史日志审计脚本需要自行提供对应动作／决策日志；尤其 `diagnose_official_logs.py` 默认查找原记录中的固定批次路径。约1.8GB开发日志、官方HTTP原始记录、缓存及登录信息没有随源码发布。

默认采用 `probes`。`mobile`、`trim`、`ring` 是主要对照；`mc-probes`、`mc-stable`、`deferred`、`phased`、`pooled`、`collect` 等属于实验分支，现有结果不支持替换默认。详情可运行 `python run.py --help` 并查阅历史报告。
