# 可运行源码

这里包含论文使用的 Q2 数值设计、Q3 `bayes-fast` 和 Q4 Ultra S21 `probes`。三个目录各自带齐运行依赖的源码；Q3、Q4 的 `vendor/` 是独立运行所需的协议与几何代码。无需从仓库旧算法目录导入模块。

## 安装与完整验证

支持 Python 3.10+，本次使用 Python 3.12.3 的全新虚拟环境验证。在 `src` 目录执行：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -X utf8 verify.py
```

Linux / macOS：

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -X utf8 verify.py
```

`verify.py` 检查源码引用、未使用代码和静态错误，分别运行三个目录的测试，再完整运行 Q2 两种情形的默认网格搜索、Q3/Q4 单局、S21 覆盖证书、历史结果数值核验和启动入口帮助。任何检查失败都会返回非零退出码；日志和源码哈希保存在新的 `results/validation-时间戳/` 中。整个验证过程只使用本地模拟器与模拟 HTTP。

仅运行算法时安装 `requirements.txt` 即可。`requirements-tested.txt` 冻结本次 Python 3.12 环境的完整包版本，供精确复现环境使用。测试应由 `verify.py` 分目录执行，或进入对应题目目录运行 `python -m pytest -q`，避免三套独立程序的同名入口互相覆盖。

## 运行入口

激活环境后，以下命令都在 `src` 执行：

```bash
# Q2：单点评估；省略 --evaluate 就运行完整搜索并导出 CSV / JSON
python Q2/run.py --scenario symmetric --R0 1200 --evaluate 840 495
python Q2/run.py --scenario asymmetric --R0 1200

# Q3 / Q4：完整离线任务
python Q3/run.py local --policy bayes-fast --seed 0
python Q4/start_s21.py --self-test
python Q4/start_s21.py --local --seed 800

# 同地图比较与覆盖布局复核
python Q3/run.py benchmark --compare baseline bayes-fast --rounds 2 --workers 2
python Q4/benchmark_layouts.py --workers 2
```

Q3/Q4 的 `run.py --help` 列出保留的本地比较策略和参数；`--output` 可指定新输出目录，已有目录不会被覆盖。Q2 默认同名搜索结果会覆盖，保留多批结果时应指定不同 `--output`。

Windows 演练与正式单局分别使用各目录的 `practice_windows.py` / `start_s21.py`、`start_formal.py`，操作见 [正式单局说明](FORMAL_TESTING.md) 和 [Q3](Q3/README.md)、[Q4](Q4/README.md) 的说明。正式单局仍需队员先在官方软件中启动案例，再传入 `--connect --case`。

## 辅助程序与清理范围

Q3 的 `analyze.py` 用于本地配对结果汇总，`summarize_practice.py` 汇总演练日志，`audit_movement.py`、`audit_routes.py`、`audit_practice.py` 用于已有动作日志的移动和路线核验。各自的 `--help` 给出输入文件要求。Q4 的 `check_s21_certificate.py` 可独立复核整数覆盖证书；`benchmark_layouts.py` 复现 S22/S21 的 30 组配对。

已移除 Q3 未接入的 MC 规划器、随机世界采样和候选生成方法，保留移动基线实际使用的确定性 `likelihood.py`。Q4 移除未用于交付与回归的实验策略、MC、共享采样及旧 Q3 巡逻代码，保留最终策略、有限完成后备和文档中的比较功能。历史实现仍在仓库原算法目录或 Git 历史中。

`SOURCE_MANIFEST.json` 与 `evidence/` 中的旧报告用于历史溯源，其哈希描述当时的文件，不作为当前运行文件清单。当前运行清单由程序根据实际源码生成。整理结果与实测边界见 [清理验证记录](CLEANUP_REPORT.md)。
