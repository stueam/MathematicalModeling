# Q3 论文表格的历史实验依据

本目录核验 `essay/essay.tex` 的“结果评价与全知者参照”。它从仓库已经保留的原批次汇总读取数据，不再复制逐局 JSON，也不运行算法或连接官方模拟器。

| 批次 | 原记录 | 论文采用的统计 |
| --- | --- | --- |
| 随机留出 | [20260911-192929-110364](../../第三问-算法3/results/20260911-192929-110364/) 中的 `bayes-fast` | 40 局，545 源，216.0097199083 秒/源 |
| 压力场景 | [20260911-193444-954565](../../第三问-算法3/results/20260911-193444-954565/) 中的 `bayes-fast` | 24 局，312 源，207.4575582596 秒/源 |
| 全知者 | [十万图汇总](../../results/oracle-tsp-mc-20260912-021030-302657/summary.json) | 纯移动 136.5153344221 秒/源；含每源 5 秒清除为 141.5153344221 秒/源 |

`summary.json` 保留原表格数值及三份原始汇总的 SHA-256，内容与清理前提交 `d92ddd1` 的 `src/Q3/evidence/paper_results/summary.json` 相同。原 Windows 文件使用 CRLF，Git 归档为 LF；`verify.py` 仅允许这种换行转换，仍按原哈希核验内容，随后检查完整完成、源数、每项动作成本和总时间并复算表格。`--paper` 还核对论文显示的数值。

在仓库根目录运行，仅需 Python 标准库：

```bash
python Q3/evidence/paper_results/verify.py --paper essay/essay.tex
```

两批 Q3 采用历史 `bayes_tsp_v6_cached_computation` 实现，配置和源码哈希在上述原批次目录的 `config.json` 中。部分历史逐动作大文件未纳入仓库，本检查直接复核已有的 64 条结果记录。原先更详细的版本对照材料可在 `d92ddd1:src/Q3/evidence/paper_results/` 查看。

当前 `src` 的清理验证和当前环境重跑见 [PAPER_VALIDATION.md](../../../src/PAPER_VALIDATION.md)。它们是不同版本的记录，不能替换本表格的历史数值。未完成与其他对照策略的原始记录继续保留，筛选规则在脚本中显式限定。
