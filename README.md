# 数学建模 B 题

当前论文算法的交付入口是 [`src/`](src/README.md)，论文与实际使用的插图在 [`essay/`](essay/README.md)。运行程序请从这两个入口开始。

| 目录 | 用途 |
| --- | --- |
| [`src/Q2`](src/Q2/README.md) | 固定接收半径下的贝叶斯选点 |
| [`src/Q3`](src/Q3/README.md) | 论文 bayes-fast 定位与路线算法 |
| [`src/Q4`](src/Q4/README.md) | 论文 S21＋probes 覆盖、定位与清除 |
| [`Q3/evidence/paper_results`](Q3/evidence/paper_results/README.md) | 论文 Q3 表格对应的历史实验核验 |
| `正式实验日志/` | 六份正式实验原始日志 |
| `Q2/`、`Q3/`、`Q4/` 的其他目录 | 历史算法、实验结果、模型和对照证据；各自 README 说明版本与限制 |
| `archive/` | 早期备赛资料 |

在 `src` 下创建虚拟环境并安装 `requirements-tested.txt`，然后运行 `python verify.py`。完整仓库会同时检查论文案例导入及历史表格；单独复制 `src` 时跳过这两项仓库集成检查。验证不会连接官方模拟器。

历史资料占据仓库主要体积，保留它们用于追溯实验，不代表仍是当前默认实现。不要混用不同目录的同名模块或将旧成绩当作当前程序成绩。当前论文不再引用的三组旧图已从 `essay/data/` 移除，其完整内容仍在 Git 提交 `0056aff` 中。
