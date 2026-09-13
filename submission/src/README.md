# 提交代码入口

总安装、运行和取舍说明见 [提交说明](../README.md)。Q2–Q4 算法与基准提交的 `src` 完全一致，各题测试保留，用于验证独立目录的算法及协议依赖。

- Q1：`../essay/data/q1_two_cases/geometry.py` 是论文半平面交与直径算法，`generate.py` 重建两组构造算例；其余 Q1 目录核验楔形与直径圆覆盖示意。
- Q2：`Q2/run.py`，固定 R0 贝叶斯选点。
- Q3：`Q3/run.py`，唯一 bayes-fast 策略。
- Q4：`Q4/run.py`，唯一 S21＋probes 策略及整数覆盖证书。
- 全知者参照：`oracle/run.py`，保留论文采用的均匀随机地图与精确开放路线求解。
- `verify.py`：保留现有 156 项测试、静态检查、完整 Q2 网格、Q3/Q4 单局、证书与案例重放；历史表格从提交包内读取。

`evidence/paper_validation.json` 与 `review_fixes_validation.json` 是历史验证记录，哈希对应各自记录的版本。包的本次检查见 `../VALIDATION.md`。正式接入说明见 [FORMAL_TESTING.md](FORMAL_TESTING.md)。
