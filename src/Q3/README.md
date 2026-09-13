# Q3：论文 bayes-fast

本目录实现 `essay.tex` 问题三对应的唯一策略。`bayes_tsp/policy.py` 直接包含决策流程；原继承链中实际调用的步骤已经合并，历史策略类与工厂已删除。

保留固定未知接收半径的确定性贝叶斯积分、按源数量约束的存在概率、补测与试探清除、已知源与未知覆盖任务的联合开放路线、七扇区覆盖站优化、实际反馈更新和有限完成后备。缓存只复用数值计算，不改变候选动作和评分。

在本目录运行：

```bash
python -m pip install -r requirements.txt
python run.py local --seed 0
python run.py reproduce --workers 4
```

`reproduce` 使用论文的随机种子 900–939，以及四类地图×三类误差×两种源数/半径的 24 个压力配置。所有入口固定使用 `Config()` 的论文参数：24×24 位置积分、1°反馈分档、存在先验 0.65、七扇区、997m 覆盖余量、12 节点以内精确开放路线；600 个动作或剩余现实时间不足 30s 时进入有限后备。

输出位于新建的 `results/时间戳/`，包含每局 `actions`、`decisions`、`summary` 和批次清单。停止条件为全部频道已清除/证明不存在，或成功清除 16 个不同频道。

Windows 演练入口为 `practice_windows.py --connect`，正式入口为 `start_formal.py --connect --case 本局编码`；使用方法见 [正式运行说明](../FORMAL_TESTING.md)。
