# Q4 NN：已验证的成本敏感注意力模型

本目录可独立运行。随包最佳权重 `models/best.pt` 在同 32 张本地开发图上为 **442.625 秒/源**，基线 **446.851 秒/源**；两组都完整清除并通过独立物理和费用审计。平均改善 0.946%，95% 配对改善区间 [−0.839%, 2.917%]，不能声称稳定优于基线。用户当前目标为 350 秒/源，尚未达到。

这是此前已完成评测的优势加权微调版本。更新中的完整 v2 注意力和 REINFORCE 实验单独进行，不包含在本目录中。上述数字来自这组开发地图，与 `../Ultra` 等方案的不同地图成绩不能直接排名。

## 运行最佳模型

Python 3.10 或更高版本。在本目录执行：

```bash
python -m pip install -r requirements.txt
python verify_release.py
python run_best.py --seed 280020000 --rounds 32 --workers 16
```

入口默认加载 `models/best.pt`，使用 `selection=advantage`、v2 特征及 ring22 布局，并同图运行基线。输出在 `results/时间戳-published-best/`，其中 `aggregate.json` 为总虚拟秒／总源数，`control.json` 为实际运行时间及进程清理记录。任务全部完成后才能报告完整成绩。

推理使用单线程 CPU；训练使用 CUDA。每轮最多 30 分钟，第 27 分钟停止派发新任务。在该轮输出目录创建 `STOP` 文件可终止工作进程。所有结果为本地模拟，入口不连接官方服务。

## 模型和数据

模型使用公开频道信念、覆盖站、后验粒子及候选动作。三组成本头学习完整专家续跑成本差，策略头采用优势加权更新、软偏好监督与 BC KL 约束，并以小学习率微调注意力。四候选规则及经验改选门槛随模型保存。完成几何、逐频道扫描计费及有限后备继续有效；网络不读取真实源位置、半径、朝向或总数。

`training_data/` 包含 213 个已完成状态、844 个实际评估过的非基线候选标签；按原始世界分组，训练 175 状态／692 标签，校准 38 状态／152 标签。缺失标签保持隔离。这里的成本为共享相容假想世界上的完整后续成本估计，并非真实最优 Q。原 BC 编码器见过部分校准世界，校准集不是整模型独立测试。

复现该训练配置并随后评测：

```bash
python run_best.py --retrain --device cuda --rounds 32 --workers 16
python -m pytest -q tests
```

训练使用原初始化、固定种子、AdamW、成本/偏好/优势/KL 损失和世界 bootstrap；不同 Torch、CUDA 或硬件版本可能产生数值差异，重训成绩仍需重新评测。旧的研究模块保留历史路径默认值，发布包的可移植入口以上述命令为准。

## 证据与来源

- [32 图配对表](evidence/paired-results.csv)、[原始实验报告](evidence/报告.md)、[费用与梯度核验](evidence/verification.json)。
- `evidence/benchmark/` 保留原三策略共 96 局的公开动作、局部模拟世界和独立审计。真实世界文件只供模拟器与事后审计使用，不能进入决策或训练特征。
- `release.json` 固定部署源码和权重 SHA256。策略、几何、特征、损失与原完成实验一致；运行控制更新为已测试的 30 分钟上限和延迟进程回收修复。
- 模型 SHA256：`7f60344c2377e7c2403cea8d2ed2fb02b43a72d43aa8e60259d363fc338cf869`。
- [方法说明](METHOD.md)。参考 [AWR](https://arxiv.org/abs/1910.00177)、[AggreVaTeD](https://proceedings.mlr.press/v70/sun17d.html)、[CAtNIPP](https://proceedings.mlr.press/v205/cao23b.html) 与 [ARiADNE](https://arxiv.org/abs/2301.11575)，属于 Q4 适配，并非这些论文的完整复现。

预留 64 图和 48 局压力验证尚未使用，本包不宣称已通过最终目标验收。证据文件中的原始本地路径用于历史追溯；运行本包不依赖这些路径。
