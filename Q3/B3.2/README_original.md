# B3 本轮优化候选

目标仍是全部清除且固定独立测试集场景均值低于200秒/个。当前没有达到这个目标；本包为离线验证候选，不能作为官方成绩或论文冻结结论。

默认 `current` 为本轮候选（配置 `spectral150`）；`latest` 为原 B3，`q2` 为朋友的 B3.1。原发布文件没有修改。

本轮只保留：搜索扫描聚焦未知频道、原有共享测向继续处理已知频道；增加一次起始侧向测量；保证清除时寻找更靠近机器狗的安全点。B3 的覆盖搜索、位置可行区域、路线算法、真实反馈更新、计时和失败兜底继续保留。

需要 Python 3.12、NumPy、SciPy。该目录没有HTTP连接入口，不会调用正式或演练模拟器。

```powershell
python -B -X utf8 -m unittest discover -s tests -v
python -B -X utf8 benchmark.py --methods latest q2 current --output outputs/reproduce20
```

程序接入点：`from solver import build_strategy`，随后 `model=build_strategy(action_callback, 'current')`、`model.run()`。回调接口与原 B3 相同。本包不提供自动连接正式测试的程序。

预先锁定的验收集仍保持未评估。详细实验和慢局见 `E:/Desktop/skill/results/Q3/experiments/b3_optimization_campaign/assessment.md`。源码来源与哈希见 `candidate_provenance.json`。
