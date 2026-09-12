# Q3：Bayes定位与动态开放路径

来自 `Q3/第三问-算法3`，默认 `bayes-fast`（V6及等价计算缓存）。固定未知接收半径后验、补测、清除与联合路线策略均保持原实现。

共享模块现存放在 `vendor/q3`，本目录可独立运行；仅调整 `bayes_tsp/shared.py` 的依赖路径。`SOURCE_MANIFEST.json` 记录迁移前源文件和哈希。

Python 3.10+，在本目录执行：

```sh
python -m pip install -r requirements.txt
python -X utf8 run.py local --policy bayes-fast --seed 0
python -m pytest -q
```

输出保存在 `results/`。模拟器真值不传给决策器。原版演练入口 `practice_windows.py` 随包保留，适配WSL Python＋Windows官方桌面软件；本次整理未启动官方测试。历史实验和报告保留在原目录，不作为本次重跑成绩。
