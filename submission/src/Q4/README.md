# Q4：论文 S21＋probes

本目录实现 `essay.tex` 问题四的唯一策略。`q4/policy.py` 包含 21 点覆盖计划、已知源与搜索任务的联合路线、移动补测候选和有限后备。

定向源的类型、朝向、位置与固定接收半径通过公开反馈更新。未知源排除使用实际负观测形成的多点几何证书；站点移动或删除前验证剩余覆盖计划，实际停止判定仍依赖已收到的反馈。

在本目录运行：

```bash
python -m pip install -r requirements.txt
python start_s21.py --self-test
python check_s21_certificate.py
python run.py local --seed 800
python run.py reproduce --workers 4
```

`reproduce` 固定使用随机种子 800–815，加上七类地图各两个压力配置，种子 9100–9113，共 30 局。全部使用相同 S21＋probes 参数。`s21_certificate.json` 是 21 点布局的整数证书，`check_s21_certificate.py` 可独立复核，`s21_layout.py` 加载时还会核验当前几何环境中的完整域覆盖。

每次新建 `results/时间戳/` 保存逐条动作反馈、决策、评估地图、批次清单与汇总。算法不会读取评估地图真值；未完整清除并获得停止证书时返回非零退出码。

Windows 演练入口为 `start_s21.py --connect`，正式入口为 `start_formal.py --connect --case 本局编码`；使用方法见 [正式运行说明](../FORMAL_TESTING.md)。
