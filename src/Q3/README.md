# Q3：Bayes定位与动态开放路径

来自 `Q3/第三问-算法3`，默认 `bayes-fast`（V6及等价计算缓存）。固定未知接收半径后验、补测、清除与联合路线策略均保持原实现。

共享模块存放在 `vendor/q3`，本目录可独立运行。已删除未接入的 MC、随机世界采样和旧候选生成代码；移动基线的确定性似然位于 `vendor/q3/likelihood.py`。`SOURCE_MANIFEST.json` 保留迁移时的原文件和哈希，可能包含后来删除的历史文件；当前运行哈希由 `code_manifest()` 按实际源码生成。

Python 3.10+，在本目录执行：

```sh
python -m pip install -r requirements.txt
python -X utf8 run.py local --policy bayes-fast --seed 0
python -m pip install pytest
python -m pytest -q
```

输出保存在新的 `results/时间戳/`，也可使用 `--output 新目录`。模拟器真值不传给决策器。演练入口 `practice_windows.py` 支持原生 Windows Python 和 WSL＋Windows 官方桌面软件。历史实验和报告保留在原目录，不作为本次重跑成绩。统一安装与验证见 [src 说明](../README.md)。

## 正式单局入口

新增 `start_formal.py`，使用同一 `bayes-fast` 核心和参数。默认 `python -X utf8 start_formal.py` 只检查可见界面，不连接接口。队员在官方模拟器中手动开始“问题3正式测试”，待倒计时结束、显示“尚未进入”后，运行：

```powershell
python -X utf8 start_formal.py --connect --case XXXX-XXXX-XXXX-XXXX
```

把示例编码换成本局真实编码；本入口不启动测试或连跑下一局。日志写入 `results/formal/案例编码-时间戳/`，不依赖正式页面不公开的源总数。正常退出后计算总虚拟时间÷成功清除数，另行导出官方加密行为日志。完整操作和异常处理见 [正式单局说明](../FORMAL_TESTING.md)。原演练入口保持不变。
