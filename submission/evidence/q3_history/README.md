# 论文历史统计

`random/` 为批次 20260911-192929-110364 的原始汇总与配置，`stress/` 为 20260911-193444-954565，`oracle/` 为 oracle-tsp-mc-20260912-021030-302657。所有数据原样复制。旧策略实现不随包分发。

`verify.py` 只改了数据定位路径，原有哈希、完成性、成本与论文均值核验保持不变。运行 `python evidence/q3_history/verify.py`；也可用 `--paper /path/to/essay.tex` 与外部论文核对。

`oracle/verification.json` 是历史验证声明。当前包没有原始十万图 NPZ，本次只能复核归档汇总；重建程序见 `src/oracle/run.py`。本地历史记录和正式加密日志不可混用。
