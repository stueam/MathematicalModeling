# 提交整理核验（2026-09-12）

- `python -m pytest -q`：81项通过。
- `python -X utf8 run.py local --policy bayes-fast --seed 0`：全部清除，证实完成，虚拟用时2730.530849 s，移动10142.654245 m，无异常。
- 仅将共享依赖路径改为本目录vendor/q3，默认Bayes策略和数值参数保持不变。
- 没有运行官方评测；以上是本地验证。
