# Ultra：方法114的S21覆盖优化版

本目录与 `../方法114` 平级，可独立运行。默认策略为 `ultra_s21_route_probes_v1`，本地 `run.py` 和官方演练入口均使用21点；原方法114保留22点，不受本目录影响。

S21是原点＋内圈8点＋外圈12点。后验积分、候选动作、联合路线与反馈停止规则沿用方法114。30组配对中S21获胜21组，平均总任务时间5861.51→5688.76 s，路程21.030→20.362 km，两版均30/30全清。随机16组单独改善2.12%，配对检验p=0.102，不宣称已证实稳定优势。详见 [实验报告](evidence/comparison_report.md) 和 [逐场数据](evidence/paired.csv)。

这是完整Python源码运行包，不是要上传给评测器的点位表。解压后启动本地程序，由程序通过 http://127.0.0.1:2026 与官方桌面评测器交互。

## Windows使用

需要Python 3.10或更高版本。保持所有文件夹结构，在解压目录打开PowerShell：

```powershell
python -m pip install -r requirements.txt
python -X utf8 start_s21.py --self-test
```

self-test只检查覆盖和策略初始化，不连接评测器。出现passed后，打开官方桌面评测器，登录自己的账号并停留在“问题4演练测试”页面，然后运行一次：

```powershell
python -X utf8 start_s21.py --connect --rounds 1
```

程序将启动一场Q4演练，默认使用S21和方法114的probes策略。无需手填队号，程序从可见演练界面读取。若已经手动启动演练并停留在“尚未进入”状态，使用：

```powershell
python -X utf8 start_s21.py --connect --rounds 1 --resume-ready-practice
```

连续演练可改为 `--rounds 10`（范围1–100）。出现接口错误、覆盖失败或结果未核验时会停止批次。每次日志保存在 results/official-practice-时间戳/，其中summary.json记录结果，actions.jsonl与http-log.json记录过程。

## 不接评测器的本地测试

```powershell
python -X utf8 start_s21.py --local --seed 800
```

## 范围与来源

本入口适配官方Windows桌面程序“问题4演练测试”；没有正式测试入口。若正式评测要求上传指定格式的程序，应按该正式接口说明另外封装，不应把此演练入口直接当成正式提交接口。

源码验证命令：`python -m pip install pytest`，然后 `python -m pytest -q`。`results/` 默认不纳入Git，账号和演练日志不随代码发布。

重新运行30组本地配对：`python -X utf8 benchmark_layouts.py`。输出至 `results/paired/`，会覆盖该目录同名实验输出。evidence内为本次已完成实验的冻结摘要；原实验报告所述本地目录结构仅用于记录原验证过程，仓库中的复现入口以上述命令为准。

S21坐标与证书来自 https://github.com/3371879035-lang/shxjm-B-RL ，提交 b4af97c4cc6fbec14fcb4dfb76acd56d4753f87c。仅替换方法114初始测站，保留Bayes、调度和实际反馈停止规则。覆盖初始化使用整数证书和原顺序多边形合并双重检查，不忽略小面积残片。

本地30组配对两版均全清，S21平均总时间降低2.95%；尚不代表官方成绩。此包生成时只做离线核验，没有启动官方演练。

需完整保留 q4/、vendor/、run.py、practice_windows.py、start_s21.py、s21_layout.py、s21_certificate.json、check_s21_certificate.py。不要只复制启动脚本。
