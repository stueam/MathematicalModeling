# 论文算法代码

本目录以 [`essay/essay.tex`](../essay/essay.tex) 为保留范围，仅提供以下算法及其运行、验证依赖：

| 目录 | 论文内容 | 唯一运行算法 |
| --- | --- | --- |
| `Q2` | 问题二：固定接收半径下的贝叶斯选点 | 反馈后完整可行区域的期望最小覆盖圆半径最小化 |
| `Q3` | 问题三 | `bayes-fast`：固定未知半径后验、联合任务路线、覆盖与有限完成 |
| `Q4` | 问题四 | S21＋probes：定向源后验、21点覆盖证书、联合路线与补测 |

Q3、Q4 分别只有一个 `Policy`。本地、演练、正式入口均调用同一实现；没有策略选择或消融开关。文件与论文的逐项对应关系见 [PAPER_SCOPE.md](PAPER_SCOPE.md)。问题一的几何计算仍在论文引用的 `essay/data/q1_two_cases/`，未复制到本目录。

## 安装和运行

实测环境为 Python 3.12。先创建并激活虚拟环境，在 `src` 下安装运行与验证依赖：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-tested.txt
python verify.py
```

`requirements-tested.txt` 固定本次验证使用的版本。只运行程序可安装 `requirements.txt`；各题也可独立复制目录并安装该目录的 `requirements.txt`。

分别运行完整本地流程：

```bash
python Q2/run.py --scenario symmetric
python Q2/run.py --scenario asymmetric
python Q3/run.py local --seed 0
python Q4/run.py local --seed 800
```

运行论文对应的完整地图批次：

```bash
python Q3/run.py reproduce --workers 4
python Q4/run.py reproduce --workers 4
```

Q3 固定为 40 个随机地图＋24 个压力地图；Q4 固定为 16 个随机地图＋14 个压力地图。`--workers` 仅并行本地独立地图。`benchmark --seed N --rounds M` 可连续运行相同论文算法，`validate` 仅运行论文压力地图。

Q3/Q4 每次执行新建结果目录，保存配置、代码哈希、动作反馈、决策与汇总，`--output` 指定目录必须尚不存在；超时、动作截断或未完成时返回非零退出码。Q2 默认输出到算例目录，保留多次结果时请指定不同的 `--output` 路径。

## 验证与模拟器入口

`verify.py` 检查运行模块可达性、未使用代码、单元测试、Q2 两个完整网格搜索、Q3/Q4 完整本地任务、S21 证书和入口帮助。测试不能使闲置的运行模块通过可达性检查。

本次实测结果见 [PAPER_VALIDATION.md](PAPER_VALIDATION.md)。本地验证不会连接官方模拟器。Windows 演练与正式运行说明见 [FORMAL_TESTING.md](FORMAL_TESTING.md)。
