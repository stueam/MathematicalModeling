# Q3 method2：高性能本地模拟器与独立 PPO

本项目针对 B 题问题三的全向干扰源。无需 method1、教师标签或在线蒙特卡洛；PPO 直接使用模拟器奖励训练。尚未接入官方 HTTP 模拟器，不消耗官方测试次数。

## 快速运行（PowerShell）

工作目录：`C:\Users\18041\Desktop\B题\MathematicalModeling\Q3\method2`。
当前 Python 环境位于仓库根目录 `.venv`，以下命令在本目录执行。

```powershell
..\..\.venv\Scripts\python.exe -m pytest -q
..\..\.venv\Scripts\python.exe -m q3ppo.benchmark --iterations 1000
..\..\.venv\Scripts\python.exe -m q3ppo.train --config configs/full.json --out runs/full_new --steps 1000000 --device cuda
..\..\.venv\Scripts\python.exe -m q3ppo.evaluate --policy ppo --checkpoint runs/full_new/final.pt --episodes 100 --out reports/full_new.json
```

每次训练使用新的输出目录，避免覆盖已有实验。`--max-seconds 600` 可在一次 PPO 更新结束后保存并停止；Ctrl+C 也会保存。`latest.pt` 每十次更新保存，`final.pt` 在结束时保存。`--resume 路径` 载入模型和优化器继续训练，但会用当前 seed 重建环境；这是继续优化，不是逐位恢复上次环境/RNG。续训应指定不同训练 seed，避免重复同一批场景。

```powershell
..\..\.venv\Scripts\python.exe -m q3ppo.train --config configs/full.json --resume runs/full_new/final.pt --seed 200000 --out runs/full_continue --steps 5000000 --device cuda
```

默认 64 个环境、每次收集 128 步，实际步数向上取整到 8192 的整数倍。`configs/single.json` 和 `configs/small.json` 是课程训练场景，不代表正式问题三；共享候选编码器支持从少频道模型迁移到 20 频道。

## 新机器安装

在仓库根目录执行：

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r Q3/method2/requirements.txt
```

本机已验证 RTX 5060 Laptop 上的 CUDA 前向和反向计算。复现版本见 `requirements-local-lock.txt`，CUDA 包来自 PyTorch 官方索引。训练设备不支持 CUDA 时会报错，不会悄悄回退 CPU；可显式指定 `--device cpu`。

## 结构与性能设计

| 文件 | 职责 |
|---|---|
| `physics.py` | NumPy 批量动作、隐藏场景、固定接收半径、可复现测角、耗时 |
| `belief.py` | Numba 编译的保守几何更新、频道状态、候选动作 |
| `env.py` | 可观察完成判据、奖励、批量自动重置、简单基线 |
| `model.py` | 共享候选编码器、Actor/Critic、GAE |
| `train.py` | 独立 PPO、日志、检查点、时间预算 |
| `evaluate.py` | 固定留出场景比较，逐局记录，分布压力测试 |
| `benchmark.py` | 分别测纯物理和完整环境吞吐率 |
| `trace.py` / `plot_report.py` | 轨迹和训练曲线，离线生成图片 |

环境以单进程的批量数组执行，不为每个环境启动一个 Python 进程；Numba 编译热点几何代码。每个频道最多四个候选动作，20 频道共 80 个槽位，不需要全图离散扫描或求期望积分。网络按候选共享参数，使用掩码汇总上下文，不记忆频道编号。循环里不写图、不等待虚拟时间、不访问网络。首次使用包含 Numba 编译和 CUDA 初始化，基准中单独排除热身。

## 题目事实与模拟假设

题面和附件来源为本机 `B题.pdf`、`附件1_B.docx`、`附件2_B.docx`。

- 正式配置 20 频道、10–16 个静态全向源，源位置在半径 1800 m 的圆内；每源接收半径在 1000–1500 m 内且局内固定。
- 检测距离 ≤5 m 返回 near；>5 m 且 ≤R 返回示向度；否则 no_signal。清除距离 ≤20 m 成功，且只清除指定频道。
- 移动速度 5 m/s；检测 5 s，变更检测频道增加 1 s；清除失败 3 s，成功 5 s，清除不改变检测频道。
- 同一源在同一测点的误差固定，返回角度保留两位小数并归一化。默认误差由世界种子、频道和坐标哈希生成；这是一种模拟误差场，不宣称与官方实现相同。
- **训练分布假设**：源总数在 10–16 上离散均匀；频道子集均匀；位置按面积均匀；半径均匀且逐源独立。题目并未规定这些概率分布。提供 edge、cluster、最小/最大半径、smooth 空间相关误差压力测试。
- 时间使用浮点秒，每次动作四舍五入到微秒后累计；附件没有公开内部取整细节，不宣称逐微秒复刻官方结果。
- 不复刻 HTTP、幂等请求、登录、真实时间窗口、网络故障和官方日志。训练环境的 `step` 为已验证合法的物理动作语义，支持越界/非法输入拒绝且不推进状态。不得直接当作正式提交客户端。

## 几何与动作的适用范围

1. 位置区域采用凸外包多边形：初始大圆用 32 边外切多边形，接收圆用 16 边外切多边形，与测角楔形做半平面裁剪。误差界为 1.005°，额外 0.005° 保守覆盖返回角度取整。
2. 本版**没有在凸区域里扣除 no_signal 的 1000 m 圆、near 下界或失败 clear 的 20 m 圆**；负反馈仅保留最近结果、测点、覆盖记录并屏蔽部分重复动作。因此区域偏大、信息利用不完全，不是完整后验，更不是最优定位算法。
3. 区域中心取包围盒中点；半径取到所有顶点的最大距离，它是外包圆半径，**不是最小包围圆半径**。半径 ≤20 m 才标记保证清除，但策略也可选择未获保证的尝试清除；失败按真实规则计费。
4. 未知频道提供原地和两个最近未扫描覆盖点；已发现频道提供区域中心、两个侧向测点和区域中心清除。候选生成器属于人工设计的动作空间；“无需 method1”不等于完全没有几何先验。策略无法选择候选集外的任意位置。
5. 覆盖点是原点及半径 1500 m 上六个 60° 等角点。在半径 ≤1000 m 内由原点覆盖；外环任意源到最近外点的方位差 ≤30°，距离平方 ≤ρ²+1500²−3000ρcos30°，在 ρ∈[1000,1800] 上最大距离约 901.93 m，故七点都无信号可认证该频道不存在（仅 Q3 全向成立）。
6. 只有所有频道已清除/认证不存在，或清除数量达到已知上限，才成功结束。**不会因为模拟器知道真实目标全被清除就自动结束**。模拟真值仅用于生成反馈和审计统计。
7. 候选动作和有限训练步数不提供任意场景的必达保证；几何证书正确性与策略最终是否及时取得证书是两回事。

## PPO 与奖励

Actor 输出掩码候选动作的概率，Critic 估计状态价值；使用裁剪 PPO（clip=0.2）、GAE（λ=0.95）、梯度裁剪、熵项和 KL 提前停止。γ=1，以无折扣总虚拟耗时为主目标。网络和价值估计只使用可观察信息，没有真实坐标、源总数或真实 R。

奖励为 `-动作耗时/1000 + Φ(next)-Φ(current)`，其中 Φ 由已发现/已清除/已认证频道和区域收缩组成。所有终止状态令 Φ=0，完整轨迹上辅助项望远镜消去；对照测试覆盖这一性质。默认超过 512 动作或 360000 虚拟秒仍未认证完成，另扣 400。512 动作是训练预算，**不是题目规定**。有限失败罚分只是代理目标，不构成全清除保证；报告必须优先展示完成率。

真实预算终止包含失败罚分，不跨越自动 reset 引导价值；收集 128 步的批次截断会用下一状态价值 bootstrap。评估使用确定性最大概率动作，因此其表现可能与训练时随机策略不同，必须独立验证。

## 评估与图表

```powershell
..\..\.venv\Scripts\python.exe -m q3ppo.evaluate --policy heuristic --episodes 100 --out reports/heuristic_uniform.json
..\..\.venv\Scripts\python.exe -m q3ppo.evaluate --policy random --episodes 100 --out reports/random_uniform.json
..\..\.venv\Scripts\python.exe -m q3ppo.evaluate --policy ppo --checkpoint runs/full_new/final.pt --episodes 100 --out reports/ppo_uniform.json
..\..\.venv\Scripts\python.exe -m q3ppo.evaluate --policy ppo --checkpoint runs/full_new/final.pt --layout edge --radius-mode min --error-mode smooth --seed 1100000 --episodes 100 --out reports/ppo_stress.json
..\..\.venv\Scripts\python.exe -m q3ppo.plot_report --run runs/full_new
..\..\.venv\Scripts\python.exe -m q3ppo.trace --checkpoint runs/full_new/final.pt
```

比较时使用相同 seed 起点和 episode 数。评估严格收集指定场景集合，避免“先完成的 N 局”导致成功/短局选择偏差。默认测试种子 1000000 起，与当前训练种子隔离；大量长训需自行为训练、验证、最终测试分配不重叠种子区间。调参使用验证集，最终结论另用未看过的测试集。

参考：[PPO 原始论文](https://arxiv.org/abs/1707.06347)、[PyTorch 官方安装](https://pytorch.org/get-started/locally/)。本地结果见 `reports`；模型和逐步训练日志见 `runs`，两者不代表官方测试结果。
