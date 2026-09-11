"""Generate a reproducible Chinese result note from raw local artifacts."""
import csv
import json
from pathlib import Path
import numpy as np


def main():
    r=Path('reports')
    ppo=json.loads((r/'ppo_test500.json').read_text(encoding='utf-8'))
    base=json.loads((r/'heuristic_test500.json').read_text(encoding='utf-8'))
    pa={e['seed']:e for e in ppo['episodes']}
    ba={e['seed']:e for e in base['episodes']}
    assert pa.keys()==ba.keys()
    paired=[s for s in sorted(pa) if pa[s]['success'] and ba[s]['success']]
    diff=np.array([ba[s]['virtual_s']-pa[s]['virtual_s'] for s in paired])
    rng=np.random.default_rng(123)
    means=diff[rng.integers(0,len(diff),(5000,len(diff)))].mean(1)
    ci=np.quantile(means,[.025,.975])
    stats={'paired_success_episodes':len(paired),'mean_saved_s':float(diff.mean()),
           'saved_s_bootstrap_95_ci':ci.tolist(),'ppo_faster_fraction':float(np.mean(diff>0)),
           'ratio_of_mean_time_reduction':1-ppo['summary']['mean_success_virtual_s']/base['summary']['mean_success_virtual_s']}
    (r/'paired_comparison.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    timing=json.loads(Path('runs/full_pilot/timing.json').read_text(encoding='utf-8'))
    progress=list(csv.DictReader(Path('runs/full_pilot/progress.csv').open(encoding='utf-8')))
    last=progress[-1]
    bench=json.loads((r/'benchmark.json').read_text(encoding='utf-8'))
    b64=next(b for b in bench if b['num_envs']==64)
    rows=[]
    for file,label in [('heuristic_test500.json','简单基线：均匀测试集'),('ppo_test500.json','PPO：均匀测试集'),
                       ('ppo_edge_stress.json','PPO：边缘 + R=1000 + 相关误差'),
                       ('ppo_cluster_stress.json','PPO：聚集 + R=1000 + 相关误差')]:
        s=json.loads((r/file).read_text(encoding='utf-8'))['summary']
        rows.append(f"| {label} | {s['episodes']} | {s['success_rate']:.1%} | {s['mean_success_virtual_s']:.2f} | {s['p90_success_virtual_s']:.2f} |")
    note=f'''# 本地模拟器与 PPO 首轮实测报告

日期：2026-09-11。完成模拟器、PPO、基线、单元验证、性能基准和一次从零训练；没有使用 method1 数据或调用官方测试接口。

## 本机性能

- RTX 5060 Laptop 8 GB；Ryzen 9 8945HX 16 核 32 线程；约 64 GB 内存。
- NumPy 批量物理模拟 + Numba 几何状态和候选动作 + PyTorch CUDA PPO。
- 64 环境完整环境吞吐：{b64['env_actions_per_s']:,.0f} 次动作/秒；纯物理：{b64['physics_actions_per_s']:,.0f} 次动作/秒。该项不包含神经网络，已热身。
- 实际 PPO：{int(last['steps']):,} 步，训练循环 {float(last['wall_s']):.2f} 秒，平均 {float(last['sps']):,.0f} 步/秒。
- 含环境/模型初始化的程序内部耗时 {timing['wall_including_setup_s']:.2f} 秒；不包含首次安装依赖、首次 JIT 编译及 Python 导入之前的耗时。
- PyTorch 峰值已分配 GPU 张量内存 {timing['peak_gpu_allocated_mb']:.1f} MiB；这不是显卡全部进程占用。
- 首轮共完成 {last['episodes']} 个训练回合。模型、优化器、配置、训练曲线和逐局日志已保存。

这些速度对应本版小网络与候选动作几何表示，不能外推到精确非凸区域、全图遍历或更大网络。没有达到某个步数就保证收敛的结论。

## 固定独立场景评估

先用 100 局验证集检查流程，再在未用于训练、未用于调整参数的 500 局场景比较；本轮未调参或选择多个检查点，使用最终检查点。两种方法使用相同的 500 个种子（2000000–2000499）。正式配置为 20 频道、每局 10–16 个源、最大 512 动作的训练预算。

| 策略及分布 | 局数 | 完成率 | 完成局平均虚拟秒 | 完成局 P90 虚拟秒 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

在共同成功的 {len(paired)} 局中，PPO 比基线平均节省 **{diff.mean():.2f} 秒（{stats['ratio_of_mean_time_reduction']:.2%}）**。配对场景重采样的平均节省 95% 区间为 [{ci[0]:.2f}, {ci[1]:.2f}] 秒；{stats['ppo_faster_fraction']:.1%} 的场景 PPO 更快。这个区间只反映这一训练模型的测试场景抽样不确定性，不包括不同训练种子的波动，也不能证明面对官方分布同样有效。

## 已验证事项

23 项测试通过：5 m/20 m/接收半径边界；清除不切频道；重复清除；附件 105/111/194/199 秒计时示例；非法输入不推进状态；固定测角及跨 0°；1600 次动作与独立标量规则对照；多次测向后真值仍在外包区域；覆盖证书；没有真实数量提前结束；观察量不依赖隐藏真值；GAE 终止/截断；势函数奖励望远镜消去；频道置换和动作掩码。

## 结论与局限

本机性能已足以支持继续开展 PPO 实验。首轮结果证明这套动作空间内从零训练能够学到有用策略，并在所测场景中优于当前简单基线；没有与 method1 比较，也没有证明全局最优。

候选集和覆盖点带有人工几何知识；随机选这些候选在最初 100 局中也能完成，但平均耗时约 39354 秒。因而成功率较高不能归功于神经网络单独学会了全部几何规则，其主要增量体现在动作排序和减少移动耗时。

当前可行区域为凸外包，未完整利用负观测圆盘排除；它不是精确后验。存在概率和完整历史也未输入网络，本版特征压缩可能损失信息。模拟分布、哈希误差场是显式工作假设。100% 是有限样本结果，不是任意场景的完成保证。尚缺多训练种子、官方演练、HTTP 客户端和真实时间预算接入。

## 文件

- `../README.md`：规则、假设、安装、训练、续训、评估命令。
- `../runs/full_pilot/final.pt`：首轮模型和优化器检查点。
- `../runs/full_pilot/progress.csv` / `episodes.jsonl`：原始训练记录。
- `training_report.png`：首轮训练及 100 局验证集曲线。
- `trajectory.png` / `trajectory.json`：固定验证场景实际路线，绘图中的真实源仅用于离线审计。
- `ppo_test500.json` / `heuristic_test500.json`：逐局测试结果。
- `paired_comparison.json`：配对统计。
'''
    (r/'本地运行报告.md').write_text(note,encoding='utf-8')
    print(json.dumps(stats))


if __name__=='__main__':
    main()
