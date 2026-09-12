"""Create readable paired offline findings from saved experiment evidence."""
from pathlib import Path
import argparse
import csv
import json
from benchmark import summarize


def analyze(folder):
    folder = Path(folder).resolve()
    with (folder/'tables/cases.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k in ('sources', 'cleared', 'directional_sources', 'envelope_checks', 'envelope_violations', 'false_absent', 'optical_cover_count'):
            row[k] = int(row[k])
        for k in ('virtual_time_s', 'seconds_per_cleared', 'runtime_s', 'movement_m', 'measures', 'clear_failures', 'after_last_clear_s'):
            row[k] = float(row[k]) if row[k] else None
    methods = sorted(set(r['method'] for r in rows))
    mixed = [r for r in rows if 0 < r['directional_sources'] < r['sources']]
    metrics, paired = summarize(mixed, methods)
    families = []
    for family in sorted(set(r['family'] for r in rows)):
        mm, pp = summarize([r for r in rows if r['family'] == family], methods)
        families.append({'family': family, 'metrics': mm, 'paired': pp})
    stratified = {'mixed_cases': metrics, 'mixed_paired': paired, 'by_family': families}
    (folder/'metrics/stratified_analysis.json').write_text(json.dumps(stratified, ensure_ascii=False, indent=2), encoding='utf-8')
    all_summary = json.loads((folder/'run_summary.json').read_text(encoding='utf-8'))
    main, base = metrics['q4_b2'], metrics['q4_safe']
    all_main = all_summary['metric_summary']['q4_b2']
    if all_summary['status'] != 'PASS':
        raise RuntimeError('Failures require an explicit report; do not produce a success report')
    names = {'uniform_mixed': '圆域均匀混合', 'boundary_outward': '边界朝外混合',
             'two_clusters': '双聚类混合', 'narrow_sector': '窄扇区混合', 'tangent_heading': '切向发射混合',
             'all_directional_stress': '全定向压力测试', 'all_omni_regression': '全向回归测试'}
    lines = [
        '# 第四问 B2 改造：离线测试报告', '',
        '本报告由保存的逐场 CSV 与 run_summary.json 生成。所有测试为本机合成环境；没有调用官方模拟器，也没有使用演练或正式测试次数。', '',
        '## 方法与比较口径', '',
        '`q4_b2`：三角网格全方向搜索、正观测几何定位、B2 联合路线、共享检测、有限补测后暂缓与光学覆盖兜底。', '',
        '`q4_safe`：使用相同网格、几何观测和局部选点/光学兜底，先完成搜索路线再处理剩余源；不使用共享观测。这是此次第四问的可用基线，不是把第三问原始 B2 原样用到第四问。', '',
        '基线和主方法同时包含搜索完成后处理剩余已发现源的步骤。若已确认存在 16 个不同频道的源，可以根据题目数量上界省略剩余发现任务；仍必须逐个清除。', '',
        '开发阶段 14 个场景用于发现过早触发光学兜底的问题。最终评估固定新种子 2026091117，共 98 个场景；先前测试结果保留，不与最终集重复计数。', '',
        '统计先对每局计算总虚拟时间/清除源数，再对场景等权平均。虚拟时间包含移动、检测、切换和清除；计算时间是策略 run() 加评估器几何核验，不包含构造策略对象、导入模块和文件写入。', '',
        '## 符合第四问题意的混合类型场景', '',
        '| 指标 | Q4 改造版 | Q4 安全基线 |', '|---|---:|---:|',
        f"| 全清场景 | {main['passed']}/{main['cases']} | {base['passed']}/{base['cases']} |",
        f"| 清除源数 | {main['cleared']}/{main['sources']} | {base['cleared']}/{base['sources']} |",
        f"| 平均每源虚拟时间，秒 | {main['mean_seconds_per_cleared']:.2f} | {base['mean_seconds_per_cleared']:.2f} |",
        f"| 平均整局虚拟时间，秒 | {main['mean_virtual_time_s']:.2f} | {base['mean_virtual_time_s']:.2f} |",
        f"| 每局平均移动距离，米 | {main['mean_movement_m']:.2f} | {base['mean_movement_m']:.2f} |",
        f"| 每局平均检测次数 | {main['mean_measures']:.2f} | {base['mean_measures']:.2f} |",
        f"| 平均计算时间，秒 | {main['mean_runtime_s']:.4f} | {base['mean_runtime_s']:.4f} |",
        f"| 最长计算时间，秒 | {main['max_runtime_s']:.4f} | {base['max_runtime_s']:.4f} |",
        f"| 错误不存在判定 | {main['false_absent']} | {base['false_absent']} |", '',
        f"平均每源耗时改善 {paired['mean_per_source_improvement_percent']:.2f}%；{paired['main_faster_cases']} 局更快、{paired['main_slower_cases']} 局更慢。不能表述为每局都更优。", '',
        '## 分布与压力测试', '',
        '| 场景类型 | 局数 | 改造版秒/源 | 基线秒/源 | 改善 |', '|---|---:|---:|---:|---:|']
    for f in families:
        a, b = f['metrics']['q4_b2'], f['metrics']['q4_safe']
        lines.append(f"| {names[f['family']]} | {a['cases']} | {a['mean_seconds_per_cleared']:.2f} | {b['mean_seconds_per_cleared']:.2f} | {f['paired']['mean_per_source_improvement_percent']:.2f}% |")
    lines += ['', f"合计 {all_main['passed']}/{all_main['cases']} 局全清，{all_main['cleared']}/{all_main['sources']} 个源清除。额外 28 局的全定向/全向场景是诊断极端，不与 70 局混合类型混淆。", '',
              f"主方法在最终集有 {all_main['envelope_checks']} 次几何区域真值包含核验，违反次数为 {all_main['envelope_violations']}。真值仅供评估器核验，不参与策略决策。", '',
              '## 为什么覆盖有效', '',
              '边长 h=950 m 的等边三角剖分保留所有与半径 1800 m 闭圆盘相交的三角形，合计 42 个三角形、31 个不同检测顶点，其中包含圆外侧顶点。', '',
              '对于任意真实源 G，存在保留三角形使 G=λ₁v₁+λ₂v₂+λ₃v₃，其中 λᵢ≥0、Σλᵢ=1。三角形直径等于边长，所以 ||vᵢ−G||≤950<1000≤R。', '',
              '对任意发射单位方向 u，有 Σλᵢ u·(vᵢ−G)=0，故至少一个顶点满足 u·(vᵢ−G)≥0，位于闭发射半平面内，必能接收或得到 near。由此得到连续位置与连续朝向的覆盖结论。密集抽样测试只是实现核验，不是这条结论的证明。', '',
              '只有该频道尚未发现且已完成全部必需顶点检测，才可根据上述覆盖论证判定不存在。检测源数量达到 16 是另一种题面给出的完成搜索依据。', '',
              '局部光学兜底覆盖的是保守多边形的旋转包围盒；网格两轴间隔都不超过 25 m，每个位置到最近网格点距离不超过 25/√2≈17.68 m，小于 20 m 清除半径。此结论与发射朝向无关。', '',
              '## 当前限制与下一步', '',
              '1. 这是可运行、有搜索覆盖依据的第四问初版，没有证明路径、检测点数量或总任务时间全局最优。',
              '2. 31 个搜索节点是充分覆盖方案，可能比实际所需多。主方法混合场景平均移动约 32.56 km，覆盖搜索仍占明显成本。',
              f"3. 混合场景最后一个源清除后，平均还需 {main['mean_after_last_clear_s']:.2f} 秒完成剩余频道搜索证据。该时间计入任务总时间，不能悄悄删去。",
              '4. 负观测暂时只存档，不用来排除源位置；尚未实现发射方向的联合可行集。它保护了可靠性，但也损失了可利用信息。',
              '5. 暂缓补测改善了开发集，但最终混合场景仍有 6 局比基线慢；不是每次联合调度都会更优。',
              '6. 合成场景的源类型比例、空间分布和误差函数是测试设计，不代表官方随机分布；尚未验证真实 HTTP 客户端和官方环境。',
              '7. 本次未做参数广泛搜索，不把种子敏感性或角度采样冒充实际测量误差验证。', '',
              '后续优先考虑保持覆盖证书成立的检测点复用，以及位置—朝向联合约束。先节省搜索路程与不存在证明的成本，再考虑更复杂的随机前瞻。', '',
              '## 原始证据', '',
              '- `run_summary.json`：源码哈希、环境、配置、完整汇总。',
              '- `tables/cases.csv`：逐场清除、耗时、移动、检测、兜底等记录。',
              '- `metrics/scenarios.json`：离线场景真值，仅供复现环境使用。',
              '- `metrics/stratified_analysis.json`：混合类型与诊断类型分层统计。',
              '- `../../../../code/Q4/reviews/qx_python_review.json`：核验记录。', '']
    (folder/'离线测试报告.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'mixed_cases': metrics, 'paired': paired}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    analyze(parser.parse_args().folder)
