"""Export a static figure from an existing audited practice batch; no simulator."""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from analyze_practice import COST_FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    args = parser.parse_args()
    cases = json.loads((args.audit/'cases.json').read_text())
    summary = json.loads((args.audit/'summary.json').read_text())
    chinese_font = Path('/mnt/c/Windows/Fonts/msyh.ttc')
    if chinese_font.exists():
        font_manager.fontManager.addfont(str(chinese_font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(chinese_font)).get_name()
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'path', 'axes.unicode_minus': False})
    fig, (ax, right) = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={'width_ratios': [2, 1]})
    y, left = np.arange(len(cases)), np.zeros(len(cases))
    colors = ('#4278a8', '#68a691', '#d99944')
    for key, label, color in zip(('search', 'localize_clear', 'completion_tail'),
                                 ('最后清除前的搜索', '已发现源的定位／清除', '最后清除后的完成证明'), colors):
        values = np.array([sum(c['phases'].get(key, {}).get(k, 0) for k in COST_FIELDS) for c in cases])
        ax.barh(y, values, left=left, label=label, color=color, height=.67)
        left += values
    ax.set_yticks(y, [f'{i+1:02d}  N={c["source_count"]}, D={c["directional_count"]}' for i, c in enumerate(cases)])
    ax.invert_yaxis()
    ax.set_xlabel('虚拟耗时（秒），分项累计；N 为总源数，D 为定向源数')
    ax.set_title('15 局官方演练：均已全清')
    ax.grid(axis='x', alpha=.16)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.04), frameon=False, fontsize=9)
    tail = summary['tail_counterfactual']
    values = [tail['mean_actual_s'], tail['mean_reordered_s'], tail['mean_pruned_s']]
    bars = right.bar(['实际收尾', '同站重新排序', '删去冗余站'], values,
                     color=['#d99944', '#8c99a6', '#4278a8'], width=.6)
    right.bar_label(bars, labels=[f'{v:.1f} s' for v in values], padding=6)
    right.set_ylim(0, max(values)*1.35)
    right.set_ylabel('平均收尾虚拟耗时（秒）')
    right.set_title('离线收尾复核：15 局均重建完成证明')
    right.text(.5, -.16, '仅使用已记录的同频道、同位置无信号反馈。\n这是离线方案分析，并非改进策略的正式运行成绩。',
               transform=right.transAxes, ha='center', fontsize=9, color='#555555')
    right.grid(axis='y', alpha=.16)
    right.set_axisbelow(True)
    fig.suptitle(f'平均 {summary["virtual_time_s"]["mean"]:.0f} 虚拟秒；移动 {summary["movement_m"]["mean"]/1000:.2f} km',
                 fontsize=16, y=1.01)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        target = args.audit/f'attribution.{suffix}'
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite an existing figure: {target}')
        fig.savefig(target, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(args.audit/'attribution.png')


if __name__ == '__main__':
    main()
