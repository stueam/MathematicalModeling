"""One combined figure from the frozen fixed-R0 Q2 search, without recomputation.

Inputs copied unchanged from the previous two selected heatmap directories.
model_core.py is copied unchanged from src/Q2/model_core.py for reproducibility.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from model_core import Design

HERE = Path(__file__).resolve().parent


def main():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'SimHei'],
                         'mathtext.fontset': 'stix', 'font.size': 15, 'axes.unicode_minus': False})
    summary = json.loads((HERE/'summary.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.7))
    fig.subplots_adjust(left=.066, right=.884, bottom=.21, top=.86, wspace=.23)
    axis = np.arange(-1800., 1801., 60.)
    # Shared limits also used in the former pair of figures.
    norm = LogNorm(10, 600)
    for ax, key, title in zip(axes, ('symmetric', 'asymmetric'), ('(a) 对称情形', '(b) 不对称情形')):
        case = summary['scenarios'][key]
        model = Design(p1=case['p1'], theta=case['theta'], reception=1200)
        with (HERE/f'{key}_grid.csv').open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        z = np.full((len(axis), len(axis)), np.nan)
        for r in rows:
            z[round((float(r['y'])+1800)/60), round((float(r['x'])+1800)/60)] = float(r['J'])
        assert np.nanmin(z) >= norm.vmin and np.nanmax(z) <= norm.vmax
        boundary = Circle((0, 0), 1800, fill=False, ec='#718096', lw=1.1, zorder=5)
        ax.add_patch(boundary)
        mesh = ax.pcolormesh(axis, axis, np.ma.masked_invalid(z), shading='nearest',
                             cmap='viridis_r', norm=norm, rasterized=True)
        mesh.set_clip_path(boundary)
        contour = ax.contour(axis, axis, z, levels=[case['best']['J']*1.1],
                             colors=['#FFFFFF'], linewidths=1.4, linestyles='--', zorder=7)
        contour.set_path_effects([pe.Stroke(linewidth=2.5, foreground='#334155'), pe.Normal()])
        ax.add_patch(Polygon(np.asarray(model.p1_region.exterior.coords),
                             fc='#FF4968', ec='#FF4968', alpha=.65, lw=1.2, zorder=6))
        end = model.p1 + 1200*np.array([np.cos(model.theta), np.sin(model.theta)])
        line, = ax.plot([model.p1[0], end[0]], [model.p1[1], end[1]],
                        ls=':', lw=1.1, c='white', zorder=8)
        line.set_clip_path(boundary)
        ax.scatter(*model.p1, marker='x', s=52, c='white', lw=2, zorder=9)
        ax.annotate(r'$\mathbf{p}_1$', model.p1, xytext=(-25, -22), textcoords='offset points',
                    color='white', fontsize=15, zorder=10,
                    path_effects=[pe.withStroke(linewidth=2, foreground='#263146')])
        best = case['best']
        stars = [(best['x'], best['y'])]
        if key == 'symmetric':
            stars.append((best['x'], -best['y']))
        for x, y in stars:
            ax.scatter(x, y, marker='*', s=150, c='#FF4968', ec='white', lw=1.1, zorder=11)
        ax.set(xlim=(-1860, 1860), ylim=(-1860, 1860), xlabel='$x$ / m', ylabel='$y$ / m')
        ax.set_aspect('equal')
        ax.set_xticks([-1800, -900, 0, 900, 1800])
        ax.set_yticks([-1800, -900, 0, 900, 1800])
        ax.tick_params(labelsize=13, length=3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(title, loc='left', pad=37, fontsize=17, weight='bold')
        px, py = model.p1
        ax.text(0, 1.035, fr'$\mathbf{{p}}_1=({px:g},{py:g})$ m，$\theta_1={case["theta"]:g}^\circ$',
                transform=ax.transAxes, fontsize=14)
    cax = fig.add_axes([.911, .24, .018, .55])
    cb = fig.colorbar(mesh, cax=cax, ticks=[10, 20, 50, 100, 200, 400, 600])
    cb.ax.set_yticklabels(['10', '20', '50', '100', '200', '400', '600'])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=13)
    cb.set_label('期望覆盖半径 $J$ / m', fontsize=14, labelpad=8)
    fig.text(.902, .815, '越小越好', fontsize=13)
    handles = [Line2D([], [], marker='x', c='#25334A', ls='none', markersize=7, label='首测点'),
               Line2D([], [], marker='*', mfc='#FF4968', mec='#FF4968', c='none', markersize=12, label='推荐点'),
               Line2D([], [], c='#475569', ls='--', label='10%近优边界'),
               Line2D([], [], c='#FF4968', lw=2, label='初始可行区域')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.49, .015), ncol=4,
               frameon=False, fontsize=14, columnspacing=1.5, handlelength=1.6)
    fig.savefig(HERE/'q2_strategy_heatmaps.png', dpi=300, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
