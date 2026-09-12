"""Constructed outcome visualization, not a validation trial or expected value.

model_core.py is copied unchanged from src/Q2/model_core.py. Geometry uses the
paper's fixed R0 model and exact continuous bearings with a prescribed error.
Each strategy observes the same fixed source and the same prescribed error.
"""
from pathlib import Path
import csv
import json
import numpy as np
import shapely
from shapely.geometry import Point, mapping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle, Patch
from matplotlib.lines import Line2D
from model_core import Design, wedge, metrics

HERE = Path(__file__).resolve().parent
SOURCE = np.array([900., 0.])
SECOND_ERROR_DEG = .35
CASES = [('沿示向前进', (300., 0.)), ('侧向移动', (0., 300.)), ('推荐测点', (840., 495.))]
PURPLE, TEAL, YELLOW, PINK = '#440154', '#21918C', '#FDE725', '#FF4968'


def draw_region(ax, region, **kwargs):
    if region.geom_type == 'Polygon':
        ax.add_patch(Polygon(np.asarray(region.exterior.coords), **kwargs))
    elif hasattr(region, 'geoms'):
        for part in region.geoms:
            draw_region(ax, part, **kwargs)


def main():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'SimHei'],
                         'mathtext.fontset': 'stix', 'font.size': 15, 'axes.unicode_minus': False})
    model = Design(p1=(0, 0), theta=0, reception=1200, nr=80, na=16, arc=128)
    assert model.p1_region.covers(Point(*SOURCE))
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 9.1))
    fig.subplots_adjust(left=.068, right=.985, bottom=.18, top=.90, hspace=.33, wspace=.25)
    records, geometries = [], []
    for col, (name, xy) in enumerate(CASES):
        q = np.asarray(xy)
        distance = float(np.linalg.norm(SOURCE-q))
        assert 5 < distance < model.reception
        theta = float(np.arctan2(*(SOURCE-q)[::-1]) + np.deg2rad(SECOND_ERROR_DEG))
        w2 = wedge(q, theta, model.eps)
        p2 = (model.p1_region.intersection(w2)
              .intersection(Point(*q).buffer(model.reception, quad_segs=model.arc))
              .difference(Point(*q).buffer(5, quad_segs=model.arc)))
        assert p2.covers(Point(*SOURCE))
        radius, diameter = metrics(p2)
        circ = shapely.minimum_bounding_circle(p2)
        x0, y0, x1, y1 = circ.bounds
        center = np.array([(x0+x1)/2, (y0+y1)/2])
        vertices = np.asarray(p2.convex_hull.exterior.coords)
        max_distance = float(np.linalg.norm(vertices-center, axis=1).max())
        assert max_distance <= radius+1e-7
        row = dict(strategy=name, qx=xy[0], qy=xy[1], distance_m=distance,
                   bearing_deg=float(np.rad2deg(theta)%360), error_deg=SECOND_ERROR_DEG,
                   radius_m=radius, diameter_m=diameter, center_x=center[0], center_y=center[1])
        records.append(row)
        geometries.append(dict(type='Feature', properties=row, geometry=mapping(p2)))
        for r in range(2):
            ax = axes[r, col]
            draw_region(ax, model.p1_region, fc=PURPLE, ec=PURPLE, alpha=.12, lw=1, zorder=2)
            draw_region(ax, w2, fc=TEAL, ec=TEAL, alpha=.10, lw=.8, zorder=1)
            draw_region(ax, p2, fc=YELLOW, ec=TEAL, lw=1.5, zorder=4)
            ax.add_patch(Circle(center, radius, fill=False, ec=PURPLE, lw=1.4, ls='--', zorder=5))
            ax.scatter(*SOURCE, c=PINK, edgecolors='white', linewidths=.8, s=44, zorder=8)
            ax.set_aspect('equal')
            ax.set_xlabel('$x$ / m')
            ax.set_ylabel('$y$ / m' if col == 0 else '')
            ax.tick_params(labelsize=14, length=3)
            ax.grid(color='#DFE3E8', alpha=.6, lw=.6, zorder=0)
            ax.spines[['top', 'right']].set_visible(False)
            if r == 0:
                ax.scatter(*model.p1, c=PURPLE, marker='x', s=40, lw=1.5, zorder=8)
                ax.scatter(*q, c=TEAL, edgecolors='white', s=55, marker='^', zorder=8)
                ax.annotate('$q$', q, xytext=(8, 7), textcoords='offset points', fontsize=14)
                ax.annotate('$p_1$', model.p1, xytext=(-18, -20), textcoords='offset points', fontsize=13)
                ax.add_patch(Rectangle((750, -150), 300, 300, fill=False, ec='#64748B', lw=1, ls=':', zorder=6))
                ax.set(xlim=(-85, 1285), ylim=(-580, 620))
                ax.set_xticks([0, 600, 1200]); ax.set_yticks([-500, 0, 500])
                ax.set_title(f'({chr(97+col)}) {name}\n$q=({xy[0]:g},{xy[1]:g})$ m', fontsize=16, pad=10)
            else:
                ax.set(xlim=(750, 1050), ylim=(-150, 150))
                ax.set_xticks([750, 900, 1050]); ax.set_yticks([-150, 0, 150])
                ax.set_title(f'局部：$R_*={radius:.2f}$ m', fontsize=15, pad=9)
    handles = [Patch(fc=PURPLE, alpha=.18, ec=PURPLE, label='初始区域 $P_1$'),
               Patch(fc=TEAL, alpha=.22, ec=TEAL, label='第二示向楔形'),
               Patch(fc=YELLOW, ec=TEAL, label='交会区域 $P_2$'),
               Line2D([], [], c=PURPLE, ls='--', label='最小覆盖圆'),
               Line2D([], [], c='none', marker='o', mfc=PINK, mec='white', markersize=7, label='构造源位置')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .015), ncol=5,
               frameon=False, fontsize=14, handlelength=1.4, columnspacing=1.1)
    fig.savefig(HERE/'q2_intersection_comparison.png', dpi=300, facecolor='white')
    plt.close(fig)
    with (HERE/'observations.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    result = dict(purpose='Constructed result visualization; not expected performance or validation',
                  source_m=SOURCE.tolist(), first_point_m=[0, 0], first_bearing_deg=0,
                  first_error_deg=0, second_error_deg=SECOND_ERROR_DEG, reception_m=1200,
                  initial_radius_m=model.initial_radius, arc_quadrant_segments=model.arc, cases=records)
    (HERE/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (HERE/'regions.geojson').write_text(json.dumps(dict(type='FeatureCollection', features=geometries),
                                                 ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
