"""Two constructed Q1 illustrations; no statistical validation or random sampling.

Run this file to reconstruct fixed observations, solve by half-plane intersection,
and redraw q1_two_cases.png. geometry.py contains the existing paper's main solver.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle, ConnectionPatch
from geometry import wedge_constraints, halfplane_deque, diameter, TOL

OUT = Path(__file__).resolve().parent
INK, MUTED = '#263748', '#687888'
BLUE, GREEN, RED = '#3275A8', '#278574', '#BA543E'


def construct(vertices):
    """One supporting wedge per edge, plus four redundant valid observations."""
    center = vertices.mean(axis=0)
    detectors, bearings = [], []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        direction = (b-a)/np.linalg.norm(b-a)
        detectors.append(a-850*direction)
        bearings.append((np.degrees(np.arctan2(direction[1], direction[0]))+1) % 360)
    for angle in np.deg2rad([20, 110, 210, 330]):
        detectors.append(center+900*np.array([np.cos(angle), np.sin(angle)]))
        bearings.append((np.degrees(angle)+180) % 360)
    detectors, bearings = np.array(detectors), np.array(bearings)
    H, g = wedge_constraints(detectors, bearings)
    assert np.max(H @ vertices.T-g[:, None]) < 1e-7
    distance = np.linalg.norm(detectors-center, axis=1)
    assert np.all((distance > 5) & (distance < 1000))
    return detectors, bearings


def solve(detectors, bearings, expected):
    H, g = wedge_constraints(detectors, bearings)
    vertices = halfplane_deque(H, g)
    assert len(vertices) == len(expected)
    assert max(min(np.linalg.norm(v-p) for p in vertices) for v in expected) < 1e-7
    assert np.max(H @ vertices.T-g[:, None]) < 1e-7
    D, a, b = diameter(vertices)
    if vertices[a, 0] > vertices[b, 0]:
        a, b = b, a
    center = (vertices[a]+vertices[b])/2
    radii = np.linalg.norm(vertices-center, axis=1)
    farthest = int(np.argmax(radii))
    excess = float(radii[farthest]-D/2)
    return dict(vertices=vertices, D=D, center=center, a=a, b=b,
                farthest=farthest, rho=float(radii[farthest]),
                excess=excess, covered=excess <= TOL)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(cases):
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'SimHei'], 'font.size': 11,
        'mathtext.fontset': 'stix', 'axes.unicode_minus': False,
        'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5))
    fig.subplots_adjust(left=.04, right=.98, top=.94, bottom=.105,
                        wspace=.20, hspace=.26)
    for row, (detectors, bearings, result) in enumerate(cases):
        overview, zoom = axes[row]
        color = GREEN if result['covered'] else RED
        v, c, D = result['vertices'], result['center'], result['D']
        overview.set_title(f'({"a" if row == 0 else "c"}) 算例{row+1}：检测点布置',
                           loc='left', fontsize=13, color=INK, pad=12)
        zoom.set_title(f'({"b" if row == 0 else "d"}) 局部放大：直径圆'
                       + ('能覆盖' if result['covered'] else '不能覆盖'),
                       loc='left', fontsize=13, color=color, pad=12)
        for ax in (overview, zoom):
            ax.set_aspect('equal')
            ax.set_axis_off()
        for i, (point, angle) in enumerate(zip(detectors, np.deg2rad(bearings))):
            length = np.linalg.norm(point-c)+130
            ends = [point+length*np.array([np.cos(angle+s), np.sin(angle+s)])
                    for s in np.deg2rad([-1, 1])]
            overview.add_patch(Polygon([point, *ends], fc=BLUE, alpha=.025, ec='none'))
            for end in ends:
                overview.plot(*np.array([point, end]).T, color=BLUE, alpha=.23, lw=.6)
            end = point+length*np.array([np.cos(angle), np.sin(angle)])
            overview.plot(*np.array([point, end]).T, color=BLUE, alpha=.38,
                          lw=.7, ls=(0, (4, 4)))
            overview.scatter(*point, color=BLUE, s=26, ec='white', lw=.5, zorder=5)
            away = (point-c)/np.linalg.norm(point-c)
            overview.annotate(rf'$p_{{{i+1}}}$', point, xytext=tuple(away*13),
                              textcoords='offset points', ha='center', va='center',
                              color=BLUE, fontsize=10)
        span = 2200
        overview.set(xlim=(c[0]-span/2, c[0]+span/2),
                     ylim=(c[1]-span/2, c[1]+span/2))
        overview.add_patch(Polygon(v, fc=color, ec=color, lw=1, zorder=6))
        overview.annotate('定位区域', c, xytext=(35, 24), textcoords='offset points',
                          fontsize=10, color=MUTED,
                          arrowprops=dict(arrowstyle='-', color=MUTED, lw=.7))
        overview.plot([c[0]-900, c[0]-400], [c[1]-970]*2, color=INK, lw=1.6)
        overview.text(c[0]-650, c[1]-910, '500 m', ha='center', color=MUTED, fontsize=9)

        width = 34.
        middle = np.array([c[0], (min(c[1]-D/2, v[:, 1].min())+
                                  max(c[1]+D/2, v[:, 1].max()))/2])
        zoom.set(xlim=(middle[0]-width/2, middle[0]+width/2),
                 ylim=(middle[1]-width/2, middle[1]+width/2))
        zoom.add_patch(Circle(c, D/2, fc='#F1F5F7', ec='none', zorder=1))
        zoom.add_patch(Polygon(v, fc=color, alpha=.20, ec='none', zorder=2))
        zoom.add_patch(Polygon(v, fill=False, ec=color, lw=1.8, zorder=4))
        zoom.add_patch(Circle(c, D/2, fill=False, ec=INK, lw=1.6,
                             ls=(0, (5, 3)), zorder=5))
        A, B = v[result['a']], v[result['b']]
        zoom.plot(*np.array([A, B]).T, color=INK, lw=2, zorder=6)
        zoom.scatter(*v.T, color=color, s=26, ec='white', lw=.6, zorder=7)
        zoom.scatter(*c, color=INK, s=25, zorder=8)
        for label, point in [('A', A), ('B', B)]:
            away = (point-c)/np.linalg.norm(point-c)
            zoom.annotate(rf'${label}$', point, xytext=tuple(away*13),
                          textcoords='offset points', ha='center', va='center',
                          fontsize=13, color=INK)
        zoom.annotate(r'$O$', c, xytext=(c[0], min(v[:, 1].min()-1.5, c[1]-2.5)),
                      ha='center', color=INK, fontsize=13)
        if not result['covered']:
            far = v[result['farthest']]
            zoom.plot(*np.array([c, far]).T, color=RED, lw=1.1, ls=':', zorder=6)
            zoom.scatter(*far, color=RED, s=50, ec='white', lw=.7, zorder=8)
            zoom.annotate(r'$C$', far, xytext=(8, 8), textcoords='offset points',
                          color=RED, fontsize=13)
        x, y = middle+[7, -14]
        zoom.plot([x, x+5], [y, y], color=INK, lw=1.7)
        zoom.text(x+2.5, y+1, '5 m', ha='center', fontsize=9, color=MUTED)
        overview.add_patch(Rectangle(middle-width/2, width, width, fill=False,
                                    ec=MUTED, lw=.8, zorder=7))
        fig.add_artist(ConnectionPatch(xyA=tuple(middle+[width/2, width/2]),
            coordsA=overview.transData, xyB=(.02, .75), coordsB=zoom.transAxes,
            color='#AAB5BD', lw=.8, ls=(0, (3, 4)), zorder=0))
    handles = [Line2D([], [], marker='o', color=BLUE, ls='none', label='检测点'),
               Polygon([[0, 0]], fc='#D6E8E1', ec=GREEN, label='定位区域'),
               Line2D([], [], color=INK, lw=2, label='直径线段 AB'),
               Line2D([], [], color=INK, ls='--', label='直径圆（半径 D/2）')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .035),
               ncol=4, frameon=False, fontsize=10, columnspacing=1.8)
    fig.savefig(OUT/'q1_two_cases.png', dpi=300)
    plt.close(fig)


def main():
    polygons = [np.array([[-10, 0], [-8, -3], [6, -4], [10, 0], [5, 4], [-6, 3]], float)
                + [90, -50],
                np.array([[-10, 0], [-6, -2], [8, -1], [10, 1], [1, 16], [-4, 12]], float)
                + [-80, 70]]
    cases, records = [], []
    for index, polygon in enumerate(polygons, 1):
        detectors, bearings = construct(polygon)
        result = solve(detectors, bearings, polygon)
        assert result['covered'] == (index == 1)
        cases.append((detectors, bearings, result))
        write_csv(OUT/f'case{index}_observations.csv',
                  [dict(detector=i+1, x_m=p[0], y_m=p[1], bearing_deg=bearings[i])
                   for i, p in enumerate(detectors)])
        write_csv(OUT/f'case{index}_vertices.csv',
                  [dict(vertex=i+1, x_m=p[0], y_m=p[1])
                   for i, p in enumerate(result['vertices'])])
        records.append(dict(case=index, detector_count=len(detectors),
            vertex_count=len(result['vertices']), diameter_m=result['D'],
            diameter_circle_radius_m=result['D']/2,
            max_vertex_distance_m=result['rho'], coverage_excess_m=result['excess'],
            covered=result['covered'], diameter_center_m=result['center'].tolist(),
            diameter_endpoints_m=result['vertices'][[result['a'], result['b']]].tolist()))
    summary = dict(purpose='Constructed illustrations of two coverage outcomes, not a sample-frequency experiment',
                   region_algorithm='sorted half-plane intersection',
                   error_half_angle_deg=1, numerical_tolerance_m=TOL, cases=records)
    (OUT/'results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    draw(cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
