"""Draw the exclusion geometry and the frozen, certified S21 station layout."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Patch
from matplotlib.lines import Line2D
from shapely.geometry import Point, Polygon as SPolygon

HERE = Path(__file__).resolve().parent
BLUE, TEAL, CORAL, INK = '#456A98', '#147D83', '#C96749', '#263747'


def main():
    data = json.loads((HERE / 'geometry.json').read_text(encoding='utf-8'))
    stations = np.array(data['stations_m'])
    points = np.array(data['example_points_m'])
    region = SPolygon(points)
    for p in points:
        region = region.intersection(Point(p).buffer(1000, quad_segs=512))
    assert not region.is_empty and region.geom_type == 'Polygon'
    vertices = np.array(region.exterior.coords)
    assert max(np.linalg.norm(vertices - p, axis=1).max() for p in points) < 1000.00001
    assert len(stations) == 21
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
        'font.size': 12, 'mathtext.fontset': 'stix', 'axes.unicode_minus': False,
        'text.color': INK, 'axes.labelcolor': INK})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.5))
    fig.subplots_adjust(left=.075, right=.985, bottom=.21, top=.9, wspace=.27)
    for ax in axes:
        ax.set_aspect('equal')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_color('#AAB3BB')
        ax.tick_params(labelsize=10, colors='#596875')
        ax.set_xlabel('$x$ / m')
        ax.set_ylabel('$y$ / m')
    ax = axes[0]
    ax.set_title('(a) 三点无信号的联合排除区域', fontsize=13, pad=12)
    ax.set(xlim=(-1350,1350), ylim=(-1100,1600))
    ax.set_xticks([-1000,0,1000]); ax.set_yticks([-1000,0,1000])
    for p in points:
        ax.add_patch(Circle(p,1000,fill=False,ec=BLUE,lw=1,alpha=.48,ls='--'))
    ax.add_patch(Polygon(points,fill=False,ec=BLUE,lw=1.6))
    ax.add_patch(Polygon(vertices,fc='#B7DAD5',ec=TEAL,lw=1.8,zorder=3))
    ax.scatter(*points.T,s=42,c=BLUE,zorder=4)
    for p,label,offset in zip(points,['$p_i$','$p_j$','$p_k$'],[(-23,-17),(9,-17),(9,7)]):
        ax.annotate(label,p,xytext=offset,textcoords='offset points',fontsize=14,color=BLUE)
    c=np.array(region.centroid.coords[0])
    ax.text(*c,'$E_{ijk}$',ha='center',va='center',fontsize=17,color=TEAL)
    ax.legend(handles=[Line2D([],[],color=BLUE,lw=1.5,label='三测点的凸包'),
        Line2D([],[],color=BLUE,ls='--',alpha=.6,label='1000 m 距离约束'),
        Patch(fc='#B7DAD5',ec=TEAL,label='可排除区域')],loc='upper center',
        bbox_to_anchor=(.5,-.16),ncol=2,frameon=False,fontsize=10,columnspacing=1)
    ax=axes[1]
    ax.set_title('(b) 中心与两圈测点覆盖目标域',fontsize=13,pad=12)
    ax.set(xlim=(-2200,2200),ylim=(-2200,2200))
    ax.set_xticks([-1800,0,1800]);ax.set_yticks([-1800,0,1800])
    ax.add_patch(Circle((0,0),1800,fc='#EEF3F7',ec=BLUE,lw=1.4))
    for radius in [998,1866]:
        ax.add_patch(Circle((0,0),radius,fill=False,ec='#AAB3BB',lw=1,ls='--'))
    ax.scatter(*stations[1:9].T,c=TEAL,s=38,zorder=4)
    ax.scatter(*stations[9:].T,c=CORAL,s=38,marker='s',zorder=4)
    ax.scatter(0,0,c=INK,s=75,marker='*',zorder=4)
    ax.text(95,80,'$O$',fontsize=13)
    ax.annotate('内圈约 998 m',xy=(-706,-706),xytext=(-450,-420),fontsize=10,color=TEAL,
        arrowprops=dict(arrowstyle='-',lw=.8,color=TEAL))
    ax.annotate('外圈约 1866 m',xy=(1616,933),xytext=(300,1350),fontsize=10,color=CORAL,
        arrowprops=dict(arrowstyle='-',lw=.8,color=CORAL))
    ax.legend(handles=[Patch(fc='#EEF3F7',ec=BLUE,label='源分布圆域（1800 m）'),
        Line2D([],[],color=TEAL,marker='o',ls='',label='内圈 8 点'),
        Line2D([],[],color=CORAL,marker='s',ls='',label='外圈 12 点')],loc='upper center',
        bbox_to_anchor=(.5,-.16),ncol=2,frameon=False,fontsize=10,columnspacing=1)
    fig.savefig(HERE/'q4_coverage_geometry.png',dpi=240,facecolor='white',bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
