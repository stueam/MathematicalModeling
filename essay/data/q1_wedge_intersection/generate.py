"""Publication schematic reconstructed from the supplied sketch, not measured data.

Angles are intentionally enlarged to match the reference diagram. Polygon
vertices are computed by half-plane intersection rather than hand-positioned.
Run: python q1_wedge_geometry.py
"""
from pathlib import Path
import itertools
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
# The script is the source of these settings; JSON records them on each run.
LAYOUT = {
    'figure_size_inches': [12.6, 7.0],
    'main_axes_bounds': [.055, .155, .45, .675],
    'main_xlim': [.32, 7.42],
    'main_ylim': [.08, 6.9],
    'main_anchor': 'W',
    'zoom_axes_bounds': [.610, .275, .35, .495],
    'panel_title_alignment': 'center',
    'panel_title_gap': .020,
    'zoom_note_y': .215,
    'aspect': 'equal',
    'output_dpi': 300,
    'connector': {
        'target_axes_fraction': [0, .94],
        'line_color': '#B4BDC5',
        'line_width_points': .8,
        'dash_pattern_points': [3, 4],
        'arrow_shape': 'equilateral_triangle',
        'arrow_side_points': 7.5,
        'arrow_fill': '#A6B2BD',
    },
}
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'SimHei'],
    'mathtext.fontset': 'stix', 'font.size': 11,
    'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'savefig.facecolor': 'white',
})
INK, MUTED = '#263748', '#6A7785'
COLORS = ['#3275A8', '#278574', '#7864A2']
ACCENT, FILL = '#B84E39', '#EFC2AC'

def xy(pixel):
    return np.array([pixel[0] / 100, (800 - pixel[1]) / 100], dtype=float)

origins = [xy(p) for p in [(71,529),(250,738),(566,739)]]
ends = [[xy(q) for q in pair] for pair in [
    [(703,529),(702,403)], [(666,305),(563,247)], [(319,64),(459,46)]
]]
directions, normals, offsets = [], [], []
for p, pair in zip(origins, ends):
    uv = [(q-p)/np.linalg.norm(q-p) for q in pair]
    uv.sort(key=lambda u: np.arctan2(u[1],u[0]))
    directions.append(uv)
    lo, hi = uv
    # Both inequalities use normal dot (x-p) >= 0.
    for normal in [np.array([-lo[1],lo[0]]), np.array([hi[1],-hi[0]])]:
        normals.append(normal)
        offsets.append(normal @ p)
A, b = np.array(normals), np.array(offsets)

def clip(poly, normal, offset):
    result = []
    for s,e in zip(poly, np.roll(poly,-1,axis=0)):
        fs, fe = normal@s-offset, normal@e-offset
        if fs >= -1e-10:
            result.append(s)
        if (fs >= 0) != (fe >= 0):
            result.append(s + fs/(fs-fe)*(e-s))
    return np.array(result)

box = np.array([[-2,-2],[10,-2],[10,11],[-2,11]], dtype=float)
partial = box.copy()
for normal, offset in zip(A[:4],b[:4]):
    partial = clip(partial,normal,offset)
poly = partial.copy()
for normal, offset in zip(A[4:],b[4:]):
    poly = clip(poly,normal,offset)
assert len(poly) >= 3 and np.all(A@poly.T >= b[:,None]-1e-8)
assert np.all((poly > -1.9) & (poly < 9.9)), 'Artificial boundary is active'

rejected = []
for i,j in itertools.combinations(range(6),2):
    matrix = A[[i,j]]
    if abs(np.linalg.det(matrix)) < 1e-10:
        continue
    q = np.linalg.solve(matrix,b[[i,j]])
    if not np.all(A@q >= b-1e-8) and 3.6<q[0]<5.5 and 2.3<q[1]<5.1:
        rejected.append(q)

fig = plt.figure(figsize=LAYOUT['figure_size_inches'])
ax = fig.add_axes(LAYOUT['main_axes_bounds'])
ax.set_anchor(LAYOUT['main_anchor'])
zoom = fig.add_axes(LAYOUT['zoom_axes_bounds'])
fig.text(.055,.94,'楔形约束与交会定位区域', fontsize=19, weight='bold', color=INK)
fig.text(.055,.895,'三组示向度约束取交集，得到干扰源的共同可行区域',fontsize=11,color=MUTED)

for panel in [ax,zoom]:
    panel.set_aspect(LAYOUT['aspect'])
    panel.set_axis_off()
    for p,uv,color in zip(origins,directions,COLORS):
        panel.add_patch(Polygon([p,p+12*uv[0],p+12*uv[1]],
                                facecolor=color,alpha=.035,edgecolor='none',zorder=0))
    panel.add_patch(Polygon(partial,facecolor='#D7E4E7',alpha=.6,edgecolor='none',zorder=1))
    for p,uv,color in zip(origins,directions,COLORS):
        for u in uv:
            points = np.array([p,p+12*u])
            panel.plot(*points.T,color=color,lw=1.3 if panel is ax else 1.65,zorder=2)
        mid = uv[0]+uv[1]
        mid /= np.linalg.norm(mid)
        points = np.array([p,p+12*mid])
        panel.plot(*points.T,color=color,lw=1.0,ls=(0,(6,5)),alpha=.72,zorder=2)
    panel.add_patch(Polygon(poly,facecolor=FILL,edgecolor=ACCENT,lw=1.9,zorder=4))
    panel.scatter(*poly.T,s=19 if panel is ax else 33,color=ACCENT,
                  edgecolors='white',linewidths=.65,zorder=5)
    if rejected:
        panel.scatter(*np.array(rejected).T,s=20 if panel is ax else 28,
                      facecolors='white',edgecolors='#9AA7B1',lw=.9,zorder=3)

ax.set(xlim=LAYOUT['main_xlim'],ylim=LAYOUT['main_ylim'])
for i,(p,color) in enumerate(zip(origins,COLORS),1):
    ax.scatter(*p,s=65,facecolor=color,edgecolor='white',linewidth=1.5,zorder=6)
    dx,dy = {1:(-.15,-.38),2:(-.35,-.38),3:(.13,-.35)}[i]
    ax.text(p[0]+dx,p[1]+dy,rf'$\mathbf{{p}}_{i}$',fontsize=16,color=color)

low = poly.min(axis=0)-np.array([.30,.27])
high = poly.max(axis=0)+np.array([.30,.27])
zoom.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]))
center = poly.mean(axis=0)
zoom.text(center[0],center[1],r'$P=\bigcap_{i=1}^{3}W_i$',ha='center',va='center',
          fontsize=17,color='#713322',zorder=8)
for i,v in enumerate(poly,1):
    vector = v-center
    vector /= np.linalg.norm(vector)
    zoom.annotate(rf'$\mathbf{{v}}_{i}$',v,xytext=tuple(vector*15),
                  textcoords='offset points',ha='center',va='center',
                  fontsize=12,color=INK,zorder=7,
                  bbox=dict(facecolor='white',edgecolor='none',pad=.5,alpha=.85))

# Equal aspect can resize the axes: anchor labels to the rendered panel bounds.
fig.canvas.draw()
for panel, label in [(ax, '(a)  多点交会'), (zoom, '(b)  定位区域局部放大')]:
    bounds = panel.get_position()
    fig.text((bounds.x0+bounds.x1)/2, bounds.y1+LAYOUT['panel_title_gap'],
             label, ha=LAYOUT['panel_title_alignment'], va='bottom',
             fontsize=12, weight='bold', color=INK)
zoom_bounds = zoom.get_position()
fig.text((zoom_bounds.x0+zoom_bounds.x1)/2, LAYOUT['zoom_note_y'],
         '仅保留同时满足全部楔形约束的交点', ha='center', fontsize=10.5, color=MUTED)

# A subtle zoom guide connects corresponding geometry without crossing labels.
pad = np.array([.09,.09])
focus_lo,focus_hi = poly.min(axis=0)-pad,poly.max(axis=0)+pad
ax.add_patch(Rectangle(focus_lo,*(focus_hi-focus_lo),fill=False,ec='#A0ADB6',lw=.8,ls=(0,(3,3)),zorder=7))
# Construct the arrow in display coordinates, so unequal figure width/height
# cannot distort its three equal sides. Stop the shaft inside the triangle base.
connector = LAYOUT['connector']
start = ax.transData.transform(focus_hi)
tip = zoom.transAxes.transform(connector['target_axes_fraction'])
direction = (tip-start)/np.linalg.norm(tip-start)
perpendicular = np.array([-direction[1], direction[0]])
side = connector['arrow_side_points']*fig.dpi/72
base = tip-np.sqrt(3)*side/2*direction
arrow_vertices = np.array([tip, base+side/2*perpendicular, base-side/2*perpendicular])
to_figure = fig.transFigure.inverted()
shaft = to_figure.transform(np.array([start, base+direction]))
fig.add_artist(Line2D(shaft[:,0], shaft[:,1], transform=fig.transFigure,
                     color=connector['line_color'], lw=connector['line_width_points'],
                     ls=(0, tuple(connector['dash_pattern_points'])), zorder=0))
fig.add_artist(Polygon(to_figure.transform(arrow_vertices), closed=True,
                       transform=fig.transFigure, facecolor=connector['arrow_fill'],
                       edgecolor='none', linewidth=0, zorder=4, clip_on=False))

handles = [
    Line2D([0],[0],color=INK,lw=1.5,label='误差边界'),
    Line2D([0],[0],color=INK,lw=1.1,ls=(0,(6,5)),label='中心示向线'),
    Line2D([0],[0],marker='o',linestyle='none',markerfacecolor='white',
           markeredgecolor='#9AA7B1',markersize=5,label='被排除的交点'),
    Polygon([[0,0]],facecolor=FILL,edgecolor=ACCENT,label='最终定位区域'),
]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.079),
           ncol=4,frameon=False,handlelength=2.4,columnspacing=2.6,fontsize=10)
fig.text(.5,.036,'几何示意图：误差角经放大以便展示，不代表实际 ±1° 数据。',
         ha='center',fontsize=9,color=MUTED)
for suffix in ['png','svg','pdf']:
    fig.savefig(OUT/f'q1_wedge_intersection.{suffix}',dpi=LAYOUT['output_dpi'])
plt.close(fig)
(OUT/'q1_wedge_geometry.json').write_text(json.dumps({
    'kind':'schematic, reconstructed from reference; not experimental data',
    'detectors':[p.tolist() for p in origins], 'polygon_vertices':poly.tolist(),
    'error_half_angles_degrees':[float(np.degrees(np.arccos(np.clip(u@v,-1,1)))/2)
                                for u,v in directions],
    'layout': LAYOUT,
    'outputs': {suffix: f'q1_wedge_intersection.{suffix}' for suffix in ['png']},
},ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Wrote q1_wedge_intersection.png; polygon has {len(poly)} vertices.')
