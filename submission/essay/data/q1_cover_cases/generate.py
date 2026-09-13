"""Two exact +/-1 degree wedge intersections and their diameter-circle coverage.

Synthetic construction using three detectors per triangle. Each detector is
600 m behind its triangle edge. Run python generate.py to regenerate the PNG
and construction JSON in this directory; add --pdf for a vector PDF.
All coordinates and intersection checks are retained from the original script.
"""
from pathlib import Path
import itertools
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle, ConnectionPatch
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pdf', action='store_true', help='Also export vector PDF')
args = parser.parse_args()
epsilon=np.deg2rad(1)

def rotate(u,a):
    return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])@u

def construct(vertices):
    p=[]; rays=[]; normals=[]; offsets=[]; bearings=[]
    for v,w in zip(vertices,np.roll(vertices,-1,axis=0)):
        u=(w-v)/np.linalg.norm(w-v)
        detector=v-600*u
        lo,hi=u,rotate(u,2*epsilon)
        p.append(detector); rays.append((lo,hi))
        for normal in [np.array([-lo[1],lo[0]]),np.array([hi[1],-hi[0]])]:
            normals.append(normal);offsets.append(normal@detector)
        bearings.append(float(np.degrees(np.arctan2(u[1],u[0])+epsilon)%360))
    N,b=np.array(normals),np.array(offsets)
    assert np.min(N@vertices.T-b[:,None]) >= -1e-9
    recovered=[]
    for i,j in itertools.combinations(range(6),2):
        M=N[[i,j]]
        if abs(np.linalg.det(M))<1e-10: continue
        q=np.linalg.solve(M,b[[i,j]])
        if np.all(N@q>=b-1e-8) and not any(np.linalg.norm(q-r)<1e-7 for r in recovered):
            recovered.append(q)
    recovered=np.array(recovered)
    assert len(recovered)==3
    vertex_error=max(min(np.linalg.norm(q-v) for v in vertices) for q in recovered)
    assert vertex_error<1e-7
    # Each wedge lies in a triangle-side halfplane and contains the triangle.
    # These inclusions also certify boundedness and exact equality, beyond vertex tests.
    source=vertices.mean(axis=0)
    angle_errors=[]
    for detector,bearing in zip(p,bearings):
        v=source-detector
        true_bearing=np.degrees(np.arctan2(v[1],v[0]))%360
        angle_errors.append(float((bearing-true_bearing+180)%360-180))
    assert max(abs(e) for e in angle_errors)<=1+1e-8
    assert max(np.linalg.norm(source-detector) for detector in p)<1000
    assert max(np.linalg.norm(v) for v in vertices)<1800
    return np.array(p),rays,dict(
        vertices_m=vertices.tolist(),detectors_m=np.array(p).tolist(),
        measured_bearings_deg=bearings,error_half_angle_deg=1,
        reconstructed_vertex_max_error_m=float(vertex_error),
        example_source_m=source.tolist(),example_source_angle_errors_deg=angle_errors,
        diameter_m=20,diameter_circle_center_m=[0,0],diameter_circle_radius_m=10,
        max_vertex_distance_from_center_m=float(max(np.linalg.norm(v) for v in vertices)),
        covered=bool(max(np.linalg.norm(v) for v in vertices)<=10+1e-8))

triangles=[np.array([[-10,0],[10,0],[0,10*np.sqrt(3)]]),
           np.array([[-10,0],[10,0],[0,10/np.sqrt(3)]])]
TRIANGLES = triangles
CASES = [construct(v) for v in TRIANGLES]
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei'],
                     'mathtext.fontset': 'stix', 'font.size': 11, 'axes.unicode_minus': False,
                     'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
INK = '#263748'
MUTED = '#687888'
COLORS = ['#3275A8', '#278574', '#7864A2']
ACCENTS = ['#B64E35', '#237E69']
NAMES = ['等边三角形', '钝角三角形']
WINDOW = np.array([[-17., -13.], [17., -13.], [17., 24.], [-17., 24.]])

def clip(poly, normal, b):
    out = []
    for s, e in zip(poly, np.roll(poly, -1, axis=0)):
        fs, fe = normal @ s - b, normal @ e - b
        if fs >= -1e-10:
            out.append(s)
        if (fs >= 0) != (fe >= 0):
            out.append(s + fs / (fs - fe) * (e - s))
    return np.array(out)

def intersection(k, count):
    poly = WINDOW.copy()
    ps, rays, _ = CASES[k]
    for p, (lo, hi) in zip(ps[:count], rays[:count]):
        for n in (np.array([-lo[1], lo[0]]), np.array([hi[1], -hi[0]])):
            poly = clip(poly, n, n @ p)
    return poly

def setup(ax, ylim=(-13,24)):
    ax.set(xlim=(-17,17), ylim=ylim, aspect='equal')
    ax.set_axis_off()

def vertex_labels(ax, k):
    for name, q, offset in zip('ABC', TRIANGLES[k], [(-9,-13),(9,-13),(0,9)]):
        ax.annotate('$'+name+'$', q, xytext=offset, textcoords='offset points',
                    ha='center', color=INK, fontsize=12, zorder=12)

def boundaries(ax, k, count=3, arrows=False):
    ps, rays, _ = CASES[k]
    for i, (p, (lo,hi)) in enumerate(zip(ps[:count], rays[:count])):
        for j, u in enumerate((lo,hi)):
            q = np.array([p, p+1800*u])
            ax.plot(*q.T, color=COLORS[i], lw=1.3 if j==0 else .9,
                    ls='-', alpha=1 if j==0 else .45, zorder=3)
        if arrows:
            v, w = TRIANGLES[k][i], TRIANGLES[k][(i+1)%3]
            mid = (v+w)/2
            n = np.array([-lo[1],lo[0]])
            ax.annotate('', xy=mid+2.5*n, xytext=mid+.3*n,
                        arrowprops=dict(arrowstyle='->',color=COLORS[i],lw=1.2),zorder=10)

def construction(ax, k, count=3, label=True):
    setup(ax)
    ps, rays, _ = CASES[k]
    for i,(p,(lo,hi)) in enumerate(zip(ps[:count], rays[:count])):
        ax.add_patch(Polygon([p,p+1800*lo,p+1800*hi],facecolor=COLORS[i],alpha=.045,edgecolor='none'))
    poly = intersection(k,count)
    ax.add_patch(Polygon(poly,facecolor='#E8D3AE' if count<3 else '#E2E7EB',edgecolor='none',alpha=.85,zorder=2))
    boundaries(ax,k,count,arrows=count==3)
    if count==3:
        for i,(v,w) in enumerate(zip(TRIANGLES[k],np.roll(TRIANGLES[k],-1,axis=0))):
            ax.plot(*np.array([v,w]).T,color=COLORS[i],lw=3.2,zorder=6)
        ax.scatter(*TRIANGLES[k].T,color=INK,edgecolor='white',s=22,zorder=8)
        if label:vertex_labels(ax,k)
    if count<3:
        ax.plot(*np.vstack([TRIANGLES[k],TRIANGLES[k][0]]).T,color='#6D7781',lw=.9,ls=':',zorder=4)

def coverage(ax,k):
    setup(ax)
    v=TRIANGLES[k];color=ACCENTS[k]
    ax.add_patch(Polygon(v,facecolor='#F2DCCF' if k==0 else '#D8EAE2',edgecolor=color,lw=1.8))
    ax.add_patch(Circle((0,0),10,fill=False,edgecolor=INK,lw=1.3,ls=(0,(6,3)),zorder=6))
    if k==0:
        arc=np.linspace(np.pi/3,2*np.pi/3,150)
        cap=np.vstack(([[-5,5*np.sqrt(3)],[0,10*np.sqrt(3)],[5,5*np.sqrt(3)]],np.column_stack((10*np.cos(arc),10*np.sin(arc)))))
        ax.add_patch(Polygon(cap,facecolor='#E5A88F',edgecolor=color,hatch='///',lw=.5,zorder=3))
    ax.plot([-10,10],[0,0],color=INK,lw=2)
    ax.plot([0,0],[0,v[2,1]],color=color,ls=':',lw=1.1)
    ax.scatter(*v.T,color=color,edgecolor='white',s=28,zorder=10)
    ax.scatter(0,0,color=INK,s=20,zorder=10)
    vertex_labels(ax,k)
    ax.annotate('$O$',(0,0),xytext=(0,-14),textcoords='offset points',ha='center',fontsize=12)
    formula = r'$OC=10\sqrt{3}\,\mathrm{m}>10\,\mathrm{m}$' if k==0 else r'$OC=(10/\sqrt{3})\,\mathrm{m}<10\,\mathrm{m}$'
    ax.text(.5,.015,formula,transform=ax.transAxes,ha='center',color=color,fontsize=13)

def footer(fig, text, y=.026):
    fig.text(.5,y,text,ha='center',color=MUTED,fontsize=10)

def save(fig,name):
    for ext in (('png','pdf') if args.pdf else ('png',)):
        fig.savefig(OUT/f'{name}.{ext}',dpi=300)
    plt.close(fig)

# 1. Establish the real detector geometry, then explicitly connect its local crop.
fig=plt.figure(figsize=(14,9.6))
gs=fig.add_gridspec(2,3,left=.06,right=.97,bottom=.10,top=.88,wspace=.25,hspace=.43)
for k in range(2):
    a,b,c=[fig.add_subplot(gs[k,j]) for j in range(3)]
    for ax,title in zip((a,b,c),('检测点与楔形布局','交会处放大：三条边的来源','同一三角形的直径圆判定')):
        ax.set_title(title,fontsize=12,color=INK,pad=12)
    ps,rays,_=CASES[k]
    a.set(xlim=(-750,750),ylim=(-650,650),aspect='equal');a.set_axis_off()
    for i,(p,(lo,hi)) in enumerate(zip(ps,rays)):
        a.add_patch(Polygon([p,p+1600*lo,p+1600*hi],facecolor=COLORS[i],alpha=.08,edgecolor='none'))
        for u in (lo,hi):a.plot(*np.array([p,p+1600*u]).T,color=COLORS[i],lw=.8)
        mid=(lo+hi)/np.linalg.norm(lo+hi)
        a.plot(*np.array([p,p+1600*mid]).T,color=COLORS[i],lw=.65,ls=(0,(6,4)),alpha=.65)
        a.scatter(*p,color=COLORS[i],s=24,zorder=4)
        a.annotate(rf'$p_{i+1}$',p,xytext=(-10,-12),textcoords='offset points',color=COLORS[i],fontsize=12)
    a.add_patch(Polygon(TRIANGLES[k],facecolor=INK))
    a.add_patch(Rectangle((-17,-13),34,37,fill=False,edgecolor=INK,lw=1.2,zorder=5))
    a.annotate('放大框',xy=(17,24),xytext=(200,210),fontsize=10,color=INK,
               arrowprops=dict(arrowstyle='-',lw=.8,color=INK))
    a.plot([-650,-350],[-570,-570],color=INK,lw=2)
    a.text(-500,-535,'300 m',ha='center',fontsize=9,color=MUTED)
    construction(b,k);coverage(c,k)
    fig.add_artist(ConnectionPatch(xyA=(17,24),coordsA=a.transData,xyB=(-17,20),coordsB=b.transData,
                                  color='#9EAAB3',lw=.8,ls=(0,(3,4)),arrowstyle='->'))
    fig.canvas.draw()
    bounds=a.get_position()
    fig.text(bounds.x0,gs[k,0].get_position(fig).y1+.080,
             ['(a) 等边三角形：不能覆盖', '(b) 钝角三角形：可以覆盖'][k],
             ha='left',va='bottom',color=ACCENTS[k],fontsize=18,weight='bold')
handles=[Line2D([0],[0],color='#526F83',lw=1.1,label='楔形边界'),
         Line2D([0],[0],color='#526F83',lw=.9,ls=(0,(6,4)),label='中心示向线'),
         Line2D([0],[0],color=INK,lw=1.3,ls=(0,(6,3)),label='直径圆')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.027),ncol=3,
           frameon=False,fontsize=11,columnspacing=3)
for k in range(2):
    recovered = intersection(k, 3)
    assert len(recovered) == 3
    assert max(min(np.linalg.norm(q-v) for v in TRIANGLES[k]) for q in recovered) < 1e-7
assert not CASES[0][2]['covered'] and CASES[1][2]['covered']
save(fig, 'q1_cover_cases')
(OUT/'q1_cover_cases.json').write_text(json.dumps({
    'construction': 'synthetic exact halfplane intersection, not measured data',
    'equal_triangle': CASES[0][2], 'covered_triangle': CASES[1][2],
    'layout': {'panels': 'two rows: overview, local intersection, circle coverage',
               'row_titles': ['(a) 等边三角形：不能覆盖', '(b) 钝角三角形：可以覆盖'],
               'figure_size_inches': [14, 9.6], 'dpi': 300},
    'outputs': {'png': 'q1_cover_cases.png'}
}, ensure_ascii=False, indent=2), encoding='utf-8')
print('Verified both triangles; wrote q1_cover_cases.png and q1_cover_cases.json.')
