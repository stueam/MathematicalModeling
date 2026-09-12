"""Three actual +/-1 degree wedges intersect in prescribed triangles.

Each detector is 600 m back along one triangle edge. Its lower angular
boundary supports that edge; the upper boundary contains all three vertices.
Thus the intersection of the six halfplanes equals the target triangle.
This is a synthetic, verified construction, not an observed experiment.
"""
from pathlib import Path
import json
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle, ConnectionPatch
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'sans-serif',
    'font.sans-serif':['Microsoft YaHei','SimHei'], 'font.size':11,
    'mathtext.fontset':'stix','axes.unicode_minus':False,
    'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
INK='#263748'; MUTED='#687888'
COLORS=['#3275A8','#278574','#7864A2']
RED='#BA543E'; GREEN='#267D69'
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
data=[]
fig=plt.figure(figsize=(11.8,8.0))
for k,vertices in enumerate(triangles):
    x0=.055+.49*k
    p,rays,record=construct(vertices);data.append(record)
    accent=RED if k==0 else GREEN
    title='(a)  等边三角形：不能覆盖' if k==0 else '(b)  钝角三角形：可以覆盖'
    fig.text(x0,.945,title,fontsize=13,weight='bold',color=accent)
    overview=fig.add_axes([x0+.055,.62,.31,.29])
    main=fig.add_axes([x0+.012,.095,.40,.49])
    for ax in [overview,main]:
        ax.set_aspect('equal');ax.set_axis_off()
        for detector,(lo,hi),color in zip(p,rays,COLORS):
            ax.add_patch(Polygon([detector,detector+1800*lo,detector+1800*hi],
                                 facecolor=color,alpha=.035,edgecolor='none',zorder=0))
            for u in [lo,hi]:
                points=np.array([detector,detector+1800*u])
                ax.plot(*points.T,lw=.75 if ax is overview else .9,color=color,alpha=.8,zorder=1)
            mid=(lo+hi)/np.linalg.norm(lo+hi)
            points=np.array([detector,detector+1800*mid])
            ax.plot(*points.T,color=color,lw=.7,ls=(0,(5,5)),alpha=.55,zorder=1)
        ax.add_patch(Polygon(vertices,facecolor='#F3DDD1' if k==0 else '#D8ECE3',
                             edgecolor=accent,lw=1.8,zorder=3))
    overview.set(xlim=(-760,760),ylim=(-680,690))
    for i,(q,color) in enumerate(zip(p,COLORS),1):
        overview.scatter(*q,s=32,color=color,edgecolor='white',lw=.8,zorder=5)
        overview.annotate(rf'$\mathbf{{p}}_{i}$',q,xytext=((-13,-10) if i==1 else (10,0)),
                          textcoords='offset points',color=color,fontsize=12,ha='center',va='center')
    overview.scatter(0,vertices[:,1].mean(),s=14,color=accent,zorder=5)
    overview.annotate('定位区域',xy=(0,vertices[:,1].mean()),xytext=(-220,200),
                       color=MUTED,fontsize=9,arrowprops=dict(arrowstyle='-',color='#A2ACB5',lw=.7))
    main.set(xlim=(-15,15),ylim=(-12.2,20.2))
    # Exact radius and equal x/y scaling are essential for this comparison.
    main.add_patch(Circle((0,0),10,facecolor='#E7EFF5',alpha=.36,edgecolor='none',zorder=0))
    main.add_patch(Circle((0,0),10,fill=False,edgecolor=INK,lw=1.3,ls=(0,(5,3)),zorder=4))
    if k==0:
        arc=np.linspace(np.pi/3,2*np.pi/3,100)
        cap=np.vstack(([[-5,5*np.sqrt(3)],[0,10*np.sqrt(3)],[5,5*np.sqrt(3)]],
                       np.column_stack((10*np.cos(arc),10*np.sin(arc)))))
        main.add_patch(Polygon(cap,facecolor='#E6A38C',edgecolor=RED,hatch='///',lw=.6,zorder=3.5))
        main.annotate('圆外区域',xy=(1.7,12.3),xytext=(7.3,17.3),fontsize=10,color=RED,
                      arrowprops=dict(arrowstyle='-',color=RED,lw=.8),ha='center')
    main.plot([-10,10],[0,0],color=INK,lw=2,zorder=5)
    main.plot([0,0],[0,vertices[2,1]],color=accent,lw=1,ls=(0,(2,3)),zorder=5)
    main.scatter(*vertices.T,s=31,color=accent,edgecolor='white',lw=.7,zorder=6)
    main.scatter(0,0,s=22,color=INK,zorder=6)
    for name,q,offset in [('A',vertices[0],(-10,-12)),('B',vertices[1],(10,-12)),
                           ('C',vertices[2],(-12,9)),('O',np.zeros(2),(0,-15))]:
        main.annotate(rf'${name}$',q,xytext=offset,textcoords='offset points',
                      ha='center',fontsize=12,color=INK,zorder=7)
    main.text(-14.5,19.4,'局部放大',fontsize=9.5,color=MUTED)

handles=[Line2D([0],[0],color='#526F83',lw=1.1,label='楔形边界'),
         Line2D([0],[0],color='#526F83',lw=.9,ls=(0,(5,5)),label='中心示向线'),
         Line2D([0],[0],color=INK,lw=1.3,ls=(0,(5,3)),label='直径圆')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.024),ncol=3,
           frameon=False,fontsize=9.5,columnspacing=3)
for ext in ['png']:
    fig.savefig(OUT/f'q1_cover_cases.{ext}',dpi=300)
plt.close(fig)
(OUT/'q1_cover_cases.json').write_text(json.dumps(dict(
    construction='synthetic exact halfplane intersection, not measured data',
    equal_triangle=data[0],covered_triangle=data[1]),ensure_ascii=False,indent=2),encoding='utf-8')
assert not data[0]['covered'] and data[1]['covered']
print('Verified: 3 vertices in both intersections; equilateral uncovered, obtuse covered.')
print('Wrote q1_cover_cases.png and reproducible coordinates.')
