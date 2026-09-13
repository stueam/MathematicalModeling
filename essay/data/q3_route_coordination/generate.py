"""Draw one constructed route-coordination figure and audit its geometry."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Patch

HERE=Path(__file__).resolve().parent
INK='#263747'; TEAL='#147D83'; BLUE='#456A98'; CORAL='#C96749'; GRAY='#AAB3BB'


def length(route):
    return float(np.linalg.norm(np.diff(np.asarray(route),axis=0),axis=1).sum())


def route_line(ax, points, color, dashed=False):
    points=np.asarray(points)
    ax.plot(*points.T,color=color,lw=1.2 if dashed else 2.3,ls='--' if dashed else '-',zorder=3,alpha=.6 if dashed else 1)
    if not dashed:
        for a,b in zip(points[:-1],points[1:]):
            ax.annotate('',xy=a+.61*(b-a),xytext=a+.44*(b-a),
                        arrowprops=dict(arrowstyle='-|>',color=color,lw=1.4,mutation_scale=11),zorder=4)


def main():
    c=json.loads((HERE/'case.json').read_text(encoding='utf-8'))
    p={k:np.array(v) for k,v in c['points'].items()}
    regions={k:np.array(v) for k,v in c['regions'].items()}
    radius=c['coverage_radius_m']
    objective=lambda q: np.linalg.norm(q-p['B'])+np.linalg.norm(q-p['C'])
    constraints={'type':'ineq','fun':lambda q: radius**2-np.sum((regions['region_2']-q)**2,axis=1)}
    opt=minimize(objective,np.array([1240.,40.]),constraints=constraints,method='SLSQP',
                 options={'ftol':1e-9,'maxiter':500})
    assert opt.success, opt.message
    p['W']=opt.x
    cover_checks={'U_region1':(p['U'],regions['region_1']), 'A_region1':(p['A'],regions['region_1']),
                  'V_region2':(p['V'],regions['region_2']), 'W_region2':(p['W'],regions['region_2'])}
    max_dist={k:float(np.linalg.norm(vertices-q,axis=1).max()) for k,(q,vertices) in cover_checks.items()}
    assert all(v<=radius+1e-5 for v in max_dist.values())
    assert all(np.linalg.norm(v)<=1800 for vertices in regions.values() for v in vertices)
    assert all(np.linalg.norm(p[k])<=1800 for k in ('A','B','C'))
    routes=[[p[k] for k in seq] for seq in (('O','A','U','B','V','C'),('O','A','B','V','C'),('O','A','B','W','C'))]
    lengths=[length(r) for r in routes]
    assert lengths[0]>lengths[1]>lengths[2]
    (HERE/'metrics.json').write_text(json.dumps({'shifted_station_m':p['W'].tolist(),'route_length_m':lengths,
        'saving_by_step_m':[lengths[0]-lengths[1],lengths[1]-lengths[2]],
        'max_vertex_distance_m':max_dist,'coverage_radius_m':radius,
        'coverage_proof':'Each search region is convex. If all its vertices lie in the convex coverage disk, the entire region lies in it.',
        'interpretation':'Constructed mechanism illustration, not an actual run or performance benchmark.'},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'font.size':12.5,'mathtext.fontset':'stix','axes.unicode_minus':False,'text.color':INK})
    fig,axes=plt.subplots(1,3,figsize=(11.4,5.8))
    fig.subplots_adjust(left=.06,right=.99,bottom=.31,top=.94,wspace=.10)
    titles=['(a) 分别访问清除点与测站','(b) 在清除点顺便扫描','(c) 调整剩余测站的位置']
    for i,ax in enumerate(axes):
        ax.set_aspect('equal'); ax.set(xlim=(-180,1880),ylim=(-400,1510))
        ax.set_xticks([0,800,1600]); ax.set_yticks([0,600,1200])
        ax.set_xlabel('$x$ / m'); ax.set_ylabel('$y$ / m' if i==0 else '')
        if i: ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=12,length=3,colors='#64717D')
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#C5CDD3')
        ax.grid(color='#E8EDF0',lw=.6,zorder=0)
        for j,vertices in enumerate(regions.values(),1):
            ax.add_patch(Polygon(vertices,facecolor='#DDEDEB',edgecolor=TEAL,lw=1.1,zorder=1))
            center=vertices.mean(axis=0)
            ax.text(*center,f'$D_{j}$',ha='center',va='center',color=TEAL,fontsize=15,zorder=6)
        centers=[p['U'],p['V']] if i==0 else ([p['A'],p['V']] if i==1 else [p['A'],p['W']])
        for q in centers:
            ax.add_patch(Circle(q,radius,fill=False,ec=TEAL,lw=1.1,ls=(0,(4,3)),alpha=.55,zorder=1))
        if i: route_line(ax,routes[i-1],GRAY,True)
        route_line(ax,routes[i],BLUE if i==0 else TEAL)
        ax.scatter(*p['O'],marker='D',s=38,c=INK,zorder=7)
        ax.annotate('$O$',p['O'],xytext=(-12,-17),textcoords='offset points',fontsize=13)
        for key,offset in [('A',(-4,-19)),('B',(-4,-19)),('C',(5,-12))]:
            ax.scatter(*p[key],marker='s',s=63,c=CORAL,edgecolors='white',linewidths=.8,zorder=7)
            ax.annotate(f'${key}$',p[key],xytext=offset,textcoords='offset points',fontsize=13,color=CORAL)
        active=['U','V'] if i==0 else (['V'] if i==1 else ['W'])
        for key in active:
            ax.scatter(*p[key],marker='o',s=63,c='white',edgecolors=BLUE,lw=1.7,zorder=8)
            label=r'$V^{\prime}$' if key=='W' else f'${key}$'
            offset=(8,-22) if key=='W' else (8,8)
            ax.annotate(label,p[key],xytext=offset,textcoords='offset points',fontsize=13,color=BLUE)
        if i:
            ax.scatter(*p['U'],marker='o',s=57,facecolors='none',edgecolors=GRAY,lw=1,zorder=5)
            ax.annotate('省去 $U$',p['U'],xytext=(-10,12),textcoords='offset points',fontsize=12,color='#77838C')
            ax.scatter(*p['A'],s=135,facecolors='none',edgecolors=TEAL,lw=1.4,zorder=6)
        if i==2:
            ax.scatter(*p['V'],marker='o',s=57,facecolors='none',edgecolors=GRAY,lw=1,zorder=5)
            ax.annotate('',xy=p['W']+[60,80],xytext=p['V']+[50,-30],
                        arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=-.12',color=CORAL,lw=1.4),zorder=5)
        ax.set_title(titles[i],fontsize=13.5,fontweight='bold',loc='left',pad=13)
        ax.text(.5,-.24,f'路程 {lengths[i]:,.0f} m',transform=ax.transAxes,ha='center',color=INK,fontsize=14,fontweight='bold')
        detail='同一组搜索区域与清除位置' if i==0 else f'较左图减少 {lengths[i-1]-lengths[i]:,.0f} m'
        ax.text(.5,-.34,detail,transform=ax.transAxes,ha='center',color='#687785',fontsize=12)
    handles=[Line2D([],[],marker='s',color='none',mfc=CORAL,mec='white',markersize=8,label='清除停留点'),
             Line2D([],[],marker='o',color='none',mfc='white',mec=BLUE,markersize=8,label='独立搜索测站'),
             Patch(facecolor='#DDEDEB',edgecolor=TEAL,label='待搜索区域'),
             Line2D([],[],color=TEAL,ls='--',lw=1.2,label='997 m 覆盖边界')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.51,.015),ncol=4,frameon=False,fontsize=12.5,
               handlelength=1.5,columnspacing=1.1)
    fig.savefig(HERE/'q3_route_coordination.png',dpi=240,facecolor='white',bbox_inches='tight',pad_inches=.10)
    plt.close(fig)
    print('Route lengths (m):', [round(v,3) for v in lengths], 'moved station:',p['W'].round(4))


if __name__=='__main__': main()
