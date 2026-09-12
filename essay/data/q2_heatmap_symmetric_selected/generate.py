"""Reproducible Q2 heatmaps: actual expected enclosing radii, not image synthesis.
Run python make_heatmaps.py; default redraws saved data; --recompute repeats numerical searches.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv, json, sys, time
import numpy as np
from model_core import Design
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parent
SCENARIOS={'symmetric':dict(p1=(0.,0.),theta=0.),
           'asymmetric':dict(p1=(1200.,400.),theta=65.)}
MODELS={}
WIDTH=.125
STEP=60.

def init_worker():
    global MODELS
    MODELS={k:Design(**v,reception=1200,nr=80,na=16,arc=128) for k,v in SCENARIOS.items()}

def evaluate(task):
    key,x,y=task
    return key,MODELS[key].evaluate((x,y),WIDTH)

def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def batch(pool,tasks,label):
    result={k:[] for k in SCENARIOS}
    for i,(key,row) in enumerate(pool.map(evaluate,tasks,chunksize=12),1):
        result[key].append(row)
        if i%300==0 or i==len(tasks):
            print(f'{label}: {i}/{len(tasks)}',flush=True)
    return result

def compute():
    init_worker()
    axis=np.arange(-1800,1800+STEP/2,STEP)
    tasks=[(k,float(x),float(y)) for k in SCENARIOS for y in axis for x in axis if x*x+y*y<=1800**2]
    with ProcessPoolExecutor(max_workers=4,initializer=init_worker) as pool:
        coarse=batch(pool,tasks,'global scan')
        fine_tasks=[]
        for key,rows in coarse.items():
            best=min(rows,key=lambda r:r['J'])
            centers=[(best['x'],best['y'])]
            if key=='symmetric':centers.append((best['x'],-best['y']))
            points={(float(x),float(y)) for cx,cy in centers
                    for x in np.arange(cx-90,cx+91,15) for y in np.arange(cy-90,cy+91,15)
                    if x*x+y*y<=1800**2}
            fine_tasks.extend((key,x,y) for x,y in sorted(points))
        fine=batch(pool,fine_tasks,'local scan')
    summary={'parameters':{'target_radius_m':1800,'R0_m':1200,'error_half_angle_deg':1,
        'angle_bin_deg':WIDTH,'radial_cells':80,'angular_cells':16,
        'candidate_spacing_m':STEP,'local_spacing_m':15,
        'objective':'Expected minimum enclosing radius; includes normal, near, no-signal outcomes',
        'geometry':'Union of wedges in each angle bin; circular boundaries polygon-approximated',
        'prior':'Uniform area on target disk; independent uniform angular errors at distinct sites',
        'search_domain':'Candidate second detectors restricted to target disk'},'scenarios':{}}
    checks=[]
    for key,rows in coarse.items():
        model=MODELS[key]
        save_csv(OUT/f'{key}_grid.csv',rows)
        save_csv(OUT/f'{key}_refined.csv',fine[key])
        best=min(rows+fine[key],key=lambda r:r['J'])
        data=dict(SCENARIOS[key],initial_radius_m=model.initial_radius,
                  initial_area_m2=model.p1_region.area,best=best,
                  reduction=1-best['J']/model.initial_radius,
                  area_partition_error=model.area_partition_relative_error,
                  probability_sum_max_error=max(abs(r['probability_sum']-1) for r in rows+fine[key]))
        if key=='symmetric':
            index={(r['x'],r['y']):r['J'] for r in rows}
            data['reflection_max_error_m']=max(abs(v-index[(x,-y)]) for (x,y),v in index.items())
            assert data['reflection_max_error_m']<1e-5
        assert data['probability_sum_max_error']<1e-9
        assert data['area_partition_error']<1e-4
        u=np.array([np.cos(model.theta),np.sin(model.theta)])
        side=np.array([-u[1],u[0]])
        baselines={'repeat':model.p1,'forward_300m':model.p1+300*u,'lateral_300m':model.p1+300*side}
        data['baselines']={k:model.evaluate(q,WIDTH) for k,q in baselines.items() if np.linalg.norm(q)<=1800}
        # At the reported candidate, separately refine position integration, angle bins and arc geometry.
        for nr,na,arc,width in [(80,16,128,WIDTH),(160,32,256,WIDTH),(160,32,256,.0625),(160,32,256,.03125)]:
            m=Design(**SCENARIOS[key],reception=1200,nr=nr,na=na,arc=arc)
            val=m.evaluate((best['x'],best['y']),width)
            checks.append(dict(scenario=key,nr=nr,na=na,arc=arc,width=width,**val))
        data['validation_J_m']=[r['J'] for r in checks if r['scenario']==key]
        summary['scenarios'][key]=data
    save_csv(OUT/'convergence.csv',checks)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)

def load_rows(name):
    with (OUT/name).open(encoding='utf-8-sig') as f:
        return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]

def plot():
    s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    datasets={k:load_rows(f'{k}_grid.csv') for k in SCENARIOS}
    all_values=[r['J'] for rows in datasets.values() for r in rows]
    lo=5*np.floor(min(all_values)/5);lo=max(5,lo)
    hi=50*np.ceil(max(all_values)/50)
    cmap=LinearSegmentedColormap.from_list('cool_spectrum',
        ['#383170','#454F9D','#357DB0','#329FAA','#72C3BC','#B9DCD8','#EDF5F4'])
    norm=LogNorm(lo,hi)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei'],
        'mathtext.fontset':'stix','font.size':11,'axes.unicode_minus':False,
        'svg.fonttype':'none','axes.labelcolor':'#344653','text.color':'#243B4B'})
    axis=np.arange(-1800,1800+STEP/2,STEP)
    for key,rows in datasets.items():
        if key != 'symmetric': continue
        model=Design(**SCENARIOS[key],reception=1200,nr=80,na=16,arc=128)
        data=s['scenarios'][key];best=data['best']
        z=np.full((len(axis),len(axis)),np.nan)
        for r in rows:z[int(round((r['y']+1800)/STEP)),int(round((r['x']+1800)/STEP))]=r['J']
        fig,ax=plt.subplots(figsize=(8.5,7.35))
        fig.subplots_adjust(left=.105,right=.835,bottom=.15,top=.86)
        disk=Circle((0,0),1800,fill=False,ec='#7C8C92',lw=1.15,zorder=4)
        ax.add_patch(disk)
        mesh=ax.pcolormesh(axis,axis,np.ma.masked_invalid(z),shading='nearest',cmap=cmap,norm=norm,rasterized=True)
        mesh.set_clip_path(disk)
        ax.contour(axis,axis,z,levels=[best['J']*1.1],colors=['#F7FBFF'],linewidths=1.45,linestyles='--',zorder=5)
        p=np.asarray(model.p1_region.exterior.coords)
        ax.add_patch(Polygon(p,fc='#FFFFFF',ec='#455D72',lw=1.05,zorder=6))
        end=model.p1+1200*np.array([np.cos(model.theta),np.sin(model.theta)])
        bearing,=ax.plot([model.p1[0],end[0]],[model.p1[1],end[1]],color='#455D72',lw=1,ls=':',zorder=7)
        bearing.set_clip_path(disk)
        ax.scatter(*model.p1,s=55,c='#19384B',edgecolors='white',linewidths=1.2,zorder=8)
        ax.annotate(r'$\mathbf{p}_1$',model.p1,xytext=(-21,-21),textcoords='offset points',fontsize=14,zorder=9,
                    bbox=dict(fc='white',ec='none',alpha=.85,pad=1))
        stars=[(best['x'],best['y'])]
        if key=='symmetric' and abs(best['y'])>1e-8:stars.append((best['x'],-best['y']))
        for x,y in stars:ax.scatter(x,y,marker='*',s=205,c='#BBE9EE',edgecolors='white',linewidths=1.1,zorder=10)
        ax.set(xlim=(-1880,1880),ylim=(-1880,1880),xlabel='$x$ / m',ylabel='$y$ / m')
        ax.set_aspect('equal');ax.set_xticks([-1800,-900,0,900,1800]);ax.set_yticks([-1800,-900,0,900,1800])
        ax.spines[['top','right']].set_visible(False)
        for spine in ['left','bottom']:ax.spines[spine].set_color('#CCD4D8')
        ax.tick_params(colors='#667781',length=3)
        title='对称情形' if key=='symmetric' else '不对称情形'
        fig.text(.105,.95,title+'：第二检测点的期望定位效果',fontsize=16,weight='bold')
        px,py=SCENARIOS[key]['p1'];theta=SCENARIOS[key]['theta']
        fig.text(.105,.904,fr'$\mathbf{{p}}_1=({px:g},{py:g})\ \mathrm{{m}}$    $\theta_1={theta:g}^\circ$    $R_0=1200\ \mathrm{{m}}$',fontsize=11,color='#647781')
        cax=fig.add_axes([.866,.205,.022,.575]);cb=fig.colorbar(mesh,cax=cax)
        ticks=[v for v in [5,10,15,20,30,50,100,200,400,600,800] if lo<=v<=hi]
        cb.set_ticks(ticks);cb.set_ticklabels([str(v) for v in ticks]);cb.ax.minorticks_off()
        cb.set_label('期望最小覆盖圆半径 $J$ / m',labelpad=12)
        cb.outline.set_edgecolor('#CDD5D8')
        fig.text(.853,.817,'越小越好',fontsize=10,color='#647781')
        legend=[Line2D([],[],marker='o',color='none',markerfacecolor='#19384B',markersize=6,label='第一检测点'),
            Line2D([],[],marker='*',color='none',markerfacecolor='#BBE9EE',markeredgecolor='#4E7283',markersize=11,label='网格推荐点'),
            Line2D([],[],color='#708BA3',ls='--',label='10% 近优边界'),
            Line2D([],[],color='#455D72',lw=2,label='初始区域')]
        fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.49,.045),ncol=4,frameon=False,fontsize=9.5,handlelength=1.6,columnspacing=1.4)
        fig.text(.5,.019,'共同对数色标  ·  候选测点限于目标圆域  ·  数值分档近似',ha='center',fontsize=9,color='#75828A')
        for ext in ['png']:fig.savefig(OUT/f'q2_heatmap_{key}_selected.{ext}',dpi=300,facecolor='white')
        plt.close(fig)

if __name__=='__main__':
    if '--recompute' in sys.argv:compute()
    plot()
