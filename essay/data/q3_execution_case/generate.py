"""Render the audited local seed-0 case from frozen CSV and JSON files."""
from pathlib import Path
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
INK='#263747'; BLUE='#456A98'; TEAL='#147D83'; CORAL='#C96749'; GOLD='#C8A24A'


def read_and_check():
    summary=json.loads((HERE/'summary.json').read_text(encoding='utf-8'))
    provenance=json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))
    assert provenance['case_kind']=='local_simulation'
    with (HERE/'actions.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    points=np.array([[float(r['x_m']),float(r['y_m'])] for r in rows])
    costs=dict(move_s=0.,switch_s=0.,measure_s=0.,clear_success_s=0.,clear_failure_s=0.)
    previous=np.zeros(2); channel=1; clock=0.
    for r,point in zip(rows,points):
        move=round(float(np.linalg.norm(point-previous))/5,6)
        switch=int(r['kind']=='measure' and int(r['channel'])!=channel)
        if r['kind']=='measure':
            key='measure_s'; operation=5; channel=int(r['channel'])
        else:
            key='clear_success_s' if r['result']=='success' else 'clear_failure_s'
            operation=5 if r['result']=='success' else 3
        costs['move_s']+=move; costs['switch_s']+=switch; costs[key]+=operation
        clock=round(clock+move+switch+operation,6)
        assert math.isclose(clock,float(r['virtual_time_s']),abs_tol=1e-6)
        previous=point
    for k,value in costs.items(): assert math.isclose(value,summary['costs'][k],abs_tol=1e-6)
    assert math.isclose(clock,summary['virtual_time_s'],abs_tol=1e-6)
    assert sum(r['result']=='success' for r in rows)==summary['cleared_count']
    return rows,points,summary


def main():
    rows,points,s=read_and_check()
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'font.size':12,'mathtext.fontset':'stix','axes.unicode_minus':False,'text.color':INK})
    fig=plt.figure(figsize=(11.4,6.6))
    ax=fig.add_axes([.075,.19,.45,.73])
    bars=fig.add_axes([.67,.36,.29,.40])
    # Every submitted point is retained in the chronological polyline. Repeated
    # in-place scans remain in the CSV but do not add movement to the display.
    route=np.vstack([np.zeros(2),points])
    ax.add_patch(Circle((0,0),1800,fill=False,ec='#BEC8D0',lw=1.1,zorder=0))
    ax.plot(*route.T,color=BLUE,lw=1.8,zorder=2)
    for a,b in zip(route[:-1],route[1:]):
        if np.linalg.norm(b-a)>250:
            ax.annotate('',xy=a+.62*(b-a),xytext=a+.47*(b-a),
                arrowprops=dict(arrowstyle='-|>',color=BLUE,lw=1.1,mutation_scale=10),zorder=3)
    measured=np.array(list(dict.fromkeys(tuple(p) for r,p in zip(rows,points) if r['kind']=='measure')))
    cleared=np.array([p for r,p in zip(rows,points) if r['result']=='success'])
    ax.scatter(*measured.T,marker='o',s=36,facecolors='white',edgecolors=TEAL,linewidths=1.25,zorder=4)
    ax.scatter(*cleared.T,marker='s',s=47,c=CORAL,edgecolors='white',linewidths=.8,zorder=6)
    # Offsets distinguish neighboring successes; numbers denote order, not channel.
    offsets=[(-18,8),(-22,7),(-25,-3),(7,-17),(3,12),(8,-11),
             (8,-12),(9,-2),(8,-17),(8,6),(8,-4),(8,7),(-3,10),(-20,6),(-23,-7)]
    for index,(point,offset) in enumerate(zip(cleared,offsets),1):
        ax.annotate(str(index),point,xytext=offset,textcoords='offset points',fontsize=12,color=CORAL,
            bbox=dict(boxstyle='round,pad=.06',fc='white',ec='none',alpha=.87),zorder=8)
    ax.scatter(0,0,marker='D',s=50,c=INK,edgecolors='white',linewidths=.6,zorder=9)
    ax.annotate('起点',(0,0),xytext=(7,7),textcoords='offset points',fontsize=11,color=INK)
    end=points[-1]
    ax.annotate('终点',end,xytext=(-57,-34),textcoords='offset points',fontsize=11,color=INK,
        arrowprops=dict(arrowstyle='-',lw=.8,color='#7D8992'),zorder=8)
    ax.set_aspect('equal'); ax.set(xlim=(-1950,1950),ylim=(-1920,1950),xlabel='$x$ / m',ylabel='$y$ / m')
    ax.set_xticks([-1800,-900,0,900,1800]); ax.set_yticks([-1800,-900,0,900,1800])
    ax.grid(color='#E8EDF0',lw=.6,zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#C5CDD3')
    ax.tick_params(labelsize=12,colors='#64717D',length=3)
    ax.set_title('(a) 实际执行轨迹',loc='left',fontsize=15,fontweight='bold',pad=16)
    values=np.array([s['costs']['move_s'],s['costs']['measure_s'],s['costs']['switch_s'],
                     s['costs']['clear_success_s']+s['costs']['clear_failure_s']])
    labels=['移动','检测','切频','清除']
    colors=[BLUE,TEAL,GOLD,CORAL]
    y=np.arange(4)
    bars.barh(y,values,color=colors,height=.42,zorder=3)
    bars.set_yticks(y,labels); bars.invert_yaxis()
    bars.set(xlim=(0,2500),xticks=[0,1000,2000],xlabel='虚拟耗时 / s')
    bars.set_ylim(3.55,-.6)
    bars.tick_params(axis='y',length=0,pad=9,labelsize=12.5)
    bars.tick_params(axis='x',length=3,labelsize=10.5,colors='#64717D')
    bars.spines[['top','right','left']].set_visible(False)
    bars.spines['bottom'].set_color('#C5CDD3')
    bars.grid(axis='x',color='#E8EDF0',lw=.7,zorder=0)
    for yi,value in zip(y,values):
        bars.text(value+45,yi,f'{value:,.1f} s',va='center',fontsize=12,color=INK)
    fig.text(.61,.94,'(b) 虚拟耗时构成',fontsize=15,fontweight='bold',color=INK)
    fig.text(.61,.87,f'完成 {s["cleared_count"]}/{s["source_count"]} 个源  ·  总计 {s["virtual_time_s"]:,.2f} s',fontsize=12.5,color=INK)
    fig.text(.61,.205,f'移动距离  {s["movement_m"]/1000:.2f} km\n检测 {s["measure_count"]} 次  ·  清除失败 {s["failed_clears"]} 次',
        fontsize=12,color=INK,linespacing=1.8)
    handles=[Line2D([],[],color=BLUE,lw=1.8,label='实际路线'),
             Line2D([],[],marker='o',color='none',mfc='white',mec=TEAL,markersize=7,label='检测位置'),
             Line2D([],[],marker='s',color='none',mfc=CORAL,mec='white',markersize=8,label='成功清除位置')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.068,.035),ncol=3,frameon=False,
               fontsize=12,handlelength=1.1,columnspacing=1.0)
    fig.text(.083,-.015,'数字为清除先后顺序；同一位置的多次检测合并显示。',fontsize=11.5,color='#687785')
    fig.savefig(HERE/'q3_execution_case.png',dpi=240,facecolor='white',bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    print('Frozen actions and all cost components verified; execution figure written.')


if __name__=='__main__': main()
