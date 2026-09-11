"""Offline scientific plots of measured training/evaluation data."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='runs/full_pilot')
    p.add_argument('--evaluations',nargs='+',default=['reports/random_uniform.json','reports/heuristic_uniform.json','reports/ppo_uniform.json'])
    p.add_argument('--out',default='reports/training_report.png')
    args=p.parse_args()
    rows=list(csv.DictReader((Path(args.run)/'progress.csv').open(encoding='utf-8')))
    steps=np.array([float(r['steps']) for r in rows])
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    fig.suptitle('Q3 / Method 2: local simulator + PPO',fontsize=17)
    ax[0,0].plot(steps,[float(r['mean_virtual_s_success_100'])/60 for r in rows],color='#2563eb')
    ax[0,0].set(xlabel='Training actions',ylabel='Virtual minutes',title='Training: last 100 successful episodes')
    ax[0,1].plot(steps,[float(r['success_rate_100']) for r in rows],color='#059669')
    ax[0,1].set(xlabel='Training actions',ylabel='Completion rate',ylim=(-.02,1.05),title='Training: last 100 episodes')
    ax[1,0].plot(steps,[float(r['sps']) for r in rows],color='#7c3aed')
    ax[1,0].set(xlabel='Training actions',ylabel='Environment actions / second',title='End-to-end PPO throughput')
    summaries=[json.loads(Path(f).read_text(encoding='utf-8'))['summary'] for f in args.evaluations]
    names=[s['policy'] for s in summaries]
    vals=[(s['mean_success_virtual_s'] or 0)/60 for s in summaries]
    bars=ax[1,1].bar(names,vals,color=['#94a3b8','#f59e0b','#2563eb'][:len(names)])
    ax[1,1].set(ylabel='Virtual minutes',title=f"Held-out {summaries[0]['episodes']} paired scenarios")
    ax[1,1].set_ylim(0,max(vals)*1.2 if vals else 1)
    for bar,s in zip(bars,summaries):
        ax[1,1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(vals)*.02,
                      f"{s['success_rate']:.0%} completed",ha='center',fontsize=9)
    for a in ax.flat:
        a.grid(alpha=.2,axis='y');a.spines[['top','right']].set_visible(False)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=160);plt.close(fig)
    print(json.dumps({'figure':str(out.resolve())}))


if __name__=='__main__':
    main()
