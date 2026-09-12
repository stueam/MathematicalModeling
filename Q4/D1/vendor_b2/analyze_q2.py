"""Summarize paired Q2 ablations and optionally plot (matplotlib is optional)."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    rows=list(csv.DictReader((args.directory/'cases.csv').open(encoding='utf-8')))
    methods=list(dict.fromkeys(r['method'] for r in rows))
    grouped={m:{r['case']:r for r in rows if r['method']==m} for m in methods}
    baseline=grouped['latest']
    metrics={}
    rng=np.random.default_rng(19031)
    for method in methods:
        rs=list(grouped[method].values())
        ids=list(grouped[method])
        if set(ids)!=set(baseline):raise ValueError('Paired case IDs differ')
        if any(r['failure'] for r in rs):
            metrics[method]={'failures':sum(bool(r['failure']) for r in rs)}
            continue
        base=np.array([float(baseline[i]['s_per_source']) for i in ids])
        values=np.array([float(grouped[method][i]['s_per_source']) for i in ids])
        delta=base-values
        indices=rng.integers(0,len(ids),(10000,len(ids)))
        gains=(1-values[indices].mean(axis=1)/base[indices].mean(axis=1))*100
        entry=dict(cases=len(rs),mean_s_per_source=float(values.mean()),
                   relative_gain_percent=float((1-values.mean()/base.mean())*100),
                   paired_bootstrap_95_percent=np.quantile(gains,[.025,.975]).tolist(),
                   faster_cases=int((delta>1e-6).sum()),slower_cases=int((delta<-1e-6).sum()),
                   tied_cases=int((abs(delta)<=1e-6).sum()),
                   mean_total_virtual_s=float(np.mean([float(r['virtual_time_s']) for r in rs])),
                   mean_movement_m=float(np.mean([float(r['movement_m']) for r in rs])),
                   mean_measurements=float(np.mean([float(r['measurements']) for r in rs])),
                   mean_failed_clears=float(np.mean([float(r['failed_clears']) for r in rs])),
                   mean_runtime_s=float(np.mean([float(r['runtime_s']) for r in rs])),
                   max_runtime_s=max(float(r['runtime_s']) for r in rs),
                   mean_planning_s=float(np.mean([float(r['planning_time_s']) for r in rs])),
                   total_planning_calls=sum(int(r['planning_calls']) for r in rs),
                   cover_searches=sum(int(r['cover_searches']) for r in rs))
        entry['families']={}
        for family in sorted({r['family'] for r in rs}):
            selected=[i for i in ids if grouped[method][i]['family']==family]
            b=np.mean([float(baseline[i]['s_per_source']) for i in selected])
            v=np.mean([float(grouped[method][i]['s_per_source']) for i in selected])
            entry['families'][family]=dict(mean_s_per_source=v,gain_percent=(1-v/b)*100)
        metrics[method]=entry
    (args.directory/'paired_analysis.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metrics,indent=2))
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
        labels={'latest':'B2 latest','mec':'+ minimum circle','q2':'+ Q2 lookahead'}
        colors=['#64748b','#e9a23b','#138a78']
        heights=[metrics[m]['mean_s_per_source'] for m in methods]
        bars=axes[0].bar([labels[m] for m in methods],heights,color=colors[:len(methods)])
        axes[0].bar_label(bars,fmt='%.2f',padding=3)
        axes[0].set_ylabel('Virtual seconds / source (lower is better)')
        axes[0].set_ylim(0,max(heights)*1.15)
        axes[0].set_title('Mean across paired cases')
        ids=list(baseline)
        base=np.array([float(baseline[i]['s_per_source']) for i in ids])
        enhanced=np.array([float(grouped['q2'][i]['s_per_source']) for i in ids])
        axes[1].scatter(base,enhanced,s=18,alpha=.65,color=colors[2])
        lo=min(base.min(),enhanced.min())-10;hi=max(base.max(),enhanced.max())+10
        axes[1].plot([lo,hi],[lo,hi],color='#94a3b8',linestyle='--')
        axes[1].set(xlabel='B2 latest: s/source',ylabel='Q2 lookahead: s/source',
                    title='Each point is one scenario; below line = faster')
        families=list(metrics['q2']['families'])
        gains=[metrics['q2']['families'][f]['gain_percent'] for f in families]
        axes[2].barh([f.replace('_',' ') for f in families],gains,
                     color=[colors[2] if x>=0 else '#c94c4c' for x in gains])
        axes[2].axvline(0,color='#64748b',linewidth=.8)
        axes[2].set(xlabel='Reduction in virtual time (%)',title='Q2 gain by scenario family')
        fig.suptitle(f'Q2 / B2 ablation — {len(ids)} paired local simulations',fontsize=14)
        fig.savefig(args.directory/'comparison.png',dpi=170)
        plt.close(fig)


if __name__=='__main__':main()
