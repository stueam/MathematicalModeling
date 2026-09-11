"""B5 paired offline experiments, action evidence, ablations and plots."""
import argparse
import csv
import datetime
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from offline_environment import StressEnvironment, make_cases, FAMILIES
from solver import build_strategy, LATEST_CONFIG
from b5_strategy import B5_CONFIG

ROOT=Path(__file__).resolve().parent


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods',nargs='+',default=['b2','q2','b4','b5'],
                   choices=['b2','latest','q2','b4','b5','b5_global','b5_single'])
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--scenarios',type=Path)
    group.add_argument('--seed',type=int)
    p.add_argument('--stratified35',action='store_true')
    p.add_argument('--limit',type=int)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot',action='store_true')
    args=p.parse_args()
    cases=make_cases(args.seed) if args.seed is not None else json.loads(args.scenarios.read_text(encoding='utf-8'))
    if args.stratified35:
        cases=[c for c in cases if c['replicate']==(len(c['targets'])+FAMILIES.index(c['family']))%4]
    if args.limit is not None:cases=cases[:args.limit]
    if not cases or len({c['id'] for c in cases})!=len(cases):raise ValueError('Nonempty distinct cases required')
    for case in cases:
        targets=case['targets']
        if not 10<=len(targets)<=16 or len({t['channel'] for t in targets})!=len(targets):
            raise ValueError('Expected 10..16 distinct-channel sources')
        for t in targets:
            if not (1<=t['channel']<=20 and 1000<=t['radius']<=1500 and np.isfinite(t['position']).all()
                    and np.linalg.norm(t['position'])<=1800.000001):raise ValueError('Invalid source')
    args.output.mkdir(parents=True,exist_ok=True)
    dump(args.output/'scenarios.json',cases)
    rows=[];start=time.perf_counter()
    for method in args.methods:
        for i,case in enumerate(cases):
            env=StressEnvironment(case)
            model=build_strategy(env.action,'b5' if method.startswith('b5') else method)
            if method=='b5_global':model.b5_config['adaptive_sectors']=False
            if method=='b5_single':model.b5_config['batching']=False
            tick=time.perf_counter();failure=None
            try:
                model.run()
                if env.cleared!=set(env.targets) or not model.absent.isdisjoint(env.targets):
                    raise AssertionError('Missed source or false absence')
            except Exception as exc:failure=repr(exc)
            trace=model.trace
            positions=[np.asarray(t['position']) for t in trace if t['position'] is not None]
            stops=sum(np.linalg.norm(b-a)>1e-6 for a,b in zip([np.zeros(2),*positions[:-1]],positions))
            def sector(x):return int(np.floor((np.arctan2(x[1],x[0])+np.pi/7)/(2*np.pi/7)))%7
            # Endpoint sector changes, excluding central disk; not path crossing count.
            sectors=[sector(x) for x in positions if np.linalg.norm(x)>1000]
            changes=sum(a!=b for a,b in zip(sectors,sectors[1:]))
            stats=dict(getattr(model,'b5_stats',{}))
            row=dict(method=method,case=case['id'],family=case['family'],sources=len(env.targets),
                cleared=len(env.cleared),virtual_time_s=env.clock,
                s_per_source=env.clock/len(env.targets) if failure is None else None,
                movement_m=env.movement,measurements=env.measures,failed_clears=env.clear_attempts-len(env.cleared),
                stops=stops,outer_sector_transitions=changes,runtime_s=time.perf_counter()-tick,
                shared_stops=stats.get('shared_stops',0),recovery_actions=stats.get('recovery_actions',0),
                max_wait_seen=stats.get('max_wait_seen',0),cover_searches=model.fallback_count,failure=failure)
            rows.append(row)
            dump(args.output/'actions'/f'{method}_{case["id"]}.json',trace)
            if hasattr(model,'b5_log'):dump(args.output/'decisions'/f'{method}_{case["id"]}.json',model.b5_log)
            # Checkpoint full rows after every case; interruptions preserve completed evidence.
            with (args.output/'cases.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(rows)
            print(f'{method} {i+1}/{len(cases)} {case["id"]}: {row["s_per_source"]} s/source, '
                  f'{row["movement_m"]:.0f} m, {row["runtime_s"]:.2f} real s, {failure or "OK"}',flush=True)
    metrics={}
    for method in args.methods:
        rs=[r for r in rows if r['method']==method];ok=all(r['failure'] is None for r in rs)
        metrics[method]=dict(cases=len(rs),complete=sum(r['failure'] is None for r in rs),
            sources=sum(r['sources'] for r in rs),cleared=sum(r['cleared'] for r in rs),
            **{f'mean_{k}':float(np.mean([r[k] for r in rs])) if ok else None for k in
               ['s_per_source','movement_m','measurements','stops','outer_sector_transitions','failed_clears','runtime_s','recovery_actions']},
            p90_s_per_source=float(np.quantile([r['s_per_source'] for r in rs],.9)) if ok else None)
    paired={};rng=np.random.default_rng(20260911)
    if 'b5' in args.methods:
        br={r['case']:r for r in rows if r['method']=='b5'}
        for baseline in [m for m in args.methods if m!='b5']:
            rs=[r for r in rows if r['method']==baseline]
            if any(r['failure'] or br[r['case']]['failure'] for r in rs):continue
            delta=np.array([br[r['case']]['s_per_source']-r['s_per_source'] for r in rs])
            samples=rng.choice(delta,size=(10000,len(delta)),replace=True).mean(axis=1)
            paired[baseline]=dict(mean_delta_s=float(delta.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),
                faster=int((delta<0).sum()),slower=int((delta>0).sum()),
                percent_change=100*(metrics['b5']['mean_s_per_source']/metrics[baseline]['mean_s_per_source']-1),
                families={family:float(np.mean([br[r['case']]['s_per_source']-r['s_per_source'] for r in rs if r['family']==family]))
                          for family in sorted({r['family'] for r in rs})})
    summary=dict(status='PASS' if all(r['failure'] is None for r in rows) else 'FAIL',
        scope='Offline synthetic paired comparison; no official tests. See report for development/holdout designation.',
        seed=args.seed,stratified35=args.stratified35,metrics=metrics,paired_b5_minus_baseline=paired,
        config=dict(base=LATEST_CONFIG,b5=B5_CONFIG),
        ablations=dict(b5_global='No soft sector eligibility filter; same backbone and progress caps',
                       b5_single='At most one known-source probe per local stop; same other rules'),
        metric_definition='Scene-equal mean of virtual time / source count, null if any scene fails. '
                          'stops counts movement destinations; outer transitions counts consecutive outer endpoint sector labels.',
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        scenario_sha256=hashlib.sha256((args.output/'scenarios.json').read_bytes()).hexdigest(),
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(ROOT.glob('*.py'))},
        elapsed_s=time.perf_counter()-start,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        network_requests=0,formal_tests_used=0)
    dump(args.output/'summary.json',summary)
    print(json.dumps(metrics,indent=2),flush=True)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,3,figsize=(13,4))
        for ax,key,label in zip(axs,['s_per_source','movement_m','measurements'],['Seconds / source','Travel (km)','Measurements']):
            values=[metrics[m]['mean_'+key] for m in args.methods]
            if any(v is None for v in values):continue
            if key=='movement_m':values=[v/1000 for v in values]
            ax.bar(args.methods,values,color=['#155e75' if m=='b5' else '#94a3b8' for m in args.methods])
            ax.set_ylabel(label);ax.tick_params(axis='x',rotation=25);ax.spines[['top','right']].set_visible(False)
        fig.suptitle(f'Paired synthetic scenarios (n={len(cases)})');fig.tight_layout()
        fig.savefig(args.output/'comparison.png',dpi=170);plt.close(fig)
    return 0 if summary['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
