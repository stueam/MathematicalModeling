"""Portable, bounded local evaluation of the published Q4 attention model."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

from nnq4.bridge import ROOT
from nnq4.experiment import dump,new_run,run_episode,summarize
from nnq4.round_control import bounded_map,supervise


def worker(out):
    torch.set_num_threads(1)
    config=json.loads((out/'config.json').read_text())
    hard=float(os.environ['Q4_ROUND_HARD_DEADLINE'])
    dispatch=float(os.environ['Q4_ROUND_DISPATCH_DEADLINE'])
    checkpoint=config['checkpoint']
    if config['retrain']:
        from nnq4.advantage_training import train_variant
        checkpoint=str(train_variant(ROOT/'training_data/manifest.json',ROOT/'models/relative-prior.pt',
                       ROOT/'models/bc-v1.pt',out/'training',True,config['device'],280130000,
                       hard_deadline=hard,dispatch_deadline=dispatch))
    jobs=[dict(seed=seed,mode=mode,checkpoint=checkpoint if mode=='nn_best' else None,
               selection='advantage',features='v2',out=str(out/'episodes'),split='development',
               real_limit=1200.,macro_budget=160,allow_fallback=True)
          for seed in range(config['seed'],config['seed']+config['rounds'])
          for mode in ('probes','nn_best')]
    rows=[]
    def consume(row):
        rows.append(row)
        dump(out/'summary.json',sorted(rows,key=lambda r:(r['seed'],r['mode'])))
        dump(out/'batch.json',dict(status='running',completed=len(rows),planned=len(jobs)))
        print(f"{len(rows)}/{len(jobs)} {row['case']} complete={row['complete']}",flush=True)
    status=bounded_map(run_episode,jobs,config['workers'],dispatch,hard,consume)
    result=summarize(rows)
    result['all_complete_and_audited']=len(rows)==len(jobs) and all(r['complete'] and r['public_complete'] and r['audit_ok'] for r in rows)
    result['local_simulator_only']=True
    dump(out/'aggregate.json',result)
    dump(out/'batch.json',status)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if not result['all_complete_and_audited']:
        raise RuntimeError('Incomplete paired evaluation; do not claim a complete score')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=280020000)
    parser.add_argument('--rounds',type=int,default=32)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--retrain',action='store_true')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not 1<=args.workers<=16 or not 1<=args.rounds<=64:
        parser.error('Require 1..16 workers and 1..64 maps')
    os.chdir(ROOT)
    config=dict(checkpoint=str(ROOT/'models/best.pt'),seed=args.seed,rounds=args.rounds,workers=args.workers,
                retrain=args.retrain,device=args.device,hard_limit_s=1800.,dispatch_limit_s=1620.,
                local_simulator_only=True,automatic_next_round=False)
    out=new_run('published-best',config)
    print('OUTPUT='+str(out),flush=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    control=supervise([sys.executable,str(ROOT/'run_best.py'),'--worker',str(out)],out,env=env)
    if control['status']!='complete':
        raise SystemExit(1)


if __name__=='__main__':
    main()
