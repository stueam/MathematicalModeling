"""Paired count-posterior diagnostic on frozen public histories, evaluation only."""
from datetime import datetime
import json
import math
from pathlib import Path
import sys

import numpy as np

from q4.core import Belief
from q4.pooled import likelihood_table,joint_categories,count_prior
from q4.posterior import Model,QuadratureError
from q4.shared import Action
from run import ROOT,dump,manifest


def main():
    directory=Path(sys.argv[1]);batch=json.loads((directory/'batch.json').read_text())
    assert batch['status']=='completed' and set(batch['completed'])==set(batch['planned'])
    out=ROOT/'results'/datetime.now().strftime('count-audit-%Y%m%d-%H%M%S-%f');out.mkdir()
    rows=json.loads((directory/'summary.json').read_text());records=[];failures=[]
    independent=np.zeros((17,17))
    for n in range(10,17):
        for d in range(n+1):independent[n,d]=.65**n*.35**(20-n)*.5**n
    print(out,flush=True)
    for row in rows:
        if row['policy']!='probes':continue
        assert row['certified_complete'] and row['all_cleared']
        actions=[json.loads(s) for s in (directory/(row['case_id']+'-actions.jsonl')).read_text().splitlines()]
        steps={20,len(actions)//3,2*len(actions)//3,len(actions)-1}
        b=Belief();model=Model()
        for i,a in enumerate(actions):
            if i in steps:
                try:
                    _,likelihood=likelihood_table(b,model)
                    predictions={}
                    for name,prior in (('independent',independent),('pooled',count_prior())):
                        _,joint,_=joint_categories(likelihood,prior)
                        n,d=np.indices(joint.shape)
                        predictions[name]={'expected_N':float((joint*n).sum()),'expected_D':float((joint*d).sum()),
                            'probability_actual_N_D':float(joint[row['source_count'],row['directional_count']])}
                    records.append({'case_id':row['case_id'],'step':i,'observed_known':len(b.known),
                                    'actual_N':row['source_count'],'actual_D':row['directional_count'],
                                    'predictions':predictions})
                except QuadratureError as exc:
                    failures.append({'case_id':row['case_id'],'step':i,'error':str(exc)})
            action=Action(a['action']['kind'],tuple(a['action']['position']),a['action']['channel'])
            b.apply(action,a['response'],a['request_id'])
    summary={'maps':len({r['case_id'] for r in records}),'paired_snapshots':len(records),'failed_snapshots':failures,
             'note':'Count calibration only; no movement or full-task performance claim.'}
    for name in ('independent','pooled'):
        summary[name]={'mean_abs_N_error':float(np.mean([abs(r['predictions'][name]['expected_N']-r['actual_N']) for r in records])),
                       'mean_abs_D_error':float(np.mean([abs(r['predictions'][name]['expected_D']-r['actual_D']) for r in records])),
                       'mean_negative_log_actual_N_D':float(np.mean([-math.log(max(r['predictions'][name]['probability_actual_N_D'],1e-300)) for r in records]))}
    dump(out/'config.json',{'input':str(directory),'code_sha256':manifest(),'truth_used_only_after_prediction':True})
    dump(out/'cases.json',records);dump(out/'summary.json',summary)
    print(summary,flush=True)


if __name__=='__main__':main()
