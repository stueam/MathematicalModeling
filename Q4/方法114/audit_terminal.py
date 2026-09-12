"""Validate chosen terminal predictions against independent complete task logs.

This is observational calibration, not an oracle action-ranking guarantee.
Future costs are never fed back into the policy.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from run import ROOT,dump


def main():
    directory=Path(sys.argv[1]);batch=json.loads((directory/'batch.json').read_text())
    assert batch['status']=='completed' and set(batch['completed'])==set(batch['planned'])
    out=ROOT/'results'/datetime.now().strftime('terminal-audit-%Y%m%d-%H%M%S-%f');out.mkdir()
    rows=json.loads((directory/'summary.json').read_text());results=[]
    for row in rows:
        if row['policy']!='deferred':continue
        assert row['certified_complete'] and row['all_cleared']
        decisions=json.loads((directory/(row['case_id']+'-decisions.json')).read_text())
        actions=[json.loads(s) for s in (directory/(row['case_id']+'-actions.jsonl')).read_text().splitlines()]
        selected=[d for d in decisions if d['reason']=='deferred_clear_route']
        # One snapshot at each task quartile prevents late cheap decisions
        # dominating the comparison. Every continuation is from a full run.
        ids=sorted(set(np.linspace(0,len(selected)-1,min(4,len(selected))).astype(int)))
        for i in ids:
            d=selected[i];step=d['step']
            before=actions[step-1]['response']['virtual_time_s'] if step else 0.
            realized=row['virtual_time_s']-before
            prediction=next(c['score_s'] for c in d['candidates'] if c['action']==d['action'])
            results.append({'case_id':row['case_id'],'step':step,'predicted_remaining_s':prediction,
                            'realized_remaining_s':realized,'error_s':prediction-realized,
                            'full_task_completed':True})
    dump(out/'config.json',{'input':str(directory),'input_config':json.loads((directory/'config.json').read_text()),
                           'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           'kind':'independent_whole_task_observational_calibration'})
    dump(out/'cases.json',results)
    summary={'cases':len({r['case_id'] for r in results}),'snapshots':len(results),
             'mean_predicted_remaining_s':float(np.mean([r['predicted_remaining_s'] for r in results])),
             'mean_realized_remaining_s':float(np.mean([r['realized_remaining_s'] for r in results])),
             'mean_error_s':float(np.mean([r['error_s'] for r in results])),
             'mean_absolute_error_s':float(np.mean([abs(r['error_s']) for r in results])),
             'note':'Realized costs of complete independent runs; finite sample, selected actions only; no optimality or expected-value guarantee.'}
    dump(out/'summary.json',summary)
    print(out);print(summary)


if __name__=='__main__':main()
