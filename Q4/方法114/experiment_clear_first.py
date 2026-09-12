"""Isolated local trial: finish nearby known tasks before distant search stops.

All search obligations stay in the inherited complete route and finite fallback.
This changes their visit order. This executable has no HTTP entry point.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import multiprocessing
from pathlib import Path

from q4.collected import CollectedPolicy
from q4.policy import Config
from q4.sector import ProbePolicy
from q4.shared import Action,distance
import run


class ClearFirst:
    def choose(self,b):
        action=super().choose(b)
        record=self.records[-1]
        if (self.completion_mode or action.kind!='measure'
                or b.channels[action.channel].status!='unresolved'
                or distance(b.position,action.position)<200):
            return action
        known=[r for r in record.get('candidates',[]) if r['task']>0]
        if not known:
            return action
        chosen=min(known,key=lambda r:r['score_s']);a=chosen['action']
        self.batch=None
        self.records.pop()
        return self.record(b,Action(a['kind'],tuple(a['position']),a['channel']),
            'known_before_distant_search',baseline_decision=record,chosen_candidate=chosen,
            search_obligations_retained=True,
            score_kind='full-task-route-proxy-with-known-first-ordering')


class ClearFirstPolicy(ClearFirst,ProbePolicy):
    def __init__(self,config=None):
        super().__init__(config)
        self.implementation='q4_v8_known_before_distant_search'


class CollectedFirstPolicy(ClearFirst,CollectedPolicy):
    def __init__(self,config=None):
        super().__init__(config)
        self.implementation='q4_v8_collected_known_before_distant_search'


original_factory=run.make_policy


def factory(name,config=None,**options):
    if name=='clear-first':return ClearFirstPolicy(config)
    if name=='collect-first':return CollectedFirstPolicy(config)
    return original_factory(name,config,**options)


def worker(job,out):
    run.make_policy=factory
    return run.run_case(job,out)


def main():
    out=run.ROOT/'results'/datetime.now().strftime('clear-first-%Y%m%d-%H%M%S-%f');out.mkdir()
    jobs=[{'seed':seed,'scenario':'uniform','error_mode':'iid','n':None,'radius':None,
           'directional_count':None,'policy':policy,'config':asdict(Config()),
           'policy_options':{},'real_limit':1200.,'max_steps':6000}
          for seed in range(300,308) for policy in ('probes','clear-first','collect-first')]
    code=run.manifest();code[str(Path(__file__).name)]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run.dump(out/'config.json',{'jobs':jobs,'code_sha256':code,'workers':16,'official_connection':False})
    batch={'status':'running','planned':[run.case_id(j) for j in jobs],'completed':[]}
    run.dump(out/'batch.json',batch);print(out,flush=True);rows=[]
    try:
        with ProcessPoolExecutor(16,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(worker,j,str(out)) for j in jobs]
            for future in as_completed(futures):
                row=future.result();rows.append(row);batch['completed'].append(row['case_id'])
                run.dump(out/'summary.json',rows);run.dump(out/'batch.json',batch)
                print(len(rows),len(jobs),row['case_id'],row['per_cleared_s'],row['stop_reason'],flush=True)
        batch['status']='completed'
    except BaseException as exc:
        batch['status']='interrupted';batch['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        run.dump(out/'batch.json',batch)
    run.dump(out/'aggregate.json',run.aggregate(rows));print(json.dumps(run.aggregate(rows),indent=2),flush=True)


if __name__=='__main__':main()
