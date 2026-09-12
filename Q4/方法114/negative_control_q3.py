"""Isolated LOCAL counterexample: single-disk coverage cannot certify Q4.

The deliberately invalid decision rule is not registered as a runnable policy
and has no HTTP path. Both branches see only their own public feedback. Only the
evaluator constructs the paired worlds and reports missing sources afterward.
"""
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

from q4.core import Belief, DOMAIN
from q4.coverage import safe_union
from q4.sector import ProbePolicy
from q4.shared import Action, disk, distance, point_key
from q4.simulator import Source, World, LocalSimulator
from run import ROOT, dump, manifest


POINTS=[(0.,0.)]+[(1550*math.cos(i*math.tau/6),1550*math.sin(i*math.tau/6)) for i in range(6)]


def invalid_absent(channel):
    return channel.status=='unresolved' and DOMAIN.difference(
        safe_union([disk(p,1000) for p in channel.negatives])).is_empty


def simulate(world,unsafe,out,tag):
    b,env,policy=Belief(),LocalSimulator(world),ProbePolicy()
    actions=[];started=time.monotonic();declared=False
    with (out/f'{tag}-actions.jsonl').open('w') as log:
        for i in range(2200):
            if b.done():
                declared=True;break
            if unsafe:
                inactive={c for c,p in b.channels.items() if p.status=='cleared' or invalid_absent(p)}
                if len(inactive)==20:
                    declared=True;break
                detected=[c for c,p in b.channels.items() if p.status=='detected']
                if detected:
                    c=min(detected,key=lambda c:distance(b.position,b.channels[c].summary()[0]))
                    options,_=policy.local_choices(b,c)
                    action=min(options,key=lambda row:row[1])[0]
                else:
                    options=[Action('measure',p,c) for p in POINTS for c in b.channels
                             if c not in inactive and point_key(p) not in b.channels[c].measured]
                    if not options:
                        break
                    action=min(options,key=lambda a:distance(b.position,a.position)/5+int(a.channel!=b.receiver))
            else:
                action=policy.choose(b)
            response=env.execute(action,str(i));b.apply(action,response,str(i))
            row={'action':asdict(action),'response':response}
            actions.append(row);log.write(json.dumps(row)+'\n')
    evaluation=world.score()
    result={'case_id':tag,'deliberately_invalid_control':unsafe,'declared_complete':declared,
            'valid_public_completion':b.done(),**evaluation,'virtual_time_s':b.virtual_time,
            'real_time_s':time.monotonic()-started,'steps':b.steps,
            'incorrectly_declared_absent':[c for c in world._sources if invalid_absent(b.channels[c])],
            'full_completion_s_per_source':b.virtual_time/evaluation['source_count'] if b.done() and evaluation['all_cleared'] else None}
    dump(out/f'{tag}-summary.json',result)
    return result


def main():
    out=ROOT/'results'/datetime.now().strftime('negative-control-%Y%m%d-%H%M%S-%f');out.mkdir()
    dump(out/'config.json',{'code_sha256':manifest(),'control_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           'q3_search_points':POINTS,'worlds':'16 boundary sources: all directional, then 15 directional + 1 omni',
                           'http':False,'unsafe_control_never_used_by_default':True})
    print(out,flush=True)
    results=[]
    for mixed in (False,True):
        sources=[]
        for i in range(16):
            theta=math.radians(11)+i*math.tau/16
            sources.append(Source(i+1,(1800*math.cos(theta),1800*math.sin(theta)),1000.,
                                  None if mixed and i==0 else math.degrees(theta)%360))
        base=World(sources,91,'extreme')
        dump(out/f'world-mixed-{mixed}.json',base.manifest())
        for unsafe in (True,False):
            tag=f'mixed-{mixed}-'+('INVALID-q3-disks' if unsafe else 'probes')
            result=simulate(base.clone(),unsafe,out,tag)
            results.append(result);print(result,flush=True)
    dump(out/'summary.json',results)


if __name__=='__main__':
    main()
