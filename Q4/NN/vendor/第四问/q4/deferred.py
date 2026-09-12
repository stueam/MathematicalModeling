"""One-observation routing proxy that can defer the measured source's clear.

Every unresolved source and every certified search obligation remains in the
terminal route. This is a declared terminal heuristic, not a complete rollout.
"""
from dataclasses import asdict
import math

import numpy as np

from .localization import guaranteed_clear, local_attempts
from .posterior import QuadratureError, finish_proxy
from .sector import ProbePolicy
from .shared import Action, distance


def observation_branches(post,point,bin_deg=2.):
    d,received=post.reception(point)
    near=received*(d<=5)
    missed=np.maximum(0.,post.weights-received)
    branches=[]
    for name,w in (('near',near),('no_signal',missed)):
        mass=float(w.sum())
        if mass>1e-15:
            branches.append((name,mass,w/mass))
    angle=np.degrees(np.arctan2((post.xy-point)[:,1],(post.xy-point)[:,0]))%360
    first=np.floor((angle-1)/bin_deg).astype(int)
    bins=first[:,None]+np.arange(math.ceil(2/bin_deg)+1)
    width=np.maximum(0.,np.minimum(angle[:,None]+1,(bins+1)*bin_deg)-
                     np.maximum(angle[:,None]-1,bins*bin_deg))/2
    masses=width*(received*(d>5))[:,None]
    bins%=round(360/bin_deg)
    for k in np.unique(bins[masses>0]):
        w=np.sum(np.where(bins==k,masses,0.),axis=1)
        mass=float(w.sum())
        if mass>1e-15:
            branches.append(('direction',mass,w/mass))
    return branches


def insertion_distance(start,route_points,point):
    """Best insertion into a fixed open route, including the free final end."""
    if not route_points:
        return distance(start,point)
    chain=[start]+list(route_points)
    saving=min(distance(a,point)+distance(point,b)-distance(a,b) for a,b in zip(chain,chain[1:]))
    return min(saving,distance(chain[-1],point))


class DeferredPolicy(ProbePolicy):
    def __init__(self,config=None):
        super().__init__(config)
        self.implementation='q4_v6_deferred_clear_terminal_route'
        self.branch_cache={}
        self.counters.update(deferred_reviews=0,deferred_changed=0,deferred_extra_candidates=0)

    def terminal_local_work(self,post,weights):
        return finish_proxy(weights@post.xy,post.xy,weights)

    def choose(self,b):
        baseline=super().choose(b)
        record=self.records[-1]
        if self.completion_mode or not record.get('candidates'):
            return baseline
        posts={}
        goals={k:p for k,p in self.points.items() if self.needed(b,p)}
        work={k:6*len(self.needed(b,p)) for k,p in goals.items()}
        try:
            for c,p in b.channels.items():
                if p.status=='detected':
                    posts[c]=self.model.posterior(p)
                    goals[c]=tuple(map(float,posts[c].mean))
                    work[c]=self.terminal_local_work(posts[c],posts[c].weights)
        except QuadratureError:
            record['deferred_review']={'reason':'quadrature_fallback','selected_baseline':True}
            return baseline
        options={}
        for row in record['candidates']:
            a=row['action'];action=Action(a['kind'],tuple(a['position']),a['channel'])
            options[action]=(row['task'],row)
        # Keep free observations and shared survey locations through the local
        # pruning that had assumed immediate completion of the measured source.
        for c,post in posts.items():
            channel=b.channels[c]
            if guaranteed_clear(b,c) or local_attempts(channel)>=self.config.local_limit:
                continue
            points=[b.position]+[goals[k] for k in sorted((j for j in goals if j<0),
                                    key=lambda j:distance(goals[j],goals[c]))[:2]]
            for point in points:
                action=Action('measure',point,c)
                if action in options or any(distance(point,o.action.position)<10 for o in channel.history if o.action.kind=='measure'):
                    continue
                options[action]=(c,None)
                self.counters['deferred_extra_candidates']+=1
        reviews=[]
        for action,(task,old) in options.items():
            q=action.position
            consumed=[k for k,p in goals.items() if k<0 and distance(p,q)<1e-6]
            remaining={k:p for k,p in goals.items() if k!=task and k not in consumed}
            order,meters=self.route(q,remaining)
            other_work=sum(work[k] for k in remaining)
            survey=sum(work[k] for k in consumed)
            movement=distance(b.position,q)/5
            if task<0:
                # Survey cost contains the first observation and every other
                # channel in the batch; never count the first action twice.
                score=movement+survey+meters/5+other_work
                detail={'branches':0,'expected_deferred_source_work_s':0.}
            else:
                post=posts[task]
                route_points=[remaining[k] for k in order]
                if action.kind=='measure':
                    key=(task,b.channels[task].revision,q,self.config.bearing_bin)
                    if key not in self.branch_cache:
                        if len(self.branch_cache)>512:
                            self.branch_cache.clear()
                        self.branch_cache[key]=observation_branches(post,q,self.config.bearing_bin)
                    branches=self.branch_cache[key]
                    operation=5+int(b.receiver!=task)
                else:
                    hit=np.linalg.norm(post.xy-q,axis=1)<=20
                    probability=float(post.weights@hit)
                    missed=post.weights*(~hit);mass=float(missed.sum())
                    branches=[('cleared',probability,None)]
                    if mass>1e-15:
                        branches.append(('clear_failed',mass,missed/mass))
                    operation=3+2*probability
                extra=0.
                for outcome,probability,weights in branches:
                    if outcome=='cleared':
                        continue
                    if outcome=='near':
                        extra+=5*probability
                        continue
                    mean=weights@post.xy
                    extra+=probability*(insertion_distance(q,route_points,mean)/5+
                                        self.terminal_local_work(post,weights))
                score=movement+operation+survey+meters/5+other_work+extra
                detail={'branches':len(branches),'branch_mass':sum(x[1] for x in branches),
                        'expected_deferred_source_work_s':extra,
                        'selected_source_still_in_terminal_route':True}
            reviews.append({'action':asdict(action),'task':task,'score_s':score,
                            'move_s':movement,'other_route_m':meters,'other_route':order,
                            'other_work_s':other_work,'same_stop_survey_s':survey,
                            'original_score_s':old['score_s'] if old else None,**detail})
        selected=min(reviews,key=lambda r:r['score_s'])
        a=selected['action'];action=Action(a['kind'],tuple(a['position']),a['channel'])
        self.batch=action.position if any(k<0 and distance(p,action.position)<1e-6 for k,p in goals.items()) else None
        self.counters['deferred_reviews']+=1
        self.counters['deferred_changed']+=int(action!=baseline)
        self.records.pop()
        return self.record(b,action,'deferred_clear_route',candidates=reviews,
                           baseline_decision=record,score_kind='one_observation_plus_all_remaining_route_proxy',
                           terminal_value_estimate=True,full_rollout=False,
                           unknown_source_localization_not_predicted=True)
