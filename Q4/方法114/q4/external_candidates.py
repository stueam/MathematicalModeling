"""Independent local ablations from the external assessment brief.

The suggestions are candidate heuristics. Actual feedback alone updates geometry;
none of these policies use an evaluator World or a probabilistic stopping rule.
"""
from functools import lru_cache
import math
import time

import numpy as np

from .coverage import certifies
from .core import DOMAIN
from .localization import guaranteed_clear, local_attempts
from .posterior import QuadratureError
from .sector import ProbePolicy
from .shared import distance


def heading_moment(post):
    """Exact first circular moment conditional on directional type.

    Uniform within each posterior heading interval; omni atoms have no heading.
    This integrates the interval analytically, without introducing 5-degree bins.
    """
    mass=post.atoms[:,1:]
    probability=float(mass.sum())
    if probability<=1e-15:
        return np.zeros(2),0.,probability
    width=post.right-post.left
    cosine=np.divide(np.sin(post.right)-np.sin(post.left),width,
                     out=np.zeros_like(width),where=width>1e-15)
    sine=np.divide(np.cos(post.left)-np.cos(post.right),width,
                   out=np.zeros_like(width),where=width>1e-15)
    mean=np.array([float((mass*cosine).sum()),float((mass*sine).sum())])/probability
    concentration=float(np.linalg.norm(mean))
    return mean/max(concentration,1e-15),concentration,probability


class HeadingPolicy(ProbePolicy):
    def __init__(self,config=None,variant='heading'):
        super().__init__(config)
        self.variant_name=variant
        self.implementation=f'q4_external_{variant}_probes_v1'
        self.counters.update(heading_candidates=0,miss_candidates=0)

    def local_choices(self,b,c):
        choices,mean=super().local_choices(b,c)
        p=b.channels[c]
        if guaranteed_clear(b,c) or local_attempts(p)>=self.config.local_limit:
            return choices,mean
        try:
            post=self.model.posterior(p)
            u,kappa,probability=heading_moment(post)
            additions=[]
            if self.variant_name in ('heading','front') and kappa>=.4 and probability>=.5:
                additions.append((tuple(np.asarray(mean)+350*u),
                                  {'candidate_family':'forward_heading','heading_kappa':kappa,
                                   'directional_probability':probability,'forward_offset_m':350.}))
            # Misses are observations, not a proof of being behind the source.
            # The fixed-R/heading posterior still prices the no-signal branch.
            last_positive=max((i for i,o in enumerate(p.history) if o.result in ('direction','near')),default=-1)
            misses=sum(o.result=='no_signal' for o in p.history[last_positive+1:])
            if self.variant_name in ('heading','miss') and misses and p.directions:
                previous=np.asarray(p.directions[-1].action.position)
                toward=previous-np.asarray(mean)
                toward/=max(float(np.linalg.norm(toward)),1.)
                sideways=np.array([-toward[1],toward[0]])
                offset=100.*min(misses,6)
                for sign in (-1,1):
                    additions.append((tuple(np.asarray(mean)+offset*toward+sign*.5*offset*sideways),
                                      {'candidate_family':'heard_side_after_miss','misses_since_signal':misses,
                                       'toward_heard_side_m':offset}))
            existing={a.position for a,_,_ in choices}
            for point,detail in additions:
                point=tuple(map(float,point))
                if point in existing or any(distance(point,o.action.position)<10 for o in p.history if o.action.kind=='measure'):
                    continue
                action,cost,branch=self.probe(b,c,point,self.config.bearing_bin)
                choices.append((action,cost,{**branch,**detail}))
                self.counters['heading_candidates' if detail['candidate_family']=='forward_heading' else 'miss_candidates']+=1
        except QuadratureError:
            pass
        return choices,mean


class WitnessPolicy(ProbePolicy):
    def __init__(self,config=None):
        super().__init__(config)
        self.variant='adaptive'
        self.implementation='q4_external_ring22_actual_witnesses_v1'


@lru_cache(maxsize=1)
def external_polar_points():
    # The exact 1850 m proposal leaves a floating polygon remnant in the
    # existing conservative union. A stated 0.1 m outward adjustment passes
    # the full existing certificate; never discard the remnant by its area.
    points=[(0.,0.)]
    points += [(980*math.cos(i*math.tau/7),980*math.sin(i*math.tau/7)) for i in range(7)]
    points += [(1850.1*math.cos(i*math.tau/14),1850.1*math.sin(i*math.tau/14)) for i in range(14)]
    if not certifies(DOMAIN,points):
        raise ValueError('External polar layout lacks full conservative certificate')
    return tuple(points)


class PolarPolicy(ProbePolicy):
    def __init__(self,config=None):
        super().__init__(config)
        self.points={-i-1:p for i,p in enumerate(external_polar_points())}
        self.implementation='q4_external_polar22_980_1850p1_v1'


class BoundaryPolicy(ProbePolicy):
    def __init__(self,config=None):
        super().__init__(config)
        self.implementation='q4_external_optional_1620_boundary_ring_v1'
        self.boundary=[(1620*math.cos(i*math.tau/12),1620*math.sin(i*math.tau/12)) for i in range(12)]
        self.boundary_visited=set()
        self.counters['boundary_visits']=0

    def choose(self,b):
        baseline=super().choose(b)
        record=self.records[-1]
        if (self.completion_mode or distance(b.position,baseline.position)<200 or not record.get('candidates')
                or b.deadline-time.monotonic()<self.config.reserve_s):
            return baseline
        if not any(p.status=='unresolved' for p in b.channels.values()):
            return baseline
        try:
            existence=self.model.existence(b)
            trials=[]
            for i,q in enumerate(self.boundary):
                if i in self.boundary_visited:
                    continue
                channels=self.needed(b,q)
                if not channels:
                    continue
                detour=distance(b.position,q)+distance(q,baseline.position)-distance(b.position,baseline.position)
                cost=detour/5+6*len(channels)
                # Explicit fixed discovery-value proxy; include the baseline
                # remainder by charging the detour back to its next destination.
                expected=sum(existence[c]*float(self.model.posterior(b.channels[c]).reception(q)[1].sum()) for c in channels)
                gain=200*expected-cost
                trials.append({'index':i,'position':q,'expected_discoveries':expected,
                               'detour_m':detour,'additional_scanning_s':6*len(channels),
                               'predicted_gain_s':gain,'value_per_discovery_s':200.})
        except QuadratureError:
            return baseline
        record['boundary_review']=trials
        if not trials or max(t['predicted_gain_s'] for t in trials)<=20:
            return baseline
        best=max(trials,key=lambda t:t['predicted_gain_s'])
        self.boundary_visited.add(best['index'])
        self.counters['boundary_visits']+=1
        self.records.pop()
        return self.scan(b,best['position'],'optional_boundary_discovery',boundary_review=trials,
                         baseline_decision=record,score_kind='discovery_value_heuristic')
