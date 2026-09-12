"""Search outer landmarks before inner ones, preserving the full suffix cost."""
import math

from .sector import ProbePolicy


class PhasedPolicy(ProbePolicy):
    def __init__(self,config=None):
        super().__init__(config)
        self.implementation='q4_v6_outer_search_then_inner_route'
        self.counters['phased_route_calls']=0

    def record(self,b,action,reason,**extra):
        # Two individually optimized phases are not a globally exact tour.
        if 'route_exact' in extra:
            extra['route_exact']=False
        if 'candidates' in extra:
            extra['candidates']=[{**r,'tail_exact':False} for r in extra['candidates']]
        extra['route_constraint']='outer_then_inner; phase composition is heuristic'
        return super().record(b,action,reason,**extra)

    def route(self,position,goals,remaining=None):
        active=goals if remaining is None else {k:goals[k] for k in remaining}
        outer={k:p for k,p in active.items() if math.hypot(*p)>1300.}
        inner={k:p for k,p in active.items() if k not in outer}
        if not outer or not inner:
            return super().route(position,active)
        self.counters['phased_route_calls']+=1
        prefix,meters=super().route(position,outer)
        suffix,extra=super().route(outer[prefix[-1]],inner)
        return prefix+suffix,meters+extra
