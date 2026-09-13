"""Measure other known channels at an actual stop when it saves local work."""
import time

from .localization import guaranteed_clear, local_attempts
from .posterior import QuadratureError, finish_proxy
from .sector import ProbePolicy
from .shared import distance


class SharedProbePolicy(ProbePolicy):
    def __init__(self, config=None):
        super().__init__(config)
        self.implementation = 'q4_v5_known_channels_at_shared_stops'
        self.counters['shared_known_measures'] = 0

    def choose(self, b):
        fallback = (self.completion_mode or b.steps >= self.config.completion_after
                    or b.deadline-time.monotonic() < self.config.reserve_s)
        pending = (self.batch is not None and distance(self.batch,b.position)<1e-6
                   and bool(self.needed(b,self.batch)))
        if fallback or pending or b.done():
            return super().choose(b)
        candidates=[]
        for c,p in b.channels.items():
            if p.status!='detected':
                continue
            safe=guaranteed_clear(b,c)
            if safe:
                if distance(safe.position,b.position)<1e-6:
                    return self.record(b,safe,'guaranteed_clear')
                continue
            if local_attempts(p)>=self.config.local_limit or any(
                    distance(b.position,o.action.position)<30 for o in p.history if o.action.kind=='measure'):
                continue
            try:
                post=self.model.posterior(p)
                before=finish_proxy(b.position,post.xy,post.weights)
                action,after,detail=self.probe(b,c,b.position,self.config.bearing_bin)
            except QuadratureError:
                continue
            saving=before-after
            if saving>10:
                candidates.append((saving,c,action,before,after,detail))
        if candidates:
            saving,c,action,before,after,detail=max(candidates,key=lambda x:x[0])
            self.counters['shared_known_measures']+=1
            return self.record(b,action,'shared_known_measure',target=c,
                               predicted_saved_s=saving,before_finish_proxy_s=before,
                               after_measure_and_finish_proxy_s=after,
                               candidate_channels=[x[1] for x in candidates],**detail)
        return super().choose(b)
