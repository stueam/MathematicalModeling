"""Accumulate actual clear-stop measurements for later joint coverage reuse.

An extra scan is an investment, not an absence certificate. Existing search
obligations are removed only by the inherited full, per-channel certificate.
"""
import time

from .coverage import full_exclusion, added_exclusion
from .routing import length
from .sector import ProbePolicy
from .shared import distance


class CollectedPolicy(ProbePolicy):
    def __init__(self,config=None,value_gate=False):
        super().__init__(config)
        self.value_gate=value_gate
        self.implementation='q4_v8_collected_clear_stop_witnesses'+('_value' if value_gate else '')
        self.collected=[(0.,0.)]
        self.counters.update(collected_scans=0,collected_value_reviews=0)

    def partial_value(self,b):
        """Fractional replacement score only; never mutates proof or plan."""
        remaining={k:p for k,p in self.points.items() if self.needed(b,p)}
        route,before=self.route(b.position,remaining)
        nearest=sorted(remaining,key=lambda k:distance(b.position,remaining[k]))[:4]
        groups=[(k,) for k in nearest]+[tuple(nearest[:n]) for n in (2,3,4) if len(nearest)>=n]
        channels={}
        for p in b.channels.values():
            if p.status=='unresolved':
                channels[(p.region.wkb,tuple(sorted(p.negatives)))]=p
        reviews=[]
        for group in groups:
            kept=[k for k in route if k not in group]
            after=length(b.position,kept,remaining)
            saving=(before-after)/5+6*sum(len(self.needed(b,remaining[k])) for k in group)
            fractions=[]
            for p in channels.values():
                points=tuple(sorted(set(p.negatives+tuple(v for k,v in self.points.items() if k not in group))))
                missing=p.region.difference(full_exclusion(points))
                if missing.is_empty:
                    fractions.append(1.);continue
                if missing.area<=0:
                    fractions.append(0.);continue
                added=added_exclusion(b.position,points)
                fractions.append(min(1.,max(0.,missing.intersection(added).area/missing.area)))
            fraction=min(fractions,default=0.)
            reviews.append({'stations':group,'fraction_of_missing_area_filled':fraction,
                            'removal_route_and_scan_proxy_s':saving,
                            'fractional_saved_proxy_s':max(0.,saving)*fraction})
        return max((r['fractional_saved_proxy_s'] for r in reviews),default=0.),reviews

    def shared_replacement(self,b):
        action=super().shared_replacement(b)
        if action is not None:
            return action
        if (not b.applied or self.completion_mode or b.steps>=self.config.completion_after
                or b.deadline-time.monotonic()<self.config.reserve_s):
            return None
        _,response=b.applied[next(reversed(b.applied))]
        channels=self.needed(b,b.position)
        if (response.get('clear_result')!='success' or not channels
                or any(distance(b.position,p)<250 for p in self.collected)):
            return None
        scanning=6*len(channels);audit=[];value=None
        if self.value_gate:
            self.counters['collected_value_reviews']+=1
            value,audit=self.partial_value(b)
            if value<=1.3*scanning+10:
                self.pending_cover_audit.append({'operation':'decline_partial_witness_scan',
                    'fractional_saved_proxy_s':value,'scan_proxy_s':scanning,'groups':audit})
                return None
        self.collected.append(b.position)
        self.counters['collected_scans']+=1
        return self.scan(b,b.position,'collected_clear_stop_scan',scan_proxy_s=scanning,
            fractional_saved_proxy_s=value,partial_replacement_reviews=audit,
            score_kind='partial-certificate-investment-heuristic',
            search_obligations_removed_by_this_decision=0,
            actual_per_channel_feedback_required=True)
