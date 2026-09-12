"""Price a known-source probe and a same-location survey as one visit."""
from dataclasses import asdict

from .localization import guaranteed_clear, local_attempts
from .posterior import QuadratureError
from .sector import ProbePolicy
from .shared import Action, distance


class BundlePolicy(ProbePolicy):
    def __init__(self, config=None):
        super().__init__(config)
        self.implementation = 'q4_v5_bundled_probe_and_survey'
        self.counters.update(bundle_candidates=0,bundle_selections=0,bundle_changed_actions=0)

    def local_choices(self,b,c):
        choices,mean=super().local_choices(b,c)
        channel=b.channels[c]
        if guaranteed_clear(b,c) or local_attempts(channel)>=self.config.local_limit:
            return choices,mean
        # A locally worse measurement can be globally cheaper because it also
        # consumes a search visit. Keep these candidates through local pruning.
        existing={a for a,_,_ in choices}
        stations=sorted(self.points,key=lambda k:distance(self.points[k],mean))[:3]
        for k in stations:
            point=self.points[k]
            action=Action('measure',point,c)
            if action in existing or not self.needed(b,point):
                continue
            if any(distance(point,o.action.position)<10 for o in channel.history if o.action.kind=='measure'):
                continue
            try:
                choices.append(self.probe(b,c,point,self.config.bearing_bin))
            except QuadratureError:
                continue
        return choices,mean

    def choose(self,b):
        baseline=super().choose(b)
        record=self.records[-1]
        candidates=record.get('candidates')
        if self.completion_mode or not candidates:
            return baseline
        goals={k:p for k,p in self.points.items() if self.needed(b,p)}
        for c,channel in b.channels.items():
            if channel.status=='detected':
                try:
                    goals[c]=tuple(map(float,self.model.posterior(channel).mean))
                except QuadratureError:
                    goals[c]=channel.summary()[0]
        revised=[]
        for original in candidates:
            row=dict(original)
            covered=[k for k,p in goals.items() if k<0 and distance(p,tuple(row['action']['position']))<1e-6]
            if row['task']>=0 and covered:
                remaining=[k for k in goals if k!=row['task'] and k not in covered]
                tail,meters=self.route(goals[row['task']],goals,remaining=remaining)
                row.update(unbundled_score_s=row['score_s'],bundled_search_tasks=covered,
                           tail_route_s=meters/5,tail_order=tail,tail_exact=len(remaining)<=self.config.exact_limit,
                           score_s=row['local_proxy_s']+meters/5+row['other_work_proxy_s'],
                           survey_cost_retained=True)
                # other_work_proxy_s still contains the survey cost. Only the
                # already-visited site's travel is removed from the suffix.
                self.counters['bundle_candidates']+=1
            revised.append(row)
        selected=min(revised,key=lambda r:r['score_s'])
        if not selected.get('bundled_search_tasks'):
            record['bundle_candidates_review']=revised
            return baseline
        a=selected['action']; action=Action(a['kind'],tuple(a['position']),a['channel'])
        self.batch=action.position  # Scan the actual location before departing.
        self.counters['bundle_selections']+=1
        self.counters['bundle_changed_actions']+=int(action!=baseline)
        self.records.pop()
        return self.record(b,action,'bundled_probe_and_survey',candidates=revised,
                           bundled_search_tasks=selected['bundled_search_tasks'],
                           baseline_decision=record,score_kind='bundled-visit-and-local-work-proxy')
