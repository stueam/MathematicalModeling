"""B4: Q2 candidate proposals plus multi-channel, travel-aware joint measurements.

Uniform area and angular errors are planning approximations. Reception is
guaranteed per selected channel for every position in its conservative polygon.
No source truth, radius prior, or assumed total source count is accessed.
"""
import time
import numpy as np
from q2_geometry import area_samples
from q2_lookahead import Q2Proposal, expected_cost


B4_CONFIG = dict(max_channels=4, max_candidates=24, coarse_power=2, fine_power=4,
                 finalists=3, max_joint_stops=10, min_gain_s=2.,
                 saved_probe_s=17., partial_credit=.25)


def detour_seconds(start, station, anchor):
    return max(0.,float((np.linalg.norm(station-start)+np.linalg.norm(anchor-station)
                        -np.linalg.norm(anchor-start))/5))


def channel_order(channels, current, primary=None):
    """Uniform switch costs: keep current channel first if available."""
    ordered=sorted(set(channels))
    if current in ordered:
        ordered.remove(current);ordered.insert(0,current)
    if primary in ordered and primary!=current:
        ordered.remove(primary);ordered.append(primary)
    return ordered


def service_seconds(channels,current):
    return 5*len(channels)+len(channels)-int(current in channels)


class B4Strategy(Q2Proposal):
    def __init__(self,action,b4_config=None,**kwargs):
        super().__init__(action,**kwargs)
        self.b4_config=dict(B4_CONFIG,**(b4_config or {}))
        self.joint_stops=0
        self.joint_measurements=0
        self.joint_planning_calls=0
        self.joint_planning_s=0.
        self.joint_log=[]
        self._after_joint=False
        self._probe_preview=None

    def select_local_measurement(self,c,poly,center,radius,default):
        key=(c,poly.tobytes(),self.position.tobytes(),self.receiver_channel,
             np.asarray(default).tobytes())
        if self._probe_preview is not None and self._probe_preview[0]==key:
            return self._probe_preview[1].copy()
        point=super().select_local_measurement(c,poly,center,radius,default)
        self._probe_preview=(key,point.copy())
        return point

    def admissible(self,c,point):
        if c not in self.polys or c in self.cleared or c in self.absent:
            return False
        if np.linalg.norm(self.polys[c]-point,axis=1).max()>999:
            return False
        previous=[p for p,_ in self.measurements[c]]+self.negative_stations[c]
        return not any(np.linalg.norm(point-p)<1 for p in previous)

    def preview_anchor(self,task):
        """Preview the actual next Q2 probe, not an uncertain source's route center."""
        kind,c,p=task
        if kind=='scan':return np.asarray(p),None
        poly=self.polys[c];center,radius=self.region_circle(poly)
        # Existing clear attempts and peripheral initial probes retain priority.
        if radius<=self.trial or radius>900:
            return None,None
        if self.initial_detected==0 and len(self.measurements[c])==1:
            return None,None
        _,_,vh=np.linalg.svd(poly-center,full_matrices=False)
        normal=np.array([-vh[0,1],vh[0,0]])
        options=[center+sign*self.lateral*normal for sign in [-1,1]]
        point=min(options,key=lambda x:np.linalg.norm(x-self.position))
        if np.linalg.norm(poly-point,axis=1).max()>999:return None,None
        if np.linalg.norm(point-self.position)<1:
            point=center+max(10,self.lateral)*normal
        point=self.select_local_measurement(c,poly,center,radius,point)
        return point,c

    def active_channels(self,primary):
        eligible=[]
        for c in sorted(set(self.polys)-self.cleared-self.absent):
            center,radius=self.region_circle(self.polys[c])
            # Small envelopes should use the existing cheap clearing attempt.
            if radius>self.trial:
                eligible.append((c,center,radius))
        eligible.sort(key=lambda item:(item[0]!=primary,np.linalg.norm(item[1]-self.position),item[0]))
        return eligible[:self.b4_config['max_channels']]

    def information_credit(self,metrics,radius):
        cfg=self.b4_config
        ready=np.clip(metrics['ready_probability'],0.,1.)
        shrink=np.clip(1-metrics['expected_radius_m']/max(radius,1e-9),0.,1.)
        # 17 s = one 5 s probe plus a 60 m lateral round trip at 5 m/s.
        return float(cfg['saved_probe_s']*(ready+cfg['partial_credit']*(1-ready)*shrink))

    def choose_subset(self,credits,primary):
        selected=[]
        if primary is not None:
            if primary not in credits:return []
            selected.append(primary)
        for c,gain in credits.items():
            if c!=primary and gain>5+int(c!=self.receiver_channel):selected.append(c)
        return channel_order(selected,self.receiver_channel,primary)

    def plan_joint(self,baseline):
        if len(self.active_channels(baseline[1] if baseline[0]=='clear' else None))<2:
            return None
        anchor,primary=self.preview_anchor(baseline)
        if anchor is None:return None
        active=self.active_channels(primary)
        if len(active)<2:return None
        cfg=self.b4_config
        states={}
        for c,center,radius in active:
            try:area_samples(self.polys[c],cfg['coarse_power'])
            except ValueError:continue
            states[c]=(self.polys[c],center,radius)
        active=[(c,center,radius) for c,(_,center,radius) in states.items()]
        if len(active)<2 or (primary is not None and primary not in states):return None
        samples={}
        cache={}

        def metrics(c,p,power):
            if not self.admissible(c,p):return None
            key=(c,power,*np.round(p,7))
            if key not in cache:
                if (c,power) not in samples:
                    samples[c,power]=area_samples(states[c][0],power)
                points,weights=samples[c,power]
                cache[key]=expected_cost(states[c][0],p,self.position,0,points,weights,
                                         2 if power==cfg['coarse_power'] else 3)
            return cache[key]

        first=[];second=[];raw=[anchor,self.position.copy()]
        for c,center,radius in active:
            poly=states[c][0]
            try:
                options=self.candidates(c,poly,center,radius,anchor)
                ranked=sorted(((p,metrics(c,p,cfg['coarse_power'])) for p in options),
                              key=lambda item:item[1]['cost_s'])
            except ValueError:
                # Area-degenerate evidence has no area prior; let B2 handle it.
                continue
            if not ranked:continue
            best=ranked[0][0];first.append(best);raw.append(best)
            _,_,vh=np.linalg.svd(poly-poly.mean(axis=0),full_matrices=False)
            normal=np.array([-vh[0,1],vh[0,0]])
            alternatives=[p for p,_ in ranked if (p-center)@normal*((best-center)@normal)<0]
            other=alternatives[0] if alternatives else (ranked[1][0] if len(ranked)>1 else best)
            second.append(other);raw.append(other)
        if not first:return None
        # Full mean, each side's mean, pair means, and means including baseline.
        for group in [first,second]:
            raw.append(np.mean([anchor,*group],axis=0))
            raw.append(np.mean(group,axis=0))
            for i,p in enumerate(group):
                raw.append((p+anchor)/2)
                for q in group[i+1:]:
                    raw.extend([(p+q)/2,(p+q+anchor)/3])
        unique=[]
        for p in raw:
            if any(np.linalg.norm(p-q)<1 for q in unique):continue
            if sum(self.admissible(c,p) for c in states)<2:continue
            if primary is not None and not self.admissible(primary,p):continue
            unique.append(p)
        # Bound runtime; prioritize candidates that add little travel to baseline.
        unique.sort(key=lambda p:detour_seconds(self.position,p,anchor))
        unique=unique[:cfg['max_candidates']]
        if not unique:return None

        def evaluate(p,power):
            credits={}
            base_primary=0.
            for c,(_,center,radius) in states.items():
                result=metrics(c,p,power)
                if result is None:continue
                benefit=self.information_credit(result,radius)
                if primary is None:
                    at_anchor=metrics(c,anchor,power)
                    # A future scan already offers information. Where its receive
                    # probability is unknown, charge maximal opportunity cost.
                    benefit-=self.information_credit(at_anchor,radius) if at_anchor else cfg['saved_probe_s']
                credits[c]=benefit
            channels=self.choose_subset(credits,primary)
            if len(channels)<2:return None
            if primary is not None:
                at_anchor=metrics(primary,anchor,power)
                if at_anchor is None:return None
                base_primary=self.information_credit(at_anchor,states[primary][2])-service_seconds([primary],self.receiver_channel)
            detour=detour_seconds(self.position,p,anchor)
            gain=sum(credits[c] for c in channels)-service_seconds(channels,self.receiver_channel)-detour-base_primary
            return dict(point=p,channels=channels,gain_s=float(gain),detour_s=detour,
                        baseline_credit_s=float(base_primary),credits={str(c):credits[c] for c in channels})

        ranked=[result for p in unique if (result:=evaluate(p,cfg['coarse_power'])) is not None]
        ranked.sort(key=lambda result:result['gain_s'],reverse=True)
        fine=[result for row in ranked[:cfg['finalists']]
              if (result:=evaluate(row['point'],cfg['fine_power'])) is not None]
        self.joint_planning_calls+=1
        if not fine:return None
        best=max(fine,key=lambda result:result['gain_s'])
        if best['gain_s']<cfg['min_gain_s']:return None
        best.update(anchor=anchor.tolist(),baseline_kind=baseline[0],baseline_key=baseline[1],
                    candidate_count=len(unique),primary=primary)
        return best

    def select_task(self,tasks,route,sites,remaining):
        baseline=super().select_task(tasks,route,sites,remaining)
        if self._after_joint or self.joint_stops>=self.b4_config['max_joint_stops']:
            return baseline
        if self.deadline-time.monotonic()<10:return baseline
        start=time.perf_counter()
        try:
            plan=self.plan_joint(baseline)
        finally:
            self.joint_planning_s+=time.perf_counter()-start
        return ['joint',plan,plan['point']] if plan is not None else baseline

    def execute_task(self,task,remaining):
        if task[0]!='joint':
            self._after_joint=False
            return super().execute_task(task,remaining)
        _,plan,point=task
        before=self.position.copy()
        executed=[]
        for c in plan['channels']:
            if len(self.cleared)==16:break
            if not self.admissible(c,point):continue
            self.measure(point,c)
            self.joint_measurements+=1;executed.append(c)
        self.joint_stops+=1
        self._after_joint=True  # force a normal task before another joint detour
        log={k:v for k,v in plan.items() if k!='point'}
        log.update(position_before=before.tolist(),point=point.tolist(),executed=executed)
        self.joint_log.append(log)
        if plan.get('primary') is not None and plan['primary'] not in self.cleared:
            # Detour scoring assumes continuation of the original source task.
            # Finish it before global replanning; other channels retain new evidence.
            c=plan['primary']
            super().execute_task(['clear',c,self.routing_center(c)],remaining)
        # Measuring selected known channels does NOT complete a sector scan.
        # Only the inherited execute_task may subsequently complete a FULL scan
        # at a proven covering site; partial joint measurements never do so.
