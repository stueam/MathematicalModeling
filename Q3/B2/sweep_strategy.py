"""Short certified scan circuit with bounded-detour source clearing."""
import math,time
import numpy as np
from q3_strategy import Strategy,envelope_circle
from fast_strategy import FastStrategy


class SweepStrategy(FastStrategy):
    def __init__(self,action,count=6,ring=1150.,detour=600.,trial_clear=False):
        Strategy.__init__(self,action,'main')
        self.count,self.ring,self.detour=count,ring,detour
        self.trial_clear=trial_clear
        # Origin covers r<=1000. For outer annulus, nearest station angular
        # gap<=pi/count; squared distance is convex in r.
        self.coverage_bound=max(1000.,*(math.sqrt(r*r+ring*ring-2*r*ring*math.cos(math.pi/count)) for r in [1000,1800]))
        if self.coverage_bound>1000+1e-8:
            raise ValueError('Scan circle does not guarantee full reception coverage')

    def localize(self,channel):
        center,radius=envelope_circle(self.polys[channel])
        if self.trial_clear and 19.8<radius<=150:
            # Cheap attempt at region centre, never counted as certain success.
            # Failure leaves the envelope intact and resumes normal localization.
            if self.clear(center,channel):return
        super().localize(channel)

    def run(self):
        entered=self.send('/enter')
        self.deadline=time.monotonic()+max(0,entered.get('remaining_real_duration_s',1200)-5)
        sites=[np.zeros(2)]+[self.ring*np.array([math.cos(k*math.tau/self.count),math.sin(k*math.tau/self.count)]) for k in range(self.count)]
        for i,point in enumerate(sites):
            for channel in range(1,21):
                if channel not in self.cleared:
                    if channel in self.polys and envelope_circle(self.polys[channel])[1]<=19.8:
                        continue
                    self.measure(point,channel)
            following=sites[i+1] if i+1<len(sites) else None
            while set(self.polys)-self.cleared:
                pending=set(self.polys)-self.cleared
                if following is None:
                    eligible=list(pending)
                else:
                    eligible=[]
                    for channel in pending:
                        center,radius=envelope_circle(self.polys[channel])
                        if len(self.measurements[channel])<2 and radius>150:
                            continue
                        extra=np.linalg.norm(center-self.position)+np.linalg.norm(center-following)-np.linalg.norm(following-self.position)
                        if extra<=self.detour:
                            eligible.append(channel)
                if not eligible:break
                channel=min(eligible,key=lambda c:(np.linalg.norm(envelope_circle(self.polys[c])[0]-self.position),c))
                self.localize(channel)
        self.absent=set(range(1,21))-set(self.polys)-self.cleared
        assert len(self.absent|self.cleared)==20
        exited=self.send('/exit')
        return {'method':'short_sweep','cleared_channels':sorted(self.cleared),'absent_channels':sorted(self.absent),
                'virtual_time_s':exited['virtual_time_s'],'actions':len(self.trace),'cover_searches':self.fallback_count,
                'ring_count':self.count,'ring_radius_m':self.ring,'detour_limit_m':self.detour}
