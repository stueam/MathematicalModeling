"""Offline-only tuning of B2: shared observations and survey geometry."""
import math,time
import numpy as np
from joint_strategy import JointStrategy,open_tour
from q3_strategy import envelope_circle


class TunedStrategy(JointStrategy):
    def __init__(self,action,count=7,radius=1000.,share=False,defer=False,found_stop=False,range_bias=0.):
        super().__init__(action,stop_at_max=True)
        self.count=count;self.ring_radius=radius;self.share=share;self.defer=defer
        self.found_stop=found_stop
        self.range_bias=range_bias
        # Origin covers r<=1000. On each remaining angular sector, squared
        # distance is maximal at an angular and radial endpoint.
        assert max(r*r+radius*radius-2*r*radius*math.cos(math.pi/count)
                   for r in [1000,1800]) < 1000**2

    def shared_observations(self):
        for c in sorted(set(self.polys)-self.cleared):
            poly=self.polys[c];center,r=envelope_circle(poly)
            if r<=19.8 or np.linalg.norm(center-self.position)<30:continue
            if np.max(np.linalg.norm(poly-self.position,axis=1))>999:continue
            # Require a useful new line of sight and avoid repeated locations.
            v=center-self.position;nv=np.linalg.norm(v)
            cross=[]
            for p,_ in self.measurements[c]:
                u=center-p;nu=np.linalg.norm(u)
                cross.append(abs(u[0]*v[1]-u[1]*v[0])/max(nu*nv,1e-9))
            if max(cross,default=0)<.25:continue
            if any(np.linalg.norm(p-self.position)<1 for p,_ in self.measurements[c]):continue
            self.measure(self.position.copy(),c)

    def run(self):
        entered=self.send('/enter');self.deadline=time.monotonic()+max(0,entered.get('remaining_real_duration_s',1200)-5)
        self.scan(np.zeros(2))
        sites=[self.ring_radius*np.array([math.cos(k*math.tau/self.count),math.sin(k*math.tau/self.count)]) for k in range(self.count)]
        remaining=set(range(self.count))
        while remaining or set(self.polys)-self.cleared:
            if len(self.cleared)==16:self.absent=set(range(1,21))-self.cleared;break
            if self.share:self.shared_observations()
            if len(self.cleared)==16:continue
            if self.found_stop and len(set(self.polys)|self.cleared)==16:
                remaining.clear();self.absent=set(range(1,21))-set(self.polys)-self.cleared
            pending=sorted(set(self.polys)-self.cleared)
            goals=[]
            for c in pending:
                center,r=envelope_circle(self.polys[c]);goal=center
                if len(self.measurements[c])==1:
                    point,bearing=self.measurements[c][0];a=math.radians(bearing);u=np.array([math.cos(a),math.sin(a)])
                    goal=center+self.range_bias*r*u
                goals.append(('clear',c,goal))
            tasks=[('scan',i,sites[i]) for i in sorted(remaining)]+goals
            route=open_tour(self.position,[t[2] for t in tasks]);self.routing_calls+=1
            chosen=route[0]
            if self.defer and remaining:
                for i in route:
                    kind,key,p=tasks[i]
                    if kind=='scan' or len(self.measurements[key])>=2 or envelope_circle(self.polys[key])[1]<=100:
                        chosen=i;break
            kind,key,p=tasks[chosen]
            if kind=='scan':self.scan(p);remaining.remove(key)
            else:self.localize(key)
            if not remaining:self.absent=set(range(1,21))-set(self.polys)-self.cleared
        result=self.send('/exit')
        return {'method':'B2_route_tuning','virtual_time_s':result['virtual_time_s'],
                'cleared_channels':sorted(self.cleared),'absent_channels':sorted(self.absent),
                'cover_searches':self.fallback_count}


from negative_strategy import NegativeEvidence


class NegativeTunedStrategy(NegativeEvidence,TunedStrategy):
    pass
