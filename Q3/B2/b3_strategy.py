"""B3 candidate: deterministic multi-start routing and shared measurements."""
import math,time
import numpy as np
from route_tuning import NegativeTunedStrategy
from route_search_probe import improved_tour
from q3_strategy import envelope_circle


class B3Strategy(NegativeTunedStrategy):
    def __init__(self,action,share_range=999.,cross_threshold=.25,range_bias=1.,station_count=7,station_radius=1000.,rotation_samples=1,skip_far=False):
        super().__init__(action,share=True,found_stop=True,range_bias=range_bias)
        self.share_range=share_range;self.cross_threshold=cross_threshold
        self.attempted=set()
        self.station_count=station_count;self.station_radius=station_radius;self.rotation_samples=rotation_samples
        self.skip_far=skip_far
        assert max(r*r+station_radius**2-2*r*station_radius*math.cos(math.pi/station_count) for r in [1000,1800])<1000**2

    def measure(self,p,c):
        self.attempted.add((c,round(float(p[0]),6),round(float(p[1]),6)))
        return super().measure(p,c)

    def scan(self,p,unknown_only=False):
        if not self.skip_far:return super().scan(p,unknown_only)
        for c in range(1,21):
            if len(self.cleared)==16:break
            if c in self.cleared or c in self.absent:continue
            if c in self.polys:
                center,r=self.region_circle(self.polys[c])
                if unknown_only or r<=19.8:continue
                if np.linalg.norm(center-p)-r>1500.01:continue
                if any(np.linalg.norm(p-q)<1 for q,_ in self.measurements[c]):continue
            self.measure(p,c)
        self.scans.append(np.asarray(p).copy())

    def shared_observations(self):
        for c in sorted(set(self.polys)-self.cleared):
            poly=self.polys[c];center,r=self.region_circle(poly);dist=np.linalg.norm(center-self.position)
            if r<=19.8 or dist<30:continue
            distance=np.max(np.linalg.norm(poly-self.position,axis=1)) if self.share_range<=999 else dist
            if distance>self.share_range:continue
            if (c,round(float(self.position[0]),6),round(float(self.position[1]),6)) in self.attempted:continue
            v=center-self.position
            crosses=[]
            for p,_ in self.measurements[c]:
                u=center-p;crosses.append(abs(u[0]*v[1]-u[1]*v[0])/max(np.linalg.norm(u)*dist,1e-9))
            if max(crosses,default=0)<self.cross_threshold:continue
            self.measure(self.position.copy(),c)

    def run(self):
        entered=self.send('/enter');self.deadline=time.monotonic()+max(0,entered.get('remaining_real_duration_s',1200)-5)
        self.scan(np.zeros(2));n=self.station_count
        best=float('inf');sites=None
        for offset in np.arange(self.rotation_samples)*math.tau/n/self.rotation_samples:
            trial=[self.station_radius*np.array([math.cos(k*math.tau/n+offset),math.sin(k*math.tau/n+offset)]) for k in range(n)]
            goals=[self.region_circle(self.polys[c])[0] for c in sorted(set(self.polys)-self.cleared)]
            points=trial+goals;route=improved_tour(self.position,points)
            cost=sum(np.linalg.norm(points[b]-(self.position if j==0 else points[route[j-1]])) for j,b in enumerate(route))
            if cost<best:best=cost;sites=trial
        remaining=set(range(n))
        while remaining or set(self.polys)-self.cleared:
            if len(self.cleared)==16:self.absent=set(range(1,21))-self.cleared;break
            self.shared_observations()
            if len(set(self.polys)|self.cleared)==16:
                remaining.clear();self.absent=set(range(1,21))-set(self.polys)-self.cleared
            goals=[]
            for c in sorted(set(self.polys)-self.cleared):
                center,r=self.region_circle(self.polys[c]);goal=center
                if len(self.measurements[c])==1:
                    _,bearing=self.measurements[c][0];a=math.radians(bearing);goal=center+self.range_bias*r*np.array([math.cos(a),math.sin(a)])
                goals.append(('clear',c,goal))
            tasks=[('scan',i,sites[i]) for i in sorted(remaining)]+goals
            if not tasks:break
            route=improved_tour(self.position,[t[2] for t in tasks]);self.routing_calls+=1
            kind,key,p=tasks[route[0]]
            if kind=='scan':self.scan(p);remaining.remove(key)
            else:self.localize(key)
            if not remaining:self.absent=set(range(1,21))-set(self.polys)-self.cleared
        assert len(self.cleared|self.absent)==20
        result=self.send('/exit');return {'method':'B3_candidate','virtual_time_s':result['virtual_time_s'],'cleared_channels':sorted(self.cleared),'absent_channels':sorted(self.absent),'cover_searches':self.fallback_count}
