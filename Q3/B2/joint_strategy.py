"""Replan search stations and discovered targets as one open tour."""
import math,time
import numpy as np
from q3_strategy import Strategy,envelope_circle
from sweep_strategy import SweepStrategy


def open_tour(start,points):
    """Deterministic nearest-neighbour start then first-improvement 2-opt."""
    if not len(points):return []
    points=np.array(points);remaining=set(range(len(points)));route=[];p=np.array(start)
    while remaining:
        i=min(remaining,key=lambda j:(np.linalg.norm(points[j]-p),j))
        route.append(i);remaining.remove(i);p=points[i]
    allp=np.vstack([start,points]);d=np.linalg.norm(allp[:,None,:]-allp[None,:,:],axis=2)
    for iteration in range(40):
        changed=False
        for i in range(len(route)):
            prev=0 if i==0 else route[i-1]+1
            for j in range(i+1,len(route)):
                a,b=route[i]+1,route[j]+1
                old=d[prev,a];new=d[prev,b]
                if j+1<len(route):
                    following=route[j+1]+1;old+=d[b,following];new+=d[a,following]
                if new<old-1e-7:
                    route[i:j+1]=reversed(route[i:j+1]);changed=True;break
            if changed:break
        if not changed:break
    return route


class JointStrategy(SweepStrategy):
    def __init__(self,action,opportunistic=False,defer_uncertain=False,stop_at_max=False):
        super().__init__(action,7,1000.,600.,False)
        self.opportunistic=opportunistic
        self.defer_uncertain=defer_uncertain
        self.stop_at_max=stop_at_max
        self.scans=[]
        self.routing_calls=0

    def scan(self,point,unknown_only=False):
        for channel in range(1,21):
            if self.stop_at_max and len(self.cleared)==16:break
            if channel in self.cleared or channel in self.absent:continue
            if channel in self.polys:
                if unknown_only or self.region_circle(self.polys[channel])[1]<=19.8:continue
                if any(np.linalg.norm(point-p)<1 for p,_ in self.measurements[channel]):continue
            self.measure(point,channel)
        self.scans.append(np.array(point).copy())

    def run(self):
        entered=self.send('/enter');self.deadline=time.monotonic()+max(0,entered.get('remaining_real_duration_s',1200)-5)
        self.scan(np.zeros(2))
        sites=[1000*np.array([math.cos(k*math.tau/7),math.sin(k*math.tau/7)]) for k in range(7)]
        remaining=set(range(7))
        while remaining or set(self.polys)-self.cleared:
            if self.stop_at_max and len(self.cleared)==16:
                self.absent=set(range(1,21))-self.cleared
                break
            pending=sorted(set(self.polys)-self.cleared)
            tasks=[('scan',i,sites[i]) for i in sorted(remaining)]+[('clear',c,self.region_circle(self.polys[c])[0]) for c in pending]
            route=open_tour(self.position,[task[2] for task in tasks]);self.routing_calls+=1
            chosen=route[0]
            if self.defer_uncertain and remaining:
                for index in route:
                    kind,key,point=tasks[index]
                    if kind=='scan' or len(self.measurements[key])>=2 or self.region_circle(self.polys[key])[1]<=100:
                        chosen=index;break
            kind,key,point=tasks[chosen]
            if kind=='scan':
                self.scan(point);remaining.remove(key)
            else:
                if self.opportunistic:
                    center,radius=self.region_circle(self.polys[key])
                    if radius<=120 and self.clear(center,key):
                        pass
                    else:self.localize(key)
                else:self.localize(key)
            if not remaining:
                self.absent=set(range(1,21))-set(self.polys)-self.cleared
        assert len(self.cleared|self.absent)==20
        response=self.send('/exit')
        return {'method':'joint_opportunity' if self.opportunistic else 'joint_route','virtual_time_s':response['virtual_time_s'],
                'cleared_channels':sorted(self.cleared),'absent_channels':sorted(self.absent),'actions':len(self.trace),
                'cover_searches':self.fallback_count,'routing_calls':self.routing_calls,
                'stop_at_max_sources':self.stop_at_max}
