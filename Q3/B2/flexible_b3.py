"""Movable guaranteed sector scans combined with B3 state and localization."""
import math,time
import numpy as np
from scipy.optimize import minimize
from radius_coupling import CoupledCloseB3
from route_search_probe import improved_tour
from q3_strategy import envelope_circle

class FlexibleB3(CoupledCloseB3):
    def __init__(self,action,sweeps=1,reuse_sector=False,sectors=7,**kw):
        super().__init__(action,**kw);self.sweeps=sweeps
        self.reuse_sector=reuse_sector
        self.sectors=sectors
        self.corners=[np.array([[r*math.cos(a),r*math.sin(a)] for r in [1000,1800] for a in [k*math.tau/sectors-math.pi/sectors,k*math.tau/sectors+math.pi/sectors]]) for k in range(sectors)]

    def adjust(self,k,prev,nxt,p):
        corners=self.corners[k]
        def objective(x):return np.linalg.norm(x-prev)+(np.linalg.norm(x-nxt) if nxt is not None else 0)
        res=minimize(objective,p,method='SLSQP',constraints=[{'type':'ineq','fun':lambda x:999.5-np.linalg.norm(corners-x,axis=1)}],options={'maxiter':15,'ftol':1e-4})
        x=res.x
        if np.max(np.linalg.norm(corners-x,axis=1))>999.50001:return p
        if np.min(corners@x)<=0:return p
        return x if objective(x)<objective(p) else p

    def run(self):
        entered=self.send('/enter');self.deadline=time.monotonic()+max(0,entered.get('remaining_real_duration_s',1200)-5)
        n=self.sectors
        self.scan(np.zeros(2));sites=[(1400 if n==6 else 1100)*np.array([math.cos(k*math.tau/n),math.sin(k*math.tau/n)]) for k in range(n)];remaining=set(range(n))
        return self.continue_run(sites,remaining)

    def select_task(self,tasks,route,sites,remaining):return tasks[route[0]]

    def routing_center(self,c):
        center,r=self.region_circle(self.polys[c]);goal=center
        if len(self.measurements[c])==1:
            _,a=self.measurements[c][0];a=math.radians(a);goal=center+self.range_bias*r*np.array([math.cos(a),math.sin(a)])
        return goal

    def execute_task(self,task,remaining):
        kind,key,p=task
        if kind=='scan':self.scan(p);remaining.remove(key)
        else:
            self.localize(key)
            if self.reuse_sector:
                covered=[k for k in remaining if np.max(np.linalg.norm(self.corners[k]-self.position,axis=1))<=999.99 and np.min(self.corners[k]@self.position)>0]
                if covered:
                    self.scan(self.position.copy());remaining.difference_update(covered)

    def continue_run(self,sites,remaining):
        while remaining or set(self.polys)-self.cleared:
            if len(self.cleared)==16:self.absent=set(range(1,21))-self.cleared;break
            self.shared_observations()
            if len(set(self.polys)|self.cleared)==16:
                remaining.clear();self.absent=set(range(1,21))-set(self.polys)-self.cleared
            tasks=[['scan',k,sites[k].copy()] for k in sorted(remaining)]
            for c in sorted(set(self.polys)-self.cleared):
                tasks.append(['clear',c,self.routing_center(c)])
            if not tasks:break
            route=improved_tour(self.position,[t[2] for t in tasks])
            for _ in range(self.sweeps):
                for j,i in enumerate(route):
                    if tasks[i][0]!='scan':continue
                    prev=self.position if j==0 else tasks[route[j-1]][2];nxt=tasks[route[j+1]][2] if j+1<len(route) else None
                    tasks[i][2]=self.adjust(tasks[i][1],prev,nxt,tasks[i][2])
                route=improved_tour(self.position,[t[2] for t in tasks])
            task=self.select_task(tasks,route,sites,remaining)
            self.execute_task(task,remaining)
            if not remaining:self.absent=set(range(1,21))-set(self.polys)-self.cleared
        result=self.send('/exit');return {'method':'flexible_B3','virtual_time_s':result['virtual_time_s'],'cleared_channels':sorted(self.cleared),'absent_channels':sorted(self.absent),'cover_searches':self.fallback_count}
