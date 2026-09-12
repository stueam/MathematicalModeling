"""Small B3 extensions: unknown-channel survey, early side view, closer safe clear."""
import math
import time
import numpy as np
from scipy.optimize import minimize
from q2_lookahead import Q2Proposal
from route_search_probe import improved_tour

CURRENT_CONFIG = dict(pilot=150, pilot_mode='spectral')


class OptimizedB3(Q2Proposal):
    def __init__(self, action, pilot=CURRENT_CONFIG['pilot'], pilot_mode=CURRENT_CONFIG['pilot_mode'], **kwargs):
        super().__init__(action, **kwargs)
        self.pilot=pilot
        self.pilot_mode=pilot_mode

    def scan(self,p,unknown_only=False):
        # Known channels still receive the inherited shared observations and
        # localization. Only this full-band discovery sweep skips them.
        for c in range(1,21):
            if len(self.cleared)==16:break
            if c in self.cleared or c in self.absent or c in self.polys:continue
            self.measure(p,c)
        self.scans.append(np.asarray(p).copy())
        if self.initial_detected is None and np.linalg.norm(p)<1e-6:
            self.initial_detected=len(set(self.polys)|self.cleared)

    def run(self):
        entered=self.send('/enter')
        self.deadline=time.monotonic()+max(0,entered.get('remaining_real_duration_s',1200)-5)
        self.scan(np.zeros(2))
        sites=[1100*np.array([math.cos(k*math.tau/self.sectors),math.sin(k*math.tau/self.sectors)]) for k in range(self.sectors)]
        goals=[self.routing_center(c) for c in sorted(set(self.polys)-self.cleared)]
        points=sites+goals
        route=improved_tour(self.position,points)
        if goals and np.linalg.norm(points[route[0]])>self.pilot:
            p=points[route[0]]/np.linalg.norm(points[route[0]])*self.pilot
            if self.pilot_mode=='perpendicular':
                p=np.array([-p[1],p[0]])
                p=min([p,-p],key=lambda x:sum(np.linalg.norm(g-x) for g in goals))
            elif self.pilot_mode=='spectral':
                bearings=[math.radians(self.measurements[c][0][1]) for c in sorted(set(self.polys)-self.cleared)]
                u=np.column_stack((np.cos(bearings),np.sin(bearings)))
                _,vectors=np.linalg.eigh(u.T@u)
                p=vectors[:,0]*self.pilot
                p=min([p,-p],key=lambda x:np.linalg.norm(x-points[route[0]]))
            else:
                raise ValueError('Unsupported fixed pilot mode')
            for c in sorted(set(self.polys)-self.cleared):
                _,r=self.region_circle(self.polys[c])
                if r>80:self.measure(p,c)
        return self.continue_run(sites,set(range(self.sectors)))

    def clear(self,point,channel,must_succeed=False):
        target=np.asarray(point)
        if must_succeed and channel in self.polys:
            poly=self.polys[channel]
            if np.linalg.norm(poly-target,axis=1).max()<=19.8:
                result=minimize(lambda p:np.linalg.norm(p-self.position),target,method='SLSQP',
                    constraints=[{'type':'ineq','fun':lambda p:19.8-np.linalg.norm(poly-p,axis=1)}],
                    options={'maxiter':30,'ftol':1e-6})
                p=result.x
                if np.isfinite(p).all() and np.linalg.norm(poly-p,axis=1).max()<=19.80001:
                    if np.linalg.norm(p-self.position)<np.linalg.norm(target-self.position):target=p
        return super().clear(target,channel,must_succeed)
