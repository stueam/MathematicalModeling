"""Q2-inspired finite-candidate expected-cost localization for B2.

Planning assumptions: uniform area over the conservative current polygon;
uniform +/-1 degree error at a NEW site. These are scoring assumptions, not
claims of an exact posterior or of the simulator's error distribution.
No radius prior: all candidates must cover the whole polygon within 999 m.
Actual observations still use every B2 geometric constraint and fallback.
"""
import time
import numpy as np
from peripheral_prior import PeripheralProposal
from q2_geometry import minimum_circle, area_samples, direction_region

Q2_CONFIG = dict(coarse_power=3, fine_power=5, coarse_errors=2, fine_errors=3,
                 finalists=3, residual_scale=1.)


def expected_cost(poly, station, current, channel_switch, points, weights,
                  error_order=3, residual_scale=1.):
    """Forward integrate positions x angular errors, not overlapping wedge areas.

    Remaining-time surrogate = travel to posterior circle center + 5 s clear
    + I[r>19.8]*(5 + 2*r/5) s: one extra measure plus a 2*r detour.
    This is a heuristic, not an optimal value function or completion-time bound.
    """
    if np.linalg.norm(poly-station,axis=1).max()>999.000001:
        raise ValueError('Expected-cost candidate must guarantee reception')
    nodes, ew = np.polynomial.legendre.leggauss(error_order)
    ew = ew/2
    distance = np.linalg.norm(points-station,axis=1)
    bearing = np.degrees(np.arctan2(points[:,1]-station[1],points[:,0]-station[0]))
    near = distance<=5
    future = float(weights[near].sum())*5  # near -> clear here, still costs 5 s
    ready = float(weights[near].sum())
    mean_radius = 0.
    cache = {}
    for j in np.flatnonzero(~near):
        for error,w in zip(nodes,ew):
            angle = round(float(bearing[j]+error),2)%360
            if angle not in cache:
                posterior = direction_region(poly,station,angle)
                center,radius = minimum_circle(posterior)
                cost = np.linalg.norm(center-station)/5+5
                if radius>19.8:
                    cost += residual_scale*(5+2*radius/5)
                cache[angle] = (cost,radius)
            cost,radius = cache[angle]
            mass = float(weights[j]*w)
            future += mass*cost
            mean_radius += mass*radius
            ready += mass*(radius<=19.8)
    return dict(cost_s=float(np.linalg.norm(station-current)/5+5+channel_switch+future),
                ready_probability=float(ready), expected_radius_m=float(mean_radius),
                near_probability=float(weights[near].sum()), no_signal_probability=0.,
                probability_mass=float(weights.sum()), evaluations=len(cache))


class MECProposal(PeripheralProposal):
    region_circle = staticmethod(minimum_circle)


class Q2Proposal(MECProposal):
    def __init__(self,action,q2_config=None,**kwargs):
        super().__init__(action,**kwargs)
        self.q2_config = dict(Q2_CONFIG,**(q2_config or {}))
        self.planning_calls = 0
        self.planning_time_s = 0.
        self.planning_degenerate = 0
        self.planning_log = []

    def candidates(self,c,poly,center,radius,default):
        _,_,vh = np.linalg.svd(poly-poly.mean(axis=0),full_matrices=False)
        normal = np.array([-vh[0,1],vh[0,0]])
        choices = [default,self.position.copy(),center]
        for offset in (10.,30.,60.,120.):
            choices.extend([center+offset*normal,center-offset*normal])
        approach = self.position-center
        norm = np.linalg.norm(approach)
        if norm>1:
            choices.extend(center+approach/norm*d for d in (min(radius,100.),min(radius,300.)))
        valid = []
        attempted = [p for p,_ in self.measurements[c]]+self.negative_stations[c]
        for p in choices:
            if np.linalg.norm(poly-p,axis=1).max()>999:
                continue
            if any(np.linalg.norm(p-old)<1 for old in attempted):
                continue  # same-position bearing error is not a new random draw
            if not any(np.linalg.norm(p-old)<.1 for old in valid):
                valid.append(np.asarray(p).copy())
        return valid

    def select_local_measurement(self,c,poly,center,radius,default):
        start = time.perf_counter()
        try:
            choices = self.candidates(c,poly,center,radius,default)
            if not choices:
                return default
            cfg = self.q2_config
            try:
                points,weights = area_samples(poly,cfg['coarse_power'])
            except ValueError:
                self.planning_degenerate += 1
                return default  # no invented probability for degenerate geometry
            switch = int(c!=self.receiver_channel)
            def score(point,samples,masses,order):
                return expected_cost(poly,point,self.position,switch,samples,masses,
                                     order,cfg['residual_scale'])
            coarse = [score(p,points,weights,cfg['coarse_errors']) for p in choices]
            finalists = list(np.argsort([s['cost_s'] for s in coarse])[:cfg['finalists']])
            # Keep the original heuristic in fine comparison if it is admissible.
            for i,p in enumerate(choices):
                if np.linalg.norm(p-default)<.1 and i not in finalists:
                    finalists.append(i)
            points,weights = area_samples(poly,cfg['fine_power'])
            fine = [(i,score(choices[i],points,weights,cfg['fine_errors'])) for i in finalists]
            best,metrics = min(fine,key=lambda item:(item[1]['cost_s'],item[0]))
            self.planning_calls += 1
            if len(self.planning_log)<20:
                self.planning_log.append(dict(channel=int(c),radius_before=float(radius),
                    position=self.position.tolist(),selected=choices[best].tolist(),
                    default=np.asarray(default).tolist(),candidates=len(choices),**metrics))
            return choices[best]
        finally:
            self.planning_time_s += time.perf_counter()-start
