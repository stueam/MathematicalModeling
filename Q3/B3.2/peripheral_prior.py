"""Observation-triggered peripheral proposal; never removes feasible positions."""
import math
import numpy as np
from flexible_b3 import FlexibleB3
from q3_strategy import envelope_circle

class PeripheralProposal(FlexibleB3):
    def __init__(self,action,proposal_radius=1750.,trigger_count=0,route_prior=False,**kwargs):
        super().__init__(action,**kwargs);self.proposal_radius=proposal_radius;self.trigger_count=trigger_count;self.initial_detected=None
        self.route_prior=route_prior

    def scan(self,p,unknown_only=False):
        super().scan(p,unknown_only)
        if self.initial_detected is None and np.linalg.norm(p)<1e-6:
            self.initial_detected=len(set(self.polys)|self.cleared)

    def localize(self,c):
        if self.initial_detected is not None and self.initial_detected<=self.trigger_count and len(self.measurements[c])==1:
            start,bearing=self.measurements[c][0];a=math.radians(bearing);u=np.array([math.cos(a),math.sin(a)])
            disc=(start@u)**2+self.proposal_radius**2-start@start
            if disc>0:
                t=-start@u+math.sqrt(disc);q=start+t*u
                if t>0 and np.max(np.linalg.norm(self.polys[c]-q,axis=1))<999:
                    state=self.measure(q,c)
                    if c in self.cleared:return
        super().localize(c)

    def routing_center(self,c):
        if self.route_prior and self.initial_detected is not None and self.initial_detected<=self.trigger_count and len(self.measurements[c])==1:
            start,bearing=self.measurements[c][0];a=math.radians(bearing);u=np.array([math.cos(a),math.sin(a)])
            disc=(start@u)**2+self.proposal_radius**2-start@start
            if disc>0:
                t=-start@u+math.sqrt(disc)
                if t>0:return start+t*u
        return super().routing_center(c)
