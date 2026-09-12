"""Use no-signal disks to trim known source envelopes conservatively."""
import numpy as np
from scipy.spatial import ConvexHull
from q3_strategy import clip
from joint_strategy import JointStrategy
from sweep_strategy import SweepStrategy


class NegativeEvidence:
    def __init__(self,*args,**kwargs):
        self.negative_stations={c:[] for c in range(1,21)}
        self.negative_cuts=0
        angles=np.arange(16)*2*np.pi/16
        self.negative_normals=np.column_stack((np.cos(angles),np.sin(angles)))
        super().__init__(*args,**kwargs)

    def trim(self,channel,station):
        poly=self.polys[channel]
        # Inscribed16-gon is inside radius1000 disk. Outside disk => outside
        # this polygon. Hull of union of complement pieces remains conservative.
        apothem=1000*np.cos(np.pi/16)-1e-5
        residual=poly@self.negative_normals.T-self.negative_normals@station-apothem
        if np.all(np.max(residual,axis=1)>=0):return
        pieces=[]
        for normal in self.negative_normals:
            try:pieces.append(clip(poly,-normal,-(normal@station+apothem)))
            except ArithmeticError:pass
        if not pieces:raise ArithmeticError('No-signal contradicts feasible source envelope')
        points=np.unique(np.round(np.vstack(pieces),9),axis=0)
        if len(points)>=3:
            points=points[ConvexHull(points).vertices]
        self.polys[channel]=points;self.negative_cuts+=1

    def measure(self,point,channel):
        state=super().measure(point,channel)
        if state=='no_signal':
            self.negative_stations[channel].append(np.array(point))
            if channel in self.polys and channel not in self.cleared:self.trim(channel,np.array(point))
        elif state=='direction':
            for station in self.negative_stations[channel]:self.trim(channel,station)
        return state


class NegativeJoint(NegativeEvidence,JointStrategy):
    pass


class NegativeSweep(NegativeEvidence,SweepStrategy):
    def __init__(self,action):super().__init__(action,7,1000.,600.,False)
