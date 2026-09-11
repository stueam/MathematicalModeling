"""Local travel-cost probes; same deterministic global routing."""
import numpy as np
from route_tuning import TunedStrategy,NegativeTunedStrategy
from q3_strategy import envelope_circle

class CloseLocalize:
    def __init__(self,*args,trial=80.,lateral=20.,**kwargs):
        self.trial=trial;self.lateral=lateral
        super().__init__(*args,**kwargs)

    def localize(self,c):
        for k in range(8):
            poly=self.polys[c];center,radius=self.region_circle(poly)
            if radius<=19.8:
                self.clear(center,c,must_succeed=True);return
            if radius<=self.trial:
                if self.clear(center,c):return
            if radius>900:break
            _,_,vh=np.linalg.svd(poly-center,full_matrices=False)
            normal=np.array([-vh[0,1],vh[0,0]])
            choices=[center+sign*self.lateral*normal for sign in [-1,1]]
            point=min(choices,key=lambda p:np.linalg.norm(p-self.position))
            if np.max(np.linalg.norm(poly-point,axis=1))>999:break
            if np.linalg.norm(point-self.position)<1:
                point=center+max(10,self.lateral)*normal
            point=self.select_local_measurement(c,poly,center,radius,point)
            state=self.measure(point,c)
            if c in self.cleared:return
            if state=='no_signal':break
        super().localize(c)

class CloseTuned(CloseLocalize,TunedStrategy):pass
class CloseNegative(CloseLocalize,NegativeTunedStrategy):pass

from b3_strategy import B3Strategy
class CloseB3(CloseLocalize,B3Strategy):pass
