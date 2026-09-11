"""Same-source fixed unknown R implies pairwise receive/no-signal bisectors."""
import numpy as np
from q3_strategy import clip
from b3_strategy import B3Strategy
from localization_probe import CloseLocalize

class RadiusCoupling:
    def __init__(self,*args,**kwargs):
        self.coupled_cuts=0
        super().__init__(*args,**kwargs)

    def measure(self,p,c):
        state=super().measure(p,c)
        if c not in self.polys or c in self.cleared:return state
        if state=='direction':
            positives=[np.asarray(p)];negatives=self.negative_stations[c]
        elif state=='no_signal':
            positives=[x for x,_ in self.measurements[c]];negatives=[np.asarray(p)]
        else:return state
        for positive in positives:
            for negative in negatives:
                # ||G-negative|| > R >= ||G-positive||.
                n=negative-positive;norm=np.linalg.norm(n)
                if norm<1e-8:raise ArithmeticError('Contradictory same-position signal states')
                bound=(negative@negative-positive@positive)/2
                self.polys[c]=clip(self.polys[c],n/norm,bound/norm+1e-6)
                self.coupled_cuts+=1
        return state

class CoupledB3(RadiusCoupling,B3Strategy):pass
class CoupledCloseB3(CloseLocalize,RadiusCoupling,B3Strategy):pass
