"""Joint source/type-count world sampler with full public-history replay."""
import math

import numpy as np

from .pooled import likelihood_table,joint_categories
from .posterior import Model,QuadratureError
from .sampling import verify_history
from .shared import point_key
from .simulator import Source,World


class JointWorldSampler:
    prior_description='uniform_N_10_16_and_Beta_1_1_directional_fraction'
    def __init__(self,model=None):
        self.model=model or Model()
        self.signature=None

    def prepare(self,b):
        signature=tuple((c,p.status,tuple(p.history)) for c,p in b.channels.items())
        if self.signature==signature:
            return
        posts,likelihood=likelihood_table(b,self.model)
        _,joint,weights=joint_categories(likelihood)
        suffix=[None]*(len(weights)+1)
        suffix[-1]=np.zeros_like(joint);suffix[-1][0,0]=1.
        for i in range(len(weights)-1,-1,-1):
            a,o,d=weights[i];old=suffix[i+1];new=a*old
            new[1:]+=o*old[:-1];new[1:,1:]+=d*old[:-1,:-1]
            suffix[i]=new
        self.posts,self.weights,self.joint,self.suffix=posts,weights,joint,suffix
        self.channels=list(b.channels);self.signature=signature

    def sample(self,b,rng):
        self.prepare(b)
        index=int(rng.choice(self.joint.size,p=self.joint.ravel()))
        n,d=divmod(index,self.joint.shape[1]);sources=[];bearings={}
        for i,c in enumerate(self.channels):
            a,o,di=self.weights[i];tail=self.suffix[i+1]
            mass=np.array([a*tail[n,d],o*tail[n-1,d] if n>0 else 0.,
                           di*tail[n-1,d-1] if n>0 and d>0 else 0.])
            if mass.sum()<=0:
                raise QuadratureError('Joint count sampling has no compatible continuation')
            category=int(rng.choice(3,p=mass/mass.sum()))
            if category==0:
                continue
            n-=1;d-=int(category==2)
            post=self.posts[c]
            atoms=post.atoms[:,1:] if category==2 else post.atoms[:,:1]
            index=int(rng.choice(atoms.size,p=atoms.ravel()/atoms.sum()))
            row,column=divmod(index,atoms.shape[1]);atom=column+1 if category==2 else 0
            xy=tuple(map(float,post.xy[row]));radius=float(rng.uniform(post.lower[row],post.upper[row,atom]))
            heading=math.degrees(float(rng.uniform(post.left[row,atom-1],post.right[row,atom-1])))%360 if atom else None
            sources.append(Source(c,xy,radius,heading))
            for obs in b.channels[c].directions:bearings[c,point_key(obs.action.position)]=obs.bearing
        if n or d:
            raise QuadratureError('Joint source/type count was not allocated completely')
        world=World(sources,seed=int(rng.integers(0,2**63)),error_mode='iid',cleared=b.cleared,known_bearings=bearings)
        verify_history(world,b)
        return world
