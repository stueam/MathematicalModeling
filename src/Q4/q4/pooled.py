"""Public-history inference for shared uncertainty in source/type counts.

Working prior: N uniform on 10..16; directional fraction Beta(1,1), so the
directional count conditional on N is uniform on 0..N. Pure types remain possible
in the scoring model. These priors never certify absence or determine exit.
"""
import math

import numpy as np

from .posterior import Model, Posterior, QuadratureError
from .sector import ProbePolicy


def count_prior(channel_count=20,minimum=10,maximum=16):
    result=np.zeros((maximum+1,maximum+1))
    for n in range(minimum,maximum+1):
        for d in range(n+1):
            result[n,d]=1/((maximum-minimum+1)*math.comb(channel_count,n)*(n+1)*math.comb(n,d))
    return result


def joint_categories(likelihood,prior=None):
    """Exact bivariate count DP and reverse derivatives for state marginals.

    Each row is the likelihood of absent / omni / directional. The count prior
    supplies probability per assignment, not merely probability per count.
    """
    likelihood=np.asarray(likelihood,dtype=float)
    prior=count_prior(len(likelihood)) if prior is None else np.asarray(prior,dtype=float)
    if (likelihood.shape[1]!=3 or np.any(likelihood<0) or not np.all(np.isfinite(likelihood))
            or np.any(likelihood.sum(axis=1)<=0)):
        raise QuadratureError('Invalid joint categorical likelihood')
    weights=likelihood/likelihood.sum(axis=1,keepdims=True)
    forward=[np.zeros_like(prior)];forward[0][0,0]=1.
    for absent,omni,directional in weights:
        old=forward[-1];new=absent*old
        new[1:]+=omni*old[:-1]
        new[1:,1:]+=directional*old[:-1,:-1]
        forward.append(new)
    joint=forward[-1]*prior;normalizer=float(joint.sum())
    if normalizer<=0:
        raise QuadratureError('No joint source/type-count mass; not an absence proof')
    marginals=np.zeros_like(weights);adjoint=prior.copy()
    for i in range(len(weights)-1,-1,-1):
        absent,omni,directional=weights[i];old=forward[i]
        marginals[i]=[absent*float((old*adjoint).sum()),
                      omni*float((old[:-1]*adjoint[1:]).sum()),
                      directional*float((old[:-1,:-1]*adjoint[1:,1:]).sum())]
        previous=absent*adjoint
        previous[:-1]+=omni*adjoint[1:]
        previous[:-1,:-1]+=directional*adjoint[1:,1:]
        adjoint=previous
    marginals/=normalizer
    return marginals,joint/normalizer,weights


def likelihood_table(b,model):
    posts={};likelihood=[]
    for c,p in b.channels.items():
        if p.status=='absent_certified':
            likelihood.append((1.,0.,0.));continue
        post=model.posterior(p);posts[c]=post
        masses=type_masses(post)
        evidence=post.evidence if p.status=='unresolved' else 1.
        likelihood.append((float(p.status=='unresolved'),
                           evidence*masses[0]/(1-model.directional_prior),
                           evidence*masses[1]/model.directional_prior))
    return posts,likelihood


def type_masses(post):
    # Summing a normalized array can round slightly above one. Subtracting
    # directional mass from one then creates a negative omni likelihood, or
    # loses a small but real omni component. Sum both nonnegative types.
    masses=np.array([post.atoms[:,0].sum(),post.atoms[:,1:].sum()])
    if np.any(masses<0) or not np.all(np.isfinite(masses)) or masses.sum()<=0:
        raise QuadratureError('Invalid posterior type masses')
    return masses/masses.sum()


class PooledModel(Model):
    def __init__(self,resolution=16,directional_prior=.5,existence_prior=.65):
        super().__init__(resolution,directional_prior,existence_prior)
        self.base=Model(resolution,directional_prior,existence_prior)
        self.conditioned={};self.presence={};self.summary={};self.signature=None

    def condition(self,b):
        signature=tuple((c,p.status,tuple(p.history)) for c,p in b.channels.items())
        if signature==self.signature:
            return
        # Remove the independent type prior already included in post.atoms.
        # A known source's common evidence factor cancels; its presence is fixed.
        posts,likelihood=likelihood_table(b,self.base)
        marginals,joint,weights=joint_categories(likelihood)
        conditioned={};presence={}
        for i,(c,p) in enumerate(b.channels.items()):
            presence[c]=float(marginals[i,1:].sum())
            if c not in posts:
                continue
            post=posts[c];old_masses=type_masses(post)
            if presence[c]<=1e-15:
                raise QuadratureError('Scoring posterior has no source mass; keep geometric support')
            new_masses=marginals[i,1:]/presence[c]
            atoms=post.atoms.copy()
            atoms[:,0]*=new_masses[0]/max(old_masses[0],1e-300)
            atoms[:,1:]*=new_masses[1]/max(old_masses[1],1e-300)
            atoms/=atoms.sum()
            conditioned[tuple(p.history)]=Posterior(post.xy,atoms,post.lower,post.upper,post.left,post.right,post.evidence)
        n,d=np.indices(joint.shape)
        self.conditioned,self.presence=conditioned,presence
        self.summary={'expected_sources':float((joint*n).sum()),
                      'expected_directional_sources':float((joint*d).sum()),
                      'source_count_probability':joint.sum(axis=1).tolist(),
                      'working_prior':'uniform_N_10_16_and_Beta_1_1_directional_fraction'}
        self.signature=signature

    def posterior(self,channel):
        return self.conditioned.get(tuple(channel.history)) or self.base.posterior(channel)

    def existence(self,b):
        self.condition(b)
        return {c:self.presence[c] for c,p in b.channels.items() if p.status=='unresolved'}


class PooledPolicy(ProbePolicy):
    def __init__(self,config=None):
        super().__init__(config)
        self.implementation='q4_v7_pooled_source_and_type_counts_massfix'
        self.model=PooledModel(self.config.resolution,self.config.directional_prior,self.config.existence_prior)
        self.counters['pooled_fallbacks']=0

    def choose(self,b):
        previous=self.model.signature
        try:
            self.model.condition(b)
            if self.model.signature!=previous:
                # Other-channel observations alter this channel's type mixture.
                # A per-channel revision alone is no longer a valid cache key.
                self.local_cache.clear()
        except QuadratureError as exc:
            self.model.conditioned={};self.model.signature=None
            self.model.summary={'fallback_to_independent_model':True,'reason':str(exc)}
            self.local_cache.clear();self.counters['pooled_fallbacks']+=1
        action=super().choose(b)
        self.records[-1]['pooled_belief']=self.model.summary
        return action
