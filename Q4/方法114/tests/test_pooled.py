from itertools import product

import numpy as np
import pytest

from q4.pooled import joint_categories,count_prior,PooledModel,likelihood_table,type_masses
from q4.core import Belief
from q4.shared import Action,Observation


def test_count_dp_matches_complete_small_enumeration():
    weights=np.array([[.3,.5,.2],[0.,.7,.3],[.4,.1,.5],[.8,.2,0.]])
    prior=count_prior(4,1,3)
    expected=np.zeros_like(weights);counts=np.zeros_like(prior);total=0.
    for assignment in product(range(3),repeat=4):
        n=sum(v>0 for v in assignment);d=sum(v==2 for v in assignment)
        if n>3:continue
        mass=prior[n,d]*np.prod([weights[i,j] for i,j in enumerate(assignment)])
        total+=mass;counts[n,d]+=mass
        for i,j in enumerate(assignment):expected[i,j]+=mass
    actual,joint,_=joint_categories(weights,prior)
    assert actual==pytest.approx(expected/total)
    assert joint==pytest.approx(counts/total)


def test_flat_likelihood_recovers_uniform_count_and_beta_binomial_types():
    marginal,joint,_=joint_categories(np.ones((20,3)))
    assert joint.sum(axis=1)[10:]==pytest.approx(np.ones(7)/7)
    for n in range(10,17):assert joint[n,:n+1]==pytest.approx(np.ones(n+1)/(7*(n+1)))
    assert marginal==pytest.approx(np.tile([.35,.325,.325],(20,1)))


def test_other_channel_evidence_updates_type_belief_without_excluding_geometry():
    b=Belief();model=PooledModel()
    for c in (1,2):b.channels[c].update(Observation(Action('measure',(0.,0.),c),'direction',0.))
    model.condition(b);before=model.posterior(b.channels[1]).directional_probability
    geometry=b.channels[1].region.wkb
    # Signal at origin but a miss only 1 m east is evidence of a directional
    # boundary; it updates the common type fraction through channel 2 only.
    b.channels[2].update(Observation(Action('measure',(1.,0.),2),'no_signal'))
    model.condition(b);after=model.posterior(b.channels[1]).directional_probability
    assert after>before
    assert b.channels[1].region.wkb==geometry and not b.done()


def test_joint_sampler_replays_observed_feedback_for_every_sample():
    from q4.joint_sampling import JointWorldSampler
    from q4.sampling import verify_history
    from q4.simulator import generate_world,LocalSimulator
    from q4.sector import ProbePolicy
    b=Belief();env=LocalSimulator(generate_world(91,10));policy=ProbePolicy()
    for i in range(30):
        action=policy.choose(b);b.apply(action,env.execute(action,str(i)),str(i))
    sampler=JointWorldSampler();rng=np.random.default_rng(13)
    for _ in range(12):
        world=sampler.sample(b,rng)
        verify_history(world,b)
        assert 10<=world.score()['source_count']<=16
        assert world._cleared==set(b.cleared)


def test_mc_uncertainty_screening_increases_required_gain():
    from q4.monte_carlo import StableMonteCarloPolicy,MonteCarloPolicy
    stable=StableMonteCarloPolicy(mc_worlds=16)
    assert stable.required_gain(10.)==17.
    assert stable.required_gain(0.)==2.
    assert MonteCarloPolicy().required_gain(10.)==2.


@pytest.mark.parametrize('omni_mass',[0.,1e-20])
def test_rounded_directional_mass_keeps_nonnegative_likelihood_and_tiny_other_type(omni_mass):
    from types import SimpleNamespace
    from q4.posterior import Posterior
    post=Posterior(np.array([[100.,0.]]),np.array([[omni_mass,1.+np.finfo(float).eps]]),
                   np.array([1000.]),np.array([[1500.,1500.]]),
                   np.array([[0.]]),np.array([[2*np.pi]]),.5)
    assert post.directional_probability>1.
    model=PooledModel()
    model.base=SimpleNamespace(directional_prior=.5,posterior=lambda channel:post)
    b=Belief()
    _,likelihood=likelihood_table(b,model.base)
    assert np.all(np.asarray(likelihood)>=0)
    assert type_masses(post)[0]==pytest.approx(omni_mass,rel=1e-12,abs=0.)
    model.condition(b)
    adjusted=model.posterior(b.channels[1])
    assert np.all(np.isfinite(adjusted.atoms)) and np.all(adjusted.atoms>=0)
    assert adjusted.atoms.sum()==pytest.approx(1.)
    assert (adjusted.atoms[:,0].sum()>0)==(omni_mass>0)
