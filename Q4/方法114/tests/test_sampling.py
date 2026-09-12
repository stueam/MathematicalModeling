import math

import numpy as np

from q4.core import Belief
from q4.policy import Config
from q4.sampling import WorldSampler, conditional_subset, verify_history
from q4.sector import ProbePolicy
from q4.simulator import generate_world, LocalSimulator
from q4.rollout import continue_world


def snapshot():
    env, b, policy = LocalSimulator(generate_world(91, 10)), Belief(), ProbePolicy()
    for i in range(36):
        if b.done():
            break
        a=policy.choose(b)
        b.apply(a,env.execute(a,str(i)),str(i))
    return b,policy


def test_count_conditioning_matches_small_closed_form():
    q=np.array([.2,.4,.6]); rng=np.random.default_rng(19)
    observed=np.zeros(4)
    for _ in range(5000):
        chosen=conditional_subset(q,15,rng)
        assert len(chosen)<=1
        observed[1+chosen[0] if chosen else 0]+=1
    zero=np.prod(1-q)
    exact=np.r_[zero,zero*q/(1-q)]
    assert np.max(abs(observed/5000-exact/exact.sum()))<.025


def test_sampled_worlds_replay_public_history_with_fixed_parameters():
    b,policy=snapshot()
    sampler=WorldSampler(policy.model); rng=np.random.default_rng(50)
    for _ in range(12):
        world=sampler.sample(b,rng)
        assert 10<=world.score()['source_count']<=16
        assert world._cleared==set(b.cleared)
        verify_history(world,b)
        for c in b.known:
            assert c in world._sources
        assert all(1000<=s.radius<=1500 for s in world._sources.values())


def test_full_rollout_and_truncation_are_distinguished():
    b,policy=snapshot(); action=policy.choose(b)
    world=WorldSampler(policy.model).sample(b,np.random.default_rng(72))
    job={'belief':b,'world':world,'config':Config(),'points':policy.points,'batch':policy.batch,
         'shared_checked':policy.shared_checked,'completion_mode':False,'action':action,
         'world_index':0,'candidate_index':0,'wall_limit_s':60.}
    full=continue_world(job)
    assert full['complete'] and full['cost_s']>0 and full['steps']>1
    truncated=continue_world({**job,'max_steps':1})
    assert not truncated['complete'] and truncated['cost_s'] is None
    assert truncated['steps']==1
