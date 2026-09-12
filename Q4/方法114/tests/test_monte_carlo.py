from concurrent.futures import Future

import pytest

from q4.core import Belief
from q4.monte_carlo import MonteCarloPolicy
from q4.shared import Action
from q4.simulator import LocalSimulator, generate_world


class CompletedPool:
    def __init__(self, fail_one=False):
        self.fail_one=fail_one

    def submit(self, function, job):
        future=Future()
        complete=not (self.fail_one and job['world_index']==0 and job['candidate_index']==0)
        future.set_result({'world_index':job['world_index'],'candidate_index':job['candidate_index'],
                           'complete':complete,'cost_s':1000-100*job['candidate_index'] if complete else None,
                           'steps':7,'movement_m':100.,'real_time_s':.1,'error':None})
        return future


@pytest.mark.parametrize('fail_one',[False,True])
def test_root_requires_every_world_candidate_to_complete(fail_one):
    b=Belief(); env=LocalSimulator(generate_world(91,10))
    for c in range(1,21):
        action=Action('measure',(0.,0.),c)
        b.apply(action,env.execute(action,str(c)),str(c))
    policy=MonteCarloPolicy(mc_worlds=2,mc_workers=1,mc_candidates=3)
    policy.pool=CompletedPool(fail_one)
    before_steps=b.steps
    action=policy.choose(b)
    assert b.steps==before_steps
    review=policy.records[-1]['mc_review']
    assert review['candidate_count']>=2 and review['worlds_per_action']==2
    if fail_one:
        assert not review['accepted_batch']
        assert policy.counters['mc_effective_rollouts']==0
        assert policy.counters['mc_changed']==0
        assert policy.counters['mc_discarded_rollouts']==2*review['candidate_count']
    else:
        assert review['accepted_batch'] and review['chosen_candidate']>0
        assert policy.counters['mc_effective_rollouts']==2*review['candidate_count']
        assert policy.counters['mc_changed']==1
        assert review['mean_total_s_per_sampled_source'][review['chosen_candidate']] < review['mean_total_s_per_sampled_source'][0]
