from dataclasses import replace
import copy
import pickle
import time

import pytest

from bayes_tsp.cached import CachedRefinementPolicy
from bayes_tsp.coupling import CoupledBelief
from bayes_tsp.refinement import RefinementConfig
from bayes_tsp.rollout_review import (ReviewWorldSampler, RolloutJob, evaluate_task,
                                    capture_continuation_state)
from bayes_tsp.shared import Action, load


def test_sampled_world_replays_fixed_radius_bearings_and_cleared_history():
    simulator = load('simulator')
    sources = [simulator.Source(1, (500., 0.), 1100.), simulator.Source(2, (800., 200.), 1250.)]
    sources += [simulator.Source(c, (float(c*50), 900.), 1400.) for c in range(3, 11)]
    env = simulator.LocalSimulator(simulator.World(sources, seed=503))
    belief = CoupledBelief()
    history = [Action('measure', (0., 0.), 1), Action('measure', (-700., 0.), 1),
               Action('clear', (490., 0.), 1), Action('measure', (-700., 0.), 2),
               Action('measure', (0., 0.), 2)]
    for index, action in enumerate(history):
        response = env.execute(action, str(index))
        belief.apply(action, response, str(index))
    before = pickle.dumps(belief)
    sampler = ReviewWorldSampler(seed=709, pool_size=128, evidence_size=512)
    worlds = sampler.sample(belief, 4)
    assert pickle.dumps(belief) == before
    for world in worlds:
        assert 10 <= world.score()['source_count'] <= 16
        assert world.score()['cleared_count'] == 1
        # Evaluator-only replay of already-observed events. Clearing is undone
        # on this extra clone solely to validate the historical feedback.
        replay = world.clone()
        replay._cleared = set()
        for action, response in belief.applied.values():
            feedback, _ = replay.feedback(action)
            assert feedback == {key: response[key] for key in feedback}
        assert replay.score()['cleared_count'] == 1
        first, _ = world.feedback(history[-1])
        repeated, _ = world.feedback(history[-1])
        assert repeated == first
        assert world.score()['cleared_count'] == 1


def test_complete_sampled_continuation_preserves_snapshot_and_never_scores_prefix():
    simulator = load('simulator')
    config = replace(RefinementConfig(resolution=8), stable_routing=False)
    true_world = simulator.generate_world(713, 10, 'uniform', 1000., 'correlated')
    env, belief = simulator.LocalSimulator(true_world), CoupledBelief()
    policy = CachedRefinementPolicy(config)
    for step in range(100):
        action = policy.choose(belief)
        belief.apply(action, env.execute(action, str(step)), str(step))
        if belief.cleared:
            break
    assert len(belief.cleared) == 1 and not belief.done()
    candidate = policy.choose(belief)
    batch = tuple(policy.batch_channels) if policy.batch_position == candidate.position else ()
    world = ReviewWorldSampler(seed=704, pool_size=128, evidence_size=512).sample(belief, 1)[0]
    before_belief, before_world = pickle.dumps(belief), pickle.dumps(world)
    job = RolloutJob(belief, world, candidate, batch, config, max_steps=1000, timeout_s=60.)
    result = evaluate_task(job)
    assert result['completed'], result
    assert result['cost'] > 0 and result['steps'] > 1 and result['error'] is None
    assert pickle.dumps(belief) == before_belief
    assert pickle.dumps(world) == before_world
    repeated = evaluate_task(job)
    assert repeated['completed'] and repeated['cost'] == result['cost']
    assert repeated['steps'] == result['steps']
    incomplete = evaluate_task(replace(job, max_steps=1))
    assert not incomplete['completed'] and incomplete['cost'] is None
    assert incomplete['steps'] == 1 and 'action limit' in incomplete['error']
    timed_out = evaluate_task(replace(job, timeout_s=1e-12))
    assert not timed_out['completed'] and timed_out['cost'] is None
    assert timed_out['steps'] == 0 and 'deadline' in timed_out['error']
    limited_belief = belief.clone()
    limited_belief.virtual_limit = limited_belief.virtual_time+.1
    over_budget = evaluate_task(replace(job, belief=limited_belief))
    assert not over_budget['completed'] and over_budget['cost'] is None
    assert over_budget['steps'] == 1 and 'virtual budget' in over_budget['error']


@pytest.mark.parametrize('completion_mode', [False, True])
def test_stateful_review_reproduces_warm_v6_every_action_and_complete_cost(monkeypatch, completion_mode):
    simulator = load('simulator')
    config = RefinementConfig(resolution=8, stable_routing=False)
    live = simulator.LocalSimulator(simulator.generate_world(0, 16, 'uniform', 1500., 'iid'))
    belief, policy = CoupledBelief(), CachedRefinementPolicy(config)
    for step in range(150):
        action = policy.choose(belief)
        belief.apply(action, live.execute(action, str(step)), str(step))
        if belief.cleared and policy.tour.revisions >= 3:
            break
    assert policy.tour.route and not belief.done()
    if completion_mode:
        # Once the live policy enters its permanent guard, resetting its
        # internal flag changes the continuation even with an identical belief.
        policy.completion_mode = True
    candidate = policy.choose(belief)
    state = capture_continuation_state(policy)
    serialized_state = pickle.dumps(state)
    world = ReviewWorldSampler(seed=713, pool_size=128, evidence_size=512).sample(belief, 1)[0]

    warm_policy, observer = copy.deepcopy(policy), belief.clone()
    observer.deadline = time.monotonic()+120.
    warm_world = world.clone()
    env = simulator.LocalSimulator(warm_world, observer.position, observer.receiver, observer.virtual_time)
    initial_time = observer.virtual_time
    expected_actions = []
    for step in range(1000):
        if observer.done():
            break
        action = candidate if step == 0 else warm_policy.choose(observer)
        request_id = f'warm-regression-{step}'
        observer.apply(action, env.execute(action, request_id), request_id)
        expected_actions.append(action)
    assert observer.done() and warm_world.score()['all_cleared']

    actual_actions = []
    execute = simulator.LocalSimulator.execute

    def record_action(env, action, request_id):
        actual_actions.append(action)
        return execute(env, action, request_id)

    monkeypatch.setattr(simulator.LocalSimulator, 'execute', record_action)
    job = RolloutJob(belief, world, candidate, config=config, continuation_state=state)
    result = evaluate_task(job)
    assert result['completed'], result
    assert actual_actions == expected_actions
    assert result['cost'] == observer.virtual_time-initial_time
    assert result['steps'] == len(expected_actions)
    assert pickle.dumps(state) == serialized_state
