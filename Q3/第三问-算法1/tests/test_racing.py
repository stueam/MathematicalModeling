import math
import time
import json

import numpy as np
import pytest

from q3.core import Action, Belief
from q3.movement import MovementPolicy
from q3.planner import rollout_result
from q3.racing import RacingPlanner, race_statistics
from q3.rollout_engine import RolloutEngine, Trial, cached_rollout, continuation_key
from q3.sampling import SamplingError
from q3.simulator import LocalSimulator, Source, World


def small_task():
    world = World([Source(1, (500., 100.), 1250)], seed=9)
    b = Belief()
    for c, p in b.channels.items():
        if c != 1:
            p.status = 'absent_certified'
    env = LocalSimulator(world)
    a = Action('measure', (0., 0.), 1)
    b.apply(a, env.execute(a, 'start'), 'start')
    return b, world


def test_full_suffix_cache_matches_uncached_cost_and_depth():
    b, world = small_task()
    policy, cache = MovementPolicy(), {}
    initial = continuation_key(b)
    actions = policy.proposals(b, 6, True)
    for a in actions+actions:
        full = rollout_result(b, world, a, policy)
        reused = cached_rollout(b, world, a, policy, cache)
        assert reused.cost == pytest.approx(full.cost, abs=1e-6)
        assert reused.steps == full.steps
    assert reused.suffix_hits == 1
    assert reused.executed_steps < reused.steps
    assert continuation_key(b) == initial
    assert not world.score()['all_cleared']


def test_cache_checks_virtual_and_action_budget():
    b, world = small_task()
    policy, cache = MovementPolicy(), {}
    a = policy.choose(b)
    result = cached_rollout(b, world, a, policy, cache)
    b.virtual_limit = b.virtual_time+result.cost-.01
    with pytest.raises(SamplingError):
        cached_rollout(b, world, a, policy, cache)
    b.virtual_limit = math.inf
    with pytest.raises(SamplingError):
        cached_rollout(b, world, a, policy, cache, max_steps=1)
    with pytest.raises(TimeoutError):
        cached_rollout(b, world, a, policy, cache, deadline=time.monotonic()-1)


def test_last_allowed_step_can_finish():
    b = Belief()
    for p in b.channels.values():
        p.status = 'absent_certified'
    b.channels[1].status = 'unresolved'
    a = Action('clear', (0., 0.), 1)
    world = World([Source(1, (0., 0.), 1000)])
    assert rollout_result(b, world, a, MovementPolicy(), max_steps=1).steps == 1


def test_process_world_order_is_deterministic():
    b, world = small_task()
    actions = MovementPolicy().proposals(b, 3, True)
    one, multi = RolloutEngine(1), RolloutEngine(2)
    try:
        assert one.evaluate(b, [world, world], actions) == multi.evaluate(b, [world, world], actions)
    finally:
        one.close()
        multi.close()
    assert multi.pool is None


def fake_planner(monkeypatch, score):
    planner = RacingPlanner(worlds=8, max_worlds=64, budget_s=30)
    actions = [Action('measure', (600.-100*i, 0.), i+1) for i in range(3)]
    monkeypatch.setattr(planner.baseline, 'choose', lambda b: actions[0])
    monkeypatch.setattr(planner.baseline, 'proposals', lambda *a: actions)
    monkeypatch.setattr(planner.sampler, 'sample', lambda b, n, deadline: list(range(n)))
    monkeypatch.setattr(planner.engine, 'evaluate',
                        lambda b, worlds, aa, deadline: [[Trial(score(w, a), 20, 10, 1) for a in aa] for w in worlds])
    return planner


def test_equivalent_actions_stop_after_first_batch(monkeypatch):
    planner = fake_planner(monkeypatch, lambda w, a: 100.)
    a = planner.choose(Belief())
    r = planner.records[-1]
    assert a.channel == 1
    assert r['batches_completed'] == [8]
    assert r['refinement_stopped'] == 'practically_resolved'
    assert r['rollout_count'] == 24
    assert r['average_rollout_steps'] == 20


def test_bad_action_eliminated_but_baseline_always_retained(monkeypatch):
    planner = fake_planner(monkeypatch, lambda w, a: 1000. if a.channel == 3 else
                           100. if a.channel == 1 else 100.+(-50 if w % 2 else 50))
    planner.choose(Belief())
    r = planner.records[-1]
    assert r['candidate_sample_counts'] == [64, 64, 8]
    assert all(0 in stage['survivors'] for stage in r['racing_stages'])
    assert r['rollout_count'] == 136  # Rectangular allocation would use 192.


def test_incomplete_later_batch_never_selects_fast_subset(monkeypatch):
    planner = fake_planner(monkeypatch, lambda w, a: 100.+a.channel*(-50 if w % 2 else 50))
    original = planner.engine.evaluate
    calls = []
    def interrupted(*args):
        calls.append(1)
        if len(calls) > 1:
            raise TimeoutError('deliberate partial stage')
        return original(*args)
    monkeypatch.setattr(planner.engine, 'evaluate', interrupted)
    planner.choose(Belief())
    r = planner.records[-1]
    assert r['status'] == 'monte_carlo'
    assert r['worlds_completed'] == 8
    assert r['candidate_sample_counts'] == [8, 8, 8]
    assert r['refinement_stopped'] == 'incomplete_later_batch_discarded'


def test_single_world_cannot_claim_precise_se():
    _, _, keep, regret = race_statistics([[1., 10.]], [0, 1])
    assert keep == [0, 1]
    assert math.isinf(regret)


def test_single_world_diagnostics_are_json_safe(monkeypatch):
    planner = fake_planner(monkeypatch, lambda w, a: float(a.channel))
    planner.worlds = planner.max_worlds = 1
    planner.choose(Belief())
    assert planner.records[-1]['racing_stages'][0]['regret_diagnostic_s'] is None
    json.dumps(planner.records, allow_nan=False)
