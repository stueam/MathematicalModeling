import pytest

from q4.compact import CoverPolicy, triangular_stations
from q4.core import Belief, DOMAIN
from q4.coverage import certifies, square_stations
from q4.simulator import LocalSimulator, generate_world


def test_compact_grid_certifies_full_domain_and_keeps_square_fallback():
    points = triangular_stations()
    assert len(points) == 31 and certifies(DOMAIN, points)
    policy = CoverPolicy()
    assert len(policy.points) == 31 and set(policy.fixed.values()) == set(square_stations())
    b = Belief()
    policy.choose(b)
    assert not b.done() and all(not c.history for c in b.channels.values())
    assert policy.valid_future(b, policy.points)


def test_rejected_pruning_cannot_change_plan_or_belief(monkeypatch):
    policy, b = CoverPolicy(), Belief()
    before = dict(policy.points)
    monkeypatch.setattr(policy, 'valid_future', lambda *a: False)
    policy.prune_stations(b)
    assert policy.points == before and not b.done()
    assert all(not c.history for c in b.channels.values())
    assert policy.counters['cover_pruned_stations'] == 0


@pytest.mark.parametrize('scenario', ['outward', 'backside'])
@pytest.mark.parametrize('mode', ['compact', 'adaptive'])
def test_compact_complete_hard_world_with_truth_retained(scenario, mode):
    world = generate_world(123, 10, scenario, 1000., 'extreme', 9)
    truth = world.manifest()['sources']
    env, b, policy = LocalSimulator(world), Belief(), CoverPolicy(mode)
    from shapely.geometry import Point
    for i in range(1800):
        if b.done():
            break
        a = policy.choose(b)
        b.apply(a, env.execute(a, str(i)), str(i))
        for source in truth:
            channel = b.channels[source['channel']]
            assert channel.status != 'absent_certified'
            assert channel.region.distance(Point(source['position'])) < 1e-7
    assert b.done() and world.score()['all_cleared']
    assert all(r['long_move_review'] for r in policy.records if r['move_m'] >= 200)


def test_compact_fallback_restores_finite_grid():
    from q4.policy import Config
    policy, b = CoverPolicy(config=Config(completion_after=1)), Belief()
    b.steps = 1
    policy.choose(b)
    assert policy.completion_mode and policy.points == policy.fixed


def test_exhausted_plan_does_not_imply_absence():
    policy, b = CoverPolicy(), Belief()
    policy.points.clear()
    action = policy.choose(b)
    assert action.kind == 'measure' and not b.done()
    assert policy.completion_mode and policy.points == policy.fixed
    assert policy.counters['cover_exhaustion_fallbacks'] == 1


@pytest.mark.parametrize('directional_count', [0, 10])
def test_explicit_pure_source_types_are_supported(directional_count):
    world = generate_world(123, 10, 'outward', 1000., 'extreme', directional_count)
    assert world.score()['directional_count'] == directional_count
