import math
from itertools import permutations

import numpy as np
import pytest
from shapely.geometry import box

from q3.core import Action, Belief, Observation, distance
from q3.movement import MovementPolicy, shortest_open_route, support
from q3.planner import RolloutPlanner, paired_statistics


def located_belief():
    b = Belief()
    b.channels[1].update(Observation(Action('measure', (0., 0.), 1), 'direction', 0.))
    return b


def test_near_and_forward_options_not_only_backups():
    b = located_belief()
    policy = MovementPolicy()
    points, costs = policy.local_menu(b.channels[1])
    assert len(points) >= 20
    assert np.linalg.norm(points, axis=1).min() < 50
    assert np.any((points[:, 0] > 100) & (np.abs(points[:, 1]) < 100))
    # Original guaranteed-reception backups remain in the menu, not forced.
    assert any(np.linalg.norm(p-[750, 500]) < .01 for p in points)
    assert np.isfinite(costs).all()


def test_information_scan_not_channel_number():
    b = Belief()
    for p in b.channels.values():
        p.status = 'absent_certified'
    b.channels[1].status = b.channels[2].status = 'unresolved'
    b.channels[1].region = box(1250, 0, 1400, 100)
    b.channels[2].region = box(100, 100, 200, 200)
    assert MovementPolicy().scan_action(b, b.position).channel == 2


@pytest.mark.parametrize('step', [0, 1, 2, 3, 4, 17, 101])
def test_every_expensive_departure_reviewed_regardless_of_step(monkeypatch, step):
    b = Belief(steps=step)
    planner = RolloutPlanner(worlds=2, max_worlds=2, interval=10000, budget_s=10)
    far = Action('measure', (600., 0.), 1)
    near = Action('measure', (100., 0.), 1)
    planner.baseline.choose = lambda state: far
    planner.baseline.proposals = lambda *args: [far, near]
    planner.sampler.sample = lambda state, n, deadline: list(range(n))
    calls = []
    def fake_rollout(state, world, action, policy, deadline):
        calls.append(action)
        return 200 if action == far else 100
    monkeypatch.setattr('q3.planner.rollout_cost', fake_rollout)
    assert planner.choose(b) == near
    assert len(calls) == 4
    assert planner.records[-1]['departure_review_required']
    assert planner.records[-1]['status'] == 'monte_carlo'


def test_adaptive_sampling_when_choices_indistinguishable(monkeypatch):
    planner = RolloutPlanner(worlds=8, max_worlds=32, budget_s=10)
    b = Belief()
    a, alternative = Action('measure', (600., 0.), 1), Action('measure', (500., 0.), 1)
    planner.baseline.choose = lambda state: a
    planner.baseline.proposals = lambda *args: [a, alternative]
    planner.sampler.sample = lambda state, n, deadline: list(range(n))
    monkeypatch.setattr('q3.planner.rollout_cost', lambda *args: 100.)
    planner.choose(b)
    assert planner.records[-1]['batches_completed'] == [8, 16, 32]


def test_open_route_matches_exhaustive_search():
    start = (30., 50.)
    points = [(100., 0.), (0., 100.), (-100., 0.), (0., -100.)]
    route = shortest_open_route(start, points)
    def cost(order):
        return distance(start, points[order[0]])+sum(distance(points[i], points[j]) for i, j in zip(order, order[1:]))
    assert cost(route) == pytest.approx(min(cost(order) for order in permutations(range(4))))


def test_support_cache_invalidates_on_new_observation():
    b = located_belief()
    c = b.channels[1]
    first = support(c)
    clone = c.clone()
    clone.update(Observation(Action('measure', (0., 100.), 1), 'direction', 352.))
    second = support(clone)
    assert first is support(c)
    assert second is not first


def test_geometry_progress_does_not_trigger_false_stagnation_lock():
    planner = RolloutPlanner()
    b = Belief()
    action = Action('measure', (600., 0.), 1)
    for i in range(30):
        b.channels[1].region = box(0, 0, 1000-i, 1000)
        assert planner._forced(b, action) is None


def test_pairwise_statistics_detect_clear_winner():
    means, best, se, ambiguous = paired_statistics([[100, 120], [101, 121], [102, 122]])
    assert best == 0 and se == 0 and not ambiguous
