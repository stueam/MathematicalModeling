import itertools
import numpy as np
import pytest
from bayes_tsp.open_routes import OpenRoutes
from bayes_tsp.policy import Config, Policy
from bayes_tsp.shared import Action, Observation
from bayes_tsp.coupling import CoupledBelief as Belief


def test_exact_open_route_and_conditioned_suffix_match_enumeration():
    points = {1: (2, 1), 2: (0, 4), 3: (6, 2), 4: (5, 7), 5: (9, 0)}
    solver = OpenRoutes(points)
    for start, remaining in [((0, 0), list(points)), ((10, 9), [1, 3, 4]), ((6, 2), [1, 2, 4, 5])]:
        expected = float('inf')
        for order in itertools.permutations(remaining):
            xy = np.array([start] + [points[k] for k in order])
            expected = min(expected, np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        length, route = solver.plan(start, remaining)
        assert set(route) == set(remaining) and len(route) == len(remaining)
        assert length == pytest.approx(expected)


def test_open_route_has_no_return_edge_and_handles_empty_tasks():
    assert OpenRoutes({}).plan((0, 0)) == (0.0, [])
    solver = OpenRoutes({1: (100, 0), 2: (200, 0), 3: (300, 0)})
    assert solver.plan((0, 0)) == (300.0, [1, 2, 3])
    assert solver.plan((1000, 0), []) == (0.0, [])


def test_large_route_heuristic_visits_each_node_once():
    points = {k: (float(k % 4) * 50, float(k // 4) * 70) for k in range(16)}
    solver = OpenRoutes(points, exact_limit=4)
    length, route = solver.plan((-10, 0))
    assert not solver.exact
    assert set(route) == set(points) and len(route) == len(points)
    reference = [(-10, 0)] + [points[k] for k in sorted(points)]
    upper = np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()
    assert length <= upper


def test_route_cache_is_endpoint_and_subset_specific_and_returns_copies():
    points = {1: (100.0, 0.0), 2: (200.0, 0.0), 3: (300.0, 0.0)}
    solver = OpenRoutes(points, exact_limit=1)
    cost, route = solver.plan((0.0, 0.0))
    route.clear()
    assert solver.plan((0.0, 0.0)) == (cost, [1, 2, 3])
    assert solver.plan((300.0, 0.0), [1, 2]) == (200.0, [2, 1])
    assert solver.plan((0.0, 0.0), [1]) == (100.0, [1])


def test_joint_forecast_does_not_change_actual_unknown_regions():
    b = Belief()
    b.channels[1].update(Observation(Action('measure', (0.0, 0.0), 1), 'direction', 0.0))
    for channel in range(2, 21):
        b.channels[channel].update(Observation(Action('measure', (0.0, 0.0), channel), 'no_signal'))
    policy = Policy(Config(resolution=8))
    post = policy.model.posterior(b.channels[1])
    original = {c: p.region.wkb for c, p in b.channels.items()}
    plans = policy._plans(b, {1: post}, {1: tuple(post.mean)})
    assert plans and any((k < 0 for plan in plans for k in plan['points']))
    assert original == {c: p.region.wkb for c, p in b.channels.items()}
    assert not b.done()


def test_refined_score_is_recomputed_instead_of_reusing_coarse_metadata():
    from bayes_tsp.posterior import Posterior

    def p(x):
        return Posterior(
            np.array([[float(x), 0.0]]),
            np.array([1.0]),
            np.array([1000.0]),
            np.array([1500.0]),
            np.array([float(x), 0.0]),
            1.0,
            1,
            8,
        )

    policy = Policy(Config(resolution=8))
    b = Belief()
    points = {1: (100.0, 0.0)}
    plan = {'points': points, 'table': OpenRoutes(points), 'local': {1: 5.0}, 'scans': {1: 0.0}, 'mask': 0}
    a = Action('clear', (100.0, 0.0), 1)
    coarse = policy._score(b, a, 1, plan, {1: p(100)})
    fine = policy._score(b, a, 1, plan, {1: p(200)})
    assert fine['score_s'] > coarse['score_s']
