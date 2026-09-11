from dataclasses import replace

import numpy as np
import pytest

from bayes_tsp.cached import CachedOpenRoutes, CachedRefinementPolicy, CachedSectorPolicy
from bayes_tsp.coupling import CoupledBelief
from bayes_tsp.open_routes import OpenRoutes
from bayes_tsp.refinement import RefinementConfig, RefinementPolicy
from bayes_tsp.sectors import SectorConfig, SectorPolicy
from bayes_tsp.shared import load


def test_cached_edges_preserve_bitwise_route_lengths_and_start_changes():
    rng = np.random.default_rng(881)
    points = dict(enumerate(rng.uniform(-1800, 1800, size=(16, 2))))
    original = OpenRoutes(points, exact_limit=4)
    cached = CachedOpenRoutes(points, exact_limit=4)
    for _ in range(50):
        start = tuple(rng.uniform(-1800, 1800, size=2))
        order = list(rng.permutation(16))
        for size in (0, 1, 9, 16):
            assert cached.length(start, order[:size]) == original.length(start, order[:size])


@pytest.mark.parametrize('count,limit', [(6, 12), (13, 4)])
def test_cached_routes_preserve_exact_and_heuristic_ties_and_subsets(count, limit):
    points = {k: (float(k % 4)*80, float(k // 4)*70) for k in range(count)}
    original, cached = OpenRoutes(points, limit), CachedOpenRoutes(points, limit)
    for start, remaining in [((-10., 0.), None), ((160., 70.), list(range(1, count, 2))),
                             ((-10., 0.), None), ((1000., -30.), [])]:
        assert cached.plan(start, remaining) == original.plan(start, remaining)


def test_decision_cache_is_discarded_after_exceptions(monkeypatch):
    policy = CachedRefinementPolicy(RefinementConfig(resolution=8))

    def fail(self, belief):
        self._decision_menus['old'] = 'stale'
        raise RuntimeError('planning failed')

    monkeypatch.setattr(RefinementPolicy, 'choose', fail)
    with pytest.raises(RuntimeError, match='planning failed'):
        policy.choose(CoupledBelief())
    assert policy._decision_menus is None
    assert policy._decision_local_costs is None


@pytest.mark.parametrize('scene,noise', [('uniform', 'correlated'), ('boundary', 'extreme')])
@pytest.mark.parametrize('original_class,cached_class,config_class', [
    (RefinementPolicy, CachedRefinementPolicy, RefinementConfig),
    (SectorPolicy, CachedSectorPolicy, SectorConfig)])
def test_cached_full_missions_preserve_every_action_feedback_and_decision(
        scene, noise, original_class, cached_class, config_class):
    simulator = load('simulator')
    config = replace(config_class(resolution=8), stable_routing=False)
    policies = [original_class(config), cached_class(config)]
    traces = []
    for policy in policies:
        world = simulator.generate_world(713, 10, scene, 1000., noise)
        env = simulator.LocalSimulator(world)
        belief = CoupledBelief()
        trace = []
        for step in range(1000):
            if belief.done():
                break
            action = policy.choose(belief)
            response = env.execute(action, str(step))
            belief.apply(action, response, str(step))
            trace.append((action, {k: v for k, v in response.items() if k != 'real_timestamp_ms'}))
        assert belief.done() and world.score()['all_cleared']
        traces.append(trace)
    assert traces[0] == traces[1]
    assert policies[0].records == policies[1].records
    assert policies[0].fallback_calls == policies[1].fallback_calls
    assert getattr(policies[0], 'recovery_filtered', 0) == getattr(policies[1], 'recovery_filtered', 0)
    assert policies[1]._decision_menus is None
    assert policies[1]._decision_local_costs is None
