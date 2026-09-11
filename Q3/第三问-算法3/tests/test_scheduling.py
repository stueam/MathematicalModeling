import itertools

import numpy as np
import pytest
from shapely.geometry import Point, box
from shapely.ops import unary_union

from bayes_tsp.efficient import EfficientConfig, EfficientPolicy
from bayes_tsp.posterior import Posterior
from bayes_tsp.scheduling import CoverTour, scan_route_value
from bayes_tsp.shared import Action, Belief, Observation, load


def posterior(xy):
    xy = np.asarray(xy, dtype=float)
    w = np.full(len(xy), 1/len(xy))
    return Posterior(xy, w, np.full(len(xy), 1000.), np.full(len(xy), 1500.),
                     w @ xy, 1., len(xy), 8)


def test_future_reception_uses_same_radius_not_independent_events():
    p = posterior([(1200, 0)])
    now = (0., 0.)
    # Any R that receives at now also receives at the nearer future position.
    review = scan_route_value(p, .6, now, [(100., 0.)], 100.)
    assert review['reception_now'] == pytest.approx(.6)
    assert review['exclusive_reception'] == 0
    assert review['unique_coverage'] == 0


def test_forecast_same_stop_does_not_create_extra_discovery_value():
    p = posterior([(300, 300), (700, 200)])
    review = scan_route_value(p, .6, (0., 0.), [(0., 0.)], 100.)
    assert review['gross_saving_s'] == pytest.approx(0)
    no_future = scan_route_value(p, .6, (0., 0.), [], 100.)
    assert no_future['gross_saving_s'] > 6


def test_early_discovery_prices_missed_route_insertion():
    p = posterior([(100., 0.)])
    # Now can insert the source on the first edge; after reaching x=900 it
    # requires a return detour. Both points are within the fixed radius.
    review = scan_route_value(p, 1., (0., 0.), [(900., 0.), (1500., 0.)], 100.)
    assert review['exclusive_reception'] == 0
    # Open route: appending after x=1500 costs 1400 m, less than the 1600 m
    # out-and-back detour from x=900. There is no mandatory return to the origin.
    assert review['early_insertion_s'] == pytest.approx(280.)


def test_cover_route_matches_full_subset_and_permutation_enumeration():
    stations = [(-7., 0.), (0., 0.), (7., 0.), (20., 0.)]
    disks = [Point(p).buffer(5) for p in stations]
    region = box(-8, -1, 8, 1)
    scan = [6., 12., 6., 6.]
    start = (-10., 2.)
    model = CoverTour(stations, disks)
    cost, route = model.plan(start, region, scan)
    expected = float('inf')
    for count in range(1, 5):
        for order in itertools.permutations(range(4), count):
            if not region.difference(unary_union([disks[j] for j in order])).is_empty:
                continue
            points = np.array([start]+[stations[j] for j in order])
            total = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()/5+sum(scan[j] for j in order)
            expected = min(expected, total)
    assert cost == pytest.approx(expected)
    assert region.difference(unary_union([disks[j] for j in route])).is_empty
    assert len(route) == len(set(route))


def test_cover_route_keeps_tiny_uncovered_component():
    stations = [(0., 0.), (100., 0.)]
    disks = [Point(p).buffer(10) for p in stations]
    region = box(-1, -1, 1, 1).union(box(100, 0, 100.000001, .000001))
    _, route = CoverTour(stations, disks).plan((0., 0.), region, [6., 6.])
    assert set(route) == {0, 1}


def test_stability_is_routing_only_not_a_clearance_certificate():
    b = Belief()
    b.channels[1].update(Observation(Action('measure', (-1000., 0.), 1), 'direction', 0.))
    policy = EfficientPolicy(EfficientConfig(resolution=8))
    stable, pending, _, _ = policy._tasks(b, {1: posterior([(-100, 0), (100, 0)])})
    assert stable == [] and pending == [1]
    assert b.channels[1].status == 'detected'
    assert not b.done()


def test_target_hysteresis_prevents_small_switch_but_allows_large_gain(monkeypatch):
    b = Belief()
    policy = EfficientPolicy(EfficientConfig(resolution=8, switch_gain_s=20))
    posts = {c: posterior([(100, 0)]) for c in (1, 2)}
    points = {c: (100., 0.) for c in posts}
    policy.active_target = 1
    def menu(_b, c, _post):
        return [(Action('clear', (100., 0.), c), 10. if c == 1 else 1., {})]
    monkeypatch.setattr(policy, '_local_candidates', menu)
    selected, _, _, _ = policy._select_task(b, posts, [1, 2], [], points)
    assert selected['task'] == 'source:1'
    monkeypatch.setattr(policy, '_local_candidates', lambda _b, c, p:
        [(Action('clear', (100., 0.), c), 50. if c == 1 else 1., {})])
    selected, _, _, _ = policy._select_task(b, posts, [1, 2], [], points)
    assert selected['task'] == 'source:2'


def test_nearby_pending_task_competes_with_far_stable_task(monkeypatch):
    b = Belief()
    policy = EfficientPolicy(EfficientConfig(resolution=8))
    posts = {1: posterior([(1000, 0)]), 2: posterior([(100, 0)])}
    points = {c: tuple(p.mean) for c, p in posts.items()}
    monkeypatch.setattr(policy, '_local_candidates', lambda _b, c, p:
        [(Action('measure', points[c], c), np.linalg.norm(points[c])/5+10, {})])
    selected, _, _, _ = policy._select_task(b, posts, [1], [2], points)
    assert selected['task'] == 'source:2'


def test_projected_coverage_is_not_applied_to_actual_belief():
    b = Belief(position=(500., 0.))
    policy = EfficientPolicy(EfficientConfig(resolution=8))
    history = {c: p.region.wkb for c, p in b.channels.items()}
    action = policy._search_route(b, dict.fromkeys(range(1, 21), .65))
    assert action.kind == 'measure'
    assert not b.done() and b.steps == 0
    assert history == {c: p.region.wkb for c, p in b.channels.items()}


@pytest.mark.parametrize('stable_routing', [False, True])
def test_full_improved_run_is_observation_only_without_random_planning(monkeypatch, stable_routing):
    sim = load('simulator')
    world = sim.generate_world(432, 10, 'near', 1000., 'correlated')
    env, b = sim.LocalSimulator(world), Belief()
    policy = EfficientPolicy(EfficientConfig(resolution=8, stable_routing=stable_routing))
    def forbidden(*args, **kwargs):
        raise AssertionError('Random planning is forbidden')
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    monkeypatch.setattr(np.random, 'random', forbidden)
    for index in range(1000):
        if b.done():
            break
        a = policy.choose(b)
        b.apply(a, env.execute(a, str(index)), str(index))
    assert b.done() and world.score()['all_cleared']
    assert all(r['status'] == 'route_decision' or r['status'].endswith('fallback')
               for r in policy.records if r['selected_move_m'] >= 50)
