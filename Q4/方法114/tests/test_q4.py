import math
from itertools import permutations

import numpy as np
import pytest
from shapely.geometry import Point, box
from shapely.errors import GEOSException

from q4.core import Belief, Channel, DOMAIN
from q4.coverage import certifies, safe_union, square_stations, triangle_exclusion
from q4.localization import finite_clear
from q4.policy import Config, Policy
from q4.posterior import Model, Posterior, count_marginals, expected_measure, history_atoms
from q4.routing import exact_route, length, open_route
from q4.shared import Action, GeometryError, Observation
from q4.simulator import LocalSimulator, Source, World, generate_world


def test_directional_near_is_gated_but_clear_is_not():
    world = World([Source(1, (0., 0.), 1000., 0.)])
    assert world.feedback(Action('measure', (-2., 0.), 1))[0]['measure_result'] == 'no_signal'
    assert world.feedback(Action('measure', (2., 0.), 1))[0]['measure_result'] == 'near'
    assert world.feedback(Action('measure', (0., 1000.), 1))[0]['measure_result'] == 'direction'
    assert world.feedback(Action('measure', (1000.01, 0.), 1))[0]['measure_result'] == 'no_signal'
    assert world.feedback(Action('clear', (-20., 0.), 1)) == ({'clear_result': 'success'}, 5)


def test_cost_retries_rejection_and_receiver():
    env = LocalSimulator(World([Source(2, (300., 400.), 1000., 0.)]))
    b = Belief()
    a = Action('clear', (300., 400.), 2)
    response = env.execute(a, 'a')
    assert response['virtual_time_s'] == 105 and env.receiver == 1
    assert b.apply(a, response, 'a') and b.receiver == 1
    assert not b.apply(a, response, 'a') and b.steps == 1
    assert env.execute(a, 'a') == response
    with pytest.raises(ValueError):
        env.execute(Action('clear', (301., 400.), 2), 'a')
    m = Action('measure', (300., 400.), 3)
    assert not b.apply(m, {'accepted': False, 'virtual_time_s': 0}, 'rejected')
    assert b.receiver == 1 and b.virtual_time == 105
    response = env.execute(m, 'm')
    b.apply(m, response, 'm')
    assert b.virtual_time == 111 and b.receiver == 3


def test_no_signal_does_not_cut_a_circle_and_history_is_transactional():
    p = Channel()
    obs = Observation(Action('measure', (0., 0.), 1), 'no_signal')
    p.update(obs)
    assert p.region.equals(DOMAIN)
    before = p.clone()
    with pytest.raises(GeometryError):
        p.update(Observation(obs.action, 'direction', 0.))
    assert p.history == before.history and p.region.equals(before.region)


def test_full_grid_certifies_and_inner_ring_does_not():
    assert certifies(DOMAIN, square_stations())
    ring = [(0., 0.)]+[(1200*math.cos(k*math.tau/7), 1200*math.sin(k*math.tau/7)) for k in range(7)]
    assert not certifies(DOMAIN, ring)
    assert not certifies(Point(1800, 0).buffer(.01), ring)


def test_triangle_certificate_is_direction_independent():
    points = ((-400., -200.), (400., -200.), (0., 450.))
    exclusion = triangle_exclusion(tuple(sorted(points)))
    assert exclusion.covers(Point(0, 0))
    for heading in np.linspace(0, 360, 145, endpoint=False):
        world = World([Source(1, (0., 0.), 1000., float(heading))])
        assert any(world.feedback(Action('measure', p, 1))[0]['measure_result'] != 'no_signal' for p in points)


def test_union_failure_preserves_only_valid_exclusions(monkeypatch):
    import q4.coverage as coverage
    from shapely.geometry import Polygon
    valid = box(0, 0, 1, 1)
    invalid = Polygon([(2, 0), (3, 1), (2, 1), (3, 0)])
    def fail(*args, **kwargs):
        raise GEOSException('Injected union error')
    monkeypatch.setattr(coverage.shapely, 'union_all', fail)
    result = safe_union([valid, invalid])
    assert result.equals(valid) and not result.covers(Point(2.5, .5))


def test_unverifiable_candidate_is_rejected(monkeypatch):
    import q4.coverage as coverage
    def fail(*args, **kwargs):
        raise GEOSException('Injected certificate error')
    monkeypatch.setattr(coverage, 'full_exclusion', fail)
    assert not coverage.certifies(DOMAIN, square_stations())


def test_coverage_is_per_channel_and_ten_clears_do_not_finish():
    b = Belief()
    for p in square_stations():
        b.channels[1].update(Observation(Action('measure', p, 1), 'no_signal'))
    assert b.channels[1].status == 'absent_certified'
    assert b.channels[2].status == 'unresolved' and not b.done()
    for c in range(2, 12):
        b.channels[c].status = 'cleared'
    assert not b.done()
    for c in range(12, 18):
        b.channels[c].status = 'cleared'
    assert b.done()


def test_unknown_heading_and_fixed_radius_marginal_against_independent_integral():
    xy = np.array([[0., 0.]])
    history = [Observation(Action('measure', (1100., 0.), 1), 'direction', 180.),
               Observation(Action('measure', (0., 1200.), 1), 'no_signal')]
    raw, *_ = history_atoms(history, xy, .5)
    # A fourth of headings illuminate both stations: R span=100.
    # A fourth illuminate only the positive: R span=400.
    expected = .5*(100/500)+.5*(.25*100/500+.25*400/500)
    assert raw.sum() == pytest.approx(expected, abs=1e-10)
    headings = (np.arange(1440)+.5)*math.tau/1440
    radii = 1000+(np.arange(1000)+.5)*.5
    positive = (np.cos(headings) >= 0)[:, None] & (radii >= 1100)[None]
    negative = (np.sin(headings) < 0)[:, None] | (radii < 1200)[None]
    integral = .5*np.mean((radii >= 1100) & (radii < 1200))+.5*np.mean(positive & negative)
    assert raw.sum() == pytest.approx(integral, abs=1e-10)


def test_prediction_conditions_on_heading_and_near_and_sums_to_one():
    xy = np.array([[0., 0.]])
    history = [Observation(Action('measure', (100., 0.), 1), 'direction', 180.)]
    raw, lo, hi, left, right = history_atoms(history, xy, .999999)
    post = Posterior(xy, raw/raw.sum(), lo, hi, left, right, 1.)
    _, front = post.reception((2., 0.))
    _, back = post.reception((-2., 0.))
    assert front.sum() == pytest.approx(1)
    assert back.sum() < .00001
    _, probabilities = expected_measure(post, (-2., 0.))
    assert sum(probabilities.values()) == pytest.approx(1)
    assert probabilities['near_probability'] < .00001
    assert probabilities['no_signal_probability'] > .99999


def test_fixed_errors_and_worlds_do_not_depend_on_query_order():
    world = generate_world(12)
    point = (300., 0.)
    e = world.error(1, point)
    world.error(3, (150., 40.))
    assert world.error(1, point) == e == world.clone().error(1, point)
    assert generate_world(12).manifest() == world.manifest()
    assert 0 < world.score()['directional_count'] < world.score()['source_count']


def test_finite_grid_clears_with_only_one_bearing_even_from_backside():
    source = Source(1, (1499., 0.), 1500., 180.)
    env, b = LocalSimulator(World([source])), Belief()
    a = Action('measure', (0., 0.), 1)
    b.apply(a, env.execute(a, 'first'), 'first')
    for i in range(400):
        if b.channels[1].status == 'cleared':
            break
        a = finite_clear(b, 1)
        b.apply(a, env.execute(a, str(i)), str(i))
    assert b.channels[1].status == 'cleared'


def test_route_cost_includes_start_but_no_return():
    points = {1: (10., 0.), 2: (20., 0.), 3: (30., 0.)}
    route, cost = open_route((0., 0.), points)
    assert route == [1, 2, 3] and cost == length((0., 0.), route, points) == 30


def test_exact_tsp_matches_exhaustive_orders_and_conditional_suffix():
    points = {i: p for i, p in enumerate([(0., 4.), (6., 1.), (2., 8.), (9., 9.), (-3., 2.), (7., -2.)])}
    start = (1., -1.)
    route, cost = exact_route(start, points)
    oracle = min(length(start, r, points) for r in permutations(points))
    assert cost == pytest.approx(oracle) and set(route) == set(points)
    for first in points:
        rest = {k: p for k, p in points.items() if k != first}
        _, suffix = exact_route(points[first], rest)
        oracle = min(length(points[first], r, rest) for r in permutations(rest))
        assert suffix == pytest.approx(oracle)
        assert exact_route(points[first], points, remaining=rest)[1] == pytest.approx(oracle)


def test_sixteen_city_exact_open_route_and_cache_coordinate_isolation():
    points = {i: (float(i+1), 0.) for i in range(16)}
    route, cost = exact_route((0., 0.), points)
    assert route == list(range(16)) and cost == pytest.approx(16.)
    shifted = {k: (p[0]+100, p[1]) for k, p in points.items()}
    assert exact_route((0., 0.), shifted)[1] == pytest.approx(116.)
    assert exact_route((16., 0.), points)[1] == pytest.approx(15.)


def test_count_conditioning_respects_total_bounds():
    assert count_marginals([.2, .8], 8) == pytest.approx([1, 1])
    assert count_marginals([.2, .8], 16) == pytest.approx([0, 0])


def test_radius_prediction_uses_fixed_radius_and_not_a_new_draw():
    xy = np.array([[0., 0.]])
    history = [Observation(Action('measure', (1400., 0.), 1), 'direction', 180.)]
    raw, lo, hi, left, right = history_atoms(history, xy, .5)
    post = Posterior(xy, raw/raw.sum(), lo, hi, left, right, 1.)
    assert post.reception((1300., 0.))[1].sum() == pytest.approx(1.)
    assert post.reception((1450., 0.))[1].sum() == pytest.approx(.5)


def test_direction_point_repetition_does_not_multiply_likelihood():
    p = Channel()
    o = Observation(Action('measure', (0., 0.), 1), 'direction', 20.)
    p.update(o)
    model = Model(8)
    first = model.posterior(p)
    p.update(o)
    second = model.posterior(p)
    assert len(p.history) == 1 and first is second


def test_move_certificate_is_not_committed_as_actual_coverage():
    b = Belief()
    policy = Policy('mobile')
    assert policy.valid_future(b, policy.points)
    assert all(p.status == 'unresolved' and not p.history for p in b.channels.values())
    assert not b.done()


@pytest.mark.parametrize('mode', ['joint', 'mobile'])
def test_joint_full_task_has_no_rng_in_decisions(monkeypatch, mode):
    world = generate_world(24, 10, 'outward', 1000., 'correlated', 9)
    env, b = LocalSimulator(world), Belief()
    policy = Policy(mode, Config(resolution=8))
    truth = world.manifest()['sources']
    monkeypatch.setattr(np.random, 'default_rng', lambda *a, **kw: pytest.fail('No world sampling in decisions'))
    for i in range(6000):
        if b.done():
            break
        a = policy.choose(b)
        b.apply(a, env.execute(a, str(i)), str(i))
        for s in truth:
            p = b.channels[s['channel']]
            assert p.status != 'absent_certified'
            assert p.region.distance(Point(s['position'])) < 1e-7
    assert b.done() and world.score()['all_cleared']


@pytest.mark.parametrize('scenario', ['uniform', 'outward', 'backside'])
def test_baseline_full_task_keeps_truth_in_public_region(scenario):
    world = generate_world(23, 10, scenario, 1000., 'extreme', 9)
    env, b = LocalSimulator(world), Belief()
    policy = Policy('baseline', Config(local_limit=4))
    truth = world.manifest()['sources']  # Test oracle only, not passed to Policy.
    for i in range(6000):
        if b.done():
            break
        a = policy.choose(b)
        b.apply(a, env.execute(a, str(i)), str(i))
        for s in truth:
            p = b.channels[s['channel']]
            assert p.status != 'absent_certified'
            assert p.region.distance(Point(s['position'])) < 1e-7
    assert b.done() and world.score()['all_cleared']
