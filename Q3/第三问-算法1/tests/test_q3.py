import math
import time
from types import SimpleNamespace

import numpy as np
import pytest
import requests
from shapely.geometry import Point

from q3.core import Action, Belief, Channel, Observation, GeometryError, disk, point_key
from q3.simulator import World, Source, LocalSimulator, generate_world
from q3.policy import Baseline
from q3.sampling import WorldSampler, likelihood, conditional_existence
from q3.planner import RolloutPlanner, rollout_cost
from q3.client import HttpClient, ProtocolError
from run import local_run


def response(kind, value, t=5, bearing=None):
    r = {'accepted': True, 'virtual_time_s': t,
         'measure_result' if kind == 'measure' else 'clear_result': value}
    if bearing is not None:
        r['svd_deg'] = bearing
    return r


def test_protocol_time_and_clear_channel():
    env, b = LocalSimulator(World([])), Belief()
    actions = [Action('measure', (300., 400.), 1), Action('measure', (300., 400.), 2),
               Action('clear', (300., 0.), 3), Action('measure', (300., 0.), 2)]
    for i, (a, expected) in enumerate(zip(actions, [105, 111, 194, 199])):
        r = env.execute(a, str(i))
        b.apply(a, r, str(i))
        assert env.virtual_time == b.virtual_time == expected
    assert b.receiver == env.receiver == 2
    assert sum(env.costs.values()) == 199


def test_idempotent_success_not_double_counted():
    env = LocalSimulator(World([Source(3, (2., 0.), 1000)]))
    b = Belief()
    a = Action('clear', (0., 0.), 3)
    first = env.execute(a, 'one')
    assert b.apply(a, first, 'one')
    assert not b.apply(a, env.execute(a, 'one'), 'one')
    assert b.cleared == {3} and b.virtual_time == 5 and b.receiver == 1
    assert env.virtual_time == 5
    with pytest.raises(ValueError):
        env.execute(Action('clear', (1., 0.), 3), 'one')


def test_rejected_response_preserves_state():
    b = Belief(virtual_time=123)
    assert not b.apply(Action('measure', (10., 20.), 2), {'accepted': False, 'virtual_time_s': 0}, 'bad')
    assert (b.position, b.receiver, b.virtual_time, b.steps) == ((0., 0.), 1, 123, 0)


@pytest.mark.parametrize('position', [(math.nan, 0), (math.inf, 0), (2000001, 0)])
def test_invalid_coordinates(position):
    with pytest.raises(ValueError):
        Action('measure', position, 1)


@pytest.mark.parametrize('mode', ['iid', 'extreme', 'correlated'])
def test_fixed_error_and_candidate_order(mode):
    w = World([Source(1, (500., 300.), 1500)], seed=42, error_mode=mode)
    a, c = Action('measure', (0., 0.), 1), Action('measure', (50., 0.), 1)
    first, _ = w.feedback(a)
    w.feedback(c)
    assert w.feedback(a)[0] == first
    other = w.clone()
    other.feedback(c)
    assert other.feedback(a)[0] == first
    assert abs(w.error(1, (12., 15.))) <= 1


def test_range_near_and_clear_boundaries():
    w = World([Source(1, (0., 0.), 1000)])
    assert w.feedback(Action('measure', (1000., 0.), 1))[0]['measure_result'] == 'direction'
    assert w.feedback(Action('measure', (1000.01, 0.), 1))[0]['measure_result'] == 'no_signal'
    assert w.feedback(Action('measure', (5., 0.), 1))[0]['measure_result'] == 'near'
    assert w.feedback(Action('clear', (20.01, 0.), 1))[0]['clear_result'] == 'no_target_in_range'
    assert w.feedback(Action('clear', (20., 0.), 1))[0]['clear_result'] == 'success'
    assert w.feedback(Action('measure', (0., 0.), 1))[0]['measure_result'] == 'no_signal'


def test_outer_and_inner_disks():
    for radius in (5, 20, 1000, 1500, 1800):
        for theta in np.linspace(0, 2*math.pi, 101):
            p = Point(radius*math.cos(theta), radius*math.sin(theta))
            assert disk((0, 0), radius, True).covers(p)
            assert not disk((0, 0), radius).contains(p)


@pytest.mark.parametrize('bearing', [0., 359.99, 90., 180.])
def test_rounding_extreme_true_source_retained(bearing):
    c = Channel()
    a = Action('measure', (0., 0.), 1)
    c.update(Observation(a, 'direction', bearing))
    for error in (-1.00499, 1.00499):
        angle = math.radians(bearing+error)
        assert c.region.covers(Point(1500*math.cos(angle), 1500*math.sin(angle)))


def test_empty_detected_is_error_not_absence():
    c = Channel()
    a = Action('measure', (0., 0.), 1)
    c.update(Observation(a, 'near'))
    with pytest.raises(GeometryError):
        c.update(Observation(a, 'no_signal'))
    assert c.status == 'detected' and not c.region.is_empty


@pytest.mark.parametrize('ring', [1200., 1500.])
def test_seven_stations_certify_absence(ring):
    c = Channel()
    for s in Baseline(ring).stations:
        c.update(Observation(Action('measure', s, 1), 'no_signal'))
    assert c.status == 'absent_certified'


def test_completion_uses_evidence_and_upper_count():
    b = Belief()
    for c in range(1, 11):
        b.channels[c].status = 'cleared'
    assert not b.done()
    for c in range(11, 17):
        b.channels[c].status = 'cleared'
    assert b.done()


def test_first_two_candidates_guaranteed_reception():
    b = Belief()
    b.apply(Action('measure', (0., 0.), 1), response('measure', 'direction', bearing=359.99), 'one')
    candidates = Baseline().measurement_candidates(b, 1)
    for r in np.linspace(5, 1500, 80):
        for e in (-1.005, 1.005):
            angle = math.radians(359.99+e)
            g = (r*math.cos(angle), r*math.sin(angle))
            for a in candidates:
                assert math.dist(g, a.position) < 1000


def test_fallback_includes_boundary_sliver_and_eventually_clears():
    true = (20.001, 79.999)
    b = Belief()
    b.apply(Action('measure', (0., 0.), 1), response('measure', 'direction', bearing=76), 'first')
    c = b.channels[1]
    # A thin external region; the closest cell center need not lie in P.
    from shapely.geometry import box
    c.region = box(19.999, 19.999, 20.002, 80.)
    c._summary = None
    env = LocalSimulator(World([Source(1, true, 1000)]), virtual_time=b.virtual_time)
    policy = Baseline()
    for i in range(10):
        a = policy.grid_clear(b, 1)
        b.apply(a, env.execute(a, str(i)), str(i))
        if b.channels[1].status == 'cleared':
            break
    assert b.channels[1].status == 'cleared'
    assert env.costs['clear_failure_s'] >= 3


def test_same_visible_state_same_policy_independent_of_environment():
    b = Belief()
    # These worlds differ in truth but expose the same empty observation history.
    w1, w2 = generate_world(1), generate_world(2)
    p1, p2 = RolloutPlanner(seed=9), RolloutPlanner(seed=9)
    assert w1 is not w2
    assert p1.choose(b.clone()) == p2.choose(b.clone())


def test_failed_completion_is_not_reported_as_success():
    args = SimpleNamespace(policy='baseline', n=16, scenario='uniform', radius=1000,
                           error_mode='iid', ring=1500, real_limit=10, max_steps=1)
    summary, _, _ = local_run(1, args)
    assert not summary['all_cleared'] and not summary['certified_complete']
    assert summary['stop_reason'] == 'action_limit'


def test_radius_interval_and_rounding_likelihood():
    c = Channel()
    c.update(Observation(Action('measure', (0., 0.), 1), 'direction', 0.))
    c.update(Observation(Action('measure', (-100., 0.), 1), 'no_signal'))
    w, lo, hi = likelihood(c, np.array([[1100., 0.], [1400., 0.], [800., 0.]]))
    assert np.allclose(lo, [1100, 1400, 1000])
    assert np.allclose(hi, [1200, 1500, 900])
    assert np.allclose(w, [.2, .2, 0])
    edge = Channel()
    edge.update(Observation(Action('measure', (0., 0.), 1), 'direction', 0.))
    xy = np.array([[500*math.cos(math.radians(x)), 500*math.sin(math.radians(x))] for x in (0, 1, 1.006)])
    weights, _, _ = likelihood(edge, xy)
    assert np.allclose(weights, [1., .5, 0.], atol=1e-10)


def test_count_dp_not_independent_after_conditioning():
    rng = np.random.default_rng(2)
    for known in (0, 10, 15, 16):
        for _ in range(50):
            z = conditional_existence([.3]*(20-known), known, rng)
            assert 10 <= known+sum(z) <= 16
    assert sum(conditional_existence([.8]*4, 16, rng)) == 0


def test_world_samples_reproduce_all_current_measurements():
    env, b = LocalSimulator(generate_world(8)), Belief()
    policy = Baseline()
    for i in range(25):
        a = policy.choose(b)
        b.apply(a, env.execute(a, str(i)), str(i))
    worlds = WorldSampler(seed=4).sample(b, 12)
    for world in worlds:
        assert 10 <= world.score()['source_count'] <= 16
        for c, state in b.channels.items():
            if state.status == 'cleared':
                continue
            for obs in state.history:
                if obs.action.kind != 'measure':
                    continue
                actual, _ = world.feedback(obs.action)
                assert actual['measure_result'] == obs.result
                if obs.result == 'direction':
                    assert actual['svd_deg'] == obs.bearing
                    # Do not allow the history cache to hide an impossible error.
                    source = world._sources[c]
                    phi = math.degrees(math.atan2(source.position[1]-obs.action.position[1],
                                                 source.position[0]-obs.action.position[0]))
                    diff = (obs.bearing-phi+180) % 360-180
                    assert abs(diff) <= 1.00500001


def test_clone_and_rollout_do_not_mutate_real_observer():
    b = Belief()
    world = generate_world(1)
    baseline = Baseline()
    action = baseline.choose(b)
    value = rollout_cost(b, world, action, baseline, deadline=time.monotonic()+5)
    assert value > 0
    assert b.steps == 0 and b.virtual_time == 0 and not b.channels[1].history
    assert world.score()['cleared_count'] == 0


def test_deadline_fallback_not_partial_scoring():
    b = Belief()
    env, base = LocalSimulator(generate_world(8)), Baseline()
    for i in range(20):
        a = base.choose(b)
        b.apply(a, env.execute(a, str(i)), str(i))
    planner = RolloutPlanner(budget_s=1e-12, interval=1)
    action = planner.choose(b)
    assert action == planner.baseline.choose(b)
    assert planner.records[-1]['status'] == 'TimeoutError'
    assert 'mean_cost_s' not in planner.records[-1]


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = iter(responses), []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        r = next(self.responses)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(status_code=r[0], json=lambda: r[1])


def test_http_retry_exact_same_payload():
    session = FakeSession([requests.Timeout(), (200, response('measure', 'no_signal'))])
    client = HttpClient('test-team', session=session)
    _, r = client.execute(Action('measure', (0., 0.), 1))
    assert r['accepted']
    assert session.calls[0][1]['data'] == session.calls[1][1]['data']


@pytest.mark.parametrize('status,accepted', [(200, False), (409, False), (500, False)])
def test_http_rejection_is_not_success(status, accepted):
    client = HttpClient('test-team', session=FakeSession([(status, {'accepted': accepted, 'virtual_time_s': 0})]))
    with pytest.raises(ProtocolError):
        client.execute(Action('clear', (0., 0.), 1))


@pytest.mark.parametrize('scenario,error_mode,n,radius', [
    ('uniform', 'iid', 10, 1000), ('boundary', 'extreme', 16, 1000),
    ('cluster', 'correlated', 16, 1500), ('near', 'iid', 10, 1500)])
def test_baseline_end_to_end(scenario, error_mode, n, radius):
    args = SimpleNamespace(policy='baseline', n=n, scenario=scenario, radius=radius,
                           error_mode=error_mode, ring=1500, real_limit=10, max_steps=4000)
    summary, _, _ = local_run(5, args)
    assert summary['error'] is None
    assert summary['all_cleared'] and summary['certified_complete']
    assert sum(summary['costs'].values()) == pytest.approx(summary['virtual_time_s'])


def test_mc_really_compares_and_finishes():
    args = SimpleNamespace(policy='mc', n=10, scenario='uniform', radius=1000,
                           error_mode='iid', ring=1500, real_limit=20, max_steps=4000,
                           planner_seed=2026, worlds=2, candidates=4, budget=1,
                           interval=8, speculative_clear=False, max_worlds=2)
    summary, _, records = local_run(3, args)
    assert summary['error'] is None
    assert summary['all_cleared'] and summary['certified_complete']
    assert summary['mc_decisions'] > 0
    assert all(r['worlds_completed'] == 2 for r in records if r['status'] == 'monte_carlo')
