"""Analytic Bayes checks, route invariants and observation-only completion."""

import itertools
import math
import numpy as np
import pytest
from bayes_tsp.policy import Policy, Config
from bayes_tsp.posterior import (
    BayesModel,
    Posterior,
    QuadratureError,
    count_conditioned_marginals,
    expected_after_measure,
    radius_likelihood,
)
from bayes_tsp.routing import DynamicTour, route_length
from bayes_tsp.shared import Action, Observation, core, load
from bayes_tsp.coupling import CoupledBelief as Belief


def obs(position, result, bearing=None, kind='measure'):
    return Observation(Action(kind, position, 1), result, bearing)


def test_fixed_radius_joint_bounds_and_duplicate_observation():
    history = [obs((0.0, 0.0), 'direction', 0.0), obs((0.0, 1000.0), 'no_signal')]
    xy = np.array([[1200.0, 0.0], [800.0, 0.0]])
    mass, lo, hi = radius_likelihood(history, xy)
    assert lo == pytest.approx([1200, 1000])
    assert hi == pytest.approx([1500, math.hypot(800, 1000)])
    assert mass == pytest.approx((hi - lo) / 500)
    assert radius_likelihood(history + [history[0]], xy)[0] == pytest.approx(mass)


def test_rounding_edge_is_kept_by_both_layers():
    angle = math.radians(0.996)
    xy = np.array([[1000 * math.cos(angle), 1000 * math.sin(angle)]])
    reading = obs((0.0, 0.0), 'direction', 2.0)
    mass, _, _ = radius_likelihood([reading], xy)
    assert mass[0] > 0
    channel = core.Channel()
    channel.update(reading)
    from shapely.geometry import Point

    assert channel.region.covers(Point(xy[0]))


def test_count_conditioning_matches_enumeration():
    q = np.array([0.2, 0.6, 0.8, 0.3])
    known = 13
    denominator = 0.0
    expected = np.zeros(4)
    for bits in itertools.product((0, 1), repeat=4):
        if 10 <= known + sum(bits) <= 16:
            mass = np.prod([p if z else 1 - p for p, z in zip(q, bits)])
            denominator += mass
            expected += mass * np.array(bits)
    assert count_conditioned_marginals(q, known) == pytest.approx(expected / denominator)
    assert count_conditioned_marginals([0.1, 0.7], 8) == pytest.approx([1.0, 1.0])


def test_feedback_prediction_accounts_for_near_and_missed_reception():
    xy = np.array([[4.0, 0.0], [500.0, 0.0], [1200.0, 0.0], [1600.0, 0.0]])
    weights = np.full(4, 0.25)
    posterior = Posterior(xy, weights, np.full(4, 1000.0), np.full(4, 1500.0), weights @ xy, 1.0, 4, 8)
    cost, probabilities = expected_after_measure(posterior, (0.0, 0.0))
    assert cost > 0
    assert probabilities['near'] == pytest.approx(0.25)
    assert probabilities['no_signal'] == pytest.approx(0.35)
    assert probabilities['direction'] == pytest.approx(0.4)
    for width in (0.5, 1.0, 2.0):
        _, p = expected_after_measure(posterior, (0.0, 0.0), width)
        assert sum((p[k] for k in ('near', 'no_signal', 'direction'))) == pytest.approx(1)


def test_clear_point_search_handles_separated_modes():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [100.0, 0.0]])
    weights = np.array([0.3, 0.3, 0.4])
    p = Posterior(xy, weights, np.full(3, 1000.0), np.full(3, 1500.0), weights @ xy, 1.0, 3, 8)
    assert p.clear_probability(p.mean) == 0
    assert p.clear_probability(p.best_clear_point) == pytest.approx(0.6)


def test_cell_reintegration_normalizes_narrow_intersection():
    channel = core.Channel()
    channel.update(obs((0.0, 0.0), 'direction', 0.0))
    channel.update(obs((900.0, 100.0), 'direction', 315.0))
    p = BayesModel(16).posterior(channel)
    assert len(p.xy) > 0
    assert p.weights.sum() == pytest.approx(1)
    assert np.all(p.hi > p.lo)
    assert np.all(radius_likelihood(channel.history, p.xy)[0] > 0)


def test_first_bearing_radial_posterior_matches_analytic_integral():
    channel = core.Channel()
    channel.update(obs((0.0, 0.0), 'direction', 0.0))

    def power_integral(power):
        first = (1000 ** (power + 1) - 5 ** (power + 1)) / (power + 1)
        second = (
            1500 * (1500 ** (power + 1) - 1000 ** (power + 1)) / (power + 1)
            - (1500 ** (power + 2) - 1000 ** (power + 2)) / (power + 2)
        ) / 500
        return first + second

    expected = power_integral(2) / power_integral(1)
    for resolution in (16, 32):
        posterior = BayesModel(resolution).posterior(channel)
        mean_radius = float(posterior.weights @ np.linalg.norm(posterior.xy, axis=1))
        assert mean_radius == pytest.approx(expected, rel=0.003)


def test_dynamic_open_tsp_removes_and_inserts_tasks_without_return_edge():
    tour = DynamicTour()
    points = {'a': (100, 0), 'b': (200, 0), 'c': (300, 0)}
    route = tour.update((0, 0), points)
    assert route_length((0, 0), route, points) == pytest.approx(300)
    changed = {'b': (200, 0), 'c': (300, 0), 'new': (150, 0)}
    route = tour.update((100, 0), changed)
    assert route == ['new', 'b', 'c']
    assert route_length((100, 0), route, changed) == pytest.approx(200)


def test_policy_does_not_draw_random_numbers(monkeypatch):

    def forbidden(*args, **kwargs):
        raise AssertionError('Random sampling inside deterministic policy')

    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    monkeypatch.setattr(np.random, 'random', forbidden)
    b = Belief()
    policy = Policy(Config(resolution=8))
    action = policy.choose(b)
    assert action.kind == 'measure' and action.position == (0.0, 0.0)
    assert action == Policy(Config(resolution=8)).choose(b)


def test_probability_failure_cannot_certify_absence(monkeypatch):
    b = Belief()
    policy = Policy(Config(resolution=8))

    def failed(*args, **kwargs):
        raise QuadratureError('deliberate empty integration')

    monkeypatch.setattr(policy.model, 'existence', failed)
    action = policy.choose(b)
    assert action.kind == 'measure'
    assert not b.done()
    assert all((p.status == 'unresolved' for p in b.channels.values()))
    assert policy.records[-1]['status'] == 'quadrature_fallback'


def test_decision_caches_are_discarded_on_success_and_exception(monkeypatch):
    policy = Policy()
    belief = Belief()
    policy.choose(belief)
    assert policy._decision_menus is None and policy._decision_local_costs is None

    def fail(_belief):
        policy._decision_menus['partial'] = 'must not survive new feedback'
        raise RuntimeError('interrupted numerical calculation')

    with monkeypatch.context() as patch:
        patch.setattr(policy, '_choose_observed', fail)
        with pytest.raises(RuntimeError, match='interrupted'):
            policy.choose(belief)
    assert policy._decision_menus is None and policy._decision_local_costs is None
    assert policy.choose(belief) == Policy().choose(belief)


def test_ten_clears_do_not_finish_and_failed_clear_keeps_target():
    b = Belief()
    for c in range(1, 11):
        b.channels[c].status = 'cleared'
    assert not b.done()
    c = b.channels[11]
    c.update(obs((0.0, 0.0), 'direction', 0.0))
    c.update(obs((500.0, 0.0), 'no_target_in_range', kind='clear'))
    assert c.status == 'detected'
    assert not c.region.is_empty


def test_finite_completion_handles_boundary_extreme_errors():
    sim = load('simulator')
    world = sim.generate_world(991, 10, 'boundary', 1000.0, 'extreme')
    env, b = (sim.LocalSimulator(world), Belief())
    policy = Policy(Config(resolution=8, completion_after=1))
    for i in range(1500):
        if b.done():
            break
        action = policy.choose(b)
        b.apply(action, env.execute(action, str(i)), str(i))
    assert b.done()
    assert world.score()['all_cleared']


def test_joint_batch_handles_near_then_resumes_at_same_stop():
    b = Belief(position=(100.0, 0.0))
    for c in (1, 2):
        b.channels[c].update(Observation(Action('measure', (0.0, 0.0), c), 'direction', 0.0))
    b.channels[1].update(Observation(Action('measure', b.position, 1), 'near'))
    policy = Policy(Config(resolution=8))
    policy.batch_position, policy.batch_channels = (b.position, [1, 2])
    first = policy.choose(b)
    assert first == Action('clear', b.position, 1)
    b.apply(first, {'accepted': True, 'clear_result': 'success', 'virtual_time_s': 5}, 'clear')
    second = policy.choose(b)
    assert second == Action('measure', b.position, 2)
    assert policy.records[-1]['status'] == 'joint_batch_measure'
