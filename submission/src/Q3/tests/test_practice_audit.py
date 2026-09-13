import numpy as np
import pytest
from bayes_tsp.repair import repair_segment
from bayes_tsp.policy import Config, Policy
from bayes_tsp.stations import repaired_station
from bayes_tsp.shared import Action, Observation, load
from bayes_tsp.coupling import CoupledBelief


def test_tiny_violation_repaired_without_relaxing_radius():

    def constraint(p):
        return np.array([997.0 - np.linalg.norm(p)])

    point = repair_segment((0.0, 0.0), (997.0 + 1e-07, 0.0), constraint)
    assert point is not None
    assert constraint(point).min() >= 0
    assert point[0] > 996.999


def test_repair_checks_all_constraints_and_rejects_large_violation():
    centers = np.array([[0.0, 0.0], [10.0, 0.0]])

    def constraint(p):
        return 20.0 - np.linalg.norm(centers - p, axis=1)

    point = repair_segment((5.0, 0.0), (20.0 + 1e-07, 0.0), constraint)
    assert point is not None and constraint(point).min() >= 0
    assert repair_segment((5.0, 0.0), (21.0, 0.0), constraint) is None
    assert repair_segment((100.0, 0.0), (20.0, 0.0), constraint) is None


def test_repaired_station_preserves_full_vertex_coverage():
    vertices = np.array([[-100.0, -100.0], [100.0, -100.0], [100.0, 100.0], [-100.0, 100.0]])
    point, detail = repaired_station(vertices, (0.0, 0.0), (1200.0, 0.0), (1200.0, 100.0), 997.0)
    assert detail['accepted']
    assert np.linalg.norm(vertices - point, axis=1).max() <= 997.0
    assert detail['saved_leg_m'] > 0


def test_failed_clear_prefers_new_parallax_measurement():
    b = CoupledBelief(position=(700.0, 0.0))
    b.channels[1].update(Observation(Action('measure', (0.0, 0.0), 1), 'direction', 0.0))
    b.channels[1].update(Observation(Action('clear', b.position, 1), 'no_target_in_range'))
    p = Policy(Config(resolution=8))
    posterior = p.model.posterior(b.channels[1])
    choices = p._local_candidates(b, 1, posterior)
    assert choices and all((a.kind == 'measure' for a, _, _ in choices))
    assert b.channels[1].status == 'detected' and (not b.done())


def test_paper_policy_finishes_without_rng_or_false_absence(monkeypatch):
    sim = load('simulator')
    world = sim.generate_world(534, 10, 'uniform', 1000.0, 'extreme')
    env, b = (sim.LocalSimulator(world), CoupledBelief())
    policy = Policy(Config(resolution=8))
    monkeypatch.setattr(np.random, 'default_rng', lambda *a, **k: pytest.fail('No random planning'))
    for index in range(1000):
        if b.done():
            break
        a = policy.choose(b)
        b.apply(a, env.execute(a, str(index)), str(index))
    assert b.done() and world.score()['all_cleared']
