import numpy as np
import pytest
from bayes_tsp.policy import Config, Policy
from bayes_tsp.posterior import Posterior
from bayes_tsp.scheduling import scan_route_value
from bayes_tsp.coupling import CoupledBelief as Belief
from bayes_tsp.shared import Action, Observation


def posterior(xy):
    xy = np.asarray(xy, dtype=float)
    w = np.full(len(xy), 1 / len(xy))
    return Posterior(xy, w, np.full(len(xy), 1000.0), np.full(len(xy), 1500.0), w @ xy, 1.0, len(xy), 8)


def test_future_reception_uses_same_radius_not_independent_events():
    p = posterior([(1200, 0)])
    now = (0.0, 0.0)
    review = scan_route_value(p, 0.6, now, [(100.0, 0.0)], 100.0)
    assert review['reception_now'] == pytest.approx(0.6)
    assert review['exclusive_reception'] == 0
    assert review['unique_coverage'] == 0


def test_forecast_same_stop_does_not_create_extra_discovery_value():
    p = posterior([(300, 300), (700, 200)])
    review = scan_route_value(p, 0.6, (0.0, 0.0), [(0.0, 0.0)], 100.0)
    assert review['gross_saving_s'] == pytest.approx(0)
    no_future = scan_route_value(p, 0.6, (0.0, 0.0), [], 100.0)
    assert no_future['gross_saving_s'] > 6


def test_early_discovery_prices_missed_route_insertion():
    p = posterior([(100.0, 0.0)])
    review = scan_route_value(p, 1.0, (0.0, 0.0), [(900.0, 0.0), (1500.0, 0.0)], 100.0)
    assert review['exclusive_reception'] == 0
    assert review['early_insertion_s'] == pytest.approx(280.0)


def test_projected_coverage_is_not_applied_to_actual_belief():
    b = Belief(position=(500.0, 0.0))
    for channel in range(1, 21):
        b.channels[channel].update(Observation(Action('measure', (0.0, 0.0), channel), 'no_signal'))
    policy = Policy(Config(resolution=8))
    history = {c: p.region.wkb for c, p in b.channels.items()}
    action = policy._search_route(b)
    assert action.kind == 'measure'
    assert not b.done() and b.steps == 0
    assert history == {c: p.region.wkb for c, p in b.channels.items()}
