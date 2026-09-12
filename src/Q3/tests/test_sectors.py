from types import SimpleNamespace

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, box

from bayes_tsp.coupling import CoupledBelief, CoupledChannel, receive_halfplane
from bayes_tsp.sectors import SectorConfig, SectorPolicy, adjust_station, covers_vertices
from bayes_tsp.shared import Action, Belief, GeometryError, Observation, core, disk, load


def reading(point, result, bearing=None, channel=1):
    return Observation(Action('measure', tuple(point), channel), result, bearing)


def census(belief_type=Belief, known=False):
    b = belief_type()
    for c in range(1, 21):
        result = 'direction' if known and c == 1 else 'no_signal'
        b.channels[c].update(reading((0., 0.), result, 0. if result == 'direction' else None, c))
    return b


@pytest.mark.parametrize('reverse', [False, True])
def test_fixed_radius_cut_rejects_a_position_that_disk_exclusion_keeps(reverse):
    records = [reading((0., 0.), 'direction', 0.), reading((1000., 1000.), 'no_signal')]
    if reverse:
        records.reverse()
    old, new = core.Channel(), CoupledChannel()
    for obs in records:
        old.update(obs)
        new.update(obs)
    assert old.region.covers(Point(1400, 0))
    assert not new.region.covers(Point(1400, 0))
    assert new.region.covers(Point(500, 0))  # compatible with e.g. fixed R=1050
    assert new.coupling_cuts > 0


def test_halfplane_retains_equality_with_margin_and_is_translation_stable():
    for shift in (np.array([0., 0.]), np.array([1e6, -1e6])):
        region = box(shift[0]-1, shift[1]-1, shift[0]+1500, shift[1]+1)
        clipped = receive_halfplane(region, shift, shift+1000)
        assert clipped.covers(Point(shift+np.array([1000., 0.])))
        assert not clipped.covers(Point(shift+np.array([1400., 0.])))


def test_coupling_is_transactional_and_respects_rejections_and_same_id():
    b = CoupledBelief()
    a = Action('measure', (0., 0.), 1)
    r = {'accepted': True, 'measure_result': 'direction', 'svd_deg': 0., 'virtual_time_s': 5.}
    b.apply(a, r, 'one')
    before = b.channels[1].region.wkb
    assert not b.apply(a, r, 'one')
    assert not b.apply(a, {'accepted': False, 'virtual_time_s': 0}, 'rejected')
    with pytest.raises(GeometryError):
        b.apply(a, {'accepted': True, 'measure_result': 'no_signal', 'virtual_time_s': 10.}, 'bad')
    assert b.channels[1].region.wkb == before
    assert len(b.channels[1].history) == 1 and b.steps == 1 and b.virtual_time == 5.
    assert 'bad' not in b.applied and b.channels[1].status == 'detected'


@pytest.mark.parametrize('error_mode', ['iid', 'extreme', 'correlated'])
def test_coupled_regions_keep_true_positions_under_legal_feedback(error_mode):
    sim = load('simulator')
    for target in ((500., 0.), (1499., 0.), (-700., 800.)):
        for radius in (1000., 1500.):
            world = sim.World([sim.Source(1, target, radius)], seed=47, error_mode=error_mode)
            env, b = sim.LocalSimulator(world), CoupledBelief()
            for i, station in enumerate(((0., 0.), (1000., 1000.), (1500., -800.),
                                         (-1000., 1000.), (500., 500.), (1600., 100.))):
                a = Action('measure', station, 1)
                b.apply(a, env.execute(a, str(i)), str(i))
                assert b.channels[1].region.covers(Point(target))


def test_task_cover_preserves_sector_seams_and_is_removed_by_real_no_signal():
    b = census()
    policy = SectorPolicy(SectorConfig(resolution=8, movable_stations=False))
    tasks = policy._coverage_tasks(b)
    whole = shapely.union_all([task['region'] for task in tasks.values()])
    assert b.channels[1].region.difference(whole).is_empty
    for key, task in tasks.items():
        pos = policy._initial_center(key, task)
        assert covers_vertices(task['vertices'], pos)
        assert task['region'].difference(disk(pos, 1000)).is_empty


def test_station_adjustment_shortens_both_legs_and_verifies_feasibility(monkeypatch):
    vertices = np.array([[-100., -100.], [100., -100.], [100., 100.], [-100., 100.]])
    original = (0., 0.)
    pos, record = adjust_station(vertices, original, (500., 100.), (500., -100.))
    assert record['accepted'] and record['saved_leg_m'] > 800
    assert covers_vertices(vertices, pos)
    import bayes_tsp.sectors as module
    monkeypatch.setattr(module, 'minimize', lambda *a, **kw:
        SimpleNamespace(success=False, x=np.array([1e9, 1e9]), nit=1))
    pos, record = adjust_station(vertices, original, (500., 100.), (500., -100.))
    assert pos == original and not record['accepted']


def test_task_scan_requires_entire_region_not_partial_intersection():
    b = census(known=True)
    b.position = (200., 0.)
    policy = SectorPolicy(SectorConfig(resolution=8))
    channels, tasks = policy._clear_service(b)
    assert channels == [] and tasks == []
    b.position = (1200., 0.)
    channels, tasks = policy._clear_service(b)
    assert tasks and channels == list(range(2, 21))
    assert all(b.channels[c].status == 'unresolved' for c in channels)
    assert not b.done()


def test_clear_service_executes_actual_channel_measurements():
    b = census(known=True)
    b.position = (1200., 0.)
    b.channels[1].update(reading(b.position, 'near'))
    policy = SectorPolicy(SectorConfig(resolution=8))
    a = policy.choose(b)
    assert a.kind == 'clear'
    b.apply(a, {'accepted': True, 'clear_result': 'success', 'virtual_time_s': 5.}, 'clear')
    a = policy.choose(b)
    assert a.kind == 'measure' and a.channel != 1 and a.position == b.position
    assert policy.records[-1]['status'] == 'task_clear_search'
    assert not b.done()


def test_planning_assignments_never_certify_absence():
    b = census(known=True)
    b.channels[1].region = box(1199., -1., 1201., 1.)
    b.channels[1]._summary = None
    policy = SectorPolicy(SectorConfig(resolution=8))
    post = policy.model.posterior(b.channels[1])
    before = {c: p.region.wkb for c, p in b.channels.items()}
    policy._plans(b, {1: post}, {1: (1200., 0.)})
    assert policy._last_task_log['sector_assignments']
    assert before == {c: p.region.wkb for c, p in b.channels.items()}
    assert not b.done()


@pytest.mark.parametrize('coupled,movable', [(False, False), (True, True)])
def test_complete_sector_policy_uses_no_random_planning(monkeypatch, coupled, movable):
    sim = load('simulator')
    world = sim.generate_world(871, 10, 'boundary', 1000., 'extreme')
    env, b = sim.LocalSimulator(world), CoupledBelief() if coupled else Belief()
    policy = SectorPolicy(SectorConfig(resolution=8, movable_stations=movable, radius_coupling=coupled))
    def forbidden(*args, **kwargs):
        raise AssertionError('No random planning')
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    for i in range(1500):
        if b.done():
            break
        a = policy.choose(b)
        b.apply(a, env.execute(a, str(i)), str(i))
    assert b.done() and world.score()['all_cleared']
