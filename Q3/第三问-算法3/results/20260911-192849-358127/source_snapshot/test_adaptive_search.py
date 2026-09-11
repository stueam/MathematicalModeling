from dataclasses import asdict

import pytest
import shapely

from bayes_tsp.adaptive_search import AdaptiveSearchConfig, AdaptiveSearchPolicy
from bayes_tsp.coupling import CoupledBelief
from bayes_tsp.refinement import RefinementConfig, RefinementPolicy
from bayes_tsp.shared import Action, Observation, disk, load


def census():
    b = CoupledBelief()
    for c in range(1, 21):
        b.channels[c].update(Observation(Action('measure', (0., 0.), c), 'no_signal'))
    return b


def test_every_rotated_partition_preserves_complete_channel_geometry():
    b = census()
    p = AdaptiveSearchPolicy(AdaptiveSearchConfig(resolution=8, sector_rotations=4,
        adaptive_sector_counts=(6, 8)))
    before = {c: channel.region.wkb for c, channel in b.channels.items()}
    for index in range(len(p._grids)):
        p._activate_grid(index)
        tasks = p._coverage_tasks(b)
        whole = shapely.union_all([task['region'] for task in tasks.values()])
        assert b.channels[1].region.difference(whole).is_empty
        for key, task in tasks.items():
            position = p._initial_center(key, task)
            assert task['region'].difference(disk(position, 1000)).is_empty
    assert before == {c: channel.region.wkb for c, channel in b.channels.items()}
    assert not b.done()


def test_plan_masks_are_distinct_and_all_planned_stops_cover_their_full_tasks():
    b = census()
    p = AdaptiveSearchPolicy(AdaptiveSearchConfig(resolution=8, sector_rotations=4))
    plans = p._plans(b, {}, {})
    assert len({plan['mask'] for plan in plans}) == len(plans) == 4
    assert any(plan['partition_index'] == 0 for plan in plans)
    assert plans[0]['proxy'] == min(plan['proxy'] for plan in plans)
    for plan in plans:
        p._activate_plan(plan)
        tasks = p._coverage_tasks(b)
        for key, position in plan['points'].items():
            assert tasks[key]['region'].difference(disk(position, 1000)).is_empty
    assert not b.done()


def test_single_partition_preserves_v6_first_action():
    b = census()
    b.position = (100., 300.)
    old = RefinementPolicy(RefinementConfig(resolution=8))
    new = AdaptiveSearchPolicy(AdaptiveSearchConfig(resolution=8, sector_rotations=1))
    assert asdict(old.choose(b)) == asdict(new.choose(b))


@pytest.mark.parametrize('noise', ['iid', 'extreme', 'correlated'])
def test_complete_adaptive_search_retains_finite_completion(noise):
    sim = load('simulator')
    world = sim.generate_world(907, 10, 'boundary', 1000., noise)
    env, b = sim.LocalSimulator(world), CoupledBelief()
    p = AdaptiveSearchPolicy(AdaptiveSearchConfig(resolution=8, sector_rotations=2))
    for index in range(1500):
        if b.done():
            break
        action = p.choose(b)
        b.apply(action, env.execute(action, str(index)), str(index))
    assert b.done() and world.score()['all_cleared']
