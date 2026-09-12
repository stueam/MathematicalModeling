import pytest
from shapely.geometry import Point

from q4.core import Belief, Channel, DOMAIN
from q4.coverage import certifies
from q4.sector import SectorPolicy, ring_stations
from q4.simulator import generate_world, LocalSimulator
from q4.shared import Action, Observation


def test_ring_backstop_covers_outer_domain_without_committing_evidence():
    b, policy = Belief(), SectorPolicy()
    assert len(ring_stations()) == 22 and certifies(DOMAIN, ring_stations())
    assert policy.valid_future(b, policy.points)
    assert not b.done() and all(not c.history for c in b.channels.values())


def test_tiny_support_is_retained_without_actual_exclusion():
    channel = Channel()
    channel.region = Point(0., 0.).buffer(1e-5)
    channel.update(Observation(Action('measure', (1500., 0.), 1), 'no_signal'))
    assert not channel.region.is_empty and channel.status == 'unresolved'
    assert channel.region.covers(Point(0., 0.))


def test_actual_negative_union_completes_in_reverse_order():
    channel = Channel()
    for point in reversed(ring_stations()):
        channel.update(Observation(Action('measure', point, 1), 'no_signal'))
    assert channel.region.is_empty and channel.status == 'absent_certified'


@pytest.mark.parametrize('scenario,n,nd', [('outward', 10, 10), ('backside', 10, 9), ('uniform', 16, 1)])
def test_sector_completion_and_per_step_truth(scenario, n, nd):
    world = generate_world(143, n, scenario, 1000., 'extreme', nd)
    sources = world.manifest()['sources']
    env, b, policy = LocalSimulator(world), Belief(), SectorPolicy()
    for i in range(2200):
        if b.done():
            break
        action = policy.choose(b)
        b.apply(action, env.execute(action, str(i)), str(i))
        for source in sources:
            channel = b.channels[source['channel']]
            assert channel.status != 'absent_certified'
            assert channel.region.distance(Point(source['position'])) < 1e-7
    assert b.done() and world.score()['all_cleared']
    assert b.virtual_time < 360000
