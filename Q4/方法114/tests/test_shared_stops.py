import pytest
from shapely.geometry import Point

from q4.compact import make_policy
from q4.core import Belief
from q4.simulator import generate_world, LocalSimulator


@pytest.mark.parametrize('name',['bundle','share','collect','collect-value'])
def test_shared_stops_complete_without_excluding_actual_sources(name):
    world=generate_world(143,10,'outward',1000.,'extreme',10)
    env,b,policy=LocalSimulator(world),Belief(),make_policy(name)
    sources=world.manifest()['sources']
    for i in range(2200):
        if b.done():
            break
        action=policy.choose(b)
        b.apply(action,env.execute(action,str(i)),str(i))
        for source in sources:
            channel=b.channels[source['channel']]
            assert channel.status!='absent_certified'
            assert channel.region.distance(Point(source['position']))<1e-7
    assert b.done() and world.score()['all_cleared']
