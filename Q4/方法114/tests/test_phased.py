import pytest

from q4.core import Belief
from q4.phased import PhasedPolicy
from q4.routing import length
from q4.shared import Action


def test_phase_route_keeps_every_obligation_and_counts_the_connection():
    policy=PhasedPolicy()
    goals={-1:(1800.,0.),-2:(0.,1800.),1:(900.,0.),2:(0.,800.)}
    order,meters=policy.route((0.,0.),goals)
    assert set(order)==set(goals) and len(order)==len(goals)
    assert set(order[:2])=={-1,-2}
    assert meters==pytest.approx(length((0.,0.),order,goals))
    policy.record(Belief(),Action('measure',(1800.,0.),1),'test',route_exact=True,
                  candidates=[{'tail_exact':True}])
    assert not policy.records[-1]['route_exact']
    assert not policy.records[-1]['candidates'][0]['tail_exact']
