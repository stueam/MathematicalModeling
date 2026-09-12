from q4.collected import CollectedPolicy
from q4.core import Belief
from q4.sector import ProbePolicy
from q4.shared import Action
from q4.simulator import LocalSimulator,Source,World


def test_collected_stop_keeps_plan_and_waits_for_actual_per_channel_feedback(monkeypatch):
    b=Belief();env=LocalSimulator(World([Source(1,(800.,0.),1000.,0.)]))
    clear=Action('clear',(800.,0.),1)
    b.apply(clear,env.execute(clear,'clear'),'clear')
    policy=CollectedPolicy();points=dict(policy.points)
    monkeypatch.setattr(ProbePolicy,'shared_replacement',lambda self,b:None)
    signature=[(p.region.wkb,tuple(p.history),p.status) for p in b.channels.values()]
    action=policy.shared_replacement(b)
    assert action.kind=='measure' and action.position==b.position and action.channel!=1
    assert policy.points==points
    assert signature==[(p.region.wkb,tuple(p.history),p.status) for p in b.channels.values()]
    b.apply(action,env.execute(action,'scan'),'scan')
    for c,p in b.channels.items():
        if c not in (1,action.channel):
            assert not p.history and p.status=='unresolved'
    assert not b.done()


def test_partial_coverage_value_cannot_remove_a_station_or_update_belief():
    b=Belief();policy=CollectedPolicy(value_gate=True);b.position=(800.,0.)
    points=dict(policy.points)
    signature=[(p.region.wkb,tuple(p.history),p.status) for p in b.channels.values()]
    value,reviews=policy.partial_value(b)
    assert value>=0 and reviews
    assert all(0<=r['fraction_of_missing_area_filled']<=1 for r in reviews)
    assert policy.points==points
    assert signature==[(p.region.wkb,tuple(p.history),p.status) for p in b.channels.values()]
