from dataclasses import asdict

from analyze_practice import costs_for, tail_counterfactual
from q4.core import Belief
from q4.shared import Action, disk


def test_clear_cost_keeps_receiver_for_next_measure():
    b = Belief(receiver=3)
    clear = Action('clear', (1000., 0.), 7)
    assert costs_for(b, clear, {'clear_result': 'success'}) == {
        'move_s': 200., 'switch_s': 0, 'clear_success_s': 5}
    b.apply(clear, {'accepted': True, 'clear_result': 'success', 'virtual_time_s': 205}, 'clear')
    assert b.receiver == 3
    assert costs_for(b, Action('measure', (1000., 0.), 7), {})['switch_s'] == 1


def test_counterfactual_prunes_redundant_recorded_station_and_reproves_completion():
    b = Belief(virtual_time=100.)
    for c in range(1, 11):
        b.channels[c].status = 'cleared'
    for c in range(11, 20):
        b.channels[c].status = 'absent_certified'
    b.channels[20].region = disk((0., 0.), 20., outer=True)
    suffix, live = [], b.clone()
    for i, pos in enumerate(((800., 800.), (-100., -100.), (100., -100.), (0., 100.))):
        assert not live.done()
        action = Action('measure', pos, 20)
        reply = {'accepted': True, 'measure_result': 'no_signal'}
        reply['virtual_time_s'] = round(live.virtual_time+sum(costs_for(live, action, reply).values()), 6)
        suffix.append({'action': asdict(action), 'response': reply})
        live.apply(action, reply, f'original-{i}')
    assert live.done()
    result = tail_counterfactual(b, suffix)
    assert result['certificate_complete'] and result['pruned_s'] < result['actual_s']
    assert result['remaining_sites'] < result['actual_sites']
    assert not b.done() and not b.channels[20].history
    recorded = {(tuple(r['action']['position']), r['action']['channel']) for r in suffix}
    assert all((tuple(r['action']['position']), r['action']['channel']) in recorded for r in result['actions'])


def test_counterfactual_refuses_unknown_positive_feedback():
    b = Belief()
    result = tail_counterfactual(b, [{'action': {'kind': 'measure', 'position': [0., 0.], 'channel': 1},
                                      'response': {'measure_result': 'near'}}])
    assert not result['available']
