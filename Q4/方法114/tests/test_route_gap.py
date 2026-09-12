from itertools import permutations

import numpy as np

from audit_route_gap import blocks_from, feasible_route, path_length, relaxed_tsp_bound, respects


def test_relaxed_bound_matches_bruteforce_open_tsp():
    xy = np.array([(0., 0.), (3., 1.), (-1., 2.), (-2., -3.), (4., -2.), (1., 4.)])
    d = np.linalg.norm(xy[:, None]-xy[None, :], axis=2)
    optimum = min(path_length((0,)+p, d) for p in permutations(range(1, len(xy))))
    result = relaxed_tsp_bound(d, path_length(list(range(len(xy))), d), seconds=10.)
    assert result['relaxed_optimal_proved']
    assert abs(result['lower_bound_m']-optimum) < 1e-6
    assert abs(path_length(result['relaxed_order'], d)-optimum) < 1e-6


def test_feasible_route_keeps_measurement_before_clear():
    xy = np.array([(0., 0.), (10., 0.), (0., 2.), (10., 2.), (1., 1.), (9., 1.)])
    d = np.linalg.norm(xy[:, None]-xy[None, :], axis=2)
    precedences = np.array([(1, 3), (2, 4), (3, 5)])
    route = feasible_route(d, precedences, starts=3)
    assert respects(route, precedences)
    assert path_length(route, d) <= path_length(list(range(len(xy))), d)
    optimum = min(path_length((0,)+p, d) for p in permutations(range(1, len(xy)))
                  if respects((0,)+p, precedences))
    relaxed = relaxed_tsp_bound(d, path_length(route, d), seconds=10.)
    assert relaxed['lower_bound_m'] <= optimum+1e-6 <= path_length(route, d)+1e-6
    constrained = relaxed_tsp_bound(d, path_length(route, d), seconds=10., precedences=precedences)
    assert constrained['relaxed_optimal_proved'] and respects(constrained['relaxed_order'], precedences)
    assert abs(constrained['lower_bound_m']-optimum) < 1e-6


def test_precedence_prefix_cut_is_valid_for_all_feasible_small_paths():
    n, measurement, clear = 5, 3, 2
    precedences = np.array([(measurement, clear)])
    for p in permutations(range(1, n)):
        order = (0,)+p
        if not respects(order, precedences):
            continue
        cycle = (n,)+order+(n,)
        for mask in range(1 << n):
            subset = {v for v in range(n) if mask & (1 << v)}
            if not {0, clear} <= subset or measurement in subset:
                continue
            internal = sum(a in subset and b in subset for a, b in zip(cycle, cycle[1:]))
            assert internal <= len(subset)-2


def test_blocks_preserve_internal_order_and_defer_clear():
    rows = [
        {'action': {'kind': 'measure', 'position': [0., 0.], 'channel': 1}, 'response': {'measure_result': 'direction'}},
        {'action': {'kind': 'clear', 'position': [100., 0.], 'channel': 1}, 'response': {'clear_result': 'no_target_in_range'}},
        {'action': {'kind': 'measure', 'position': [100., 0.], 'channel': 2}, 'response': {'measure_result': 'no_signal'}},
        {'action': {'kind': 'clear', 'position': [150., 0.], 'channel': 1}, 'response': {'clear_result': 'success'}}]
    blocks, precedences = blocks_from(rows)
    assert len(blocks) == 3 and blocks[1]['rows'] == rows[1:3]
    assert set(map(tuple, precedences)) == {(0, 2), (1, 2)}
    assert not respects([0, 2, 1], precedences)
