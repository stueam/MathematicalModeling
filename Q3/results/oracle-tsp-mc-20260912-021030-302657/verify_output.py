"""Independent forward Held-Karp checks and full saved-output reconciliation."""
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'oracle-tsp-mc-20260912-020854-434182'


def forward_dp(xy):
    n = len(xy)
    dist = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    table = np.full((1 << n, n), np.inf)
    table[1 << np.arange(n), np.arange(n)] = np.linalg.norm(xy, axis=1)
    masks = np.arange(1, 1 << n)
    cardinality = np.array([int(s).bit_count() for s in masks])
    for count in range(2, n + 1):
        layer = masks[cardinality == count]
        for j in range(n):
            selected = layer[(layer & (1 << j)) != 0]
            table[selected, j] = np.min(table[selected ^ (1 << j)] + dist[:, j], axis=1)
    return float(table[-1].min())


data = np.load(HERE / 'worlds_and_exact_routes.npz')
previous = np.load(PREVIOUS / 'worlds_and_exact_routes.npz')
summary = json.loads((HERE / 'summary.json').read_text())
ns, xy, order, lengths = (data[k] for k in ('source_counts', 'source_xy', 'optimal_order', 'optimal_length_m'))
assert len(ns) == 100000
assert all(np.array_equal(data[k][:20000], previous[k], equal_nan=True) for k in data.files)
assert int(ns.sum()) == 1299298
assert np.nanmax(np.linalg.norm(xy, axis=2)) <= 1800
maximum_route_error = 0.
dp_checks = []
for n in range(10, 17):
    ids = np.flatnonzero(ns == n)
    points, routes = xy[ids, :n], order[ids, :n]
    assert np.all(np.sort(routes, axis=1) == np.arange(n))
    path = np.concatenate((np.zeros((len(ids), 1, 2)), points[np.arange(len(ids))[:, None], routes]), axis=1)
    sums = np.linalg.norm(np.diff(path, axis=1), axis=2).sum(axis=1)
    error = float(np.max(np.abs(sums - lengths[ids])))
    maximum_route_error = max(maximum_route_error, error)
    assert error < 1e-7
    value = forward_dp(points[0])
    assert abs(value - lengths[ids[0]]) < 1e-7
    dp_checks.append({'N': n, 'world_index': int(ids[0]), 'forward_dp_m': value,
                      'stored_tsp_m': float(lengths[ids[0]])})
    print(n, len(ids), float(sums.mean()), float((sums / (5 * n)).mean()))
assert abs(float((lengths / 5 / ns).mean()) - summary['primary_expected_T_div_N_s_per_source']['mean']) < 1e-10
result = {'status': 'passed', 'all_100000_routes_visit_every_source_once': True,
          'all_source_coordinates_inside_1800m_disk': True,
          'first_20000_worlds_and_routes_exactly_equal_previous_batch': True,
          'maximum_saved_route_length_error_m': maximum_route_error,
          'independent_forward_dp_checks': dp_checks,
          'saved_mean_recomputed': True,
          'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(HERE / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
