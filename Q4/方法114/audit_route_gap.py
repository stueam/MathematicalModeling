"""Offline distance gaps on fixed logged tasks; no HTTP and no hidden truth.

Exact clear-point TSP is a hindsight geometric reference. A separate feasible
route retains every recorded action and forces each channel's measurements and
failed clears before its successful clear. A relaxed TSP supplies a lower bound
for that fixed-task routing problem, not for the unknown full mission.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from analyze_practice import action_from, costs_for
from q4.core import Belief
from q4.routing import exact_route
from q4.shared import distance
from run import ROOT, dump, manifest


def blocks_from(rows):
    blocks = [{'position': (0., 0.), 'rows': []}]
    for row in rows:
        pos = tuple(row['action']['position'])
        if pos != blocks[-1]['position']:
            blocks.append({'position': pos, 'rows': []})
        blocks[-1]['rows'].append(row)
    clears = {}
    for i, block in enumerate(blocks):
        for row in block['rows']:
            if row['response'].get('clear_result') == 'success':
                clears[row['action']['channel']] = i
    precedences = set()
    for i, block in enumerate(blocks):
        for row in block['rows']:
            c = row['action']['channel']
            if c in clears and i != clears[c] and row['response'].get('clear_result') != 'success':
                assert i < clears[c], 'Original log contains observations after a successful clear'
                precedences.add((i, clears[c]))
    return blocks, np.array(sorted(precedences), dtype=int).reshape(-1, 2)


def path_length(order, d):
    return float(sum(d[a, b] for a, b in zip(order, order[1:])))


def respects(order, precedences):
    if order[0] != 0 or len(set(order)) != len(order):
        return False
    indices = np.empty(len(order), dtype=int)
    indices[np.asarray(order)] = np.arange(len(order))
    return bool(np.all(indices[precedences[:, 0]] < indices[precedences[:, 1]]))


def improve(order, d, precedences):
    """Best improving 2-opt and relocation, preserving all clear dependencies."""
    order = list(order)
    n = len(order)
    for _ in range(300):
        trials = []
        for i in range(1, n):
            a, x = order[i-1], order[i]
            z = order[i+1] if i+1 < n else None
            removal = d[a, x]+(d[x, z]-d[a, z] if z is not None else 0.)
            for j in range(1, n):
                if j in (i, i+1):
                    continue
                b, y = order[j-1], order[j]
                saving = removal-(d[b, x]+d[x, y]-d[b, y])
                if saving > 1e-6:
                    trials.append((float(saving), 'relocate', i, j))
            if i != n-1:
                saving = removal-d[order[-1], x]
                if saving > 1e-6:
                    trials.append((float(saving), 'relocate', i, n))
            for j in range(i+1, n):
                y = order[j]
                saving = d[a, x]-d[a, y]
                if j+1 < n:
                    z2 = order[j+1]
                    saving += d[y, z2]-d[x, z2]
                if saving > 1e-6:
                    trials.append((float(saving), 'reverse', i, j))
        accepted = False
        for _, kind, i, j in sorted(trials, reverse=True):
            candidate = list(order)
            if kind == 'reverse':
                candidate[i:j+1] = reversed(candidate[i:j+1])
            else:
                x = candidate.pop(i)
                candidate.insert(j-int(j > i), x)
            if respects(candidate, precedences):
                assert path_length(candidate, d) < path_length(order, d)-1e-7
                order, accepted = candidate, True
                break
        if not accepted:
            break
    return order


def feasible_route(d, precedences, starts=12):
    n = len(d)
    original = list(range(n))
    candidates = [improve(original, d, precedences)]
    parents = {i: set(precedences[precedences[:, 1] == i, 0]) for i in range(n)}
    rng = np.random.default_rng(123)
    for seed in range(starts):
        order, visited = [0], {0}
        while len(order) < n:
            eligible = sorted((i for i in range(n) if i not in visited and parents[i] <= visited),
                              key=lambda i: (d[order[-1], i], i))
            if not eligible:
                raise ValueError('Cyclic measurement-to-clear dependencies')
            j = 0 if seed == 0 else int(rng.integers(min(3, len(eligible))))
            order.append(eligible[j])
            visited.add(eligible[j])
        candidates.append(improve(order, d, precedences))
    return min(candidates, key=lambda r: (path_length(r, d), r))


def relaxed_tsp_bound(d, upper_m, seconds=45., precedences=None):
    """Undirected TSP with a dummy endpoint and iterated subtour cuts.

    Without precedences this is a distance lower bound. With precedences, a
    forbidden prefix S containing start and clear but excluding a required
    measurement must have at least four cut edges in the dummy cycle. This gives
    the valid cut internal_edges(S) <= |S|-2. Add violated prefixes until the
    optimal cycle respects every dependency, or return the current bound.
    """
    n = len(d)
    dummy = n
    pairs = [(i, j) for i in range(n+1) for j in range(i+1, n+1)]
    index = {edge: k for k, edge in enumerate(pairs)}
    objective = np.array([d[i, j] if j != dummy else 0. for i, j in pairs])
    lo, hi = np.zeros(len(pairs)), np.ones(len(pairs))
    lo[index[(0, dummy)]] = 1
    cuts, seen = [], set()
    best_lower, connected, solved, iterations, precedence_cuts = 0., None, False, 0, 0
    started = time.monotonic()
    while time.monotonic()-started < seconds:
        rr, cc, vv, lower, upper = [], [], [], [], []
        def add(indices, values, low, high):
            row = len(lower)
            rr.extend([row]*len(indices)); cc.extend(indices); vv.extend(values)
            lower.append(low); upper.append(high)
        for v in range(n+1):
            indices = [k for k, edge in enumerate(pairs) if v in edge]
            add(indices, [1.]*len(indices), 2., 2.)
        add(list(range(len(pairs))), objective.tolist(), -np.inf, upper_m+1e-5)
        for component, maximum in cuts:
            indices = [k for k, (i, j) in enumerate(pairs) if i in component and j in component]
            add(indices, [1.]*len(indices), -np.inf, maximum)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(lower), len(pairs))).tocsc()
        remaining = max(.01, seconds-(time.monotonic()-started))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected.*')
            result = milp(objective, integrality=np.ones(len(pairs)), bounds=Bounds(lo, hi),
                          constraints=LinearConstraint(matrix, lower, upper),
                          options={'time_limit': remaining, 'mip_rel_gap': 0., 'threads': 1})
        iterations += 1
        bound = getattr(result, 'mip_dual_bound', None)
        if bound is not None and math.isfinite(bound):
            best_lower = max(best_lower, float(bound))
        if result.x is None:
            break
        neighbors = {i: [] for i in range(n+1)}
        for x, (i, j) in zip(result.x, pairs):
            if x > .5:
                neighbors[i].append(j); neighbors[j].append(i)
        if any(len(v) != 2 for v in neighbors.values()):
            raise RuntimeError('Nonintegral/invalid TSP incumbent')
        left, components = set(range(n+1)), []
        while left:
            stack, component = [min(left)], set()
            while stack:
                v = stack.pop()
                if v not in component:
                    component.add(v); stack.extend(neighbors[v])
            left -= component
            components.append(frozenset(component))
        if len(components) == 1:
            order, prev, current = [0], dummy, 0
            while True:
                nxt = next(v for v in neighbors[current] if v != prev)
                if nxt == dummy:
                    break
                order.append(nxt)
                prev, current = current, nxt
            assert len(order) == n
            if precedences is not None and not respects(order, precedences):
                positions = {v: i for i, v in enumerate(order)}
                for measurement, clear in precedences:
                    if positions[measurement] > positions[clear]:
                        prefix = frozenset(order[:positions[clear]+1])
                        cut = (prefix, len(prefix)-2.)
                        if cut not in seen:
                            seen.add(cut); cuts.append(cut)
                            precedence_cuts += 1
                continue
            connected = order
            solved = result.status == 0 and path_length(order, d)-best_lower < 1e-4
            break
        for component in components:
            cut = (component, len(component)-1.)
            if cut not in seen:
                cuts.append(cut); seen.add(cut)
    return {'lower_bound_m': best_lower, 'relaxed_order': connected,
            'relaxed_optimal_proved': solved, 'iterations': iterations,
            'real_s': time.monotonic()-started, 'time_limit_s': seconds,
            'precedence_cuts': precedence_cuts, 'precedences_enforced': precedences is not None}


def replay(blocks, order):
    b, saved, first_done = Belief(), [], None
    for i in order:
        for original in blocks[i]['rows']:
            a = action_from(original)
            cost = costs_for(b, a, original['response'])
            response = dict(original['response'], virtual_time_s=round(b.virtual_time+sum(cost.values()), 6))
            row = {'original_request_id': original['request_id'], 'action': original['action'],
                   'response': response, 'costs': cost, 'block': i}
            b.apply(a, response, f'fixed-task-replay-{len(saved)}')
            saved.append(row)
            if b.done() and first_done is None:
                first_done = {'actions': len(saved), 'virtual_time_s': b.virtual_time}
    assert b.done()
    return {'complete': b.done(), 'cleared_count': len(b.cleared), 'virtual_time_s': b.virtual_time,
            'first_public_completion': first_done, 'actions': saved}


def run_case(case, time_limit, precedence_seconds=60.):
    folder = Path(case['folder'])
    rows = [json.loads(s) for s in (folder/'actions.jsonl').read_text().splitlines()]
    blocks, precedences = blocks_from(rows)
    xy = np.asarray([block['position'] for block in blocks])
    d = np.linalg.norm(xy[:, None]-xy[None, :], axis=2)
    before = path_length(list(range(len(blocks))), d)
    order = feasible_route(d, precedences)
    feasible_m = path_length(order, d)
    bound = relaxed_tsp_bound(d, feasible_m, time_limit)
    relaxed_order = bound['relaxed_order']
    if relaxed_order and respects(relaxed_order, precedences):
        if path_length(relaxed_order, d) < feasible_m:
            order, feasible_m = relaxed_order, path_length(relaxed_order, d)
    constrained = relaxed_tsp_bound(d, feasible_m, precedence_seconds, precedences)
    if constrained['relaxed_order'] is not None:
        new_order = constrained['relaxed_order']
        assert respects(new_order, precedences)
        if path_length(new_order, d) < feasible_m:
            order, feasible_m = new_order, path_length(new_order, d)
    fixed_task_lower = max(bound['lower_bound_m'], constrained['lower_bound_m'])
    counterfactual = replay(blocks, order)
    assert len(counterfactual['actions']) == len(rows)
    assert counterfactual['cleared_count'] == case['source_count']
    assert feasible_m <= before+1e-5 and fixed_task_lower <= feasible_m+1e-3
    clears = {r['action']['channel']: tuple(r['action']['position']) for r in rows
              if r['response'].get('clear_result') == 'success'}
    clear_order, clear_optimum = exact_route((0., 0.), clears)
    return {'case_code': case['case_code'], 'source_count': case['source_count'],
            'original_m': before, 'original_virtual_s': case['virtual_time_s'],
            'clear_points_exact_m': clear_optimum, 'clear_points_exact_order': clear_order,
            'clear_only_reference_gap_m': before-clear_optimum,
            'clear_order_only_gap_m': case['fixed_clear_points_order_gap_m'],
            'fixed_task_blocks': len(blocks), 'precedences': precedences.tolist(),
            'feasible_fixed_task_m': feasible_m, 'fixed_task_lower_bound_m': fixed_task_lower,
            'feasible_saved_m': before-feasible_m,
            'fixed_task_optimal_saving_lower_m': before-feasible_m,
            'fixed_task_optimal_saving_upper_m': before-fixed_task_lower,
            'feasible_movement_saved_s': (before-feasible_m)/5,
            'feasible_total_saved_s': case['virtual_time_s']-counterfactual['virtual_time_s'],
            'fixed_task_optimal_proved': feasible_m-fixed_task_lower < 1e-4,
            'relaxed_solver': bound, 'constrained_solver': constrained, 'route_order': order,
            'blocks': [{'position': b['position'], 'actions': len(b['rows'])} for b in blocks],
            'counterfactual': counterfactual,
            'note': 'Hindsight fixed-log schedule, preserving every action and measurement-before-clear constraints. Not a live policy or a global optimum for unknown sources.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit', type=Path)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seconds', type=float, default=45.)
    p.add_argument('--precedence-seconds', type=float, default=60.)
    args = p.parse_args()
    cases = json.loads((args.audit/'cases.json').read_text())
    cfg = json.loads((args.audit/'config.json').read_text())
    assert cfg['algorithm_sha256'] == manifest(), 'Algorithm differs from audited version'
    out = ROOT/'results'/datetime.now().strftime('route-gap-%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'config.json', {'input': str(args.audit), 'workers': args.workers,
         'seconds_per_relaxed_solver': args.seconds,
         'seconds_per_precedence_solver': args.precedence_seconds,
         'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'algorithm_sha256': manifest(), 'official_requests': 0})
    state = {'status': 'running', 'planned_cases': len(cases), 'completed_cases': 0}
    dump(out/'batch.json', state)
    print(f'Output: {out}', flush=True)
    records = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(run_case, c, args.seconds, args.precedence_seconds) for c in cases]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                dump(out/f'{record["case_code"]}.json', record)
                state['completed_cases'] = len(records)
                dump(out/'batch.json', state)
                print(f'{len(records)}/{len(cases)} {record["case_code"]}: saved {record["feasible_saved_m"]:.1f} m', flush=True)
        dump(out/'cases.json', records)
        keys = ('original_m', 'original_virtual_s', 'clear_points_exact_m', 'clear_only_reference_gap_m',
                'clear_order_only_gap_m', 'feasible_fixed_task_m', 'fixed_task_lower_bound_m',
                'feasible_saved_m', 'fixed_task_optimal_saving_lower_m', 'fixed_task_optimal_saving_upper_m',
                'feasible_movement_saved_s', 'feasible_total_saved_s')
        summary = {'cases': len(records), 'public_replays_complete': sum(r['counterfactual']['complete'] for r in records),
                   'fixed_task_optimal_proved_cases': sum(r['fixed_task_optimal_proved'] for r in records),
                   'relaxed_optimal_proved_cases': sum(r['relaxed_solver']['relaxed_optimal_proved'] for r in records),
                   'means': {k: float(np.mean([r[k] for r in records])) for k in keys},
                   'minimum_feasible_saved_m': min(r['feasible_saved_m'] for r in records),
                   'maximum_feasible_saved_m': max(r['feasible_saved_m'] for r in records)}
        dump(out/'summary.json', summary)
        state['status'] = 'completed'
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        dump(out/'batch.json', state)


if __name__ == '__main__':
    main()
