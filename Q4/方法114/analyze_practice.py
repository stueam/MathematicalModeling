"""Audit and attribute complete official PRACTICE histories; never send HTTP.

Source counts are read only from saved end screens. No true source coordinates,
headings, radii, or hidden error fields are available to this analysis.
"""
import argparse
import csv
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import math
import multiprocessing
from pathlib import Path

import numpy as np

from practice_windows import has_title, result_composition, result_count
from q4.core import Belief
from q4.coverage import certifies
from q4.routing import exact_route, length, open_route
from q4.shared import Action, distance, point_key
from run import ROOT, dump, manifest

COST_FIELDS = ('move_s', 'switch_s', 'measure_s', 'clear_success_s', 'clear_failure_s')


def action_from(row):
    a = row['action']
    return Action(a['kind'], tuple(a['position']), a['channel'])


def costs_for(b, action, response):
    move = round(distance(b.position, action.position)/5, 6)
    switch = int(action.kind == 'measure' and b.receiver != action.channel)
    operation = 'measure_s' if action.kind == 'measure' else (
        'clear_success_s' if response['clear_result'] == 'success' else 'clear_failure_s')
    return {'move_s': move, 'switch_s': switch,
            operation: 3 if operation == 'clear_failure_s' else 5}


def tail_replay(start, groups, order):
    """Reorder only already recorded negative readings; fixed-location feedback.

    This is an offline counterfactual, not an implemented live-policy result.
    It must rebuild a complete public certificate after every deletion/reorder.
    """
    b = start.clone()
    costs, actions = Counter(), []
    for key in order:
        pending = {row['action']['channel']: row for row in groups[key]}
        channels = sorted(pending, key=lambda c: (c != b.receiver, c))
        for c in channels:
            if b.channels[c].status in ('cleared', 'absent_certified'):
                continue
            row = pending[c]
            a = action_from(row)
            if a.kind != 'measure' or row['response']['measure_result'] != 'no_signal':
                raise ValueError('Counterfactual supports recorded negative-only tails')
            cost = costs_for(b, a, row['response'])
            response = dict(row['response'], virtual_time_s=round(b.virtual_time+sum(cost.values()), 6))
            b.apply(a, response, f'offline-tail-{len(actions)}')
            costs.update(cost)
            actions.append({'action': row['action'], 'response': response})
            if b.done():
                return {'complete': True, 'cost_s': b.virtual_time-start.virtual_time,
                        'costs': dict(costs), 'actions': actions}
    return {'complete': b.done(), 'cost_s': b.virtual_time-start.virtual_time,
            'costs': dict(costs), 'actions': actions}


def tail_counterfactual(start, suffix):
    if not suffix:
        return {'available': True, 'actual_s': 0., 'reordered_s': 0., 'pruned_s': 0.,
                'reordering_saved_s': 0., 'pruning_additional_saved_s': 0.,
                'actual_sites': 0, 'remaining_sites': 0, 'certificate_complete': True,
                'route_exact': True, 'actions': []}
    if any(r['action']['kind'] != 'measure' or r['response'].get('measure_result') != 'no_signal' for r in suffix):
        return {'available': False, 'reason': 'Suffix contains feedback other than no_signal'}
    grouped = defaultdict(list)
    for row in suffix:
        grouped[tuple(row['action']['position'])].append(row)
    points = dict(enumerate(grouped))
    groups = {k: grouped[p] for k, p in points.items()}
    actual = suffix[-1]['response']['virtual_time_s']-start.virtual_time

    def valid(kept):
        checked = set()
        for c, channel in start.channels.items():
            if channel.status in ('cleared', 'absent_certified'):
                continue
            if channel.status != 'unresolved':
                return False
            future = tuple(points[k] for k in kept if any(r['action']['channel'] == c for r in groups[k]))
            key = (channel.region.wkb, tuple(sorted(set(channel.negatives+future))))
            if key in checked:
                continue
            checked.add(key)
            if not certifies(channel.region, channel.negatives+future):
                return False
        return True

    order, _ = open_route(start.position, points)
    reordered = tail_replay(start, groups, order)
    if not reordered['complete']:
        raise ValueError('Reordered recorded tail lost its public completion certificate')
    kept = dict(points)
    removed, checks = [], 0
    # Greedy station removal, always checking ALL channel regions and readings.
    # This optimizes the recorded suffix only, not the full unknown mission.
    while kept:
        order, before = open_route(start.position, kept)
        trials = []
        for i, k in enumerate(order):
            prev = start.position if i == 0 else points[order[i-1]]
            gain = distance(prev, points[k])
            if i+1 < len(order):
                following = points[order[i+1]]
                gain += distance(points[k], following)-distance(prev, following)
            trials.append((gain/5+6*len(groups[k]), k))
        accepted = False
        for _, key in sorted(trials, reverse=True):
            proposed = {k: p for k, p in kept.items() if k != key}
            checks += 1
            if valid(proposed):
                kept = proposed
                removed.append(key)
                accepted = True
                break
        if not accepted:
            break
    order, _ = open_route(start.position, kept)
    pruned = tail_replay(start, groups, order)
    if not pruned['complete']:
        raise ValueError('Pruned recorded tail lost its public completion certificate')
    return {'available': True, 'actual_s': actual, 'reordered_s': reordered['cost_s'],
            'pruned_s': pruned['cost_s'], 'reordering_saved_s': actual-reordered['cost_s'],
            'pruning_additional_saved_s': reordered['cost_s']-pruned['cost_s'],
            'actual_sites': len(points), 'remaining_sites': len(kept),
            'removed_sites': [points[k] for k in removed], 'certificate_checks': checks,
            'certificate_complete': pruned['complete'], 'route_exact': len(points) <= 16,
            'actions': pruned['actions']}


def audit_case(folder, optimize_tail=True):
    folder = Path(folder)
    original = json.loads((folder/'summary.json').read_text())
    rows = [json.loads(s) for s in (folder/'actions.jsonl').read_text().splitlines()]
    decisions = {d['step']: d for d in json.loads((folder/'decisions.json').read_text())}
    http = json.loads((folder/'http-log.json').read_text())
    ui = json.loads((folder/'practice-after.json').read_text())
    total = result_count(ui)
    assert has_title(ui, 'test-run-title') and original['case_code'] in [i['name'] for i in ui]
    assert original['exited'] and original['certified_complete'] and original['error'] is None
    assert total == original['cleared_count']
    assert http[0]['path'] == 'enter' and http[-1]['path'] == 'exit'
    assert http[-1]['response'].get('exit_reason') == 'user_exit'
    assert all(h.get('http_status') == 200 and h['response']['accepted'] is True for h in http if 'response' in h)
    clear_indices = [i for i, r in enumerate(rows) if r['response'].get('clear_result') == 'success']
    last_clear = clear_indices[-1]
    final_channels = {r['action']['channel'] for r in rows if r['response'].get('clear_result') == 'success'}
    b, seen = Belief(), set()
    stats, costs = Counter(), Counter()
    phases = defaultdict(Counter)
    channels = defaultdict(Counter)
    sites, source_points, large_moves, proxy_errors, route_sizes = {}, {}, [], [], []
    snapshots, last_clear_snapshot = [], None
    for step, row in enumerate(rows):
        assert not b.done() and row['request_id'] not in seen
        seen.add(row['request_id'])
        a, response = action_from(row), row['response']
        channel = b.channels[a.channel]
        pre_status = channel.status
        decision = decisions[step]
        assert decision['action'] == row['action']
        move = distance(b.position, a.position)
        cost = costs_for(b, a, response)
        elapsed = sum(cost.values())
        expected = round(b.virtual_time+elapsed, 6)
        assert abs(expected-response['virtual_time_s']) < 1e-5
        stats['max_cost_error_s'] = max(stats['max_cost_error_s'], abs(expected-response['virtual_time_s']))
        purpose = 'search' if a.kind == 'measure' and pre_status == 'unresolved' else 'localize_clear'
        phase = 'completion_tail' if step > last_clear else purpose
        phases[phase].update(cost)
        phases[phase]['actions'] += 1
        phases[phase]['movement_m'] += move
        costs.update(cost)
        stats[f'{purpose}_s'] += elapsed
        stats[f'{purpose}_movement_m'] += move
        stats[f'reason:{decision["reason"]}'] += 1
        stats['planning_s'] += row['planning_s']
        stats['moving_actions'] += int(move > 1e-6)
        stats['outside_domain_actions'] += int(math.hypot(*a.position) > 1800)
        stats['completion_mode_actions'] += int(decision['completion_mode'])
        if move >= 200:
            assert decision['long_move_review']
            stats['long_move_m'] += move
            stats['long_move_s'] += move/5
            large_moves.append({'step': step, 'move_m': move, 'purpose': purpose, 'phase': phase,
                                'known': len(b.known), 'cleared': len(b.cleared), 'position': a.position})
        if a.kind == 'measure':
            stats[f'{purpose}_measurements'] += 1
            stats['repeated_measurements'] += int(point_key(a.position) in channel.measured)
            channels[a.channel]['measurements'] += 1
            stats[f'feedback:{response["measure_result"]}'] += 1
            if a.channel not in final_channels:
                stats['eventually_absent_measurements'] += 1
                stats['eventually_absent_operation_s'] += 5+cost['switch_s']
            if purpose == 'search':
                sites.setdefault(a.position, set()).add(a.channel)
                if response['measure_result'] in ('direction', 'near'):
                    stats['discoveries_at_origin'] += int(distance(a.position, (0., 0.)) < 1e-6)
                    channels[a.channel]['first_signal_step'] = step
                    channels[a.channel]['first_signal_time_s'] = response['virtual_time_s']
                    snapshots.append({'step': step, 'virtual_time_s': response['virtual_time_s'],
                                      'event': 'first_signal', 'channel': a.channel})
        else:
            channels[a.channel]['clear_attempts'] += 1
            channels[a.channel]['clear_failures'] += int(response['clear_result'] != 'success')
            if response['clear_result'] == 'success':
                assert a.channel not in source_points
                source_points[a.channel] = a.position
                channels[a.channel]['clear_step'] = step
                channels[a.channel]['clear_time_s'] = response['virtual_time_s']
        if pre_status == 'detected' or a.kind == 'clear':
            channels[a.channel]['local_target_s'] += elapsed
            channels[a.channel]['local_target_movement_m'] += move
        if 'planned_route' in decision:
            route = decision['planned_route']
            route_sizes.append(len(route))
            stats['root_route_decisions'] += 1
            stats['root_route_exact_decisions'] += int(decision['route_exact'])
            stats['survey_route_nodes'] += sum(k < 0 for k in route)
            stats['source_route_nodes'] += sum(k > 0 for k in route)
            selected = next(c for c in decision['candidates'] if c['action'] == row['action'])
            realized_remaining = original['virtual_time_s']-b.virtual_time
            proxy_errors.append(selected['score_s']-realized_remaining)
        b.apply(a, response, row['request_id'])
        steps = b.steps
        assert b.apply(a, response, row['request_id']) is False and b.steps == steps
        if step == last_clear:
            last_clear_snapshot = b.clone()
    assert b.done() and len(b.cleared) == total
    assert abs(b.virtual_time-original['virtual_time_s']) < 1e-5
    assert len(rows) == original['actions']
    assert all(abs(costs[k]-original['costs'].get(k, 0)) < 1e-4 for k in costs)
    stats['search_sites'] = len(sites)
    stats['search_sites_outside_domain'] = sum(math.hypot(*p) > 1800 for p in sites)
    stats['after_last_clear_s'] = b.virtual_time-last_clear_snapshot.virtual_time
    stats['after_last_clear_actions'] = len(rows)-last_clear-1
    source_order = list(source_points)
    _, fixed_source_optimal_m = exact_route((0., 0.), source_points)
    source_order_m = length((0., 0.), source_order, source_points)
    cf = tail_counterfactual(last_clear_snapshot, rows[last_clear+1:]) if optimize_tail else {'available': False}
    return {'case_code': original['case_code'], 'folder': str(folder), 'source_count': total,
            **result_composition(ui), 'complete': True, 'virtual_time_s': b.virtual_time,
            'real_time_s': original['real_time_s'], 'planning_time_s': original['planning_time_s'],
            'movement_m': original['movement_m'], 'actions': len(rows), 'costs': dict(costs),
            'phases': {k: dict(v) for k, v in phases.items()}, 'stats': dict(stats),
            'counters': original['counters'], 'channels': dict(channels),
            'root_route_nodes_mean': float(np.mean(route_sizes)) if route_sizes else 0.,
            'root_route_nodes_max': max(route_sizes, default=0),
            'proxy_minus_realized_remaining_mean_s': float(np.mean(proxy_errors)) if proxy_errors else None,
            'fixed_clear_points_actual_order_m': source_order_m,
            'fixed_clear_points_exact_order_m': fixed_source_optimal_m,
            'fixed_clear_points_order_gap_m': source_order_m-fixed_source_optimal_m,
            'reference_only': 'Exact open TSP through executed successful clear points; not a feasible discovery policy.',
            'large_moves': sorted(large_moves, key=lambda r: r['move_m'], reverse=True),
            'events': snapshots, 'search_sites': [{'position': p, 'channels': sorted(cs)} for p, cs in sites.items()],
            'http_retries_or_errors': sum('error' in h for h in http),
            'evaluation_recovered': not original.get('official_result_verified', False),
            'tail_counterfactual': cf}


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'median': float(np.median(a)),
            'p90': float(np.quantile(a, .9)), 'p95': float(np.quantile(a, .95)),
            'min': float(a.min()), 'max': float(a.max())}


def aggregate(cases):
    n = len(cases)
    if not n:
        return {'cases': 0}
    total = sum(r['virtual_time_s'] for r in cases)
    costs, stats, counters = Counter(), Counter(), Counter()
    phases = defaultdict(Counter)
    for r in cases:
        costs.update(r['costs'])
        stats.update(r['stats'])
        counters.update(r['counters'])
        for k, v in r['phases'].items():
            phases[k].update(v)
    by_count = {}
    for count in sorted({r['source_count'] for r in cases}):
        group = [r for r in cases if r['source_count'] == count]
        by_count[count] = {'cases': len(group), 'mean_virtual_s': float(np.mean([r['virtual_time_s'] for r in group])),
                           'mean_tail_s': float(np.mean([r['stats']['after_last_clear_s'] for r in group])),
                           'mean_measurements': float(np.mean([r['stats']['search_measurements']+
                                                              r['stats'].get('localize_clear_measurements', 0) for r in group]))}
    tails = [r['tail_counterfactual'] for r in cases if r['tail_counterfactual']['available']]
    clear_point_references = {key: float(np.mean([r[key] for r in cases])) for key in (
        'fixed_clear_points_actual_order_m', 'fixed_clear_points_exact_order_m', 'fixed_clear_points_order_gap_m')}
    return {'cases': n, 'complete': sum(r['complete'] for r in cases),
            'source_count_total': sum(r['source_count'] for r in cases),
            'directional_count_total': sum(r['directional_count'] or 0 for r in cases),
            'zero_omnidirectional_cases': sum(r['omnidirectional_count'] == 0 for r in cases),
            'zero_directional_cases': sum(r['directional_count'] == 0 for r in cases),
            'virtual_time_s': distribution([r['virtual_time_s'] for r in cases]),
            'movement_m': distribution([r['movement_m'] for r in cases]),
            'real_time_s': distribution([r['real_time_s'] for r in cases]),
            'planning_time_s': distribution([r['planning_time_s'] for r in cases]),
            'mean_measurements': float(np.mean([r['stats']['search_measurements']+
                                               r['stats'].get('localize_clear_measurements', 0) for r in cases])),
            'per_source_pooled_s': total/sum(r['source_count'] for r in cases),
            'costs_mean_s': {k: costs[k]/n for k in COST_FIELDS},
            'cost_fractions': {k: costs[k]/total for k in COST_FIELDS},
            'phases': {k: {'mean_s': sum(v[c] for c in COST_FIELDS)/n,
                           'fraction': sum(v[c] for c in COST_FIELDS)/total,
                           'mean_movement_m': v['movement_m']/n,
                           'mean_actions': v['actions']/n,
                           'costs_mean_s': {c: v[c]/n for c in COST_FIELDS}} for k, v in phases.items()},
            'stats_totals': dict(stats), 'counters_totals': dict(counters),
            'max_clock_error_s': max(r['stats']['max_cost_error_s'] for r in cases),
            'http_retries_or_errors': sum(r['http_retries_or_errors'] for r in cases),
            'recovered_evaluations': [r['case_code'] for r in cases if r['evaluation_recovered']],
            'by_source_count': by_count,
            'exact_root_route_fraction': stats['root_route_exact_decisions']/max(1, stats['root_route_decisions']),
            'mean_root_route_nodes': float(np.mean([r['root_route_nodes_mean'] for r in cases])),
            'clear_point_geometry_references': clear_point_references,
            'tail_counterfactual': {'cases': len(tails),
                'complete_certificates': sum(t['certificate_complete'] for t in tails),
                'exact_original_suffix_route_cases': sum(t['route_exact'] for t in tails),
                'mean_actual_s': float(np.mean([t['actual_s'] for t in tails])) if tails else None,
                'mean_reordered_s': float(np.mean([t['reordered_s'] for t in tails])) if tails else None,
                'mean_pruned_s': float(np.mean([t['pruned_s'] for t in tails])) if tails else None,
                'mean_reordering_saved_s': float(np.mean([t['reordering_saved_s'] for t in tails])) if tails else None,
                'mean_pruning_additional_saved_s': float(np.mean([t['pruning_additional_saved_s'] for t in tails])) if tails else None,
                'mean_original_sites': float(np.mean([t['actual_sites'] for t in tails])) if tails else None,
                'mean_pruned_sites': float(np.mean([t['remaining_sites'] for t in tails])) if tails else None,
                'note': 'Offline recorded negative-only suffixes. Same fixed-location feedback; all final public certificates rebuilt. Not live-policy performance.'}}


def write_csv(path, cases):
    fields = ('case_code', 'source_count', 'omnidirectional_count', 'directional_count',
              'virtual_time_s', 'movement_m', 'real_time_s', 'planning_time_s', 'actions',
              'search_before_last_clear_s', 'localize_clear_s', 'completion_tail_s',
              'search_sites', 'absent_channel_measurements', 'clear_failures',
              'tail_reordered_s', 'tail_pruned_s')
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in cases:
            row = {k: r[k] for k in fields if k in r}
            row.update(search_before_last_clear_s=sum(r['phases'].get('search', {}).get(c, 0) for c in COST_FIELDS),
                       localize_clear_s=r['stats']['localize_clear_s'],
                       completion_tail_s=r['stats']['after_last_clear_s'], search_sites=r['stats']['search_sites'],
                       absent_channel_measurements=r['stats'].get('eventually_absent_measurements', 0),
                       clear_failures=r['costs'].get('clear_failure_s', 0)/3,
                       tail_reordered_s=r['tail_counterfactual'].get('reordered_s'),
                       tail_pruned_s=r['tail_counterfactual'].get('pruned_s'))
            writer.writerow(row)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', type=Path, nargs='+')
    p.add_argument('--expected', type=int, required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--skip-tail-counterfactual', action='store_true')
    args = p.parse_args()
    if args.workers < 1 or args.expected < 1:
        p.error('Invalid case or process count')
    folders, input_batches, ignored_markers = [], [], []
    current = manifest()
    for directory in args.inputs:
        state = json.loads((directory/'batch.json').read_text())
        cfg = json.loads((directory/'config.json').read_text())
        assert state['status'] != 'running', 'Wait for the active experiment to stop before final attribution'
        for key, digest in current.items():
            assert cfg['code_sha256'][key] == digest, f'Algorithm changed: {key}'
        input_batches.append({'path': str(directory), 'state': state})
        for folder in sorted(directory.glob('practice-*'), key=lambda f: int(f.name.split('-')[-1])):
            marker = folder/'STOP_REQUESTED.json'
            if marker.exists() and not (folder/'summary.json').exists():
                ignored_markers.append(str(marker))
                continue
            assert (folder/'summary.json').exists(), f'Unfinished or unentered case: {folder}'
            folders.append(folder)
    assert len(folders) == args.expected, (len(folders), args.expected)
    out = ROOT/'results'/datetime.now().strftime('practice-attribution-%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'config.json', {'inputs': input_batches, 'expected_cases': args.expected,
         'workers': args.workers, 'ignored_unstarted_stop_markers': ignored_markers,
         'algorithm_sha256': current, 'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'counterfactual_tail': not args.skip_tail_counterfactual,
         'original_batch_errors_preserved': True})
    state = {'status': 'running', 'expected_cases': args.expected, 'completed_cases': 0}
    dump(out/'batch.json', state)
    print(f'Attribution: {out}', flush=True)
    cases = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(audit_case, str(folder), not args.skip_tail_counterfactual): folder for folder in folders}
            for future in as_completed(futures):
                case = future.result()
                assert all(r['case_code'] != case['case_code'] for r in cases)
                cases.append(case)
                dump(out/f'{case["case_code"]}.json', case)
                state['completed_cases'] = len(cases)
                dump(out/'batch.json', state)
                print(f'Audited {len(cases)}/{len(folders)}: {case["case_code"]}', flush=True)
        case_order = {str(f): i for i, f in enumerate(folders)}
        cases.sort(key=lambda r: case_order[r['folder']])
        dump(out/'cases.json', cases)
        dump(out/'summary.json', aggregate(cases))
        write_csv(out/'cases.csv', cases)
        state['status'] = 'completed'
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        dump(out/'batch.json', state)
        print(f'Results: {out}', flush=True)


if __name__ == '__main__':
    main()
