"""Offline Q4 diagnostics; reads saved actions only, never accesses HTTP.

The absent-reading deletion experiment uses hindsight channel identities and
recorded future positions. It is a feasible fixed-action replay, NOT an online
policy score, an optimum, or a guarantee of achievable savings.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from q4.core import Belief, DOMAIN
from q4.coverage import certifies
from q4.shared import Action, SHARED_DIR, distance


ROOT = Path(__file__).resolve().parent


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def describe(values):
    a = np.asarray(values, dtype=float)
    return ({'n': len(a), 'sum': float(a.sum()), 'mean': float(a.mean()),
             'median': float(np.median(a)), 'p90': float(np.quantile(a, .9)),
             'max': float(a.max())} if len(a) else {'n': 0})


def replay(rows, retained=None):
    b, cost = Belief(), Counter()
    executed = []
    for i, row in enumerate(rows):
        if retained is not None and i not in retained:
            continue
        if b.done():
            break
        raw = row['action']
        a = Action(raw['kind'], tuple(raw['position']), raw['channel'])
        response = dict(row['response'])
        move = round(distance(b.position, a.position) / 5, 6)
        switch = int(a.kind == 'measure' and a.channel != b.receiver)
        operation = 5 if a.kind == 'measure' or response.get('clear_result') == 'success' else 3
        response['virtual_time_s'] = round(b.virtual_time + move + switch + operation, 6)
        if retained is None:
            assert abs(response['virtual_time_s'] - row['response']['virtual_time_s']) < 1e-5
        b.apply(a, response, f'offline-{i}')
        cost.update(move_s=move, switch_s=switch, operation_s=operation)
        executed.append(i)
    assert b.done()
    return {'virtual_time_s': b.virtual_time, 'costs': dict(cost),
            'complete': b.done(), 'retained_original_steps': executed,
            'cleared_count': len(b.cleared)}


def prune_absent(rows):
    present = {r['action']['channel'] for r in rows if r['response'].get('clear_result') == 'success'}
    absent = set(range(1, 21)) - present
    points, groups = defaultdict(set), defaultdict(list)
    for i, r in enumerate(rows):
        c = r['action']['channel']
        if c in absent:
            assert r['response'].get('measure_result') == 'no_signal'
            p = tuple(r['action']['position'])
            points[c].add(p)
            groups[p].append(i)
    before = replay(rows)
    retained = set(range(len(rows)))
    checks, removed = 0, []
    if len(present) == 16:
        for ids in groups.values():
            retained.difference_update(ids)
    else:
        # Order by initial direct shortcut plus measurement savings. A failed
        # deletion cannot become valid after removing further negative evidence.
        def priority(p):
            ids = groups[p]
            left, right = min(ids) - 1, max(ids) + 1
            prev = (0., 0.) if left < 0 else tuple(rows[left]['action']['position'])
            gain = distance(prev, p)
            if right < len(rows):
                nxt = tuple(rows[right]['action']['position'])
                gain += distance(p, nxt) - distance(prev, nxt)
            return gain / 5 + 6 * len(ids)
        for p in sorted(groups, key=priority, reverse=True):
            signatures, valid = set(), True
            for c in absent:
                trial = tuple(sorted(points[c] - {p}))
                if trial in signatures:
                    continue
                signatures.add(trial)
                checks += 1
                if not certifies(DOMAIN, trial):
                    valid = False
                    break
            if valid:
                retained.difference_update(groups[p])
                for c in absent:
                    points[c].discard(p)
                removed.append(p)
    after = replay(rows, retained)
    assert after['cleared_count'] == len(present)
    return {'hindsight_only': True, 'certificate_checks': checks,
            'removed_absent_scan_positions': removed,
            'removed_actions': len(rows) - len(after['retained_original_steps']),
            'saved_s': before['virtual_time_s'] - after['virtual_time_s'],
            'before_s': before['virtual_time_s'], 'after': after,
            'saved_movement_s': before['costs']['move_s'] - after['costs']['move_s'],
            'saved_operation_and_switch_s': sum(before['costs'][k] - after['costs'][k]
                                                for k in ('switch_s', 'operation_s'))}


def analyze(folder, prune=False):
    rows = [json.loads(s) for s in (folder / 'actions.jsonl').read_text().splitlines()]
    decisions = json.loads((folder / 'decisions.json').read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    assert len(rows) == len(decisions)
    total = rows[-1]['response']['virtual_time_s']
    present = {r['action']['channel'] for r in rows if r['response'].get('clear_result') == 'success'}
    assert summary['certified_complete'] and len(present) == summary['cleared_count']
    known, cleared, scan_sites = set(), set(), set()
    channels = defaultdict(Counter)
    phases, radial = defaultdict(Counter), defaultdict(Counter)
    probes, proxies, long_moves, milestones = [], [], [], []
    t, position, receiver = 0., (0., 0.), 1
    max_known = 0
    for step, (r, d) in enumerate(zip(rows, decisions)):
        assert d['step'] == step and d['action'] == r['action']
        a, response = r['action'], r['response']
        c, p = a['channel'], tuple(a['position'])
        move = distance(position, p)
        switch = int(a['kind'] == 'measure' and receiver != c)
        op = 5 if a['kind'] == 'measure' or response.get('clear_result') == 'success' else 3
        elapsed = round(move / 5, 6) + switch + op
        assert abs(t + elapsed - response['virtual_time_s']) < 1e-5
        phase = 'search' if a['kind'] == 'measure' and c not in known else 'localize_clear'
        phases[phase].update(move_s=move / 5, switch_s=switch, operation_s=op, actions=1)
        if move >= 1000:
            long_moves.append({'step': step, 'distance_m': move, 'purpose': phase,
                               'known': len(known), 'cleared': len(cleared), 'reason': d['reason']})
        chosen = next((x for x in d.get('candidates', []) if x['action'] == a), None)
        if chosen and 'score_s' in chosen:
            ranking = sorted(x['score_s'] for x in d['candidates'])
            proxies.append({'step': step, 'time_fraction': t / total,
                            'error_s': chosen['score_s'] - (total - t),
                            'candidate_gap_s': ranking[1] - ranking[0] if len(ranking) > 1 else None,
                            'task_count': len(d.get('planned_route', [])), 'selected_kind': phase})
        if a['kind'] == 'measure':
            receiver = c
            outcome = response['measure_result']
            channels[c]['measurements'] += 1
            channels[c]['post_signal_measurements' if c in known else 'pre_signal_measurements'] += 1
            channels[c]['post_signal_misses' if c in known else 'pre_signal_misses'] += int(outcome == 'no_signal')
            if phase == 'search':
                scan_sites.add(p)
                band = 'origin' if distance(p, (0., 0.)) < 1e-6 else ('inside' if np.hypot(*p) <= 1800 else 'outside')
                radial[band].update(measurements=1, move_s=move / 5, operation_switch_s=5 + switch,
                                    discoveries=int(outcome != 'no_signal'),
                                    absent_measurements=int(c not in present))
                if outcome != 'no_signal':
                    known.add(c)
                    channels[c].update(first_signal_step=step, first_signal_s=response['virtual_time_s'],
                                       first_signal_radius_m=float(np.hypot(*p)), first_signal_site=len(scan_sites))
            elif chosen and 'no_signal_probability' in chosen:
                probes.append({'step': step, 'channel': c, 'p_no_signal': chosen['no_signal_probability'],
                               'miss': int(outcome == 'no_signal'), 'move_m': move,
                               'local_proxy_s': chosen['local_proxy_s']})
        else:
            channels[c]['clear_attempts'] += 1
            channels[c]['clear_failures'] += int(response['clear_result'] != 'success')
            if response['clear_result'] == 'success':
                cleared.add(c)
                channels[c].update(clear_step=step, clear_s=response['virtual_time_s'])
        if phase == 'localize_clear':
            channels[c]['local_s'] += elapsed
            channels[c]['local_movement_m'] += move
        max_known = max(max_known, len(known - cleared))
        milestones.append({'time_s': response['virtual_time_s'], 'known': len(known), 'cleared': len(cleared),
                           'scan_sites': len(scan_sites)})
        t, position = response['virtual_time_s'], p
    per_progress = {}
    for frac in (.25, .5, .75, .9):
        eligible = [m for m in milestones if m['time_s'] <= frac * total]
        per_progress[str(frac)] = eligible[-1] if eligible else {'known': 0, 'cleared': 0, 'scan_sites': 0}
    for c in present:
        channels[c]['detection_to_clear_s'] = channels[c]['clear_s'] - channels[c]['first_signal_s']
    last_discovery = max(channels[c]['first_signal_s'] for c in present)
    last_clear = max(channels[c]['clear_s'] for c in present)
    outcome = {'case_code': summary['case_code'], 'folder': str(folder.relative_to(ROOT)),
               'sources': len(present), 'time_s': total, 's_per_source': total / len(present),
               'phases': dict(phases), 'radial': dict(radial), 'channels': dict(channels),
               'probes': probes, 'proxies': proxies, 'long_moves': long_moves,
               'milestones': per_progress, 'max_detected_backlog': max_known,
               'last_discovery_s': last_discovery, 'last_clear_s': last_clear,
               'after_last_discovery_s': total - last_discovery,
               'after_last_clear_s': total - last_clear,
               'original_replay': replay(rows)}
    if prune:
        outcome['absent_prune'] = prune_absent(rows)
    return outcome


def aggregate(cases):
    probes = [p for c in cases for p in c['probes']]
    source_channels = [v for c in cases for v in c['channels'].values() if 'clear_s' in v]
    bands = {}
    for b in ('origin', 'inside', 'outside'):
        sums = Counter()
        for c in cases:
            sums.update(c['radial'].get(b, {}))
        bands[b] = dict(sums)
    grouped = {}
    for low, high in ((0., .25), (.25, .5), (.5, .75), (.75, 1.01)):
        ps = [p for c in cases for p in c['proxies'] if low <= p['time_fraction'] < high]
        grouped[str(low)] = describe([p['error_s'] for p in ps])
    return {'cases': len(cases), 'sources': sum(c['sources'] for c in cases),
            'time_s': describe([c['time_s'] for c in cases]),
            's_per_source': describe([c['s_per_source'] for c in cases]),
            'radial': bands, 'proxy_error_by_time_quartile': grouped,
            'after_last_discovery_s': describe([c['after_last_discovery_s'] for c in cases]),
            'after_last_clear_s': describe([c['after_last_clear_s'] for c in cases]),
            'detection_to_clear_s': describe([c['detection_to_clear_s'] for c in source_channels]),
            'post_signal_measurements': describe([c['post_signal_measurements'] for c in source_channels]),
            'post_signal_misses': sum(c['post_signal_misses'] for c in source_channels),
            'local_s': describe([c['local_s'] for c in source_channels]),
            'probe_calibration': {'n': len(probes), 'expected_misses': sum(p['p_no_signal'] for p in probes),
                                  'actual_misses': sum(p['miss'] for p in probes),
                                  'brier': float(np.mean([(p['p_no_signal'] - p['miss']) ** 2 for p in probes])) if probes else None},
            'progress': {str(f): {k: describe([c['milestones'][str(f)][k] for c in cases])
                                  for k in ('known', 'cleared', 'scan_sites')} for f in (.25, .5, .75, .9)},
            'long_moves': describe([p['distance_m'] for c in cases for p in c['long_moves']]),
            'absent_prune_saved_s': describe([c['absent_prune']['saved_s'] for c in cases if 'absent_prune' in c]),
            'absent_prune_removed_actions': describe([c['absent_prune']['removed_actions'] for c in cases if 'absent_prune' in c])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prune-absent', action='store_true')
    parser.add_argument('--local-reference', action='store_true',
                        help='Separately audit existing probes seed 800..815 logs; no new simulation')
    args = parser.parse_args()
    attribution = ROOT / 'results/practice-attribution-20260911-193651-437807/cases.json'
    inputs = json.loads(attribution.read_text())
    out = ROOT / 'results' / ('official-diagnostic-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    out.mkdir()
    files = [ROOT / x['folder'] / name for x in inputs for name in ('actions.jsonl', 'decisions.json', 'summary.json')]
    local_root = ROOT / 'results/20260911-214856-359889'
    if args.local_reference:
        files += sorted(local_root.glob('*-probes-actions.jsonl'))
        files += [local_root / 'summary.json']
    code = [Path(__file__)] + sorted((ROOT / 'q4').glob('*.py'))
    code += sorted(SHARED_DIR.glob('*.py'))
    dump(out / 'config.json', {'prune_absent': args.prune_absent, 'local_reference': args.local_reference,
         'inputs': [str(p.relative_to(ROOT)) for p in files],
         'sha256': {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files + code},
         'interpretation': 'Historical mobile only. Full-history absent pruning uses hindsight, not online policy performance.'})
    dump(out / 'batch.json', {'status': 'running', 'expected': len(inputs)})
    cases = []
    for item in inputs:
        c = analyze(ROOT / item['folder'], args.prune_absent)
        cases.append(c)
        dump(out / (c['case_code'] + '.json'), c)
        print(c['case_code'], round(c['time_s'], 2), 'prune_saved', round(c.get('absent_prune', {}).get('saved_s', 0), 2), flush=True)
    dump(out / 'summary.json', aggregate(cases))
    if args.local_reference:
        local = []
        for s in json.loads((local_root / 'summary.json').read_text()):
            if s['policy'] != 'probes':
                continue
            rows = [json.loads(line) for line in (local_root / (s['case_id'] + '-actions.jsonl')).read_text().splitlines()]
            result = {'case_id': s['case_id'], 'source_count': s['source_count'],
                      'costs': s['costs'], 'counters': s['counters'], 'prune': prune_absent(rows)}
            local.append(result)
        assert len(local) == 16
        dump(out / 'local-reference.json', {'source': str(local_root.relative_to(ROOT)),
             'cases': local, 'saved_s': describe([c['prune']['saved_s'] for c in local]),
             'saved_s_excluding_n16': describe([c['prune']['saved_s'] for c in local if c['source_count'] < 16])})
    dump(out / 'batch.json', {'status': 'completed', 'expected': len(inputs), 'completed': len(cases)})
    print(out, flush=True)


if __name__ == '__main__':
    main()
