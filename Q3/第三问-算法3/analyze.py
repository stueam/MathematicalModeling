"""Summarize complete paired batches, including failures and route diagnostics."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from run import POLICIES, code_manifest, dump, new_output


def analyze(folder, target=None):
    rows = json.loads((folder/'summary.json').read_text())
    manifest = json.loads((folder/'batch.json').read_text())
    groups = defaultdict(list)
    for row in rows:
        groups[row['policy']].append(row)
    result = {'input': str(folder), 'batch_status': manifest['status'], 'policies': {}}
    for name, records in groups.items():
        summary = {'runs': len(records), 'completed': sum(r['all_cleared'] and r['certified_complete'] and not r['error'] for r in records)}
        for field in ('virtual_time_s', 'movement_m', 'real_time_s', 'average_per_cleared_s', 'measure_count', 'failed_clears',
                      'shared_measurements', 'incidental_scans', 'quadrature_fallbacks', 'completion_fallbacks',
                      'target_switches', 'coverage_departures', 'deferred_known_measurements',
                      'after_last_clear_s', 'after_last_clear_m', 'clear_stop_scans', 'refinement_reviews', 'review_rejections',
                      'task_clear_scans', 'clear_sector_services', 'coupling_cuts',
                      'station_adjustments', 'station_adjustments_accepted'):
            data = [r[field] for r in records if field in r and r[field] is not None]
            summary[field] = {'mean': float(np.mean(data)), 'median': float(np.median(data)),
                              'max': float(np.max(data)), 'p95': float(np.quantile(data, .95))} if data else None
        result['policies'][name] = summary
    result['pairs'] = []
    def key(row):
        return row['seed'], row['scenario'], row['error_mode'], row['n_input'], row['radius_input']
    target = target or next((name for name in ('bayes-sector', 'bayes-joint', 'bayes-route-scan', 'bayes-efficient', 'bayes-tsp') if name in groups), 'bayes-tsp')
    result['target_policy'] = target
    main = {key(r): r for r in groups.get(target, [])}
    for name, records in groups.items():
        if name == target:
            continue
        other = {key(r): r for r in records}
        if set(main) != set(other) or manifest['status'] != 'completed':
            result['pairs'].append({'reference': name, 'status': 'incomplete_pair_set'})
            continue
        keys = sorted(main, key=repr)
        a = np.array([main[k]['virtual_time_s'] for k in keys])
        b = np.array([other[k]['virtual_time_s'] for k in keys])
        all_complete = all(main[k]['all_cleared'] and main[k]['certified_complete'] and not main[k]['error'] and
                           other[k]['all_cleared'] and other[k]['certified_complete'] and not other[k]['error'] for k in keys)
        result['pairs'].append({'reference': name, 'paired_runs': len(keys), 'all_complete': all_complete,
            'bayes_wins': int((a < b).sum()), 'mean_delta_s': float((a-b).mean()),
            'reduction_of_means_pct': float(100*(b.mean()-a.mean())/b.mean()) if all_complete else None})
    diagnostics = {}
    for path in folder.glob(f'*-{target}-decisions.json'):
        decisions = json.loads(path.read_text())
        status = Counter(d['status'] for d in decisions)
        tasks = Counter()
        for d in decisions:
            if d['status'] == 'route_decision':
                selected = next((r for r in d['candidates'] if r['action'] == d['selected']),
                                min(d['candidates'], key=lambda item: item['score_s']))
                tasks[selected['task'].split(':')[0]] += 1
        diagnostics[path.name] = {'status': dict(status), 'chosen_tasks': dict(tasks)}
    result['diagnostics'] = diagnostics
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', type=Path, nargs='+')
    parser.add_argument('--output')
    parser.add_argument('--target', choices=POLICIES)
    args = parser.parse_args()
    out = new_output(args.output)
    dump(out/'config.json', {'inputs': list(map(str, args.folders)), 'code_sha256': code_manifest()})
    for index, folder in enumerate(args.folders):
        result = analyze(folder, args.target)
        dump(out/f'{index}-analysis.json', result)
        print(str(folder), result['batch_status'])
        for name, summary in result['policies'].items():
            print(json.dumps({'policy': name, 'completed': summary['completed'], 'runs': summary['runs'],
                **{field: summary[field]['mean'] for field in ('virtual_time_s', 'movement_m', 'real_time_s',
                       'average_per_cleared_s', 'measure_count', 'failed_clears')}}, ensure_ascii=False))
        print(json.dumps(result['pairs'], ensure_ascii=False))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
