"""Paired offline comparison against the previous Q4 B2 implementation."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from datetime import datetime
import numpy as np
from adaptive_strategy import AdaptiveB2
from strategy import build_strategy
from offline import make_cases
from benchmark import run_case, source_hashes

ROOT = Path(__file__).resolve().parent


def factory(action, method):
    return AdaptiveB2(action) if method == 'q4_adaptive' else build_strategy(action, method)


def stats(rows):
    output = {}
    for method in sorted(set(r['method'] for r in rows)):
        group = [r for r in rows if r['method'] == method]
        good = all(r['status'] == 'PASS' for r in group)
        output[method] = {'cases': len(group), 'passed': sum(r['status'] == 'PASS' for r in group),
                          'sources': sum(r['sources'] for r in group), 'cleared': sum(r['cleared'] for r in group),
                          'false_absent': sum(r['false_absent'] for r in group),
                          'envelope_checks': sum(r['envelope_checks'] for r in group),
                          'envelope_violations': sum(r['envelope_violations'] for r in group)}
        for key in ('seconds_per_cleared', 'virtual_time_s', 'runtime_s', 'movement_m', 'measures',
                    'after_last_clear_s', 'scanned_sites', 'retired_sites', 'reused_scans', 'moved_sites',
                    'clear_failures', 'optical_cover_count', 'certificate_calls', 'negative_pair_cuts', 'history_pair_cuts'):
            output[method]['mean_'+key] = float(np.mean([r[key] for r in group])) if good else None
        output[method]['max_runtime_s'] = max(r['runtime_s'] for r in group)
    paired = None
    if {'q4_adaptive', 'q4_b2'} <= set(output) and all(r['status'] == 'PASS' for r in rows):
        a = {r['case_id']: r for r in rows if r['method'] == 'q4_adaptive'}
        b = {r['case_id']: r for r in rows if r['method'] == 'q4_b2'}
        paired = {'faster': sum(a[k]['virtual_time_s'] < b[k]['virtual_time_s']-1e-6 for k in a),
                  'slower': sum(a[k]['virtual_time_s'] > b[k]['virtual_time_s']+1e-6 for k in a),
                  'improvement_percent': 100*(1-output['q4_adaptive']['mean_seconds_per_cleared']/output['q4_b2']['mean_seconds_per_cleared'])}
    return {'methods': output, 'paired': paired}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2026091104)
    parser.add_argument('--count', type=int, default=14)
    parser.add_argument('--scenarios', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if (out/'run_summary.json').exists():
        raise FileExistsError('Choose a new output directory')
    for child in ('metrics', 'tables'):
        (out/child).mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.scenarios.read_text(encoding='utf-8')) if args.scenarios else make_cases(args.seed, args.count)
    (out/'metrics/scenarios.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    started_hashes = source_hashes()
    rows = []
    for case in cases:
        for method in ('q4_adaptive', 'q4_b2'):
            row, failure = run_case(case, method, factory)
            rows.append(row)
            if failure:
                (out/'logs').mkdir(exist_ok=True)
                (out/'logs'/f"{case['id']}_{method}.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding='utf-8')
            with (out/'tables/cases.csv').open('w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"{case['id']} {method} {row['status']} {row['cleared']}/{row['sources']} {row['seconds_per_cleared']:.2f}s/source runtime={row['runtime_s']:.2f}s scans={row['scanned_sites']} replaced={row['retired_sites']}", flush=True)
    mixed = [r for r in rows if 0 < r['directional_sources'] < r['sources']]
    summary = {'question': 'Q4', 'round': out.name, 'decision_id': 'q4_adaptive_search_optimization',
               'status': 'PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL',
               'roles': {'q4_adaptive': 'main_candidate', 'q4_b2': 'usable_baseline'},
               'all_cases': stats(rows), 'mixed_cases': stats(mixed),
               'by_family': {family: stats([r for r in rows if r['family'] == family]) for family in sorted(set(r['family'] for r in rows))},
               'seed': args.seed if args.scenarios is None else None,
               'input_sha256': hashlib.sha256((out/'metrics/scenarios.json').read_bytes()).hexdigest(),
               'source_hashes': started_hashes, 'source_unchanged_during_run': started_hashes == source_hashes(),
               'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()},
               'created_at': datetime.now().astimezone().isoformat(),
               'outputs': ['tables/cases.csv', 'metrics/scenarios.json'],
               'warnings': ['Local synthetic cases only; no official simulator or HTTP connection.',
                            'Completion uses actual scan history, not future planned scans.',
                            'Coverage cells are checked continuously; no posterior mass cutoff is used.',
                            'Runtime covers strategy run() plus evaluator checks, excluding imports and constructor.']}
    if not summary['source_unchanged_during_run']:
        summary['status'] = 'FAIL_SOURCE_CHANGED'
    (out/'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary['mixed_cases'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
