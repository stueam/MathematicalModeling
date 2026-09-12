"""Reproducible offline paired evaluation. No network or official simulator client."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import time
import traceback
from datetime import datetime
import numpy as np
from offline import OfflineEnvironment, make_cases
from strategy import build_strategy

ROOT = Path(__file__).resolve().parent


def source_hashes():
    paths = sorted(ROOT.glob('*.py'))+sorted((ROOT/'vendor_b2').glob('*.py'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_case(case, method, strategy_factory=None):
    env = OfflineEnvironment(case)
    checked, envelope_violations = {}, []
    envelope_checks = 0

    def audit_envelopes():
        # Evaluator-only assertion: never returns source truth to the strategy.
        nonlocal envelope_checks
        for c, poly in strategy.polys.items():
            if checked.get(c) is poly:
                continue
            checked[c] = poly
            envelope_checks += 1
            truth = np.asarray(env.targets[c]['position'])
            edges = np.roll(poly, -1, axis=0)-poly
            delta = truth-poly
            cross = edges[:, 0]*delta[:, 1]-edges[:, 1]*delta[:, 0]
            inside = (np.all(cross >= -1e-5) or np.all(cross <= 1e-5))
            inside = inside and np.all(truth >= poly.min(axis=0)-1e-5) and np.all(truth <= poly.max(axis=0)+1e-5)
            if not inside:
                envelope_violations.append({'channel': c, 'action_index': len(strategy.trace)})

    # A plain callback exposes only responses; the strategy never receives env or case.
    def action(path, point=None, channel=None):
        audit_envelopes()
        return env.action(path, point, channel)
    strategy = (strategy_factory or build_strategy)(action, method)
    start = time.perf_counter()
    output, error = {}, None
    try:
        output = strategy.run()
        audit_envelopes()
    except Exception:
        error = traceback.format_exc()
    runtime = time.perf_counter()-start
    total = len(case['targets'])
    false_absent = sorted(set(strategy.absent)&set(env.targets))
    success = error is None and len(env.cleared) == total and not false_absent and not envelope_violations
    row = {'case_id': case['id'], 'family': case['family'], 'method': method,
           'status': 'PASS' if success else 'FAIL', 'sources': total,
           'directional_sources': sum(t.get('heading_deg') is not None for t in case['targets']),
           'cleared': len(env.cleared), 'false_absent': len(false_absent),
           'envelope_checks': envelope_checks, 'envelope_violations': len(envelope_violations),
           'virtual_time_s': env.clock, 'seconds_per_cleared': env.clock/len(env.cleared) if env.cleared else None,
           'runtime_s': runtime, 'movement_m': env.movement, 'measures': env.measures,
           'clear_attempts': env.clear_attempts, 'clear_failures': env.clear_failures,
           'optical_cover_count': strategy.fallback_count,
           'scanned_sites': len(strategy.visited), 'required_sites': len(strategy.sites),
           'shared_measures': strategy.shared_measures, 'actions': len(strategy.trace),
           'deferred_localizations': strategy.deferred_localizations,
           'direction_readings': env.states['direction'], 'no_signal_readings': env.states['no_signal'],
           'completion_reason': output.get('completion_reason'),
           'after_last_clear_s': env.clock-env.clear_times[-1] if env.clear_times else None}
    for name in ('coverage_cells', 'certified_cells', 'retired_sites', 'reused_scans', 'moved_sites', 'certificate_calls', 'negative_pair_cuts', 'history_pair_cuts'):
        row[name] = output.get(name, 0)
    failure = None if success else {'error': error, 'false_absent_channels': false_absent,
                                  'envelope_violations': envelope_violations,
                                  'uncleared_channels': sorted(set(env.targets)-env.cleared), 'trace': strategy.trace}
    return row, failure


def summarize(rows, methods):
    result = {}
    for method in methods:
        records = [r for r in rows if r['method'] == method]
        ok = all(r['status'] == 'PASS' for r in records)
        result[method] = {'cases': len(records), 'passed': sum(r['status'] == 'PASS' for r in records),
                          'sources': sum(r['sources'] for r in records),
                          'cleared': sum(r['cleared'] for r in records),
                          'false_absent': sum(r['false_absent'] for r in records),
                          'envelope_checks': sum(r['envelope_checks'] for r in records),
                          'envelope_violations': sum(r['envelope_violations'] for r in records),
                          'total_runtime_s': sum(r['runtime_s'] for r in records),
                          'mean_runtime_s': float(np.mean([r['runtime_s'] for r in records])),
                          'max_runtime_s': max(r['runtime_s'] for r in records),
                          'optical_cover_count': sum(r['optical_cover_count'] for r in records)}
        for name in ('virtual_time_s', 'seconds_per_cleared', 'movement_m', 'measures', 'clear_failures', 'after_last_clear_s'):
            result[method]['mean_'+name] = float(np.mean([r[name] for r in records])) if ok else None
        result[method]['p90_seconds_per_cleared'] = float(np.percentile([r['seconds_per_cleared'] for r in records], 90)) if ok else None
        result[method]['max_seconds_per_cleared'] = max(r['seconds_per_cleared'] for r in records) if ok else None
    paired = None
    if {'q4_b2', 'q4_safe'} <= set(methods) and all(r['status'] == 'PASS' for r in rows):
        a = {r['case_id']: r for r in rows if r['method'] == 'q4_b2'}
        b = {r['case_id']: r for r in rows if r['method'] == 'q4_safe'}
        paired = {'main_faster_cases': sum(a[k]['virtual_time_s'] < b[k]['virtual_time_s']-1e-6 for k in a),
                  'main_slower_cases': sum(a[k]['virtual_time_s'] > b[k]['virtual_time_s']+1e-6 for k in a),
                  'mean_per_source_improvement_percent': 100*(1-result['q4_b2']['mean_seconds_per_cleared']/result['q4_safe']['mean_seconds_per_cleared'])}
    return result, paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2026091104)
    parser.add_argument('--count', type=int, default=28)
    parser.add_argument('--methods', nargs='+', choices=['q4_b2', 'q4_safe'], default=['q4_b2', 'q4_safe'])
    parser.add_argument('--scenarios', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'results/Q4/experiments/round1')
    args = parser.parse_args()
    out = args.output.resolve()
    if (out/'run_summary.json').exists():
        raise FileExistsError('Use a new output directory to preserve experiment evidence')
    for folder in ('metrics', 'tables'):
        (out/folder).mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.scenarios.read_text(encoding='utf-8')) if args.scenarios else make_cases(args.seed, args.count)
    if not cases:
        raise ValueError('No scenarios')
    scenario_path = out/'metrics/scenarios.json'
    scenario_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    for case in cases:
        for method in args.methods:
            row, failure = run_case(case, method)
            rows.append(row)
            if failure:
                (out/'logs').mkdir(exist_ok=True)
                (out/'logs'/f"{case['id']}_{method}.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding='utf-8')
            with (out/'tables/cases.csv').open('w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"{case['id']} {method}: {row['status']} {row['cleared']}/{row['sources']} {row['virtual_time_s']:.2f}s virtual {row['runtime_s']:.3f}s compute", flush=True)
    metrics, paired = summarize(rows, args.methods)
    summary = {'question': 'Q4', 'round': out.name, 'decision_id': 'q4_b2_adaptation',
               'approved_methods': args.methods, 'roles': {'q4_b2': 'main_candidate', 'q4_safe': 'usable_baseline'},
               'status': 'PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL',
               'seed': args.seed if args.scenarios is None else None,
               'inputs': {'scenario_path': str(scenario_path), 'sha256': hashlib.sha256(scenario_path.read_bytes()).hexdigest()},
               'outputs': ['tables/cases.csv', 'metrics/scenarios.json'],
               'metric_summary': metrics, 'paired': paired,
               'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()},
               'source_hashes': source_hashes(), 'created_at': datetime.now().astimezone().isoformat(),
               'warnings': ['Offline synthetic cases only; no official simulator used.',
                            'All-directional and all-omni families are diagnostic extremes, not mixed-type official-format cases.',
                            'Timing means weight scenarios equally; successful runs do not establish global optimality.'],
               'fallback_trigger_state': {'finite_optical_cover_uses': {k: v['optical_cover_count'] for k, v in metrics.items()}}}
    (out/'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'metrics': metrics, 'paired': paired}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
