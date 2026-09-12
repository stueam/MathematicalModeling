"""Reproducible offline comparison. No network or official simulator access."""
import argparse
import csv
import datetime
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from offline_environment import StressEnvironment, make_cases
from solver import build_strategy, LATEST_CONFIG, VALIDATED_CONFIG

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=['latest', 'b2', 'validated', 'mec', 'q2'],
                        default=['latest', 'b2'])
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--scenarios', type=Path)
    source.add_argument('--generate-seed', type=int, help='Generate all 140 cases with a new seed')
    parser.add_argument('--q2-fine-power', type=int, choices=range(3,11))
    parser.add_argument('--q2-residual-scale', type=float)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/benchmark')
    args = parser.parse_args()
    if args.q2_residual_scale is not None and (not np.isfinite(args.q2_residual_scale) or args.q2_residual_scale<=0):
        parser.error('Residual scale must be finite and positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.generate_seed is not None:
        cases=make_cases(args.generate_seed)
        args.scenarios=args.output/'scenarios.json'
        args.scenarios.write_text(json.dumps(cases,indent=2)+'\n',encoding='utf-8')
    else:
        args.scenarios=args.scenarios or ROOT/'scenarios/screening20.json'
        cases = json.loads(args.scenarios.read_text(encoding='utf-8'))
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Cases must have distinct IDs and must not be empty')
    for case in cases:
        targets = case['targets']
        if not 10 <= len(targets) <= 16 or len({t['channel'] for t in targets}) != len(targets):
            raise ValueError('Expected 10..16 sources on distinct channels')
        for t in targets:
            if not (1 <= t['channel'] <= 20 and 1000 <= t['radius'] <= 1500
                    and np.isfinite(t['position']).all() and np.linalg.norm(t['position']) <= 1800.000001):
                raise ValueError('Source violates scenario contract')
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.perf_counter()
    for method in args.methods:
        for case in cases:
            env = StressEnvironment(case)
            # Ground truth is held only by the evaluator; the strategy gets this callback.
            model = build_strategy(env.action, method)
            if method=='q2':
                if args.q2_fine_power is not None:model.q2_config['fine_power']=args.q2_fine_power
                if args.q2_residual_scale is not None:model.q2_config['residual_scale']=args.q2_residual_scale
            tick = time.perf_counter()
            failure = None
            try:
                model.run()
                if env.cleared != set(env.targets) or not model.absent.isdisjoint(env.targets):
                    raise AssertionError('Missed source or false absence')
            except Exception as exc:
                failure = repr(exc)
                logs = args.output/'failures'
                logs.mkdir(exist_ok=True)
                (logs/f'{method}_{case["id"]}.json').write_text(
                    json.dumps(model.trace, indent=2)+'\n', encoding='utf-8')
            rows.append(dict(method=method, case=case['id'], family=case['family'],
                             sources=len(env.targets), cleared=len(env.cleared),
                             virtual_time_s=env.clock,
                             s_per_source=env.clock/len(env.cleared) if env.cleared else None,
                             movement_m=env.movement, measurements=env.measures,
                             failed_clears=env.clear_attempts-len(env.cleared),
                             runtime_s=time.perf_counter()-tick,
                             planning_calls=getattr(model,'planning_calls',0),
                             planning_time_s=getattr(model,'planning_time_s',0.),
                             cover_searches=model.fallback_count, failure=failure))
            if getattr(model,'planning_log',None):
                logs=args.output/'planning'
                logs.mkdir(exist_ok=True)
                (logs/f'{method}_{case["id"]}.json').write_text(
                    json.dumps(model.planning_log,indent=2)+'\n',encoding='utf-8')
            if len(rows)%10==0:
                print(f'{method}: case {case["id"]}, elapsed {time.perf_counter()-start:.1f}s',flush=True)
        print(f'{method}: {len(cases)} cases completed', flush=True)
    with (args.output/'cases.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {}
    for method in args.methods:
        rs = [r for r in rows if r['method'] == method]
        ok = all(r['failure'] is None for r in rs)
        values = np.array([r['s_per_source'] for r in rs]) if ok else None
        metrics[method] = dict(cases=len(rs), full_clear_cases=sum(r['failure'] is None for r in rs),
            total_sources=sum(r['sources'] for r in rs), total_cleared=sum(r['cleared'] for r in rs),
            mean_s_per_source=float(values.mean()) if ok else None,
            median_s_per_source=float(np.median(values)) if ok else None,
            p90_s_per_source=float(np.quantile(values,.9)) if ok else None,
            worst_s_per_source=float(values.max()) if ok else None,
            cases_under200=int((values < 200).sum()) if ok else None)
    from q2_lookahead import Q2_CONFIG
    q2_config=dict(Q2_CONFIG)
    if args.q2_fine_power is not None:q2_config['fine_power']=args.q2_fine_power
    if args.q2_residual_scale is not None:q2_config['residual_scale']=args.q2_residual_scale
    summary = dict(status='PASS' if all(r['failure'] is None for r in rows) else 'FAIL',
        scope=('Fresh seeded simulation comparison, same scenario generator; not official evaluation'
               if args.generate_seed is not None else 'Development screening or supplied cases; not official evaluation'),
        generation_seed=args.generate_seed,
        metric_definition='Unweighted mean of each case total virtual seconds / successfully cleared sources; null on any failure',
        methods=metrics, configurations=dict(latest=LATEST_CONFIG, validated=VALIDATED_CONFIG,
                                            b2=dict(stop_at_max=True),
                                            mec=dict(base=LATEST_CONFIG,circle='minimum'),
                                            q2=dict(base=LATEST_CONFIG,**q2_config)),
        scenario_sha256=hashlib.sha256(args.scenarios.read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))},
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        elapsed_s=time.perf_counter()-start, completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        network_requests=0, formal_tests_used=0)
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2))
    return 0 if summary['status'] == 'PASS' else 1

if __name__ == '__main__':
    raise SystemExit(main())
