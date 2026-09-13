"""Local Q4 worlds and complete paired experiments. No official HTTP entry."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import shapely

from q4.core import Belief
from q4.coverage import DIAGNOSTICS
from q4.policy import Config
from q4.compact import POLICIES, make_policy
from q4.shared import SHARED_DIR, distance
from q4.simulator import LocalSimulator, SCENARIOS, generate_world

ROOT = Path(__file__).resolve().parent


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def manifest():
    files = sorted(ROOT.glob('*.py')) + sorted((ROOT/'q4').glob('*.py'))
    files += sorted(SHARED_DIR.glob('*.py'))
    files += [ROOT/'s21_certificate.json']
    return {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def case_id(job):
    return f"{job['scenario']}-{job['error_mode']}-n{job['n']}-r{job['radius']}-d{job['directional_count']}-{job['seed']}-{job['policy']}"


def run_case(job, out):
    tag = case_id(job)
    diagnostics_before = dict(DIAGNOSTICS)
    out = Path(out)
    world = generate_world(job['seed'], job['n'], job['scenario'], job['radius'], job['error_mode'], job['directional_count'])
    # Evaluation-only world line. Policy receives neither this file nor World.
    dump(out/f'{tag}-world.json', world.manifest())
    env, belief = LocalSimulator(world), Belief()
    policy = make_policy(job['policy'], Config(**job['config']))
    started = time.monotonic()
    belief.deadline = started+job['real_limit']
    planning, actions = [], []
    reason, error = 'action_limit', None
    with (out/f'{tag}-actions.jsonl').open('w', encoding='utf-8') as log, \
         (out/f'{tag}-decisions.jsonl').open('w', encoding='utf-8') as decision_log:
        try:
            for i in range(job['max_steps']):
                if belief.done():
                    reason = 'certified_complete'
                    break
                if time.monotonic() >= belief.deadline:
                    reason = 'real_time_limit'
                    break
                if belief.virtual_time >= belief.virtual_limit:
                    reason = 'virtual_time_limit'
                    break
                tick = time.monotonic()
                action = policy.choose(belief)
                planning.append(time.monotonic()-tick)
                if time.monotonic() >= belief.deadline:
                    reason = 'real_time_limit'
                    break
                request_id = f'local-{i}'
                response = env.execute(action, request_id)
                row = {'request_id': request_id, 'action': asdict(action), 'response': response,
                       'planning_s': planning[-1], 'move_m': distance(belief.position, action.position)}
                actions.append(row)
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                log.flush()
                belief.apply(action, response, request_id)
                if policy.records:
                    decision_log.write(json.dumps({'request_id':request_id,**policy.records[-1]},
                                                  ensure_ascii=False,allow_nan=False)+'\n')
                    decision_log.flush()
        except Exception as exc:
            error, reason = f'{type(exc).__name__}: {exc}', 'error'
    elapsed = time.monotonic()-started
    if belief.done():
        reason = 'certified_complete'
    successes = [i for i, a in enumerate(actions) if a['response'].get('clear_result') == 'success']
    last = successes[-1] if successes else None
    evaluation = world.score()
    summary = {'case_id': tag, **job, **evaluation, 'implementation': policy.implementation,
               'certified_complete': belief.done(), 'stop_reason': reason, 'error': error,
               'virtual_time_s': belief.virtual_time, 'movement_m': sum(a['move_m'] for a in actions),
               'real_time_s': elapsed, 'steps': belief.steps, 'costs': env.costs,
               'measure_count': sum(a['action']['kind'] == 'measure' for a in actions),
               'failed_clears': sum(a['response'].get('clear_result') == 'no_target_in_range' for a in actions),
               'per_cleared_s': belief.virtual_time/len(belief.cleared) if belief.cleared else None,
               'after_last_clear_s': belief.virtual_time-actions[last]['response']['virtual_time_s'] if last is not None else None,
               'planning_p95_s': float(np.quantile(planning, .95)) if planning else 0.,
               'counters': policy.counters, 'channel_status': {c: p.status for c, p in belief.channels.items()}}
    summary['geometry_diagnostics'] = {k: v-diagnostics_before[k] for k, v in DIAGNOSTICS.items()}
    dump(out/f'{tag}-summary.json', summary)
    dump(out/f'{tag}-decisions.json', policy.records)
    return summary


def aggregate(rows):
    result = {}
    for policy in sorted({r['policy'] for r in rows}):
        subset = [r for r in rows if r['policy'] == policy]
        result[policy] = {'cases': len(subset),
                          'fully_completed': sum(r['all_cleared'] and r['certified_complete'] for r in subset),
                          'failures': [r['case_id'] for r in subset if not (r['all_cleared'] and r['certified_complete'])],
                          **{f'mean_{key}': float(np.mean([r[key] for r in subset])) for key in
                             ('virtual_time_s', 'movement_m', 'real_time_s', 'measure_count', 'failed_clears')}}
        complete = [r for r in subset if r['all_cleared'] and r['certified_complete']]
        result[policy]['mean_s_per_source'] = (float(np.mean([r['per_cleared_s'] for r in subset]))
            if len(complete) == len(subset) else None)
        result[policy]['weighted_s_per_source'] = (sum(r['virtual_time_s'] for r in subset)
            /sum(r['source_count'] for r in subset) if len(complete) == len(subset) else None)
        result[policy]['complete_under_350_s_per_source'] = sum(r['per_cleared_s'] <= 350 for r in complete)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('local', 'benchmark', 'validate'))
    p.add_argument('--policy', choices=POLICIES, default='probes')
    p.add_argument('--compare', nargs='+', choices=POLICIES, default=['mobile', 'probes'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rounds', type=int, default=6)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--scenario', choices=SCENARIOS, default='uniform')
    p.add_argument('--error-mode', choices=('iid', 'extreme', 'correlated'), default='iid')
    p.add_argument('--n', type=int)
    p.add_argument('--radius', type=float)
    p.add_argument('--directional-count', type=int)
    p.add_argument('--resolution', type=int, default=16)
    p.add_argument('--bearing-bin', type=float, default=2.)
    p.add_argument('--directional-prior', type=float, default=.5)
    p.add_argument('--exact-limit', type=int, default=16, choices=range(17))
    p.add_argument('--real-limit', type=float, default=1200.)
    p.add_argument('--max-steps', type=int, default=6000)
    p.add_argument('--output', help='New output directory; existing directories are rejected')
    args = p.parse_args(argv)
    if args.workers < 1 or args.rounds < 1 or args.real_limit <= 0 or args.max_steps < 1:
        p.error('Invalid execution limits')
    config = Config(resolution=args.resolution, bearing_bin=args.bearing_bin,
                    directional_prior=args.directional_prior, exact_limit=args.exact_limit)
    policies = [args.policy] if args.mode == 'local' else list(dict.fromkeys(args.compare))
    scenarios = [(args.seed+i, args.scenario, args.error_mode, args.n, args.radius, args.directional_count)
                 for i in range(1 if args.mode == 'local' else args.rounds)]
    if args.mode == 'validate':
        scenarios = [(args.seed+10000+i, scene, noise, n, r, nd)
                     for scene in SCENARIOS for noise in ('iid', 'extreme', 'correlated')
                     for i, (n, r, nd) in enumerate(((10, 1000., 9), (16, 1500., 1)))]
    jobs = [{'seed': seed, 'scenario': scene, 'error_mode': noise, 'n': n, 'radius': r,
             'directional_count': nd, 'policy': policy, 'config': asdict(config),
             'real_limit': args.real_limit, 'max_steps': args.max_steps}
            for seed, scene, noise, n, r, nd in scenarios for policy in policies]
    out = Path(args.output) if args.output else ROOT/'results'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'config.json', {'arguments': vars(args), 'jobs': jobs, 'code_sha256': manifest(),
                             'environment': {'python': sys.version, 'numpy': np.__version__, 'shapely': shapely.__version__}})
    batch = {'status': 'running', 'planned': [case_id(j) for j in jobs], 'completed': []}
    dump(out/'batch.json', batch)
    print(f'output: {out}', flush=True)
    rows = []
    try:
        if args.workers == 1:
            results = (run_case(j, out) for j in jobs)
            pool = None
        else:
            pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'))
            futures = [pool.submit(run_case, j, str(out)) for j in jobs]
            results = (future.result() for future in as_completed(futures))
        for row in results:
            rows.append(row)
            batch['completed'].append(row['case_id'])
            dump(out/'summary.json', rows)
            dump(out/'batch.json', batch)
            print(f"{len(rows)}/{len(jobs)} {row['case_id']}: {row['cleared_count']}/{row['source_count']} "
                  f"{row['virtual_time_s']:.2f}s {row['movement_m']:.0f}m real={row['real_time_s']:.2f}s "
                  f"{row['stop_reason']} {row['error'] or ''}", flush=True)
        batch['status'] = 'completed'
    except BaseException as exc:
        batch['status'], batch['error'] = 'interrupted', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        dump(out/'batch.json', batch)
        dump(out/'aggregate.json', aggregate(rows))
        if 'pool' in locals() and pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    print(json.dumps(aggregate(rows), ensure_ascii=False, indent=2))
    raise SystemExit(any(r['error'] or not r['all_cleared'] or not r['certified_complete'] for r in rows))


if __name__ == '__main__':
    main()
