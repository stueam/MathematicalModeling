"""Bayesian + dynamic open TSP. Local by default; no sampled-world planning."""

from dataclasses import asdict
from bayes_tsp.policy import Policy, Config
from bayes_tsp.coupling import CoupledBelief
from bayes_tsp.shared import SHARED_DIR, load
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import sys
import numpy as np
import scipy
import shapely

ROOT = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def new_output(path=None):
    out = Path(path) if path else ROOT / 'results' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    return out


def code_manifest():
    files = sorted(ROOT.glob('*.py')) + sorted((ROOT / 'bayes_tsp').glob('*.py'))
    files += sorted(SHARED_DIR.glob('*.py'))
    return {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def run_case(job):
    seed, scene, noise, n, radius, config_dict, real_limit, max_steps = job
    simulator = load('simulator')
    world = simulator.generate_world(seed, n, scene, radius, noise)
    env = simulator.LocalSimulator(world)
    config = Config(**config_dict)
    policy = Policy(config)
    belief = CoupledBelief()
    started = time.monotonic()
    belief.deadline = started + real_limit
    actions, error = ([], None)
    reason = 'action_limit'
    for index in range(max_steps):
        if belief.done():
            reason = 'certified_complete'
            break
        if time.monotonic() >= belief.deadline:
            reason = 'real_time_limit'
            break
        if belief.virtual_time >= belief.virtual_limit:
            reason = 'virtual_time_limit'
            break
        try:
            tick = time.monotonic()
            action = policy.choose(belief)
            planning = time.monotonic() - tick
            if time.monotonic() >= belief.deadline:
                reason = 'real_time_limit'
                break
            request_id = f'local-{index}'
            response = env.execute(action, request_id)
            actions.append(
                {
                    'request_id': request_id,
                    'action': asdict(action),
                    'response': response,
                    'planning_s': planning,
                }
            )
            belief.apply(action, response, request_id)
        except Exception as exc:
            error, reason = (f'{type(exc).__name__}: {exc}', 'error')
            break
    elapsed = time.monotonic() - started
    if belief.done():
        reason = 'certified_complete'
    score = world.score()
    decisions = getattr(policy, 'records', [])
    last_success = max(
        (i for i, row in enumerate(actions) if row['response'].get('clear_result') == 'success'), default=-1
    )
    after_clear_s = (
        env.virtual_time - actions[last_success]['response']['virtual_time_s'] if last_success >= 0 else None
    )
    after_clear_m = (
        float(
            np.linalg.norm(
                np.diff(np.asarray([row['action']['position'] for row in actions[last_success:]]), axis=0),
                axis=1,
            ).sum()
        )
        if last_success >= 0
        else None
    )
    summary = {
        'seed': seed,
        'policy': 'bayes-fast',
        'scenario': scene,
        'error_mode': noise,
        'n_input': n,
        'radius_input': radius,
        'implementation': policy.implementation,
        'policy_config': asdict(config),
        'belief_implementation': type(belief).__name__,
        **score,
        'certified_complete': belief.done(),
        'stop_reason': reason,
        'virtual_time_s': env.virtual_time,
        'movement_m': env.costs['move_s'] * 5,
        'average_per_cleared_s': env.virtual_time / score['cleared_count']
        if score['cleared_count']
        else None,
        'real_time_s': elapsed,
        'steps': len(actions),
        'error': error,
        'costs': env.costs,
        'measure_count': sum((a['action']['kind'] == 'measure' for a in actions)),
        'failed_clears': sum((a['response'].get('clear_result') == 'no_target_in_range' for a in actions)),
        'planning_p95_s': float(np.quantile([a['planning_s'] for a in actions], 0.95)) if actions else 0.0,
        'shared_measurements': sum((d['status'] == 'shared_measure' for d in decisions)),
        'incidental_scans': sum((d['status'] == 'incidental_scan' for d in decisions)),
        'quadrature_fallbacks': sum((d['status'] == 'quadrature_fallback' for d in decisions)),
        'completion_fallbacks': sum((d['status'] == 'completion_fallback' for d in decisions)),
        'long_departures': sum((d['selected_move_m'] >= 50 for d in decisions)),
        'long_departure_reviews': sum(
            (d['selected_move_m'] >= 50 and d['status'] == 'route_decision' for d in decisions)
        ),
        'target_switches': sum((d.get('target_switched', False) for d in decisions)),
        'after_last_clear_s': after_clear_s,
        'after_last_clear_m': after_clear_m,
        'coupling_checks': sum((getattr(p, 'coupling_checks', 0) for p in belief.channels.values())),
        'coupling_cuts': sum((getattr(p, 'coupling_cuts', 0) for p in belief.channels.values())),
        'coupling_area_removed_m2': sum(
            (getattr(p, 'coupling_area_removed_m2', 0.0) for p in belief.channels.values())
        ),
        'task_clear_scans': sum((d['status'] == 'task_clear_search' for d in decisions)),
        'clear_sector_services': getattr(policy, 'clear_sector_services', 0),
        'station_adjustments': getattr(policy, 'station_adjustments', 0),
        'station_adjustments_accepted': getattr(policy, 'station_adjustments_accepted', 0),
        'static_station_leg_saving_m': getattr(policy, 'station_leg_saving_m', 0.0),
        'station_repair_accepts': getattr(policy, 'repair_accepts', 0),
        'recovery_filtered_candidates': getattr(policy, 'recovery_filtered', 0),
        'coverage_departures': sum((d.get('coverage_phase', False) for d in decisions)),
        'deferred_known_measurements': sum(
            (
                sum(
                    (
                        r['future_gain_s'] >= r['now_gain_s'] - 2 and r['now_gain_s'] > 2
                        for r in d.get('known_sensing_review', [])
                    )
                )
                for d in decisions
            )
        ),
        'channel_status': {c: p.status for c, p in belief.channels.items()},
    }
    return (summary, actions, decisions)


def job_key(job):
    return f'{job[1]}-{job[2]}-n{job[3]}-r{job[4]}-{job[0]}'


def execute_batch(jobs, out, workers):
    manifest = {'status': 'running', 'planned': [job_key(j) for j in jobs], 'completed': []}
    dump(out / 'batch.json', manifest)
    summaries = []

    def save(job, result):
        summary, actions, decisions = result
        key = job_key(job)
        dump(out / f'{key}-summary.json', summary)
        dump(out / f'{key}-actions.json', actions)
        dump(out / f'{key}-decisions.json', decisions)
        summaries.append(summary)
        manifest['completed'].append(key)
        dump(out / 'batch.json', manifest)
        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        'seed',
                        'policy',
                        'scenario',
                        'error_mode',
                        'all_cleared',
                        'certified_complete',
                        'virtual_time_s',
                        'movement_m',
                        'real_time_s',
                        'error',
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    try:
        if workers == 1:
            for job in jobs:
                save(job, run_case(job))
        else:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context('spawn')
            ) as pool:
                futures = {pool.submit(run_case, job): job for job in jobs}
                for future in as_completed(futures):
                    save(futures[future], future.result())
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest['status'] = 'interrupted'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        dump(out / 'batch.json', manifest)
        summaries.sort(key=lambda row: (row['scenario'], row['error_mode'], row['seed'], row['policy']))
        dump(out / 'summary.json', summaries)
    return summaries


def paper_cases():
    """The paper's 40 random maps and 24 stress maps, in a fixed order."""
    cases = [(seed, 'uniform', 'iid', None, None) for seed in range(900, 940)]
    cases += [
        (10000 + i, scene, noise, n, radius)
        for scene in ('uniform', 'boundary', 'cluster', 'near')
        for noise in ('iid', 'extreme', 'correlated')
        for i, (n, radius) in enumerate(((10, 1000.0), (16, 1500.0)))
    ]
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('local', 'benchmark', 'validate', 'reproduce'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--workers', type=int, default=1, help='Independent local maps; HTTP stays serial')
    parser.add_argument('--n', type=int, choices=range(10, 17))
    parser.add_argument('--radius', type=float, choices=(1000.0, 1250.0, 1500.0))
    parser.add_argument('--scenario', choices=('uniform', 'boundary', 'cluster', 'near'), default='uniform')
    parser.add_argument('--error-mode', choices=('iid', 'extreme', 'correlated'), default='iid')
    parser.add_argument('--real-limit', type=float, default=1200.0)
    parser.add_argument('--max-steps', type=int, default=4000)
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    if args.workers < 1 or args.rounds < 1 or args.real_limit <= 0 or (args.max_steps < 1):
        parser.error('workers, rounds, real-limit and max-steps must be positive')
    config = Config()
    Policy(config)
    if args.mode == 'reproduce':
        cases = paper_cases()
    elif args.mode == 'validate':
        cases = paper_cases()[40:]
    else:
        count = args.rounds if args.mode == 'benchmark' else 1
        cases = [
            (seed, args.scenario, args.error_mode, args.n, args.radius)
            for seed in range(args.seed, args.seed + count)
        ]
    jobs = [(*case, asdict(config), args.real_limit, args.max_steps) for case in cases]
    out = new_output(args.output)
    dump(
        out / 'config.json',
        {
            **vars(args),
            'algorithm_config': asdict(config),
            'jobs': jobs,
            'code_sha256': code_manifest(),
            'environment': {
                'python': sys.version,
                'numpy': np.__version__,
                'scipy': scipy.__version__,
                'shapely': shapely.__version__,
            },
        },
    )
    results = execute_batch(jobs, out, args.workers)
    print(f'Results: {out}', flush=True)
    return int(any((r['error'] or not r['all_cleared'] or (not r['certified_complete']) for r in results)))


if __name__ == '__main__':
    raise SystemExit(main())
