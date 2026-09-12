"""Bayesian + dynamic open TSP. Local by default; no sampled-world planning."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
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

from bayes_tsp.policy import BayesTSPPolicy, Config
from bayes_tsp.efficient import EfficientConfig, EfficientPolicy
from bayes_tsp.joint import JointConfig, JointPolicy
from bayes_tsp.coupling import CoupledBelief
from bayes_tsp.sectors import SectorConfig, SectorPolicy
from bayes_tsp.refinement import RefinementConfig, RefinementPolicy
from bayes_tsp.cached import CachedRefinementPolicy, CachedSectorPolicy
from bayes_tsp.shared import Belief, SHARED_DIR, load


ROOT = Path(__file__).resolve().parent
SECTOR_POLICIES = ('bayes-radius', 'bayes-task-scan', 'bayes-mobile', 'bayes-sector-fixed', 'bayes-sector')
REFINEMENT_POLICIES = ('bayes-station-fix', 'bayes-recovery', 'bayes-v6', 'bayes-v6-six', 'bayes-v6-eight')
FAST_POLICIES = ('bayes-fast', 'bayes-sector-fast')
POLICIES = FAST_POLICIES+REFINEMENT_POLICIES+SECTOR_POLICIES+('bayes-tsp4', 'bayes-service', 'bayes-joint', 'bayes-joint-probe', 'bayes-reviewed',
            'bayes-efficient', 'bayes-route-scan', 'bayes-stable', 'bayes-no-cover', 'bayes-strict-stable',
            'bayes-tsp', 'bayes-nearest', 'bayes-no-shared', 'baseline', 'baseline-ring')


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def new_output(path=None):
    out = Path(path) if path else ROOT/'results'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    return out


def code_manifest():
    files = list((ROOT/'bayes_tsp').glob('*.py'))+[ROOT/'run.py', ROOT/'analyze.py', ROOT/'audit_routes.py',
                                               ROOT/'practice_windows.py']
    files += [SHARED_DIR/f'{name}.py' for name in ('core', 'policy', 'simulator', 'client', 'movement', 'sampling')]
    return {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def make_policy(name, config):
    if name == 'bayes-fast':
        return CachedRefinementPolicy(RefinementConfig(**{**asdict(config), 'stable_routing': False}))
    if name == 'bayes-sector-fast':
        return CachedSectorPolicy(replace(config, stable_routing=False))
    if name in REFINEMENT_POLICIES:
        config = replace(RefinementConfig(**asdict(config)), stable_routing=False)
        if name == 'bayes-station-fix':
            config = replace(config, recovery_measure=False)
        elif name == 'bayes-recovery':
            config = replace(config, repair_stations=False)
        elif name == 'bayes-v6-six':
            config = replace(config, sectors=6)
        elif name == 'bayes-v6-eight':
            config = replace(config, sectors=8)
        return RefinementPolicy(config)
    if name in SECTOR_POLICIES:
        config = replace(SectorConfig(**asdict(config)), stable_routing=False)
        if name == 'bayes-radius':
            config = replace(config, movable_stations=False)
            policy = JointPolicy(config)
            policy.implementation = 'bayes_tsp_v5_radius_only'
            return policy
        if name in ('bayes-task-scan', 'bayes-mobile'):
            config = replace(config, radius_coupling=False)
        if name in ('bayes-task-scan', 'bayes-sector-fixed'):
            config = replace(config, movable_stations=False)
        return SectorPolicy(config)
    config = JointConfig(**{k: v for k, v in asdict(config).items() if k in JointConfig.__dataclass_fields__})
    if name in ('bayes-tsp4', 'bayes-service', 'bayes-joint', 'bayes-joint-probe', 'bayes-reviewed'):
        config = replace(JointConfig(**asdict(config)), stable_routing=False)
        if name == 'bayes-tsp4':
            config = replace(config, joint_coverage=False, service_scans=False)
        elif name == 'bayes-service':
            config = replace(config, joint_coverage=False, exact_routes=False)
        elif name == 'bayes-joint-probe':
            config = replace(config, route_probes=True)
        elif name == 'bayes-reviewed':
            config = replace(config, route_probes=True, review_departures=True)
        return JointPolicy(config)
    config = EfficientConfig(**{k: v for k, v in asdict(config).items() if k in EfficientConfig.__dataclass_fields__})
    if name in ('baseline', 'baseline-ring'):
        return load('movement').MovementPolicy(1500. if name == 'baseline' else config.ring)
    if name in ('bayes-efficient', 'bayes-route-scan', 'bayes-stable', 'bayes-no-cover', 'bayes-strict-stable'):
        config = EfficientConfig(**asdict(config))
        if name == 'bayes-route-scan':
            config = replace(config, stable_routing=False)
        elif name == 'bayes-stable':
            config = replace(config, route_measurements=False)
        elif name == 'bayes-no-cover':
            config = replace(config, coverage_routing=False)
        elif name == 'bayes-strict-stable':
            config = replace(config, pending_detours=False)
        return EfficientPolicy(config)
    config = Config(**{k: v for k, v in asdict(config).items() if k in Config.__dataclass_fields__})
    if name == 'bayes-nearest':
        config = replace(config, dynamic_tsp=False)
    elif name == 'bayes-no-shared':
        config = replace(config, shared=False)
    return BayesTSPPolicy(config)


def make_belief(policy):
    coupled = getattr(getattr(policy, 'config', None), 'radius_coupling', False)
    return CoupledBelief() if coupled else Belief()


def run_case(job):
    seed, name, scene, noise, n, radius, config_dict, real_limit, max_steps = job
    simulator = load('simulator')
    world = simulator.generate_world(seed, n, scene, radius, noise)
    env = simulator.LocalSimulator(world)
    config = SectorConfig(**config_dict)
    policy = make_policy(name, config)
    belief = make_belief(policy)
    started = time.monotonic()
    belief.deadline = started+real_limit
    actions, error = [], None
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
            planning = time.monotonic()-tick
            if time.monotonic() >= belief.deadline:
                reason = 'real_time_limit'
                break
            request_id = f'local-{index}'
            response = env.execute(action, request_id)
            actions.append({'request_id': request_id, 'action': asdict(action),
                            'response': response, 'planning_s': planning})
            belief.apply(action, response, request_id)
        except Exception as exc:
            error, reason = f'{type(exc).__name__}: {exc}', 'error'
            break
    elapsed = time.monotonic()-started
    if belief.done():
        reason = 'certified_complete'
    # Only the evaluator sees this score, after the policy has finished.
    score = world.score()
    decisions = getattr(policy, 'records', [])
    last_success = max((i for i, row in enumerate(actions) if row['response'].get('clear_result') == 'success'), default=-1)
    after_clear_s = env.virtual_time-actions[last_success]['response']['virtual_time_s'] if last_success >= 0 else None
    after_clear_m = (float(np.linalg.norm(np.diff(np.asarray([row['action']['position'] for row in actions[last_success:]]),
                                                 axis=0), axis=1).sum()) if last_success >= 0 else None)
    summary = {'seed': seed, 'policy': name, 'scenario': scene, 'error_mode': noise,
        'n_input': n, 'radius_input': radius, 'implementation': getattr(policy, 'implementation',
            'bayes_tsp_v2_guide_stops' if name.startswith('bayes') else 'sibling_movement'),
        'policy_config': asdict(policy.config) if isinstance(policy, BayesTSPPolicy) else {'ring': 1500. if name == 'baseline' else config.ring},
        'belief_implementation': type(belief).__name__,
        **score, 'certified_complete': belief.done(), 'stop_reason': reason,
        'virtual_time_s': env.virtual_time, 'movement_m': env.costs['move_s']*5,
        'average_per_cleared_s': env.virtual_time/score['cleared_count'] if score['cleared_count'] else None,
        'real_time_s': elapsed, 'steps': len(actions), 'error': error, 'costs': env.costs,
        'measure_count': sum(a['action']['kind'] == 'measure' for a in actions),
        'failed_clears': sum(a['response'].get('clear_result') == 'no_target_in_range' for a in actions),
        'planning_p95_s': float(np.quantile([a['planning_s'] for a in actions], .95)) if actions else 0.,
        'shared_measurements': sum(d['status'] == 'shared_measure' for d in decisions),
        'incidental_scans': sum(d['status'] == 'incidental_scan' for d in decisions),
        'quadrature_fallbacks': sum(d['status'] == 'quadrature_fallback' for d in decisions),
        'completion_fallbacks': sum(d['status'] == 'completion_fallback' for d in decisions),
        'long_departures': sum(d['selected_move_m'] >= 50 for d in decisions),
        'long_departure_reviews': sum(d['selected_move_m'] >= 50 and d['status'] == 'route_decision' for d in decisions),
        'target_switches': sum(d.get('target_switched', False) for d in decisions),
        'after_last_clear_s': after_clear_s, 'after_last_clear_m': after_clear_m,
        'clear_stop_scans': sum(d['status'] == 'clear_stop_search' for d in decisions),
        'refinement_reviews': sum(d.get('departure_check', {}).get('status') == 'fine_quadrature' for d in decisions),
        'review_rejections': sum(d.get('departure_check', {}).get('accepted') is False for d in decisions),
        'coupling_checks': sum(getattr(p, 'coupling_checks', 0) for p in belief.channels.values()),
        'coupling_cuts': sum(getattr(p, 'coupling_cuts', 0) for p in belief.channels.values()),
        'coupling_area_removed_m2': sum(getattr(p, 'coupling_area_removed_m2', 0.) for p in belief.channels.values()),
        'task_clear_scans': sum(d['status'] == 'task_clear_search' for d in decisions),
        'clear_sector_services': getattr(policy, 'clear_sector_services', 0),
        'station_adjustments': getattr(policy, 'station_adjustments', 0),
        'station_adjustments_accepted': getattr(policy, 'station_adjustments_accepted', 0),
        'static_station_leg_saving_m': getattr(policy, 'station_leg_saving_m', 0.),
        'station_repair_accepts': getattr(policy, 'repair_accepts', 0),
        'recovery_filtered_candidates': getattr(policy, 'recovery_filtered', 0),
        'coverage_departures': sum(d.get('coverage_phase', False) for d in decisions),
        'deferred_known_measurements': sum(sum(r['future_gain_s'] >= r['now_gain_s']-2 and r['now_gain_s'] > 2
            for r in d.get('known_sensing_review', [])) for d in decisions),
        'channel_status': {c: p.status for c, p in belief.channels.items()}}
    return summary, actions, decisions


def job_key(job):
    return f'{job[2]}-{job[3]}-n{job[4]}-r{job[5]}-{job[0]}-{job[1]}'


def execute_batch(jobs, out, workers):
    manifest = {'status': 'running', 'planned': [job_key(j) for j in jobs], 'completed': []}
    dump(out/'batch.json', manifest)
    summaries = []

    def save(job, result):
        summary, actions, decisions = result
        key = job_key(job)
        dump(out/f'{key}-summary.json', summary)
        dump(out/f'{key}-actions.json', actions)
        dump(out/f'{key}-decisions.json', decisions)
        summaries.append(summary)
        manifest['completed'].append(key)
        dump(out/'batch.json', manifest)
        print(json.dumps({k: summary[k] for k in ('seed', 'policy', 'scenario', 'error_mode',
            'all_cleared', 'certified_complete', 'virtual_time_s', 'movement_m', 'real_time_s', 'error')}, ensure_ascii=False), flush=True)

    try:
        if workers == 1:
            for job in jobs:
                save(job, run_case(job))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = {pool.submit(run_case, job): job for job in jobs}
                for future in as_completed(futures):
                    save(futures[future], future.result())
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest['status'] = 'interrupted'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        dump(out/'batch.json', manifest)
        summaries.sort(key=lambda row: (row['scenario'], row['error_mode'], row['seed'], row['policy']))
        dump(out/'summary.json', summaries)
    return summaries


def official_run(args, config, out):
    client = load('client').HttpClient(args.robot_id, args.base_url)
    policy = make_policy(args.policy, config)
    belief = make_belief(policy)
    started = time.monotonic()
    entered, exited, error = False, False, None
    reason = 'action_limit'
    try:
        _, response = client.enter()
        entered = True
        belief.deadline = client.deadline-5
        belief.virtual_time = float(response['virtual_time_s'])
        belief.virtual_limit = float(response['max_virtual_duration_s'])
        for _ in range(args.max_steps):
            if belief.done():
                reason = 'certified_complete'
                break
            if time.monotonic() >= belief.deadline or belief.virtual_time >= belief.virtual_limit:
                reason = 'time_budget'
                break
            action = policy.choose(belief)
            if time.monotonic() >= belief.deadline:
                reason = 'time_budget'
                break
            request_id, response = client.execute(action)
            with (out/'accepted-actions.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps({'request_id': request_id, 'action': asdict(action), 'response': response})+'\n')
            belief.apply(action, response, request_id)
        if belief.done():
            reason = 'certified_complete'
        if time.monotonic() < client.deadline and belief.virtual_time < belief.virtual_limit:
            client.exit()
            exited = True
    except Exception as exc:
        # No replacement request or exit after an uncertain action outcome.
        error, reason = f'{type(exc).__name__}: {exc}', 'error_or_uncertain_response'
    finally:
        dump(out/'http-log.json', client.log)
        dump(out/'decisions.json', policy.records)
        dump(out/'summary.json', {'entered': entered, 'exited': exited, 'error': error,
             'stop_reason': reason, 'certified_complete': belief.done(), 'source_count': None,
             'cleared_count': len(belief.cleared), 'virtual_time_s': belief.virtual_time,
             'average_per_cleared_s': belief.virtual_time/len(belief.cleared) if belief.cleared else None,
             'real_time_s': time.monotonic()-started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['local', 'benchmark', 'validate', 'official'])
    parser.add_argument('--policy', choices=POLICIES, default='bayes-fast')
    parser.add_argument('--compare', nargs='+', choices=POLICIES, default=['bayes-sector', 'bayes-v6', 'bayes-fast'])
    parser.add_argument('--validation-compare', nargs='+', choices=POLICIES,
                        help='Optional paired policies for the fixed stress configurations')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--workers', type=int, default=1, help='Independent local cases only; HTTP stays serial')
    parser.add_argument('--n', type=int, choices=range(10, 17))
    parser.add_argument('--radius', type=float, choices=[1000., 1250., 1500.])
    parser.add_argument('--scenario', choices=['uniform', 'boundary', 'cluster', 'near'], default='uniform')
    parser.add_argument('--error-mode', choices=['iid', 'extreme', 'correlated'], default='iid')
    parser.add_argument('--ring', type=float, default=1200.)
    parser.add_argument('--resolution', type=int, default=24)
    parser.add_argument('--bearing-bin', type=float, default=1.)
    parser.add_argument('--p0', type=float, default=.65)
    parser.add_argument('--stable-radius', type=float, default=100.)
    parser.add_argument('--switch-gain', type=float, default=20.)
    parser.add_argument('--future-stops', type=int, default=3)
    parser.add_argument('--exact-limit', type=int, default=12)
    parser.add_argument('--sectors', type=int, default=7)
    parser.add_argument('--station-sweeps', type=int, default=2)
    parser.add_argument('--speculative-clear', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--real-limit', type=float, default=1200.)
    parser.add_argument('--max-steps', type=int, default=4000)
    parser.add_argument('--output')
    parser.add_argument('--connect', action='store_true')
    parser.add_argument('--robot-id')
    parser.add_argument('--base-url', default='http://127.0.0.1:2026')
    args = parser.parse_args()
    if args.workers < 1 or args.rounds < 1 or args.real_limit <= 0 or args.max_steps < 1:
        parser.error('workers, rounds, real-limit and max-steps must be positive')
    if args.mode == 'official' and (not args.connect or not args.robot_id or args.policy not in
                                   ('bayes-tsp', 'bayes-efficient', 'bayes-route-scan', 'bayes-joint')+SECTOR_POLICIES+FAST_POLICIES):
        parser.error('Official mode requires --connect --robot-id ... and a supported Bayesian policy')
    config = SectorConfig(ring=args.ring, resolution=args.resolution, bearing_bin=args.bearing_bin, stable_routing=True,
                    p0=args.p0, speculative_clear=args.speculative_clear, stable_radius_m=args.stable_radius,
                    switch_gain_s=args.switch_gain, future_stops=args.future_stops, exact_limit=args.exact_limit,
                    sectors=args.sectors, station_sweeps=args.station_sweeps)
    SectorPolicy(config)  # Validate before creating an output or opening a connection.
    out = new_output(args.output)
    dump(out/'config.json', {**vars(args), 'algorithm_config': asdict(config), 'code_sha256': code_manifest(),
                            'environment': {'python': sys.version, 'numpy': np.__version__,
                                            'scipy': scipy.__version__, 'shapely': shapely.__version__}})
    if args.mode == 'official':
        official_run(args, config, out)
        return
    jobs = []
    if args.mode == 'validate':
        for scene in ('uniform', 'boundary', 'cluster', 'near'):
            for noise in ('iid', 'extreme', 'correlated'):
                for index, (n, radius) in enumerate(((10, 1000.), (16, 1500.))):
                    for name in args.validation_compare or [args.policy]:
                        jobs.append((10000+index, name, scene, noise, n, radius,
                                     asdict(config), args.real_limit, args.max_steps))
    else:
        for seed in range(args.seed, args.seed+(args.rounds if args.mode == 'benchmark' else 1)):
            for name in (args.compare if args.mode == 'benchmark' else [args.policy]):
                jobs.append((seed, name, args.scenario, args.error_mode, args.n, args.radius,
                             asdict(config), args.real_limit, args.max_steps))
    results = execute_batch(jobs, out, args.workers)
    print(f'Results: {out}', flush=True)
    failed = any(r['error'] or not r['all_cleared'] or not r['certified_complete'] for r in results)
    raise SystemExit(bool(failed))


if __name__ == '__main__':
    main()
