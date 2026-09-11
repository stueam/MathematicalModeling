"""Local runs by default; explicit `official --connect` for a ready GUI session."""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from q3.core import Belief
from q3.policy import Baseline
from q3.movement import MovementPolicy
from q3.mission import MissionPolicy
from q3.planner import RolloutPlanner
from q3.racing import RacingPlanner
from q3.simulator import LocalSimulator, generate_world


def make_policy(args):
    if args.policy == 'legacy':
        return Baseline(args.ring)
    if args.policy == 'baseline':
        return MovementPolicy(args.ring)
    if args.policy.startswith('mission-'):
        return MissionPolicy(args.policy[8:], args.ring)
    planner_type = RacingPlanner if args.policy == 'mc-race' else RolloutPlanner
    extra = {'workers': getattr(args, 'workers', 1),
             'tolerance_s': getattr(args, 'tolerance', 5.),
             'reuse': getattr(args, 'suffix_reuse', True)} if args.policy == 'mc-race' else {}
    return planner_type(seed=args.planner_seed, worlds=args.worlds, candidates=args.candidates,
                          budget_s=args.budget, interval=args.interval, ring_radius=args.ring,
                          speculative_clear=args.speculative_clear,
                          max_worlds=getattr(args, 'max_worlds', 32), **extra)


def local_run(seed, args):
    world = generate_world(seed, args.n, args.scenario, args.radius, args.error_mode)
    env, b, policy = LocalSimulator(world), Belief(), make_policy(args)
    started = time.monotonic()
    b.deadline = started+args.real_limit
    logs, error = [], None
    for i in range(args.max_steps):
        if b.done() or time.monotonic() >= b.deadline or b.virtual_time >= b.virtual_limit:
            break
        try:
            tick = time.monotonic()
            action = policy.choose(b)
            planning = time.monotonic()-tick
            if time.monotonic() >= b.deadline:
                break
            request_id = f'local-{i}'
            response = env.execute(action, request_id)
            logs.append({'request_id': request_id, 'action': asdict(action), 'response': response,
                         'planning_s': planning})
            b.apply(action, response, request_id)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            break
    if hasattr(policy, 'close'):
        policy.close()
    elapsed = time.monotonic()-started
    score = world.score()  # Evaluator only, after decisions have finished.
    records = getattr(policy, 'records', [])
    mc = [r for r in records if r['status'] == 'monte_carlo']
    stop_reason = ('error' if error else 'certified_complete' if b.done() else
                   'real_time_limit' if time.monotonic() >= b.deadline else
                   'virtual_time_limit' if env.virtual_time >= b.virtual_limit else 'action_limit')
    summary = {'seed': seed, 'policy': args.policy, 'scenario': args.scenario,
               'implementation': ('v3_racing' if args.policy == 'mc-race' else
                                  'experimental_mission' if args.policy.startswith('mission-') else
                                  'v1_legacy' if args.policy == 'legacy' else 'v2_movement_aware'),
               'error_mode': args.error_mode, 'radius': args.radius, 'ring': args.ring,
               **score, 'certified_complete': b.done(), 'virtual_time_s': env.virtual_time,
               'average_per_cleared_s': env.virtual_time/score['cleared_count'] if score['cleared_count'] else None,
               'real_time_s': elapsed, 'steps': len(logs), 'error': error,
               'stop_reason': stop_reason,
               'costs': env.costs, 'mc_decisions': len(mc), 'mc_changed_actions': sum(r['changed'] for r in mc),
               'mc_fallbacks': sum(r['status'] in ('TimeoutError', 'SamplingError', 'GeometryError') for r in records),
               'planning_p95_s': float(np.quantile([l['planning_s'] for l in logs], .95)) if logs else 0,
               'major_departures': sum(r.get('selected_move_m', 0) >= 500 for r in records),
               'major_departures_mc_reviewed': sum(r.get('selected_move_m', 0) >= 500 and r['status'] == 'monte_carlo' for r in records),
               'departure_reviews_attempted': sum(r.get('departure_review_attempted', False) for r in records),
               'channel_status': {c: p.status for c, p in b.channels.items()}}
    return summary, logs, records


def output_dir(path):
    p = Path(path) if path else Path(__file__).parent/'results'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    p.mkdir(parents=True, exist_ok=False)  # Never silently overwrite an earlier experiment.
    return p


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['local', 'benchmark', 'official'])
    policies = ['legacy', 'baseline', 'mc', 'mc-race', 'mission-route', 'mission-nearest',
                'mission-clockwise', 'mission-counterclockwise', 'mission-survey']
    parser.add_argument('--policy', choices=policies, default='baseline')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--compare', nargs='+', choices=policies, default=['legacy', 'baseline', 'mc'])
    parser.add_argument('--n', type=int, choices=range(10, 17))
    parser.add_argument('--scenario', choices=['uniform', 'boundary', 'cluster', 'near'], default='uniform')
    parser.add_argument('--radius', type=float, choices=[1000., 1250., 1500.])
    parser.add_argument('--error-mode', choices=['iid', 'extreme', 'correlated'], default='iid')
    parser.add_argument('--ring', type=float, default=1500.)
    parser.add_argument('--worlds', type=int, default=8)
    parser.add_argument('--max-worlds', type=int, default=32)
    parser.add_argument('--workers', type=int, default=1, help='mc-race process workers; worlds only, never parallel HTTP')
    parser.add_argument('--tolerance', type=float, default=5., help='mc-race practical cost tolerance in virtual seconds')
    parser.add_argument('--suffix-reuse', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--candidates', type=int, default=10)
    parser.add_argument('--budget', type=float, default=8.)
    parser.add_argument('--interval', type=int, default=None, help='Deprecated compatibility flag; never skips a departure review')
    parser.add_argument('--planner-seed', type=int, default=2026)
    parser.add_argument('--speculative-clear', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--real-limit', type=float, default=1200.)
    parser.add_argument('--max-steps', type=int, default=4000)
    parser.add_argument('--output')
    parser.add_argument('--robot-id')
    parser.add_argument('--base-url', default='http://127.0.0.1:2026')
    parser.add_argument('--connect', action='store_true', help='Explicitly enter the already-ready simulator session')
    args = parser.parse_args()
    if args.mode == 'official' and (not args.connect or not args.robot_id):
        parser.error('Official adapter requires --connect and --robot-id; no connection was made')
    out = output_dir(args.output)
    dump(out/'config.json', vars(args))
    if args.mode == 'official':
        official_run(args, out)
        return
    summaries = []
    for seed in range(args.seed, args.seed+(args.rounds if args.mode == 'benchmark' else 1)):
        for name in (args.compare if args.mode == 'benchmark' else [args.policy]):
            args.policy = name
            summary, log, decisions = local_run(seed, args)
            summaries.append(summary)
            prefix = f'{seed}-{name}'
            dump(out/f'{prefix}-summary.json', summary)
            dump(out/f'{prefix}-actions.json', log)
            dump(out/f'{prefix}-decisions.json', decisions)
            print(json.dumps({k: summary[k] for k in ('seed', 'policy', 'all_cleared', 'certified_complete',
                              'virtual_time_s', 'real_time_s', 'mc_decisions', 'mc_changed_actions', 'error')}, ensure_ascii=False), flush=True)
    dump(out/'summary.json', summaries)
    print(f'Results: {out}', flush=True)


def official_run(args, out):
    from q3.client import HttpClient
    client = HttpClient(args.robot_id, args.base_url)
    b, policy = Belief(), make_policy(args)
    error, logs = None, []
    entered, exited = False, False
    started = time.monotonic()
    stop_reason = 'action_limit'
    try:
        _, response = client.enter()
        entered = True
        b.deadline = client.deadline-5
        b.virtual_limit = float(response['max_virtual_duration_s'])
        b.virtual_time = float(response['virtual_time_s'])
        for _ in range(args.max_steps):
            if b.done():
                client.exit()
                exited, stop_reason = True, 'certified_complete'
                break
            if time.monotonic() >= b.deadline or b.virtual_time >= b.virtual_limit:
                stop_reason = 'time_budget'
                break
            action = policy.choose(b)
            if time.monotonic() >= b.deadline:
                stop_reason = 'time_budget'
                break
            request_id, response = client.execute(action)
            logs.append({'request_id': request_id, 'action': asdict(action), 'response': response})
            # Persist every accepted action, not just at process termination.
            with (out/'accepted-actions.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(logs[-1], ensure_ascii=False)+'\n')
            b.apply(action, response, request_id)
        if not exited and time.monotonic() < client.deadline and b.virtual_time < b.virtual_limit:
            client.exit()
            exited = True
            if b.done():
                stop_reason = 'certified_complete'
        # Never send a new action/exit after an uncertain request outcome.
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        stop_reason = 'error_or_uncertain_response'
    finally:
        if hasattr(policy, 'close'):
            policy.close()
        dump(out/'http-log.json', client.log)
        dump(out/'decisions.json', getattr(policy, 'records', []))
        summary = {'entered': entered, 'exited': exited, 'stop_reason': stop_reason,
                   'certified_complete': b.done(), 'cleared_count': len(b.cleared),
                   'virtual_time_s': b.virtual_time, 'error': error,
                   'average_per_cleared_s': b.virtual_time/len(b.cleared) if b.cleared else None,
                   'real_time_s': time.monotonic()-started,
                   'source_count': None, 'note': 'Official truth is not available; no clearance fraction invented.'}
        dump(out/'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
