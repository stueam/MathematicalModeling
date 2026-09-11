"""Algorithm two: local paired benchmarks or explicitly authorized HTTP runs."""
import os
# Set before NumPy is imported, also inherited by all 16 worker processes.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time
import pomcp
import numpy as np
from q3.core import Belief
from q3.movement import MovementPolicy
from q3.racing import RacingPlanner
from q3.simulator import LocalSimulator, generate_world
from pomcp.belief import ParticleBelief
from pomcp.config import Config
from pomcp.planner import Planner
from pomcp.action_gen import MacroPolicy, scan
from pomcp.generative_model import MacroAction


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def make_policy(name, config):
    if name == 'pomcp':
        return Planner(config)
    if name == 'macro-baseline':
        return MacroPolicy()
    if name == 'algorithm1':
        return MovementPolicy()
    if name == 'algorithm1-mc':
        return RacingPlanner(seed=config.seed, workers=config.workers, worlds=config.coarse_worlds,
                             max_worlds=config.fine_worlds, budget_s=config.budget_s,
                             candidates=config.candidates)
    raise ValueError(name)


def control(b, policy, execute, out, max_steps):
    logs, error = [], None
    started = time.monotonic()
    try:
        while not b.done() and b.steps < max_steps:
            if time.monotonic() >= b.deadline or b.virtual_time >= b.virtual_limit:
                break
            tick = time.monotonic()
            if isinstance(policy, MacroPolicy):
                macro = scan(b, b.position, range(1, 21), 'INITIAL_SCAN') if b.steps == 0 else policy.choose(b)
            else:
                macro = policy.choose(b)
                if not isinstance(macro, MacroAction):
                    macro = MacroAction.atomic(macro)
            planning = time.monotonic()-tick
            for action in macro.actions:
                if b.done() or time.monotonic() >= b.deadline or b.virtual_time >= b.virtual_limit or b.steps >= max_steps:
                    break
                request_id, response = execute(action, b.steps)
                row = {'request_id': request_id, 'macro': macro.kind, 'action': asdict(action),
                       'response': response, 'planning_s': planning}
                # Persist accepted response before belief calculations, so even
                # a numerical failure leaves enough evidence to replay the run.
                with (out/'actions.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(row, ensure_ascii=False)+'\n')
                b.apply(action, response, request_id)
                logs.append(row)
            if hasattr(policy, 'records') and policy.records:
                with (out/'decisions.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(policy.records[-1], ensure_ascii=False)+'\n')
            if isinstance(b, ParticleBelief):
                with (out/'belief.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'step': b.steps, 'channels': b.diagnostics()}, ensure_ascii=False)+'\n')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if hasattr(policy, 'close'):
            policy.close()
    reason = ('error' if error else 'certified_complete' if b.done() else 'real_time_limit'
              if time.monotonic() >= b.deadline else 'virtual_time_limit'
              if b.virtual_time >= b.virtual_limit else 'action_limit')
    return {'certified_complete': b.done(), 'virtual_time_s': b.virtual_time,
            'real_time_s': time.monotonic()-started, 'physical_steps': len(logs),
            'stop_reason': reason, 'error': error}, logs


def local_run(args, cfg, name, seed, out):
    world = generate_world(seed, args.n, args.scenario, args.radius, args.error_mode)
    env = LocalSimulator(world)
    b = ParticleBelief(cfg.particles, cfg.seed) if name in ('pomcp', 'macro-baseline') else Belief()
    b.deadline = time.monotonic()+args.real_limit
    policy = make_policy(name, cfg)
    def execute(a, step):
        key = f'local-{step}'
        return key, env.execute(a, key)
    summary, logs = control(b, policy, execute, out, args.max_steps)
    score = world.score()  # Evaluator only, after all decisions have stopped.
    summary.update(seed=seed, policy=name, **score, costs=env.costs,
                   movement_m=env.costs['move_s']*5,
                   measure_count=int(env.costs['measure_s']/5),
                   switch_count=int(env.costs['switch_s']),
                   clear_failures=int(env.costs['clear_failure_s']/3),
                   average_per_cleared_s=(b.virtual_time/score['cleared_count'] if score['cleared_count'] else None),
                   average_per_source_s=b.virtual_time/score['source_count'])
    first, durations = {}, {}
    for row in logs:
        r, c = row['response'], row['action']['channel']
        if r.get('measure_result') in ('direction', 'near'):
            first.setdefault(c, r['virtual_time_s'])
        if r.get('clear_result') == 'success' and c in first:
            durations[c] = r['virtual_time_s']-first[c]
    summary['discovery_to_clear_s'] = durations
    records = getattr(policy, 'records', [])
    summary['mc_decisions'] = sum(r['status'] == 'monte_carlo' for r in records)
    summary['mc_changed'] = sum(r['status'] == 'monte_carlo' and r['changed'] for r in records)
    summary['mc_no_batch'] = sum(r['status'] in ('TimeoutError', 'SamplingError', 'GeometryError') for r in records)
    dump(out/'summary.json', summary)
    return summary


def official(args, cfg, out):
    from q3.client import HttpClient
    client = HttpClient(args.robot_id, args.base_url)
    b, entered, exited, summary = ParticleBelief(cfg.particles, cfg.seed), False, False, {}
    try:
        _, response = client.enter()
        entered = True
        b.deadline = client.deadline-5
        b.virtual_limit = float(response['max_virtual_duration_s'])
        b.virtual_time = float(response['virtual_time_s'])
        summary, _ = control(b, Planner(cfg), lambda a, _: client.execute(a), out, args.max_steps)
        # Strict exit only. An error/uncertain request never triggers a new exit.
        if b.done() and summary['error'] is None and time.monotonic() < client.deadline:
            client.exit()
            exited = True
    except Exception as exc:
        summary['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        summary.update(entered=entered, exited=exited, certified_complete=b.done(),
                       cleared_count=len(b.cleared), source_count=None)
        dump(out/'http-log.json', client.log)
        dump(out/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['local', 'benchmark', 'official'])
    names = ['pomcp', 'macro-baseline', 'algorithm1', 'algorithm1-mc']
    p.add_argument('--policy', choices=names, default='pomcp')
    p.add_argument('--compare', nargs='+', choices=names, default=['algorithm1', 'pomcp'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--n', type=int, choices=range(10, 17))
    p.add_argument('--scenario', choices=['uniform', 'boundary', 'cluster', 'near'], default='uniform')
    p.add_argument('--radius', type=float)
    p.add_argument('--error-mode', choices=['iid', 'extreme', 'correlated'], default='iid')
    p.add_argument('--real-limit', type=float, default=1200.)
    p.add_argument('--max-steps', type=int, default=4000)
    p.add_argument('--compute-scale', type=int, default=1,
                   help='Multiply computational counts/depth/time only; preserve rules, seed and workers')
    for name, field in Config.__dataclass_fields__.items():
        option = '--planner-seed' if name == 'seed' else '--'+name.replace('_', '-')
        p.add_argument(option, dest='cfg_'+name, type=type(field.default), default=field.default)
    p.add_argument('--output')
    p.add_argument('--connect', action='store_true')
    p.add_argument('--robot-id')
    p.add_argument('--base-url', default='http://127.0.0.1:2026')
    args = p.parse_args()
    if args.mode == 'official' and (not args.connect or not args.robot_id):
        p.error('Connecting requires explicit --connect and --robot-id')
    cfg = Config(**{name: getattr(args, 'cfg_'+name) for name in Config.__dataclass_fields__}).scaled(args.compute_scale)
    out = Path(args.output) if args.output else Path(__file__).parent/'results'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'config.json', {'arguments': vars(args), 'planner': asdict(cfg)})
    root = Path(__file__).parent
    sources = list(root.glob('*.py'))+list((root/'pomcp').glob('*.py'))
    sources += list((root.parent/'第三问-算法1/q3').glob('*.py'))
    dump(out/'source-hashes.json', {str(f.relative_to(root.parent)): hashlib.sha256(f.read_bytes()).hexdigest()
                                     for f in sources})
    if args.mode == 'official':
        official(args, cfg, out)
        return
    results = []
    for seed in range(args.seed, args.seed+(args.rounds if args.mode == 'benchmark' else 1)):
        for name in (args.compare if args.mode == 'benchmark' else [args.policy]):
            folder = out/f'{seed}-{name}'
            folder.mkdir()
            result = local_run(args, cfg, name, seed, folder)
            results.append(result)
            dump(out/'summary.json', results)
            print(json.dumps({k: result[k] for k in ('seed', 'policy', 'all_cleared', 'certified_complete',
                  'virtual_time_s', 'real_time_s', 'mc_decisions', 'mc_no_batch', 'error')}, ensure_ascii=False), flush=True)
    if args.mode == 'benchmark' and 'pomcp' in args.compare:
        pairs = []
        for baseline in [n for n in args.compare if n != 'pomcp']:
            for seed in sorted({r['seed'] for r in results}):
                a = next(r for r in results if r['seed'] == seed and r['policy'] == baseline)
                b = next(r for r in results if r['seed'] == seed and r['policy'] == 'pomcp')
                valid = all(r['all_cleared'] and r['certified_complete'] for r in (a, b))
                pairs.append({'seed': seed, 'baseline': baseline, 'both_complete': valid,
                              'saved_s': a['virtual_time_s']-b['virtual_time_s'] if valid else None,
                              'saved_fraction': 1-b['virtual_time_s']/a['virtual_time_s'] if valid else None})
        dump(out/'paired.json', pairs)
    print(f'Results: {out}', flush=True)


if __name__ == '__main__':
    main()
