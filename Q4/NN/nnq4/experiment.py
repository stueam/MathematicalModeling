"""Local complete Q4 missions, episode-grouped demonstrations and paired metrics."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .bridge import ROOT, Belief, ProbePolicy, Config, distance


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def new_run(name, config):
    out = ROOT/'results'/f'{datetime.now():%Y%m%d-%H%M%S-%f}-{name}'
    out.mkdir(parents=True)
    sources = list((ROOT/'nnq4').glob('*.py'))+list(ROOT.glob('*.py'))+list((ROOT/'vendor').rglob('*.py'))
    hashes = {}
    for p in sources:
        rel = p.relative_to(ROOT)
        target = out/'source_snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        content = p.read_bytes()
        target.write_bytes(content)
        hashes[str(rel)] = hashlib.sha256(content).hexdigest()
    dump(out/'config.json', config)
    dump(out/'source_sha256.json', hashes)
    inputs = {}
    for key in ('checkpoint', 'data'):
        value = config.get(key)
        if value and Path(value).is_file():
            inputs[key] = {'path': str(Path(value).resolve()), 'sha256': hashlib.sha256(Path(value).read_bytes()).hexdigest()}
    dump(out/'input_sha256.json', inputs)
    dump(out/'environment.json', {'python': platform.python_version(), 'numpy': np.__version__,
                                  'torch': str(torch.__version__), 'environment_kind': 'local_simulator',
                                  'official_connection': False})
    return out


def pack_episode(frames, clocks, total, path):
    if not frames:
        raise ValueError('No supervised neural decisions')
    n = max(len(f.nodes) for f in frames)
    k = max(len(f.actions) for f in frames)
    count = len(frames)
    arrays = {'nodes': torch.zeros((count, n, frames[0].nodes.shape[-1])), 'node_mask': torch.zeros((count, n), dtype=torch.bool),
              'candidates': torch.zeros((count, k, frames[0].candidates.shape[-1])), 'candidate_mask': torch.zeros((count, k), dtype=torch.bool),
              'global_features': torch.stack([torch.as_tensor(f.global_features) for f in frames]),
              'target': torch.tensor([f.target for f in frames], dtype=torch.int64),
              'remaining': (total-torch.tensor(clocks, dtype=torch.float64)).float()}
    for i, frame in enumerate(frames):
        arrays['nodes'][i, :len(frame.nodes)] = torch.as_tensor(frame.nodes)
        arrays['node_mask'][i, :len(frame.nodes)] = True
        arrays['candidates'][i, :len(frame.actions)] = torch.as_tensor(frame.candidates)
        arrays['candidate_mask'][i, :len(frame.actions)] = True
    np.savez_compressed(path, **{k: v.numpy() for k, v in arrays.items()})
    return count


def run_episode(job):
    # Only this evaluator imports a simulator. Controllers receive only belief.
    from q4.simulator import generate_world, LocalSimulator
    from .policy import NeuralPolicy
    import torch
    torch.set_num_threads(1)
    seed, mode = job['seed'], job['mode']
    out = Path(job['out'])
    case = f"{job.get('scenario','uniform')}-{job.get('error_mode','iid')}-{seed}-{mode}"
    folder = out/case
    folder.mkdir(parents=True, exist_ok=False)
    world = generate_world(seed, n=job.get('n'), scenario=job.get('scenario', 'uniform'),
                           radius=job.get('radius'), error_mode=job.get('error_mode', 'iid'),
                           directional_count=job.get('directional_count'))
    manifest = world.manifest()
    dump(folder/'world.json', manifest)
    world_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    config = Config(**job.get('planner_config', {}))
    if mode == 'probes':
        policy = ProbePolicy(config)
    else:
        policy = NeuralPolicy(job.get('checkpoint'), teacher=mode == 'teacher',
                              macro_budget=job.get('macro_budget', 160),
                              allow_fallback=job.get('allow_fallback', True),
                              selection=job.get('selection', 'policy'), config=config,
                              sample=job.get('sample', False), sampling_seed=job.get('sampling_seed', 42),
                              temperature=job.get('temperature', 1.), menu=job.get('menu', 'full'),
                              layout=job.get('layout', 'ring22'), features=job.get('features', 'v1'))
    b = Belief()
    env = LocalSimulator(world)
    started = time.monotonic()
    b.deadline = started+job.get('real_limit', 1200)
    error = None
    frames, clocks, events = [], [], []
    macro_counts = dict.fromkeys(('network', 'teacher', 'macro_continuation', 'fallback', 'probes',
                                 'network_override', 'baseline'), 0)
    executor_costs = dict.fromkeys(macro_counts, 0.)
    with gzip.open(folder/'public.jsonl.gz', 'wt') as stream:
        try:
            for i in range(job.get('max_steps', 6000)):
                if b.done():
                    break
                if time.monotonic() >= b.deadline or b.virtual_time >= b.virtual_limit:
                    raise TimeoutError('Actual mission deadline or virtual limit')
                before = b.virtual_time
                action = policy.choose(b)
                if mode == 'probes':
                    decision = {'executor': 'probes', 'reason': policy.records[-1]['reason']}
                    policy.records.clear()
                else:
                    decision = policy.last_decision
                    if job.get('collect') and policy.last_frame is not None:
                        frames.append(policy.last_frame)
                        clocks.append(before)
                response = env.execute(action, f'q4-nn-{i}')
                event = {'request_id': f'q4-nn-{i}', 'action': asdict(action), 'response': response,
                         'move_m': distance(b.position, action.position), **decision}
                b.apply(action, response, f'q4-nn-{i}')
                executor_costs[decision['executor']] += b.virtual_time-before
                macro_counts[decision['executor']] += 1
                events.append(event)
                stream.write(json.dumps(event, ensure_ascii=False)+'\n')
                stream.flush()
            if not b.done():
                raise RuntimeError('Mission did not produce a public completion certificate')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    score = world.score()
    mission_wall_s = time.monotonic()-started
    complete = bool(b.done() and score['all_cleared'] and error is None)
    audit_started = time.monotonic()
    from .audit import audit_episode
    audit = audit_episode(manifest, events)
    dump(folder/'physical-audit.json', audit)
    if not audit['ok'] or not audit['complete']:
        error = error or 'Independent local physical audit failed: '+str(audit['errors'])
        complete = False
    data_path = folder/'episode.npz'
    steps = pack_episode(frames, clocks, b.virtual_time, data_path) if job.get('collect') and complete else 0
    result = {'case': case, 'seed': seed, 'mode': mode, **score, 'world_sha256': world_hash,
              'complete': complete, 'public_complete': b.done(), 'error': error,
              'virtual_time_s': b.virtual_time, 'seconds_per_source': b.virtual_time/score['source_count'] if complete else None,
              'costs': env.costs, 'movement_m': sum(r['move_m'] for r in events),
              'long_moves_over_1km': sum(r['move_m'] > 1000 for r in events),
              'long_move_distance_m': sum(r['move_m'] for r in events if r['move_m'] > 1000),
              'real_time_s': mission_wall_s, 'audit_time_s': time.monotonic()-audit_started,
              'environment_kind': 'local_simulator', 'official_connection': False,
              'audit_ok': audit['ok'], 'actions': len(events),
              'measure_count': sum(r['action']['kind'] == 'measure' for r in events),
              'failed_clears': sum(r['response'].get('clear_result') == 'no_target_in_range' for r in events),
              'executor_actions': macro_counts, 'executor_costs_s': executor_costs,
              'stats': policy.stats if mode != 'probes' else policy.counters,
              'menu_stats': policy.planner.stats if mode != 'probes' else {},
              'menu': job.get('menu', 'full') if mode != 'probes' else 'full',
              'layout': job.get('layout', 'ring22') if mode != 'probes' else 'ring22',
              'features': job.get('features', 'v1'),
              'planner_config': job.get('planner_config', {}),
              'folder': str(folder.resolve()), 'supervised_states': steps,
              'source_group': f"{job.get('scenario','uniform')}:{job.get('error_mode','iid')}:{seed}:{job.get('n')}:{job.get('radius')}:{job.get('directional_count')}",
              'split': job.get('split', 'test')}
    if steps:
        result['path'] = str(data_path.resolve())
    dump(folder/'summary.json', result)
    return result


def summarize(rows):
    groups = {}
    for mode in sorted({r['mode'] for r in rows}):
        selected = [r for r in rows if r['mode'] == mode]
        complete = all(r['complete'] for r in selected)
        cost = torch.tensor([r['virtual_time_s'] for r in selected], dtype=torch.float64)
        n = torch.tensor([r['source_count'] for r in selected], dtype=torch.float64)
        ix = torch.randint(len(selected), (10000, len(selected)), generator=torch.Generator().manual_seed(812))
        quantiles = torch.tensor([.025, .975], dtype=torch.float64)
        groups[mode] = {'episodes': len(selected), 'complete': sum(r['complete'] for r in selected),
                        'aggregate_s_per_source': float(cost.sum()/n.sum()) if complete else None,
                        'map_bootstrap_95pct': torch.quantile(cost[ix].sum(1)/n[ix].sum(1), quantiles).tolist() if complete else None,
                        'mean_movement_m': sum(r['movement_m'] for r in selected)/len(selected),
                        'mean_real_time_s': sum(r['real_time_s'] for r in selected)/len(selected),
                        'mean_actions': sum(r['actions'] for r in selected)/len(selected),
                        'fallback_episodes': sum(r['executor_actions'].get('fallback', 0) > 0 for r in selected),
                        'errors': [r['case'] for r in selected if not r['complete']]}
    paired = {}
    base = {r['world_sha256']: r for r in rows if r['mode'] == 'probes'}
    for mode in groups:
        if mode == 'probes':
            continue
        selected = [r for r in rows if r['mode'] == mode]
        if not base or len(selected) != len(base) or any(r['world_sha256'] not in base for r in selected):
            continue
        if not all(r['complete'] and base[r['world_sha256']]['complete'] for r in selected):
            continue
        a = torch.tensor([base[r['world_sha256']]['virtual_time_s'] for r in selected], dtype=torch.float64)
        c = torch.tensor([r['virtual_time_s'] for r in selected], dtype=torch.float64)
        ix = torch.randint(len(a), (10000, len(a)), generator=torch.Generator().manual_seed(812))
        paired[mode] = {'relative_improvement': float(1-c.sum()/a.sum()),
                        'paired_map_95pct': torch.quantile(1-c[ix].sum(1)/a[ix].sum(1), quantiles).tolist()}
    return {'groups': groups, 'paired_to_probes': paired}


def execute(jobs, out, workers=16):
    torch.set_num_threads(1)
    out = Path(out)
    rows = []
    dump(out/'batch.json', {'status': 'running', 'planned': len(jobs), 'completed': 0})
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(run_episode, {**job, 'out': str(out.resolve())}) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            dump(out/'summary.json', rows)
            if len(rows) % 8 == 0 or len(rows) == len(jobs):
                dump(out/'aggregate.json', summarize(rows))
            dump(out/'batch.json', {'status': 'running', 'planned': len(jobs), 'completed': len(rows)})
            print(f"{len(rows)}/{len(jobs)} {row['case']} complete={row['complete']} s/src={row['seconds_per_source']} real={row['real_time_s']:.2f} {row['error'] or ''}", flush=True)
    dump(out/'batch.json', {'status': 'complete', 'planned': len(jobs), 'completed': len(rows),
                           'all_missions_complete': all(r['complete'] for r in rows)})
    if any(job.get('collect') for job in jobs):
        records = [{k: r[k] for k in ('path', 'split', 'seed', 'source_group', 'complete', 'menu', 'layout', 'features', 'planner_config')}
                   for r in rows if r.get('path')]
        dump(out/'manifest.json', {'records': records, 'failed_episodes': [r['case'] for r in rows if not r['complete']]})
    return rows
