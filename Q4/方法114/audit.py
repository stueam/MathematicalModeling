"""Read-only log audit and independent complete local-tail checks on saved worlds.

Writes a NEW result directory. Never contacts a simulator or official service.
World truth is accessed only by the evaluator and cloned local environments.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Point

from q4.core import Belief
from q4.compact import make_belief
from q4.localization import finite_clear
from q4.policy import Config, Policy
from q4.shared import Action, distance
from q4.simulator import LocalSimulator, Source, World
from run import dump, manifest, case_id

ROOT = Path(__file__).resolve().parent


def complete_tail(belief, world_data, cleared, action, config):
    sources = [Source(s['channel'], tuple(s['position']), s['radius'], s['heading_deg']) for s in world_data['sources']]
    world = World(sources, world_data['seed'], world_data['error_mode'], cleared=cleared)
    b = belief.clone()
    env = LocalSimulator(world, b.position, b.receiver, b.virtual_time)
    policy = Policy('bayes', Config(**config))
    start, c = b.virtual_time, action.channel
    actions = []
    error = None
    try:
        for i in range(1000):
            if b.channels[c].status == 'cleared':
                break
            if i:
                choices, _ = policy.local_choices(b, c)
                action = min(choices, key=lambda r: r[1])[0]
            response = env.execute(action, f'audit-tail-{i}')
            b.apply(action, response, f'audit-tail-{i}')
            actions.append({'action': asdict(action), 'response': response})
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    complete = b.channels[c].status == 'cleared'
    return {'complete': complete, 'realized_virtual_s': b.virtual_time-start if complete else None,
            'spent_virtual_s': b.virtual_time-start, 'error': error, 'actions': actions}


def audit_case(directory, row, tail_limit, out, surround_spacings=()):
    tag = row['case_id']
    data = json.loads((directory/f'{tag}-world.json').read_text())
    sources = {s['channel']: s for s in data['sources']}
    decisions = json.loads((directory/f'{tag}-decisions.json').read_text())
    decisions = {r['step']: r for r in decisions}
    b, cleared, seen = make_belief(row['policy']), set(), set()
    pos, receiver, clock, maximum_error = (0., 0.), 1, 0., 0.
    tails, channels_checked = [], set()
    for i, line in enumerate((directory/f'{tag}-actions.jsonl').read_text().splitlines()):
        record = json.loads(line)
        action = Action(record['action']['kind'], tuple(record['action']['position']), record['action']['channel'])
        response = record['response']
        assert response['accepted'] is True and not b.done(), (tag, i, 'invalid post-completion action')
        assert record['request_id'] not in seen, (tag, i, 'duplicate logged action')
        seen.add(record['request_id'])
        decision = decisions[i]
        chosen = next((r for r in decision.get('candidates', []) if r['action'] == record['action'] and r['task'] > 0), None)
        if chosen and len(tails) < tail_limit and action.channel not in channels_checked:
            checked = complete_tail(b, data, cleared, action, row['config'])
            detail = {'case_id': tag, 'step': i, 'channel': action.channel,
                      'predicted_local_s': chosen['local_proxy_s'], **checked}
            dump(out/f'{tag}-tail-{i}.json', detail)
            tails.append({k: v for k, v in detail.items() if k != 'actions'})
            channels_checked.add(action.channel)
        source = sources.get(action.channel)
        active = source is not None and action.channel not in cleared
        d = distance(action.position, source['position']) if active else math.inf
        if action.kind == 'clear':
            success = d <= 20
            assert response['clear_result'] == ('success' if success else 'no_target_in_range'), (tag, i, 'clear mismatch')
            operation = 5 if success else 3
            if success:
                cleared.add(action.channel)
        else:
            operation = 5+int(receiver != action.channel)
            receiver = action.channel
            illuminated = True
            if active and source['heading_deg'] is not None:
                th = math.radians(source['heading_deg'])
                dx, dy = action.position[0]-source['position'][0], action.position[1]-source['position'][1]
                illuminated = math.cos(th)*dx+math.sin(th)*dy >= -8*np.finfo(float).eps*max(1., d)
            signal = active and d <= source['radius'] and illuminated
            expected = 'near' if signal and d <= 5 else 'direction' if signal else 'no_signal'
            assert response['measure_result'] == expected, (tag, i, 'measure mismatch')
            if expected == 'direction':
                phi = math.degrees(math.atan2(source['position'][1]-action.position[1], source['position'][0]-action.position[0]))
                error = abs((response['svd_deg']-phi+180) % 360-180)
                assert error <= 1.0050001, (tag, i, 'bearing outside allowed error')
        clock = round(clock+round(distance(pos, action.position)/5, 6)+operation, 6)
        maximum_error = max(maximum_error, abs(clock-response['virtual_time_s']))
        assert maximum_error < 1e-6, (tag, i, 'cost mismatch')
        pos = action.position
        b.apply(action, response, record['request_id'])
        for c, s in sources.items():
            channel = b.channels[c]
            assert channel.status != 'absent_certified', (tag, i, 'real source marked absent')
            assert channel.region.distance(Point(s['position'])) < 1e-7, (tag, i, 'truth excluded from conservative support')
    assert b.done() == row['certified_complete']
    assert (len(cleared) == len(sources)) == row['all_cleared']
    surround=None
    if surround_spacings:
        from surround_audit import audit_absence
        surround=audit_absence(b,surround_spacings)
        assert surround['all_sample_checks_pass'], (tag,'independent angular counterexample')
    return {'case_id': tag, 'steps': b.steps, 'max_clock_error_s': maximum_error,
            'certified_complete': b.done(), 'all_cleared': len(cleared) == len(sources),
            'tails': tails, 'surround_audit':surround}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--tails', type=int, default=0, help='Complete local-tail checks per selected case')
    parser.add_argument('--tail-policy', default='mobile')
    parser.add_argument('--surround-spacings',type=float,nargs='+',default=[],
                        help='Independent angle-gap samples in meters; diagnostics only')
    args = parser.parse_args()
    if any(s<10 or s>1800 for s in args.surround_spacings):
        parser.error('Surround audit spacing must be 10..1800 m')
    out = ROOT/'results'/datetime.now().strftime('audit-%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'config.json', {'inputs': [str(p) for p in args.inputs], 'tails_per_case': args.tails,
                             'tail_policy': args.tail_policy, 'code_sha256': manifest(),
                             'surround_spacings_m':args.surround_spacings,
                             'surround_audit_sha256':hashlib.sha256((ROOT/'surround_audit.py').read_bytes()).hexdigest(),
                             'audit_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    status = {'status': 'running', 'completed': []}
    dump(out/'batch.json', status)
    print(f'output: {out}', flush=True)
    results, worlds, versions = [], {}, set()
    try:
        for directory in args.inputs:
            batch = json.loads((directory/'batch.json').read_text())
            assert batch['status'] == 'completed' and set(batch['completed']) == set(batch['planned']), 'Incomplete experiment'
            cfg = json.loads((directory/'config.json').read_text())
            versions.add(json.dumps(cfg['code_sha256'], sort_keys=True))
            rows = json.loads((directory/'summary.json').read_text())
            assert {r['case_id'] for r in rows} == set(batch['planned']), 'Missing paired case'
            # Isolated local prototypes record explicit jobs without a CLI
            # argument namespace. Validate their complete plan just as strictly.
            if 'arguments' in cfg:
                policies = set(cfg['arguments']['compare']) if cfg['arguments']['mode'] != 'local' else {cfg['arguments']['policy']}
            else:
                assert {case_id(j) for j in cfg['jobs']} == set(batch['planned']), 'Job manifest differs from plan'
                policies = {j['policy'] for j in cfg['jobs']}
            groups = {}
            for row in rows:
                pair = tuple(row[k] for k in ('seed', 'scenario', 'error_mode', 'n', 'radius', 'directional_count'))
                groups.setdefault(pair, set()).add(row['policy'])
                data = json.loads((directory/f"{row['case_id']}-world.json").read_text())
                encoded = json.dumps(data, sort_keys=True)
                assert pair not in worlds or worlds[pair] == encoded, 'Paired worlds differ'
                worlds[pair] = encoded
                result = audit_case(directory, row, args.tails if row['policy'] == args.tail_policy else 0, out,args.surround_spacings)
                results.append(result)
                status['completed'].append(row['case_id'])
                dump(out/'batch.json', status)
                print(f"audited {len(results)}: {row['case_id']}", flush=True)
            assert all(v == policies for v in groups.values()), 'Unpaired policy group'
        tails = [t for r in results for t in r['tails']]
        complete = [t for t in tails if t['complete']]
        summary = {'cases': len(results), 'all_completed': all(r['all_cleared'] and r['certified_complete'] for r in results),
                   'actions': sum(r['steps'] for r in results), 'source_code_versions': len(versions),
                   'max_clock_error_s': max(r['max_clock_error_s'] for r in results),
                   'surround_spacings_m':args.surround_spacings,
                   'surround_all_sample_checks_pass':all(r['surround_audit']['all_sample_checks_pass'] for r in results) if args.surround_spacings else None,
                   'tail_cases': len(tails), 'tail_completed': len(complete),
                   'tail_mean_predicted_s': float(np.mean([t['predicted_local_s'] for t in complete])) if complete else None,
                   'tail_mean_realized_s': float(np.mean([t['realized_virtual_s'] for t in complete])) if complete else None,
                   'tail_note': 'Complete local continuations on independent maps, not full-mission rollouts or strict expectation calibration.'}
        dump(out/'cases.json', results)
        dump(out/'summary.json', summary)
        status['status'] = 'completed'
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except BaseException as exc:
        status['status'], status['error'] = 'failed', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        dump(out/'batch.json', status)


if __name__ == '__main__':
    main()
