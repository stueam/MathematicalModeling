"""Learn from public practice traces, without contacting the simulator.

Replay uses only each prefix of observed feedback. End-of-run source counts and
clear locations are used for retrospective statistics, never for policy calls.
Static proposed route gains are not claimed as achievable mission savings.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

import bayes_tsp.sectors as sectors
from bayes_tsp.shared import distance, point_key
from bayes_tsp.repair import repair_segment
from run import code_manifest, dump, make_belief, make_policy, new_output


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_case(case, folder, config, replay):
    rows = read_jsonl(folder/'actions.jsonl')
    decisions = {r['step']: r for r in read_jsonl(folder/'decisions.jsonl')}
    policy = make_policy('bayes-sector', sectors.SectorConfig(**config))
    b = make_belief(policy)
    stats = Counter()
    per_channel = defaultdict(Counter)
    failures, large_moves, mismatches, optimizer_samples = [], [], [], []
    all_cleared = {r['action']['channel'] for r in rows if r['response'].get('clear_result') == 'success'}
    last_clear = max(i for i, r in enumerate(rows) if r['response'].get('clear_result') == 'success')
    real_minimize = sectors.minimize
    step = -1

    def instrument(fun, x0, *args, **kwargs):
        stats['replay_optimizer_calls'] += 1
        result = real_minimize(fun, x0, *args, **kwargs)
        stats['replay_optimizer_successes'] += int(result.success)
        constraints = kwargs['constraints'][0]['fun']
        feasible_margin = float(np.min(constraints(result.x)))
        proposed_gain = float(fun(x0)-fun(result.x))
        if result.success and proposed_gain > 1e-6 and feasible_margin < 0:
            stats['optimizer_success_but_infeasible'] += 1
            if feasible_margin >= -1e-5:
                stats['optimizer_tiny_constraint_violation'] += 1
            repaired = repair_segment(x0, result.x, constraints)
            repaired_gain = float(fun(x0)-fun(repaired)) if repaired is not None else None
            if repaired_gain is not None and repaired_gain > 1e-6:
                stats['audit_repair_keeps_feasible_improvement'] += 1
            if len(optimizer_samples) < 30:
                optimizer_samples.append({'step': step, 'proposed_gain_m': proposed_gain,
                    'min_constraint_margin_m': feasible_margin, 'iterations': int(result.nit),
                    'repaired_gain_m': repaired_gain,
                    'repaired_min_constraint_margin_m': float(np.min(constraints(repaired))) if repaired is not None else None})
        if result.success and proposed_gain > 1e-6 and feasible_margin >= 0:
            stats['replay_optimizer_accepted'] += 1
        if not result.success and proposed_gain > 1e-6 and feasible_margin >= 0:
            stats['optimizer_feasible_improvement_flagged_failure'] += 1
        return result

    if replay:
        sectors.minimize = instrument
    try:
        for step, row in enumerate(rows):
            a = row['action']
            action = sectors.Action(a['kind'], tuple(a['position']), a['channel'])
            channel = b.channels[action.channel]
            decision = decisions[step]
            move = distance(b.position, action.position)
            switch = int(action.kind == 'measure' and action.channel != b.receiver)
            result = row['response'].get('measure_result', row['response'].get('clear_result'))
            purpose = ('survey' if (decision.get('selected_task') or '').startswith('survey:') or
                       decision.get('coverage_phase') else 'localize_clear')
            stats[f'{purpose}_movement_m'] += move
            stats[f'{purpose}_move_actions'] += int(move > 1e-6)
            stats[f'status:{decision["status"]}'] += 1
            if move > 500:
                large_moves.append({'step': step, 'movement_m': move, 'status': decision['status'],
                    'task': decision.get('selected_task'), 'cleared_so_far': len(b.cleared),
                    'known_sources': sum(p.status in ('cleared', 'detected') for p in b.channels.values())})
            if step > last_clear:
                stats['after_last_clear_m'] += move
            if action.kind == 'measure':
                category = 'search' if channel.status == 'unresolved' else 'known_localization'
                stats[f'{category}_measurements'] += 1
                stats[f'{category}_operation_s'] += 5+switch
                stats['same_channel_same_point_repeats'] += int(point_key(action.position) in channel.measured)
                if action.channel not in all_cleared:
                    stats['eventually_absent_channel_measurements'] += 1
                    stats['eventually_absent_operation_s'] += 5+switch
                if channel.status == 'unresolved' and result in ('direction', 'near'):
                    stats['discovered_at_origin' if distance(action.position, (0., 0.)) < 1e-6 else 'discovered_elsewhere'] += 1
                per_channel[action.channel]['measurements'] += 1
            if action.kind == 'clear':
                per_channel[action.channel]['clear_attempts'] += 1
                if result == 'no_target_in_range':
                    per_channel[action.channel]['failed_clears'] += 1
                    posterior = policy.model.posterior(channel)
                    following = next((r for r in rows[step+1:] if r['action']['channel'] == action.channel), None)
                    failures.append({'step': step, 'channel': action.channel,
                        'predicted_clear_probability': posterior.clear_probability(action.position),
                        'conservative_radius_m': channel.summary()[1], 'movement_before_failed_clear_m': move,
                        'finite_grid': any(c.get('finite_grid') and c['action'] == a for c in decision.get('candidates', [])),
                        'next_same_channel_action': following['action']['kind'] if following else None})
            for adjustment in decision.get('station_adjustments', []):
                stats['station_optimizer_calls'] += 1
                stats['station_optimizer_successes'] += int(adjustment['optimizer_success'])
                stats['station_optimizer_accepted'] += int(adjustment['accepted'])
            if replay:
                predicted = policy.choose(b)
                if predicted.kind != action.kind or predicted.channel != action.channel or distance(predicted.position, action.position) > 1e-6:
                    mismatches.append({'step': step, 'position_delta_m': distance(predicted.position, action.position),
                                       'predicted_kind': predicted.kind, 'predicted_channel': predicted.channel})
            b.apply(action, row['response'], row['request_id'])
    finally:
        sectors.minimize = real_minimize
    if not b.done():
        raise ValueError(f'Public history did not certify completion: {case["case_code"]}')
    return {'case_code': case['case_code'], 'source_count': case['source_count'],
        'virtual_time_s': case['virtual_time_s'], 'time_per_source_s': case['average_per_cleared_s'],
        'movement_m': case['movement_m'], 'costs': case['costs'], 'stats': dict(stats),
        'after_last_clear_s': case['virtual_time_s']-rows[last_clear]['response']['virtual_time_s'],
        'failures': failures, 'channels': dict(per_channel), 'large_moves': large_moves,
        'optimizer_samples': optimizer_samples, 'replay_mismatches': mismatches,
        'replay_executed': replay}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('aggregate', type=Path)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    cases = json.loads((args.aggregate/'cases.json').read_text())
    metadata = json.loads((args.aggregate/'config.json').read_text())
    current = code_manifest()
    checked = [k for k in metadata['code_sha256'] if '/bayes_tsp/' in k or '/q3/' in k]
    if any(current.get(k) != metadata['code_sha256'][k] for k in checked):
        raise ValueError('Algorithm source changed: do not label a new version as an exact historical replay')
    out = new_output()
    dump(out/'config.json', {'input': str(args.aggregate), 'replay': args.replay,
                             'policy_config': metadata['policy_config'], 'code_sha256': current})
    audits = []
    for case in cases:
        batch = Path(case['source_batch'])
        folders = [p.parent for p in batch.glob('practice-*/summary.json')
                   if json.loads(p.read_text()).get('case_code') == case['case_code']]
        if len(folders) != 1:
            raise ValueError('Case directory is not unique')
        record = audit_case(case, folders[0], metadata['policy_config'], args.replay)
        audits.append(record)
        dump(out/f'{case["case_code"]}.json', record)
        print(json.dumps({'case': record['case_code'], 'after_last_clear_s': record['after_last_clear_s'],
            'failures': len(record['failures']), 'replay_mismatches': len(record['replay_mismatches']),
            'stats': record['stats']}, ensure_ascii=False), flush=True)
    totals, costs = Counter(), Counter()
    for record in audits:
        totals.update(record['stats'])
        costs.update(record['costs'])
    summary = {'rounds': len(audits), 'totals': dict(totals), 'costs': dict(costs),
        'total_virtual_time_s': sum(a['virtual_time_s'] for a in audits),
        'mean_after_last_clear_s': float(np.mean([a['after_last_clear_s'] for a in audits])),
        'mean_after_last_clear_m': totals['after_last_clear_m']/len(audits),
        'replay_mismatches': sum(len(a['replay_mismatches']) for a in audits)}
    dump(out/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
