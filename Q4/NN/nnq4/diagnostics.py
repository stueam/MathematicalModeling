"""Read-only physical-log audits and after-the-fact Q4 cost attribution.

Manifest truth is used only by this evaluator. The public-state categories
come from replayed observations; truth phases additionally use the final
source count and must never be exposed to the controller.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import gzip
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

from .audit import audit_episode, CLOCK_TOLERANCE_S, PROTOCOL_COMPARISON
from .bridge import Action, Belief, ROOT, distance, point_key


def _dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_events(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _accumulate(table, key, values):
    bucket = table.setdefault(key, {})
    for name, value in values.items():
        bucket[name] = bucket.get(name, 0) + value


def _progress_stage(cleared):
    return 'cleared_0_3' if cleared < 4 else 'cleared_4_7' if cleared < 8 else 'cleared_8_11' if cleared < 12 else 'cleared_12_16'


def diagnose_episode(world_manifest, events):
    """Attribute every accepted unique action's cost to disjoint phases."""
    b = Belief()
    source_count = len(world_manifest['sources'])
    tables = {name: {} for name in ('action_category', 'public_phase', 'truth_phase', 'progress_stage',
                                    'executor', 'radial_movement', 'decision_stage', 'decision_category')}
    outcomes = Counter()
    survey_points, outer_points = set(), set()
    route = []
    decisions = matches = survey_decisions = 0
    last_clock = 0.0
    for event in events:
        response = event['response']
        if response.get('accepted') is not True or event['request_id'] in b.applied:
            continue
        raw = event['action']
        action = Action(raw['kind'], tuple(raw['position']), raw['channel'])
        status = b.channels[action.channel].status
        detected = sum(state.status == 'detected' for state in b.channels.values())
        cleared_count = len(b.cleared)
        known_count = len(b.known)
        public_phase = 'known_targets_pending' if detected else 'no_positive_yet' if not known_count else 'unknown_channels_only'
        truth_phase = ('certification_tail' if cleared_count == source_count else
                       'all_discovered_clearing' if known_count == source_count else 'discovering')
        progress = _progress_stage(cleared_count)
        radius = math.hypot(*action.position)
        outer = radius >= 1600.0
        if action.kind == 'measure':
            category = 'survey_unknown' if status == 'unresolved' else 'localize_known' if status == 'detected' else 'measure_completed'
            if category == 'survey_unknown':
                key = point_key(action.position)
                survey_points.add(key)
                if outer:
                    outer_points.add(key)
        else:
            category = 'clear_known' if status == 'detected' else 'clear_without_detection'
        outcome = response.get('measure_result', response.get('clear_result'))
        outcomes[outcome] += 1
        move = distance(b.position, action.position)
        move_s = round(move / 5, 6)
        switch_s = int(action.kind == 'measure' and action.channel != b.receiver)
        total_s = response['virtual_time_s'] - last_clock
        values = {'actions': 1, 'move_m': move, 'move_s': move_s, 'virtual_s': total_s,
                  'switch_s': switch_s, 'measure_s': 5 if action.kind == 'measure' else 0,
                  'clear_success_s': 5 if outcome == 'success' else 0,
                  'clear_failure_s': 3 if outcome == 'no_target_in_range' else 0,
                  'moving_actions': int(move > 1e-6)}
        for table, key in (('action_category', category), ('public_phase', public_phase), ('truth_phase', truth_phase),
                           ('progress_stage', progress), ('executor', event.get('executor', 'unspecified')),
                           ('radial_movement', category + ('_outer' if outer else '_inner'))):
            _accumulate(tables[table], key, values)
        if event.get('executor') == 'network':
            matched = event.get('matches_reference') is True
            is_survey = event.get('survey_macro') is True
            decision_values = {'decisions': 1, 'matches_reference': int(matched), 'changed': int(not matched),
                               'survey_decisions': int(is_survey), 'selected_move_m': move,
                               'candidates': event.get('candidates', 0)}
            _accumulate(tables['decision_stage'], progress, decision_values)
            _accumulate(tables['decision_category'], category, decision_values)
            decisions += 1
            matches += int(matched)
            survey_decisions += int(is_survey)
        if move > 1e-6:
            route.append({'step': b.steps, 'kind': action.kind, 'channel': action.channel,
                          'position': list(action.position), 'move_m': move, 'radius_m': radius,
                          'category': category, 'truth_phase': truth_phase,
                          'executor': event.get('executor'), 'matches_reference': event.get('matches_reference')})
        b.apply(action, response, event['request_id'])
        last_clock = response['virtual_time_s']
    return {'source_count': source_count, 'directional_count': sum(row.get('heading_deg') is not None for row in world_manifest['sources']),
            'steps': b.steps, 'virtual_s': b.virtual_time, 'move_m': sum(row['move_m'] for row in route),
            'survey_unique_sites': len(survey_points), 'outer_survey_unique_sites': len(outer_points),
            'network_decisions': decisions, 'reference_matches': matches, 'reference_changes': decisions - matches,
            'reference_change_rate': (decisions - matches) / decisions if decisions else None,
            'survey_decisions': survey_decisions, 'outcomes': dict(outcomes), 'tables': tables, 'moving_route': route}


def _audit_one(job):
    folder = Path(job['folder'])
    files = {name: folder / name for name in ('world.json', 'public.jsonl.gz', 'summary.json')}
    hashes = {name: sha256(path) for name, path in files.items()}
    manifest = json.loads(files['world.json'].read_text())
    events = load_events(files['public.jsonl.gz'])
    source_summary = json.loads(files['summary.json'].read_text())
    audit = audit_episode(manifest, events)
    mismatches = []
    actual_world_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if source_summary.get('world_sha256') != actual_world_hash:
        mismatches.append('world_sha256 differs from the actual manifest')
    for key, audited in (('complete', audit['complete']), ('source_count', audit['source_count']),
                         ('all_cleared', audit['all_cleared']), ('actions', audit['accepted_steps'])):
        if source_summary.get(key) != audited:
            mismatches.append(f'{key}: saved={source_summary.get(key)!r}, audited={audited!r}')
    for key, audited in (('virtual_time_s', audit['spent_virtual_s']), ('movement_m', audit['move_m'])):
        tolerance = CLOCK_TOLERANCE_S if key == 'virtual_time_s' else 1e-5
        if abs(source_summary.get(key, math.inf) - audited) > tolerance:
            mismatches.append(f'{key}: saved={source_summary.get(key)!r}, audited={audited!r}')
    if audit['complete'] and abs(source_summary.get('seconds_per_source', math.inf) - audit['seconds_per_source']) > CLOCK_TOLERANCE_S:
        mismatches.append('seconds_per_source differs')
    for key, value in audit['costs'].items():
        if abs(source_summary['costs'].get(key, math.inf) - value) > CLOCK_TOLERANCE_S:
            mismatches.append(f'cost component differs: {key}')
    diagnostics = diagnose_episode(manifest, events) if audit['ok'] else None
    # Ensure input files were not concurrently modified while being audited.
    for name, path in files.items():
        if sha256(path) != hashes[name]:
            mismatches.append(f'Input changed during audit: {name}')
    return {'batch': job['batch'], 'case': source_summary['case'], 'mode': source_summary['mode'],
            'seed': source_summary['seed'], 'split': source_summary.get('split'),
            'source_folder': str(folder.resolve()), 'world_sha256': actual_world_hash,
            'input_sha256': hashes, 'audit': audit, 'summary_matches_audit': not mismatches,
            'summary_mismatches': mismatches, 'diagnostics': diagnostics}


def aggregate_diagnostics(rows):
    output = {}
    groups = defaultdict(list)
    for row in rows:
        groups[(row['batch'], row['mode'])].append(row)
    for (batch, mode), selected in sorted(groups.items()):
        table = {}
        counts = Counter()
        outcomes = Counter()
        total_s = movement = 0.0
        for row in selected:
            diag = row['diagnostics']
            if diag is None:
                continue
            total_s += diag['virtual_s']
            movement += diag['move_m']
            for name in ('source_count', 'steps', 'survey_unique_sites', 'outer_survey_unique_sites',
                         'network_decisions', 'reference_matches', 'reference_changes', 'survey_decisions'):
                counts[name] += diag[name]
            outcomes.update(diag['outcomes'])
            for name, buckets in diag['tables'].items():
                target = table.setdefault(name, {})
                for category, values in buckets.items():
                    _accumulate(target, category, values)
        complete = all(row['audit']['complete'] and row['summary_matches_audit'] for row in selected)
        denominator = counts['source_count']
        for name, buckets in table.items():
            for category, values in buckets.items():
                if 'virtual_s' in values:
                    values['s_per_source'] = values['virtual_s'] / denominator if denominator else None
                    values['mean_move_m_per_map'] = values['move_m'] / len(selected)
                if 'decisions' in values:
                    values['change_rate'] = values['changed'] / values['decisions']
        output[batch + '/' + mode] = {'batch': batch, 'mode': mode, 'episodes': len(selected),
            'all_complete_and_consistent': complete, 'total_virtual_s': total_s, 'total_move_m': movement,
            'seconds_per_source': total_s / denominator if complete and denominator else None,
            'mean_move_m': movement / len(selected), 'mean_actions': counts['steps'] / len(selected),
            'mean_survey_sites': counts['survey_unique_sites'] / len(selected),
            'mean_outer_survey_sites': counts['outer_survey_unique_sites'] / len(selected),
            'reference_change_rate': counts['reference_changes'] / counts['network_decisions'] if counts['network_decisions'] else None,
            'counts': dict(counts), 'outcomes': dict(outcomes), 'tables': table}
    return output


def paired_diagnostics(rows, development_batch):
    selected = [row for row in rows if row['batch'] == development_batch]
    by_world = defaultdict(dict)
    for row in selected:
        by_world[row['world_sha256']][row['mode']] = row
    pairs = []
    for world_hash, modes in sorted(by_world.items()):
        if 'neural' not in modes or 'probes' not in modes:
            continue
        neural, probes = modes['neural'], modes['probes']
        n, p = neural['diagnostics'], probes['diagnostics']
        if n is None or p is None:
            continue
        nevents = load_events(Path(neural['source_folder']) / 'public.jsonl.gz')
        pevents = load_events(Path(probes['source_folder']) / 'public.jsonl.gz')
        prefix = 0
        for a, b in zip(nevents, pevents):
            if a['action'] != b['action']:
                break
            prefix += 1
        pairs.append({'seed': neural['seed'], 'world_sha256': world_hash,
                      'source_count': n['source_count'], 'delta_virtual_s': n['virtual_s'] - p['virtual_s'],
                      'delta_s_per_source': (n['virtual_s'] - p['virtual_s']) / n['source_count'],
                      'delta_move_m': n['move_m'] - p['move_m'], 'delta_steps': n['steps'] - p['steps'],
                      'neural_changes': n['reference_changes'], 'neural_decisions': n['network_decisions'],
                      'identical_action_prefix': prefix,
                      'delta_outer_survey_sites': n['outer_survey_unique_sites'] - p['outer_survey_unique_sites']})
    return {'pairs': sorted(pairs, key=lambda row: row['delta_virtual_s'], reverse=True),
            'neural_faster': sum(row['delta_virtual_s'] < -CLOCK_TOLERANCE_S for row in pairs),
            'neural_slower': sum(row['delta_virtual_s'] > CLOCK_TOLERANCE_S for row in pairs),
            'equal': sum(abs(row['delta_virtual_s']) <= CLOCK_TOLERANCE_S for row in pairs)}


def _write_report(out, summary, grouped, paired, development_batch):
    lines = ['# Q4 本地日志独立物理审计与费用诊断', '',
             f"审计 {summary['episodes']} 条完整日志，物理通过 {summary['physics_ok']} 条，完整完成 {summary['complete']} 条，原 summary 一致 {summary['summary_consistent']} 条。",
             f"独立检查 {summary['accepted_steps']} 次有效动作；最大计费误差 {summary['max_clock_error_s']:.9f} 秒。",
             '本报告只针对冻结本地模拟器；不是官方演练或正式测试结果。', '',
             '| 批次/方法 | 轨迹 | 秒/源 | 平均步数 | 平均移动米 |', '|---|---:|---:|---:|---:|']
    for key, group in grouped.items():
        rate = f"{group['seconds_per_source']:.3f}" if group['seconds_per_source'] is not None else '未通过完整审计'
        lines.append(f"| {key} | {group['episodes']} | {rate} | {group['mean_actions']:.3f} | {group['mean_move_m']:.3f} |")
    neural = grouped.get(development_batch + '/neural') if development_batch else None
    probes = grouped.get(development_batch + '/probes') if development_batch else None
    if neural and probes and neural['seconds_per_source'] is not None and probes['seconds_per_source'] is not None:
        delta = neural['seconds_per_source'] - probes['seconds_per_source']
        lines += ['', f"开发集 neural 比 probes 增加 {delta:.3f} 秒/源（{100*delta/probes['seconds_per_source']:.3f}%）。",
                  f"同图配对：改善 {paired['neural_faster']} 图，退步 {paired['neural_slower']} 图，相同 {paired['equal']} 图。",
                  f"网络 {neural['counts']['network_decisions']} 次决策中改选 {neural['counts']['reference_changes']} 次（{100*neural['reference_change_rate']:.3f}%）。",
                  '改选率是同一公开状态下与参考动作的分歧率，不能单独解释为错误率。', '']
        for name, title in (('action_category', '按执行动作分解'), ('truth_phase', '按事后任务阶段分解'),
                            ('radial_movement', '按目标半径分解；外侧定义为半径至少 1600 米')):
            lines += [f'## {title}', '', '| 类别 | probes 秒/源 | neural 秒/源 | 差值 | probes 移动米/图 | neural 移动米/图 |',
                      '|---|---:|---:|---:|---:|---:|']
            for category in sorted(set(probes['tables'][name]) | set(neural['tables'][name])):
                p = probes['tables'][name].get(category, {})
                n = neural['tables'][name].get(category, {})
                ps, ns = p.get('s_per_source', 0), n.get('s_per_source', 0)
                lines.append(f"| {category} | {ps:.3f} | {ns:.3f} | {ns-ps:+.3f} | {p.get('mean_move_m_per_map',0):.3f} | {n.get('mean_move_m_per_map',0):.3f} |")
            lines.append('')
        lines += ['## 公开进度与网络改选', '', '| 已清数量阶段 | 决策次数 | 改选次数 | 改选率 |', '|---|---:|---:|---:|']
        for stage, values in sorted(neural['tables']['decision_stage'].items()):
            lines.append(f"| {stage} | {values['decisions']} | {values['changed']} | {100*values['change_rate']:.3f}% |")
        lines += ['', '## 解释限制', '',
                  '动作/阶段费用是已执行轨迹的精确归因，不是对替代动作的因果收益估计；路径分歧会改变后续公开状态。',
                  'certification_tail 使用真实总数事后确定“已经全清、仍在证明其他频道不存在”的尾段，只用于评估。',
                  '外侧测量不一定是固定外环站位；半径分类同时允许经过移动的候选站，完整坐标仅保存在每局诊断路径中。',
                  '完整 prior、物理协议对照和未覆盖的官方要求见 protocol-comparison.json。', '']
    (out / '报告.md').write_text('\n'.join(lines))


def audit_batches(inputs, out, workers=16, development_batch=None):
    started = time.monotonic()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    jobs = []
    batch_inputs = {}
    for value in inputs:
        folder = Path(value).resolve()
        batch = json.loads((folder / 'batch.json').read_text())
        rows = json.loads((folder / 'summary.json').read_text())
        if batch.get('status') != 'complete' or batch['planned'] != batch['completed'] or batch['completed'] != len(rows):
            raise ValueError(f'Input batch is not complete: {folder}')
        batch_inputs[folder.name] = {name: sha256(folder / name) for name in
                                   ('batch.json', 'config.json', 'summary.json', 'aggregate.json') if (folder / name).exists()}
        for row in rows:
            jobs.append({'batch': folder.name, 'folder': str(folder / row['case'])})
    code = {str(path.relative_to(ROOT)): sha256(path) for parent in (ROOT / 'nnq4', ROOT / 'vendor')
            for path in sorted(parent.rglob('*.py'))}
    _dump(out / 'config.json', {'inputs': [str(Path(p).resolve()) for p in inputs], 'workers': workers,
                               'development_batch': development_batch, 'input_sha256': batch_inputs,
                               'code_sha256': code, 'audit_scope': 'frozen_local_q4_simulator', 'official_validation': False})
    _dump(out / 'protocol-comparison.json', PROTOCOL_COMPARISON)
    _dump(out / 'batch.json', {'status': 'running', 'planned': len(jobs), 'completed': 0})
    rows = []
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(_audit_one, job) for job in jobs]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                _dump(out / 'episodes' / row['batch'] / (row['case'] + '.json'), row)
                if len(rows) % 16 == 0 or len(rows) == len(jobs):
                    _dump(out / 'batch.json', {'status': 'running', 'planned': len(jobs), 'completed': len(rows)})
                    print(f'Audited {len(rows)}/{len(jobs)}', flush=True)
        grouped = aggregate_diagnostics(rows)
        paired = paired_diagnostics(rows, development_batch) if development_batch else {}
        summary = {'episodes': len(rows), 'physics_ok': sum(row['audit']['ok'] for row in rows),
                   'complete': sum(row['audit']['complete'] for row in rows),
                   'summary_consistent': sum(row['summary_matches_audit'] for row in rows),
                   'accepted_steps': sum(row['audit']['accepted_steps'] for row in rows),
                   'max_clock_error_s': max(row['audit']['max_clock_error_s'] for row in rows),
                   'truth_support_checks': sum(row['audit']['truth_support_checks'] for row in rows),
                   'absence_certificates_checked': sum(row['audit']['absence_certificates_checked'] for row in rows),
                   'exact_surround_episodes': sum(bool(row['audit'].get('exact_surround_certificates')) for row in rows),
                   'exact_surround_channels': sum(len(row['audit'].get('exact_surround_certificates', [])) for row in rows),
                   'real_time_s': time.monotonic() - started,
                   'errors': [{'batch': row['batch'], 'case': row['case'], 'errors': row['audit']['errors'],
                               'summary_mismatches': row['summary_mismatches']} for row in rows
                              if not row['audit']['ok'] or not row['audit']['complete'] or not row['summary_matches_audit']]}
        _dump(out / 'summary.json', summary)
        _dump(out / 'groups.json', grouped)
        _dump(out / 'paired-development.json', paired)
        _dump(out / 'episode-index.json', [{key: row[key] for key in ('batch', 'case', 'mode', 'seed', 'input_sha256')} for row in rows])
        _write_report(out, summary, grouped, paired, development_batch)
        _dump(out / 'batch.json', {'status': 'complete', 'planned': len(jobs), 'completed': len(rows),
                                  'all_ok': not summary['errors']})
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return out
    except BaseException as exc:
        _dump(out / 'batch.json', {'status': 'failed', 'planned': len(jobs), 'completed': len(rows),
                                  'error': f'{type(exc).__name__}: {exc}'})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--development-batch')
    args = parser.parse_args()
    out = args.out or ROOT / 'results' / datetime.now().strftime('%Y%m%d-%H%M%S-%f-physical-audit')
    print(f'OUTPUT={out}', flush=True)
    audit_batches(args.inputs, out, args.workers, args.development_batch)


if __name__ == '__main__':
    main()
