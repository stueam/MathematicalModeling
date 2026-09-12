"""One explicitly bounded local round: reuse 213 states, train, compare 32 maps."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

from .bridge import ROOT, Config
from .experiment import dump, new_run, run_episode, summarize
from .relative import CANDIDATE_RULE
from .relative_data import completed_records, reuse_episode, build_manifest, sha256
from .relative_training import train_relative
from .round_control import bounded_map, supervise


DEFAULT_CHECKPOINT = ROOT / 'results/20260912-165053-233852-train/training/best.pt'
DEFAULT_DATA = ROOT / 'results/20260912-170247-695356-cost-labels/labels/manifest.json'
DEV_SEEDS = tuple(range(280020000, 280020032))


def decision_analysis(folder):
    choices, owner_costs, baseline_cost = [], {}, 0.
    with gzip.open(Path(folder) / 'public.jsonl.gz', 'rt') as stream:
        for line in stream:
            event = json.loads(line)
            after = event['response']['virtual_time_s']
            cost = after - baseline_cost
            baseline_cost = after
            owner = event.get('macro_owner') if event['executor'] == 'macro_continuation' else event['executor']
            owner = owner or 'macro_continuation'
            owner_costs[owner] = owner_costs.get(owner, 0.) + cost
            if event['executor'] not in ('baseline', 'network_override'):
                continue
            changed = event['selected'] != event['reference']
            selected = event['selected_slot']
            if changed != (event['executor'] == 'network_override'):
                raise ValueError('Override executor attribution mismatch')
            if any(head[0] != 0. for head in event['head_delta_s']):
                raise ValueError('Nonzero baseline residual in deployed trace')
            if changed and not event['upper_delta_s'][selected] < -20.:
                raise ValueError('Override violates conservative threshold')
            choices.append({key: event[key] for key in
                            ('request_id', 'executor', 'reference_action', 'selected_action', 'comparison_indices',
                             'comparison_reasons', 'selected_slot', 'predicted_delta_s', 'ensemble_std_s',
                             'upper_delta_s', 'calibration_margin_s', 'move_m')})
    return {'decisions': len(choices), 'overrides': sum(c['executor'] == 'network_override' for c in choices),
            'costs_by_decision_owner_s_including_macro_channels': owner_costs, 'choices': choices}


def evaluate_gate(rows, expected_seeds=DEV_SEEDS):
    reasons = []
    expected = {(seed, mode) for seed in expected_seeds for mode in ('probes', 'neural', 'hybrid')}
    observed = [(r['seed'], r['mode']) for r in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        reasons.append('incomplete_or_duplicate_32_map_three_policy_pairs')
    complete = all(r['complete'] and r['audit_ok'] and r['public_complete'] for r in rows)
    if not complete:
        reasons.append('mission_or_independent_audit_failed')
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r['seed'], []).append(r)
    if any(len({r['world_sha256'] for r in group}) != 1 for group in by_seed.values()):
        reasons.append('paired_map_hash_mismatch')
    aggregate = summarize(rows) if rows else {'groups': {}, 'paired_to_probes': {}}
    pair = aggregate['paired_to_probes'].get('hybrid')
    if not pair or pair['relative_improvement'] < .02:
        reasons.append('relative_improvement_below_2_percent')
    if not pair or pair['paired_map_95pct'][0] <= 0:
        reasons.append('paired_bootstrap_lower_bound_not_positive')
    changes = sum(r.get('stats', {}).get('network_overrides', 0) for r in rows if r['mode'] == 'hybrid')
    if not changes:
        reasons.append('no_network_override')
    return {'continue_condition_met': not reasons, 'conclusion': 'continue_eligible' if not reasons else 'stop',
            'reasons': reasons, 'network_overrides': changes, 'aggregate': aggregate,
            'automatic_next_round': False, 'final_300_target_met': False,
            'final_test_run': False, 'stress_test_run': False,
            'metric': '1 - total_hybrid_virtual_seconds / total_baseline_virtual_seconds',
            'bootstrap': '10000 paired map resamples, seed 812, descriptive 95% interval',
            'local_simulator_only': True}


def write_report(out, rows, gate, training):
    lines = ['# Q4 相对成本首轮（本地仿真）', '',
             '结论：' + ('满足继续条件；本轮结束，不自动启动下一轮。' if gate['continue_condition_met'] else
                        '停止本轮优化；未满足继续条件，不扩大采集。'), '',
             '| 策略 | 完成 | 秒/源 | 米/图 | >1km移动次数 | 失败清除 | 实际秒/图 | 后备虚拟秒 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode, label in (('probes', '基线'), ('neural', 'BC-v1'), ('hybrid', '相对成本混合')):
        group = [r for r in rows if r['mode'] == mode]
        if not group:
            continue
        info = gate['aggregate']['groups'][mode]
        seconds = info['aggregate_s_per_source']
        score = f'{seconds:.3f}' if seconds is not None else '未完成'
        lines.append(f"| {label} | {info['complete']}/{len(group)} | {score} | {info['mean_movement_m']:.3f} | "
                     f"{sum(r['long_moves_over_1km'] for r in group)} | {sum(r['failed_clears'] for r in group)} | "
                     f"{info['mean_real_time_s']:.3f} | {sum(r['executor_costs_s'].get('fallback', 0) for r in group):.3f} |")
    pair = gate['aggregate']['paired_to_probes'].get('hybrid')
    if pair:
        low, high = pair['paired_map_95pct']
        lines += ['', f"相对基线改善 {100*pair['relative_improvement']:.3f}%；配对 bootstrap 95% 区间 "
                  f"[{100*low:.3f}%, {100*high:.3f}%]。网络改选 {gate['network_overrides']} 次。"]
    lines += ['', '未通过条件：' + ('；'.join(gate['reasons']) or '无'), '',
              '只读取原 manifest 中 213 个已完成成本状态。原始世界按固定哈希分组；旧 BC 编码器已见过部分世界，'
              '校准组不作为整个模型的独立测试。三个成本头分别按世界重采样，冻结编码器，'
              '100 秒尺度加权 Huber 回归；不使用旧绝对 Q 头或硬分类最优标签。', '',
              f"校准余量 {training['calibration_margin_s']:.3f} 秒；最佳 epoch {training['selected_epochs']}。"
              '仅当平均预测差 + 2 倍模型间标准差 + 余量 < −20 秒时改选。这是经验控制，不是性能保证。', '',
              '基线、BC-v1 和混合策略均重新执行同 32 张开发地图，未复用旧基线日志。'
              '真值只用于环境和事后独立物理审计；控制器与公开重放不读取 world.json。'
              '逐频道扫描宏全部请求和计费，后备费用另列。', '',
              '300 秒/源目标尚未在预留主测试集验收；64 图预留测试、48 局压力验证、'
              '新状态采样及外部候选/S21 消融均未执行。本轮后不自动串接实验。', '',
              '模型见 training/best.pt；逐图结果见 benchmark/summary.json；改选及宏费用归因见 decisions/；'
              '运行时间与进程清理见 control.json；数据覆盖与隔离见 reuse/manifest.json。', '',
              '方法背景：[CAtNIPP](https://proceedings.mlr.press/v205/cao23b.html)、'
              '[ARiADNE](https://arxiv.org/abs/2301.11575) 的注意力表示；'
              '[AggreVaTe](https://arxiv.org/abs/1406.5979) 的后续成本监督。'
              '本实现是 Q4 的混合适配，未复现这些论文的训练或性能结果。', '']
    (out / '报告.md').write_text('\n'.join(lines))


def pipeline(out):
    torch.set_num_threads(1)
    out = Path(out)
    config = json.loads((out / 'config.json').read_text())
    dispatch = float(os.environ['Q4_ROUND_DISPATCH_DEADLINE'])
    hard = float(os.environ['Q4_ROUND_HARD_DEADLINE'])
    rows = []
    try:
        jobs, provenance = completed_records(config['data'])
        old_hashes = json.loads((Path(config['data']).parent.parent / 'source_sha256.json').read_text())
        current_hashes = json.loads((out / 'source_sha256.json').read_text())
        frozen_keys = [key for key in old_hashes if key.startswith('vendor/')]
        if not frozen_keys or any(old_hashes[key] != current_hashes.get(key) for key in frozen_keys):
            raise ValueError('Saved cost continuation simulator/planner version has changed')
        provenance['frozen_continuation_source_hashes_verified'] = True
        dump(out / 'reuse-inputs.json', provenance)
        reuse = out / 'reuse'
        (reuse / 'states').mkdir(parents=True)
        jobs = [{**job, 'checkpoint': config['checkpoint'], 'seed': config['seed'],
                 'out': str(reuse / 'states'), 'hard_deadline': hard} for job in jobs]

        def record_reuse(result):
            manifest = build_manifest(reuse)
            print(f"public reuse: {manifest['states']}/213 states, {manifest['labeled_alternatives']} evaluated alternatives", flush=True)

        status = bounded_map(reuse_episode, jobs, config['workers'], dispatch, hard, record_reuse)
        dump(out / 'reuse-batch.json', status)
        manifest = build_manifest(reuse)
        if manifest['states'] != 213 or status['status'] != 'complete':
            raise TimeoutError('Public replay did not complete before dispatch cutoff')
        if time.monotonic() >= dispatch:
            raise TimeoutError('No new training after 12 minute dispatch cutoff')
        checkpoint = train_relative(reuse / 'manifest.json', config['checkpoint'], out / 'training',
                                    seed=config['seed'], hard_deadline=hard, device=config['device'])
        if time.monotonic() >= dispatch:
            raise TimeoutError('No new benchmark after 12 minute dispatch cutoff')
        benchmark = out / 'benchmark'
        benchmark.mkdir()
        planner = asdict(Config())
        config_hash = hashlib.sha256(json.dumps(planner, sort_keys=True).encode()).hexdigest()
        code_hashes = json.loads((out / 'source_sha256.json').read_text())
        simulator = {k: v for k, v in code_hashes.items() if k.startswith('vendor/')}
        simulator_hash = hashlib.sha256(json.dumps(simulator, sort_keys=True).encode()).hexdigest()
        dump(benchmark / 'provenance.json', {'baseline_logs_reused': False, 'planner_config': planner,
                                           'planner_config_sha256': config_hash,
                                           'simulator_sha256': simulator_hash,
                                           'checkpoint_sha256': sha256(checkpoint),
                                           'bc_checkpoint_sha256': sha256(config['checkpoint']),
                                           'seeds': DEV_SEEDS})
        jobs = [{'seed': seed, 'mode': mode, 'checkpoint': str(checkpoint) if mode == 'hybrid' else config['checkpoint'],
                 'selection': 'relative' if mode == 'hybrid' else 'policy',
                 'features': 'v2' if mode == 'hybrid' else 'v1', 'out': str(benchmark),
                 'macro_budget': 160, 'allow_fallback': True, 'split': 'development',
                 'real_limit': 1200.}
                for seed in DEV_SEEDS for mode in ('probes', 'neural', 'hybrid')]

        def record_episode(row):
            row.update(simulator_sha256=simulator_hash, planner_config_sha256=config_hash)
            rows.append(row)
            dump(benchmark / 'summary.json', sorted(rows, key=lambda r: (r['seed'], r['mode'])))
            dump(benchmark / 'batch.json', dict(status='running', planned=96, completed=len(rows)))
            print(f"benchmark {len(rows)}/96 {row['case']} complete={row['complete']} s/src={row['seconds_per_source']}", flush=True)

        status = bounded_map(run_episode, jobs, config['workers'], dispatch, hard, record_episode)
        dump(benchmark / 'batch.json', status)
        gate = evaluate_gate(rows)
        dump(benchmark / 'aggregate.json', gate['aggregate'])
        for row in rows:
            if row['mode'] == 'hybrid':
                detail = decision_analysis(row['folder'])
                if detail['overrides'] != row['stats']['network_overrides']:
                    raise ValueError('Override log count mismatch')
                dump(out / 'decisions' / f"{row['seed']}.json", detail)
        training = json.loads((out / 'training/training-summary.json').read_text())
        dump(out / 'conclusion.json', gate)
        write_report(out, rows, gate, training)
        dump(out / 'batch.json', dict(status='complete' if status['status'] == 'complete' else 'dispatch_cutoff',
                                      planned=96, completed=len(rows), automatic_next_round=False))
    except BaseException as exc:
        if (out / 'reuse/states').exists():
            build_manifest(out / 'reuse')
        dump(out / 'batch.json', dict(status='interrupted_or_failed', error=f'{type(exc).__name__}: {exc}',
                                      planned=96, completed=len(rows), automatic_next_round=False))
        dump(out / 'conclusion.json', {'conclusion': 'stop', 'continue_condition_met': False,
                                      'reasons': [f'{type(exc).__name__}: {exc}'], 'automatic_next_round': False,
                                      'local_simulator_only': True})
        raise


def launch(checkpoint=DEFAULT_CHECKPOINT, data=DEFAULT_DATA, workers=16, seed=280120000, device='cuda'):
    config = dict(checkpoint=str(Path(checkpoint).resolve()), data=str(Path(data).resolve()), workers=workers,
                  seed=seed, candidate_rule=CANDIDATE_RULE, cost_states=213, development_maps=32,
                  hard_limit_s=900, dispatch_limit_s=720, baseline_logs_reused=False, automatic_next_round=False,
                  device=device)
    out = new_run('relative-round', config)
    print('OUTPUT=' + str(out), flush=True)
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = supervise([sys.executable, '-m', 'nnq4.relative_round', '--worker', str(out)], out, env=env)
    if result['status'] != 'complete':
        if (out / 'reuse/states').exists():
            build_manifest(out / 'reuse')
        benchmark_summary = out / 'benchmark/summary.json'
        finished = len(json.loads(benchmark_summary.read_text())) if benchmark_summary.exists() else 0
        dump(out / 'batch.json', {'status': result['status'], 'planned': 96, 'completed': finished,
                                 'automatic_next_round': False})
        if (out / 'benchmark').exists():
            dump(out / 'benchmark/batch.json', {'status': result['status'], 'planned': 96, 'completed': finished})
        dump(out / 'conclusion.json', {'conclusion': 'stop', 'continue_condition_met': False,
                                      'reasons': [result['status']], 'automatic_next_round': False,
                                      'local_simulator_only': True})
    return out, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=280120000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.worker:
        pipeline(args.worker)
    else:
        _, result = launch(args.checkpoint, args.data, args.workers, args.seed, args.device)
        if result['status'] != 'complete':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
