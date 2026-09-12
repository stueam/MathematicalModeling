"""One 15-minute local cost/advantage training and ablation experiment."""
import argparse
import csv
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

from .advantage import GUARD, choose_supported
from .advantage_data import restore_episode, manifest
from .advantage_training import train_variant
from .bridge import ROOT, Config
from .experiment import dump, new_run, run_episode, summarize
from .relative_data import completed_records, sha256
from .relative_round import DEFAULT_DATA, DEFAULT_CHECKPOINT, DEV_SEEDS
from .round_control import bounded_map, supervise


PRIOR = ROOT / 'results/20260912-202941-255694-relative-round/training/best.pt'
MODES = ('probes', 'frozen_advantage', 'finetuned_advantage')
SOURCES = {
    'AWR': 'https://arxiv.org/abs/1910.00177',
    'AggreVaTeD': 'https://arxiv.org/abs/1703.01030',
    'AWAC': 'https://arxiv.org/abs/2006.09359',
    'IQL': 'https://arxiv.org/abs/2110.06169',
    'SPIBB': 'https://proceedings.mlr.press/v97/laroche19a.html'}


def paired_improvement(baseline, candidate):
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError('Paired improvement requires identical seed sets')
    keys = sorted(baseline)
    base = torch.tensor([baseline[k] for k in keys], dtype=torch.float64)
    new = torch.tensor([candidate[k] for k in keys], dtype=torch.float64)
    ix = torch.randint(len(keys), (10000, len(keys)), generator=torch.Generator().manual_seed(812))
    return {'relative_improvement': float(1-new.sum()/base.sum()),
            'paired_95pct': torch.quantile(1-new[ix].sum(1)/base[ix].sum(1),
                                          torch.tensor([.025, .975], dtype=torch.float64)).tolist()}


def analyze_decisions(folder):
    records, cost_by_owner, clock = [], {}, 0.
    with gzip.open(Path(folder) / 'public.jsonl.gz', 'rt') as stream:
        for line in stream:
            event = json.loads(line)
            after = event['response']['virtual_time_s']
            owner = event.get('macro_owner', event['executor']) if event['executor'] == 'macro_continuation' else event['executor']
            cost_by_owner[owner] = cost_by_owner.get(owner, 0.) + after - clock
            clock = after
            if event['executor'] not in ('network_override', 'baseline'):
                continue
            result = choose_supported(torch.tensor(event['head_delta_s']), torch.tensor(event['actor_logits']))
            if result[0] != event['selected_slot'] or event['guard'] != GUARD:
                raise ValueError('Recorded action does not reproduce the declared deployment guard')
            if (event['executor'] == 'network_override') != (event['selected_slot'] != 0):
                raise ValueError('Override attribution mismatch')
            records.append({key: event[key] for key in ('request_id', 'executor', 'reference_action', 'selected_action',
                            'comparison_indices', 'comparison_reasons', 'head_delta_s', 'upper_delta_s',
                            'actor_logits', 'selected_slot', 'guard', 'move_m')})
    return {'decisions': len(records), 'overrides': sum(r['executor'] == 'network_override' for r in records),
            'cost_by_owner_s': cost_by_owner, 'records': records}


def conclusion(rows):
    expected = {(seed, mode) for seed in DEV_SEEDS for mode in MODES}
    observed = [(r['seed'], r['mode']) for r in rows]
    complete = (len(set(observed)) == len(observed) and set(observed) == expected and
                all(r['complete'] and r['audit_ok'] and r['public_complete'] for r in rows))
    for seed in DEV_SEEDS:
        same = [r for r in rows if r['seed'] == seed]
        if any(len({r[k] for r in same}) > 1 for k in ('world_sha256', 'simulator_sha256', 'planner_config_sha256')):
            complete = False
    aggregate = summarize(rows) if rows else {'groups': {}, 'paired_to_probes': {}}
    evaluations = {}
    for mode in MODES[1:]:
        pair = aggregate['paired_to_probes'].get(mode)
        changes = sum(r['stats'].get('network_overrides', 0) for r in rows if r['mode'] == mode)
        reasons = []
        if not complete:
            reasons.append('incomplete_pairs_or_failed_audit')
        if not pair or pair['relative_improvement'] < .02:
            reasons.append('improvement_below_2_percent')
        if not pair or pair['paired_map_95pct'][0] <= 0:
            reasons.append('bootstrap_lower_bound_not_positive')
        if not changes:
            reasons.append('no_network_override')
        evaluations[mode] = {'continue_condition_met': not reasons, 'reasons': reasons, 'overrides': changes}
    cross = None
    if complete:
        cross = paired_improvement({r['seed']: r['virtual_time_s'] for r in rows if r['mode'] == MODES[1]},
                                   {r['seed']: r['virtual_time_s'] for r in rows if r['mode'] == MODES[2]})
    return {'all_complete_and_audited': complete, 'aggregate': aggregate, 'variants': evaluations,
            'finetuned_vs_frozen': cross, 'automatic_next_round': False,
            'conclusion': 'continue_eligible' if any(v['continue_condition_met'] for v in evaluations.values()) else 'stop',
            'final_300_target_validated': False, 'reserved_test_or_stress_run': False, 'local_simulator_only': True}


def report(out, rows, result):
    paired = []
    for seed in DEV_SEEDS:
        group = {r['mode']: r for r in rows if r['seed'] == seed}
        if set(group) != set(MODES):
            continue
        record = {'seed': seed, 'source_count': group['probes']['source_count']}
        for mode in MODES:
            row = group[mode]
            for key in ('virtual_time_s', 'movement_m', 'real_time_s', 'audit_time_s', 'failed_clears', 'long_moves_over_1km'):
                record[mode + '_' + key] = row[key]
            record[mode + '_fallback_s'] = row['executor_costs_s'].get('fallback', 0.)
            record[mode + '_complete'] = row['complete']
            if mode != 'probes':
                record[mode + '_saved_s'] = group['probes']['virtual_time_s'] - row['virtual_time_s']
                record[mode + '_overrides'] = row['stats']['network_overrides']
        paired.append(record)
    if paired:
        with (out / 'paired-results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    lines = ['# Q4 优势加权与注意力微调实验（本地仿真）', '',
             '结论：' + ('至少一个方案满足继续条件；本轮结束，不自动串接。' if result['conclusion'] == 'continue_eligible'
                        else '未满足继续条件，本轮结束。'), '',
             '| 策略 | 完成 | 秒/源 | 米/图 | >1km移动 | 失败清除 | 实际秒/图 | 后备虚拟秒 | 改选 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        group = [r for r in rows if r['mode'] == mode]
        if not group:
            continue
        info = result['aggregate']['groups'][mode]
        score = info['aggregate_s_per_source']
        value = f'{score:.3f}' if score is not None else '未完成'
        lines.append(f"| {mode} | {info['complete']}/{len(group)} | {value} | {info['mean_movement_m']:.3f} | "
                     f"{sum(r['long_moves_over_1km'] for r in group)} | {sum(r['failed_clears'] for r in group)} | "
                     f"{info['mean_real_time_s']:.3f} | {sum(r['executor_costs_s'].get('fallback',0) for r in group):.3f} | "
                     f"{sum(r['stats'].get('network_overrides',0) for r in group)} |")
    for mode, pair in result['aggregate']['paired_to_probes'].items():
        low, high = pair['paired_map_95pct']
        lines += ['', f"{mode} 相对基线改善 {100*pair['relative_improvement']:.3f}%，配对 bootstrap 95% 区间 "
                  f"[{100*low:.3f}%, {100*high:.3f}%]。"]
    if result['finetuned_vs_frozen']:
        cross = result['finetuned_vs_frozen']
        lines += ['', f"微调相对冻结对照改善 {100*cross['relative_improvement']:.3f}%；这组对比才用于判断注意力微调的增量效果。"]
    lines += ['', '本轮重新利用原 manifest 的 213 个完整状态和全部 844 个非基线候选成本标签。'
              '通过公开历史重放补全 v2 特征；旧数据目录未恢复、没有新增成本世界采样。'
              '按原始世界固定哈希划分训练/校准，所有变体使用完全相同的数据和划分。', '',
              '训练损失为标准误加权 Huber + 0.5×软偏好损失 + 0.3×优势加权策略损失 + 0.1×BC KL 约束。'
              '偏好目标用标准误软化，避免将三世界均值最小者当成必然正确的硬标签。'
              '微调版注意力学习率 3e-5，成本/策略头 1e-3；冻结版仅更新新增头。'
              '三个成本头按世界 bootstrap，微调版共享可训练编码器，不能把模型间差异当成完整独立模型不确定性。', '',
              '改选规则在开发评测前固定：平均成本差 + 1 倍头间标准差 < −10 秒、三个头均预测节省，'
              '且策略头对该动作的概率相对基线超过 1.1 倍。候选仍采用统一四槽规则。'
              '这替换了上轮固定 129.96 秒余量，不是通过本轮开发地图调出来的门槛，也不提供性能保证。', '',
              '校准用于选 epoch 和报告误差；遇到未标注的拟改选动作会单独记录，不用已评价候选替换后冒充完整策略评估。'
              '旧编码器曾见过部分校准世界；整局收益只按配对开发组报告，不冒充最终留出测试。', '',
              '完整性仍由原逐频道几何证明和有限后备维护；测量扫描宏逐频道计费，真值仅供模拟器及事后审计。'
              '未运行预留 64 图、48 局压力测试或官方接口。300 秒/源目标尚未验收。', '',
              '模型：training-frozen/best.pt 和 training-finetuned/best.pt；逐图费用：paired-results.csv；'
              '改选记录：decisions/；原始日志与物理审计：benchmark/；退出与进程清理：control.json。', '',
              '论文依据：采用 [AWR](https://arxiv.org/abs/1910.00177) 的优势加权更新思想和 '
              '[AggreVaTeD](https://arxiv.org/abs/1703.01030) 的可微成本敏感学习思想。'
              'AWAC/IQL 提供离线学习参考，但本实现没有 Bellman 自举，不称为 AWAC 或 IQL 的复现；'
              '也不把 SPIBB 在其假设下的安全改进保证套用到当前神经模型。', '']
    (out / '报告.md').write_text('\n'.join(lines))


def pipeline(out):
    out = Path(out)
    torch.set_num_threads(1)
    config = json.loads((out / 'config.json').read_text())
    dispatch, hard = (float(os.environ[k]) for k in ('Q4_ROUND_DISPATCH_DEADLINE', 'Q4_ROUND_HARD_DEADLINE'))
    rows = []
    dump(out / 'batch.json', {'status': 'running', 'phase': 'public_replay', 'planned': 96, 'completed': 0})
    try:
        jobs, provenance = completed_records(config['data'])
        old_hashes = json.loads((Path(config['data']).parent.parent / 'source_sha256.json').read_text())
        new_hashes = json.loads((out / 'source_sha256.json').read_text())
        vendor = {k: v for k, v in old_hashes.items() if k.startswith('vendor/')}
        if not vendor or any(v != new_hashes.get(k) for k, v in vendor.items()):
            raise ValueError('Frozen cost continuation code changed')
        dump(out / 'reuse-inputs.json', provenance)
        cache = out / 'data'
        (cache / 'states').mkdir(parents=True)
        bc_sha = sha256(config['bc_checkpoint'])
        jobs = [{**job, 'bc_checkpoint': config['bc_checkpoint'], 'bc_sha256': bc_sha,
                 'out': str(cache / 'states'), 'hard_deadline': hard} for job in jobs]
        def commit(_):
            data = manifest(cache)
            print(f"public states {data['states']}/213, all labeled alternatives {data['labeled_alternatives']}/844", flush=True)
        status = bounded_map(restore_episode, jobs, config['workers'], dispatch, hard, commit)
        dump(out / 'reuse-batch.json', status)
        data = manifest(cache)
        if status['status'] != 'complete' or data['states'] != 213 or data['labeled_alternatives'] != 844:
            raise TimeoutError('Incomplete data recovery before cutoff')
        trained = {}
        for finetune in (False, True):
            if time.monotonic() >= dispatch:
                raise TimeoutError('Training dispatch cutoff')
            name = 'finetuned' if finetune else 'frozen'
            trained[name + '_advantage'] = train_variant(cache / 'manifest.json', config['checkpoint'], config['bc_checkpoint'],
                    out / ('training-' + name), finetune, config['device'], config['seed'],
                    hard_deadline=hard, dispatch_deadline=dispatch)
        benchmark = out / 'benchmark'
        benchmark.mkdir()
        simulator_hash = hashlib.sha256(json.dumps(vendor, sort_keys=True).encode()).hexdigest()
        planner_hash = hashlib.sha256(json.dumps(asdict(Config()), sort_keys=True).encode()).hexdigest()
        dump(benchmark / 'provenance.json', {'simulator_sha256': simulator_hash, 'planner_config_sha256': planner_hash,
                                           'seeds': DEV_SEEDS, 'baseline_logs_reused': False, 'guard': GUARD})
        jobs = [{'seed': seed, 'mode': mode, 'checkpoint': str(trained[mode]) if mode != 'probes' else None,
                 'selection': 'advantage', 'features': 'v2' if mode != 'probes' else 'v1',
                 'out': str(benchmark), 'macro_budget': 160, 'allow_fallback': True, 'split': 'development',
                 'real_limit': 1200.} for seed in DEV_SEEDS for mode in MODES]
        def consume(row):
            row.update(simulator_sha256=simulator_hash, planner_config_sha256=planner_hash)
            rows.append(row)
            dump(benchmark / 'summary.json', sorted(rows, key=lambda r: (r['seed'], r['mode'])))
            dump(benchmark / 'batch.json', {'status': 'running', 'planned': 96, 'completed': len(rows)})
            print(f"benchmark {len(rows)}/96 {row['case']} complete={row['complete']} s/src={row['seconds_per_source']}", flush=True)
        status = bounded_map(run_episode, jobs, config['workers'], dispatch, hard, consume)
        dump(benchmark / 'batch.json', status)
        result = conclusion(rows)
        dump(benchmark / 'aggregate.json', result['aggregate'])
        for row in rows:
            if row['mode'] != 'probes':
                detail = analyze_decisions(row['folder'])
                if detail['overrides'] != row['stats']['network_overrides']:
                    raise ValueError('Override count differs between logs and summary')
                dump(out / 'decisions' / (row['case'] + '.json'), detail)
        dump(out / 'conclusion.json', result)
        report(out, rows, result)
        dump(out / 'batch.json', {'status': status['status'], 'planned': 96, 'completed': len(rows),
                                 'all_complete_and_audited': result['all_complete_and_audited']})
    except BaseException as exc:
        if (out / 'data/states').exists():
            manifest(out / 'data')
        dump(out / 'batch.json', {'status': 'failed_or_interrupted', 'planned': 96, 'completed': len(rows),
                                 'error': f'{type(exc).__name__}: {exc}'})
        raise


def launch(data=DEFAULT_DATA, checkpoint=PRIOR, bc_checkpoint=DEFAULT_CHECKPOINT, workers=16, device='cuda', seed=280130000):
    config = dict(data=str(Path(data).resolve()), checkpoint=str(Path(checkpoint).resolve()),
                  bc_checkpoint=str(Path(bc_checkpoint).resolve()), workers=workers, device=device, seed=seed,
                  guard=GUARD, candidate_slots=4, training_all_evaluated_candidates=True,
                  new_cost_sampling=False, hard_limit_s=900, dispatch_limit_s=720, automatic_next_round=False,
                  baseline_logs_reused=False, research_sources=SOURCES)
    out = new_run('advantage-round', config)
    print('OUTPUT=' + str(out), flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    control = supervise([sys.executable, '-m', 'nnq4.advantage_round', '--worker', str(out)], out, env=env)
    if control['status'] != 'complete':
        if (out / 'data/states').exists():
            manifest(out / 'data')
        summary = out / 'benchmark/summary.json'
        rows = json.loads(summary.read_text()) if summary.exists() else []
        dump(out / 'batch.json', {'status': control['status'], 'planned': 96, 'completed': len(rows)})
        dump(out / 'conclusion.json', {'conclusion': 'stop', 'reason': control['status'], 'automatic_next_round': False})
    return out, control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--checkpoint', type=Path, default=PRIOR)
    parser.add_argument('--bc-checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=280130000)
    args = parser.parse_args()
    if args.worker:
        pipeline(args.worker)
    else:
        _, result = launch(args.data, args.checkpoint, args.bc_checkpoint, args.workers, args.device, args.seed)
        if result['status'] != 'complete':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
