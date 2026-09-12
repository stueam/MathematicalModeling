"""Finish unstarted development maps after the user extends this round to 30 min.

The original start time remains authoritative. No training or data collection
is repeated, and completed pairs are preserved with their original provenance.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

from .advantage import GUARD
from .advantage_round import DEV_SEEDS, MODES, analyze_decisions, conclusion, report
from .experiment import dump, run_episode
from .relative_data import sha256
from .round_control import bounded_map, live_group_members, supervise


def remaining_budget(start_epoch, now_epoch):
    elapsed = max(0., now_epoch - start_epoch)
    # One second also covers the timestamp write and supervisor launch overhead.
    hard, dispatch = 1800. - elapsed - 1., 1620. - elapsed - 1.
    if dispatch <= 0 or hard <= dispatch:
        raise TimeoutError('Original round has passed the extended dispatch deadline')
    return hard, dispatch


def worker(out):
    torch.set_num_threads(1)
    out = Path(out)
    config = json.loads((out / 'config.json').read_text())
    benchmark = out / 'benchmark'
    rows = json.loads((benchmark / 'summary.json').read_text())
    provenance = json.loads((benchmark / 'provenance.json').read_text())
    if provenance['guard'] != GUARD:
        raise ValueError('Deployment guard changed across budget extension')
    expected = {(seed, mode) for seed in DEV_SEEDS for mode in MODES}
    observed = {(row['seed'], row['mode']) for row in rows}
    if len(observed) != len(rows) or not observed <= expected:
        raise ValueError('Unexpected existing episode identities')
    if not all(row['complete'] and row['audit_ok'] and row['public_complete'] for row in rows):
        raise ValueError('Failed episodes must not be silently replaced')
    jobs = []
    for seed in DEV_SEEDS:
        for mode in MODES:
            if (seed, mode) in observed:
                continue
            checkpoint = out / ('training-' + mode.removesuffix('_advantage')) / 'best.pt'
            jobs.append(dict(seed=seed, mode=mode, checkpoint=str(checkpoint) if mode != 'probes' else None,
                             selection='advantage', features='v2' if mode != 'probes' else 'v1',
                             out=str(benchmark), macro_budget=160, allow_fallback=True,
                             split='development', real_limit=1200.))
    dispatch, hard = (float(os.environ[k]) for k in ('Q4_ROUND_DISPATCH_DEADLINE', 'Q4_ROUND_HARD_DEADLINE'))

    def consume(row):
        row.update({k: provenance[k] for k in ('simulator_sha256', 'planner_config_sha256')})
        rows.append(row)
        dump(benchmark / 'summary.json', sorted(rows, key=lambda r: (r['seed'], r['mode'])))
        dump(out / 'batch.json', dict(status='running', planned=96, completed=len(rows), phase='extended_budget'))
        print(f"benchmark {len(rows)}/96 {row['case']} complete={row['complete']} s/src={row['seconds_per_source']}", flush=True)

    status = bounded_map(run_episode, jobs, config['workers'], dispatch, hard, consume)
    dump(out / 'extension-batch.json', status)
    result = conclusion(rows)
    for row in rows:
        if row['mode'] != 'probes':
            detail = analyze_decisions(row['folder'])
            if detail['overrides'] != row['stats']['network_overrides']:
                raise ValueError('Override count mismatch')
            dump(out / 'decisions' / (row['case'] + '.json'), detail)
    dump(out / 'conclusion.json', result)
    dump(benchmark / 'aggregate.json', result['aggregate'])
    final = dict(status=status['status'], planned=96, completed=len(rows),
                 all_complete_and_audited=result['all_complete_and_audited'])
    dump(benchmark / 'batch.json', final)
    dump(out / 'batch.json', final)
    report(out, rows, result)


def launch(out):
    out = Path(out).resolve()
    original = json.loads((out / 'control.json').read_text())
    process_path = out / 'process.json'
    process = json.loads(process_path.read_text())
    if original['live_worker_pids_after_shutdown'] or live_group_members(process['worker_pgid']):
        raise RuntimeError('Original workers have not exited')
    if original['status'] != 'complete':
        raise ValueError('Only clean dispatch-cutoff completion can be extended')
    if json.loads((out / 'benchmark/batch.json').read_text())['status'] != 'dispatch_cutoff':
        raise ValueError('No cleanly unstarted development jobs to extend')
    start_epoch = process_path.stat().st_mtime
    hard, dispatch = remaining_budget(start_epoch, time.time())
    archive = out / 'initial-budget'
    archive.mkdir(exist_ok=False)
    for name in ('control.json', 'process.json', 'batch.json', 'conclusion.json', '报告.md',
                 'benchmark/batch.json', 'benchmark/summary.json', 'benchmark/aggregate.json'):
        source, target = out / name, archive / name
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    dump(out / 'budget-extension.json', dict(reason='User explicitly extended the active round to 30 minutes',
         original_start_epoch_s=start_epoch, hard_deadline_epoch_s=start_epoch+1800.,
         dispatch_deadline_epoch_s=start_epoch+1620., original_hard_limit_s=original['hard_limit_s'],
         new_round=False, training_repeated=False, new_cost_sampling=False,
         checkpoint_sha256={name: sha256(out / name) for name in ('training-frozen/best.pt', 'training-finetuned/best.pt')}))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    control = supervise([sys.executable, '-m', 'nnq4.advantage_resume', '--worker', str(out)], out,
                        hard_limit_s=hard, dispatch_limit_s=dispatch, env=env)
    control.update(total_round_wall_s=time.time()-start_epoch, round_hard_limit_s=1800.,
                   round_dispatch_limit_s=1620., execution_segments=2, original_control='initial-budget/control.json')
    dump(out / 'control.json', control)
    if control['status'] != 'complete':
        rows = json.loads((out / 'benchmark/summary.json').read_text())
        dump(out / 'batch.json', dict(status=control['status'], planned=96, completed=len(rows)))
        dump(out / 'conclusion.json', dict(conclusion='stop', reason=control['status'], automatic_next_round=False))
    return control


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.out)
    elif launch(args.out)['status'] != 'complete':
        raise SystemExit(1)
