"""Paired full-task 1x/3x compute comparison, with unchanged hidden worlds."""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location('q3_compute_runner', Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def planning_statistics(folder):
    records = [json.loads(s) for s in (folder/'decisions.jsonl').read_text().splitlines()]
    decisions = [r for r in records if r.get('departure_review_attempted')]
    stages = [s for r in decisions for s in r['stages']]
    return {
        'reviewed_decisions': len(decisions),
        'no_completed_batch': sum(not r['stages'] for r in decisions),
        'coarse_only_decisions': sum(len(r['stages']) == 1 for r in decisions),
        'fine_completed_decisions': sum(len(r['stages']) == 2 for r in decisions),
        'decisions_with_discarded_work': sum(r.get('discarded_worlds', 0) > 0 for r in decisions),
        'discarded_paired_worlds': sum(r.get('discarded_worlds', 0) for r in decisions),
        'valid_complete_rollouts': sum(s['rollout_count'] for s in stages),
        'valid_physical_rollout_steps': sum(c['physical_steps'] for s in stages for row in s['details'] for c in row),
        'mean_actual_candidates': sum(len(r.get('actions', [])) for r in decisions)/max(1, len(decisions)),
        'completion_reserve_decisions': sum(r['status'] == 'finite_completion_reserve' for r in records),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=20)
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    root = Path(__file__).parent
    out = root/'results'/datetime.now().strftime('compute-paired-%Y%m%d-%H%M%S')
    out.mkdir()
    configs = {str(scale): asdict(runner.Config(workers=args.workers).scaled(scale)) for scale in (1, 3)}
    runner.dump(out/'config.json', {'seed': args.seed, 'rounds': args.rounds, 'configs': configs,
                                  'scenario': 'uniform', 'error_mode': 'iid', 'real_limit': 1200})
    sources = list(root.glob('*.py'))+list((root/'pomcp').glob('*.py'))
    sources += list((root.parent/'第三问-算法1/q3').glob('*.py'))
    runner.dump(out/'source-hashes.json', {str(f.relative_to(root.parent)): hashlib.sha256(f.read_bytes()).hexdigest()
                                          for f in sources})
    results = []
    for seed in range(args.seed, args.seed+args.rounds):
        for scale in (1, 3):
            folder = out/f'{seed}-{scale}x'
            folder.mkdir()
            cfg = runner.Config(workers=args.workers).scaled(scale)
            case = argparse.Namespace(n=None, scenario='uniform', radius=None, error_mode='iid',
                                      real_limit=1200, max_steps=4000)
            summary = runner.local_run(case, cfg, 'pomcp', seed, folder)
            summary.update(compute_scale=scale, planning=planning_statistics(folder))
            runner.dump(folder/'summary.json', summary)
            results.append(summary)
            runner.dump(out/'summary.json', results)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    paired = []
    for seed in range(args.seed, args.seed+args.rounds):
        a, b = [r for r in results if r['seed'] == seed]
        valid = all(r['all_cleared'] and r['certified_complete'] for r in (a, b))
        paired.append({'seed': seed, 'both_complete': valid,
                       'saved_virtual_s': a['virtual_time_s']-b['virtual_time_s'] if valid else None,
                       'saved_fraction': 1-b['virtual_time_s']/a['virtual_time_s'] if valid else None,
                       'real_time_ratio': b['real_time_s']/a['real_time_s']})
    runner.dump(out/'paired.json', paired)
    print(f'Results: {out}', flush=True)


if __name__ == '__main__':
    main()
