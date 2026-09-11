"""Full-rule stress cases. Seeds here are disjoint from development seeds 0..4."""
import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--budget-s', type=float, default=5.)
    p.add_argument('--worlds', type=int, default=16)
    args = p.parse_args()
    cfg = run.Config(workers=args.workers, coarse_worlds=args.worlds,
                     fine_worlds=max(args.worlds, 48), budget_s=args.budget_s)
    out = Path(__file__).parent/'results'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir()
    cases = [('boundary', 10, 1000., 'extreme'), ('boundary', 16, 1500., 'correlated'),
             ('cluster', 10, 1000., 'correlated'), ('near', 16, 1500., 'extreme')]
    run.dump(out/'config.json', {'planner': asdict(cfg), 'cases': cases, 'seed': args.seed})
    rows = []
    for i, (scenario, n, radius, error_mode) in enumerate(cases):
        case = argparse.Namespace(n=n, scenario=scenario, radius=radius, error_mode=error_mode,
                                  real_limit=300., max_steps=4000)
        for name in ('algorithm1', 'pomcp'):
            folder = out/f'{i}-{name}'
            folder.mkdir()
            result = run.local_run(case, cfg, name, args.seed+i, folder)
            result.update(scenario=scenario, error_mode=error_mode, radius=radius)
            rows.append(result)
            run.dump(out/'summary.json', rows)
            print(run.json.dumps({k: result[k] for k in ('seed', 'policy', 'scenario', 'error_mode',
                  'all_cleared', 'certified_complete', 'virtual_time_s', 'real_time_s', 'error')},
                  ensure_ascii=False), flush=True)
    print(out, flush=True)
    if not all(r['all_cleared'] and r['certified_complete'] and r['error'] is None for r in rows):
        raise SystemExit('At least one stress case failed; inspect preserved logs')


if __name__ == '__main__':
    main()
