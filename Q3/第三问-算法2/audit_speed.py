"""Fixed-work serial/16-process comparison: identical complete cost matrix."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import math
import multiprocessing
from pathlib import Path
import time
import numpy as np
import pomcp
from q3.core import Action
from pomcp.belief import ParticleBelief
from pomcp.action_gen import MacroPolicy
from pomcp.rollout import evaluate_world
from pomcp.world_sampler import ParticleWorldSampler


def worker(job):
    return os.getpid(), evaluate_world(job)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('actions', type=Path)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--worlds', type=int, default=16)
    args = p.parse_args()
    b = ParticleBelief()
    for row in [json.loads(s) for s in args.actions.read_text().splitlines()][:20]:
        d = row['action']
        b.apply(Action(d['kind'], tuple(d['position']), d['channel']), row['response'], row['request_id'])
    policy = MacroPolicy()
    actions = policy.prune(b, policy.generate(b), 8)
    worlds = ParticleWorldSampler(310009).sample(b, args.worlds)
    jobs = [(b, w, actions, 6, 'completion', math.inf) for w in worlds]
    start = time.monotonic()
    serial = [evaluate_world(job) for job in jobs]
    serial_s = time.monotonic()-start
    start = time.monotonic()
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        parallel = list(pool.map(worker, jobs))
    parallel_s = time.monotonic()-start
    assert serial == [r for _, r in parallel], 'Parallel costs or steps differ'
    result = {'worlds_per_action': len(worlds), 'candidates': len(actions),
              'complete_rollouts': len(worlds)*len(actions),
              'physical_steps': sum(cell['physical_steps'] for row in serial for cell in row),
              'serial_s': serial_s, 'parallel_s_including_startup_shutdown': parallel_s,
              'speedup': serial_s/parallel_s, 'workers_requested': args.workers,
              'distinct_workers_used': len({pid for pid, _ in parallel}),
              'identical_costs_and_steps': True, 'results': serial}
    out = Path(__file__).parent/'results'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir()
    (out/'speed.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}, ensure_ascii=False))
    print(out)


if __name__ == '__main__':
    main()
