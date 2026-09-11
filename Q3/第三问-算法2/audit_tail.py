"""Independent posterior-world audit of the geometric leaf value, from public logs."""
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
from pomcp.generative_model import MacroAction
from pomcp.heuristic import components
from pomcp.rollout import evaluate_world
from pomcp.world_sampler import ParticleWorldSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('actions', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--worlds', type=int, default=32)
    parser.add_argument('--snapshots', type=int, default=4)
    args = parser.parse_args()
    records = [json.loads(s) for s in args.actions.read_text().splitlines()]
    indices = set(np.linspace(20, max(20, len(records)-10), args.snapshots, dtype=int))
    b, snapshots = ParticleBelief(), []
    for i, row in enumerate(records, 1):
        data = row['action']
        a = Action(data['kind'], tuple(data['position']), data['channel'])
        b.apply(a, row['response'], row['request_id'])
        if i in indices and not b.done():
            snapshots.append(b.clone())
    out = Path(__file__).parent/'results'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir()
    result = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for b in snapshots:
            sampler = ParticleWorldSampler(970001+b.steps)
            worlds = sampler.sample(b, args.worlds)
            dummy = MacroAction.atomic(MacroPolicy().base.choose(b))
            # horizon=0: the entire continuation is the tail, sampled from an
            # independent current posterior; no true source coordinates read.
            jobs = [(b, w, [dummy], 0, 'completion', math.inf) for w in worlds]
            values = [row[0]['cost'] for row in pool.map(evaluate_world, jobs)]
            parts = components(b)
            record = {'step': b.steps, 'components_s': parts, 'geometric_H_s': sum(parts.values()),
                      'independent_mean_remaining_s': float(np.mean(values)),
                      'independent_se_s': float(np.std(values, ddof=1)/math.sqrt(len(values))),
                      'worlds': len(values), 'complete_tail_costs_s': values,
                      'bias_s': sum(parts.values())-float(np.mean(values))}
            result.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != 'complete_tail_costs_s'}, ensure_ascii=False), flush=True)
    (out/'tail-audit.json').write_text(json.dumps({'input': str(args.actions), 'results': result},
                                     ensure_ascii=False, indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    main()
