import argparse
import json
from pathlib import Path
import time
import numpy as np
from .physics import Config, WorldBatch
from .env import RadioEnv, heuristic_actions, random_actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--envs',type=int,nargs='+',default=[1,16,64,256])
    p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--out',default='reports/benchmark.json')
    args = p.parse_args()
    results = []
    for n in args.envs:
        cfg = Config(num_envs=n)
        w = WorldBatch(cfg,11000)
        rng = np.random.default_rng(0)
        # Pre-generated inputs isolate physics from controller and RNG allocation.
        pos = rng.uniform(-1800,1800,(args.iterations,n,2))
        channels = rng.integers(0,20,(args.iterations,n))
        clear = rng.random((args.iterations,n)) < .1
        start = time.perf_counter()
        for k in range(args.iterations):
            w.step(pos[k],channels[k],clear[k])
            if w.time.max() >= cfg.max_virtual_s:
                w.reset(np.arange(n),np.arange(n)+12000+k*n)
        physics_s = time.perf_counter()-start
        env = RadioEnv(cfg,13000)
        for _ in range(5):
            env.step(heuristic_actions(env.obs))
        start = time.perf_counter()
        for _ in range(args.iterations):
            env.step(heuristic_actions(env.obs))
        full_s = time.perf_counter()-start
        records = list(env.completed)
        row = {'num_envs':n,'actions':n*args.iterations,'physics_s':physics_s,
               'physics_actions_per_s':n*args.iterations/physics_s,
               'env_belief_candidates_s':full_s,'env_actions_per_s':n*args.iterations/full_s,
               'heuristic_episodes':len(records),
               'heuristic_success_rate':np.mean([r['success'] for r in records]) if records else None}
        print(json.dumps(row),flush=True)
        results.append(row)
    out = Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
