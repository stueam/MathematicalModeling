"""POMCP-lite root rollout, common worlds, complete-batch successive halving."""
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict
import math
import multiprocessing
import os
import time
import numpy as np
from q3.core import distance, GeometryError
from q3.sampling import SamplingError
from .config import Config
from .world_sampler import ParticleWorldSampler
from .action_gen import MacroPolicy, scan
from .generative_model import MacroAction
from .rollout import evaluate_world


class Planner:
    def __init__(self, config=Config()):
        for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        self.config = config
        self.policy = MacroPolicy()
        self.sampler = ParticleWorldSampler(config.seed)
        self.executor = ProcessPoolExecutor(max_workers=config.workers,
                            mp_context=multiprocessing.get_context('spawn')) if config.workers > 1 else None
        self.records = []
        self.completion_mode = False

    def close(self):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

    def choose(self, b):
        cfg, start = self.config, time.monotonic()
        base = MacroAction.atomic(self.policy.base.choose(b), 'BASELINE')
        record = {'step': b.steps, 'position': b.position, 'status': 'pending',
                  'departure_review_attempted': False, 'stages': [], 'discarded_worlds': 0}
        chosen = base
        # Initial census is an explicitly recommended macro, with all 20
        # channels and the initial receiver 1 charged by the physical kernel.
        if b.steps == 0:
            chosen = scan(b, b.position, range(1, 21), 'INITIAL_SCAN')
            record['status'] = 'initial_census'
        elif time.monotonic() > b.deadline-30 or b.steps >= 500:
            self.completion_mode = True
            record['status'] = 'finite_completion_reserve'
        else:
            forced = [c for c, p in b.channels.items() if p.status == 'detected' and
                      (len(p.history) >= 18 or sum(o.result == 'no_target_in_range' for o in p.history) >= 6)]
            if forced:
                chosen = MacroAction.atomic(self.policy.base.for_channel(b, forced[0]), 'FINITE_COMPLETION')
                record['status'] = 'finite_target_completion'
            elif self.completion_mode:
                record['status'] = 'finite_completion_reserve'
            elif base.actions[0].kind == 'clear' and distance(base.position, b.position) < 1e-8:
                record['status'] = 'inplace_clear'
            else:
                deadline = min(start+cfg.budget_s, b.deadline-30)
                try:
                    raw = self.policy.generate(b)
                    actions = self.policy.prune(b, raw, cfg.candidates)
                    record.update(raw_candidates=len(raw), actions=[asdict(a) for a in actions],
                                  departure_review_attempted=True)
                    # Each stage uses its own homogeneous horizon. Fine scores
                    # replace coarse scores, never mix incomparable depths.
                    active = list(range(len(actions)))
                    for count, horizon in ((cfg.coarse_worlds, cfg.horizon), (cfg.fine_worlds, cfg.fine_horizon)):
                        if time.monotonic() >= deadline:
                            break
                        worlds = self.sampler.sample(b, count, deadline)
                        jobs = [(b, w, [actions[i] for i in active], horizon, cfg.tail, deadline) for w in worlds]
                        futures = [self.executor.submit(evaluate_world, job) for job in jobs] if self.executor else None
                        try:
                            results = [f.result() for f in futures] if futures else [evaluate_world(job) for job in jobs]
                        except (TimeoutError, SamplingError, GeometryError, BrokenProcessPool):
                            record['discarded_worlds'] += count
                            if futures:
                                for f in futures:
                                    f.cancel()
                            raise
                        costs = np.array([[cell['cost'] for cell in row] for row in results])
                        means = costs.mean(axis=0)
                        rank = np.argsort(means, kind='stable')
                        chosen = actions[active[int(rank[0])]]
                        record['status'] = 'monte_carlo'
                        record['stages'].append({'worlds_per_action': count, 'horizon': horizon,
                                                 'indices': active.copy(), 'costs': costs.tolist(),
                                                 'means': means.tolist(), 'details': results,
                                                 'rollout_count': count*len(active)})
                        # Always retain the baseline for the final comparison.
                        active = [active[int(i)] for i in rank[:cfg.finalists]]
                        anchors = [i for i in (0, 1) if i < len(actions)]
                        for anchor in anchors:
                            if anchor not in active:
                                replace = next((j for j in range(len(active)-1, -1, -1)
                                                if active[j] not in anchors), None)
                                if replace is not None:
                                    active[replace] = anchor
                except (SamplingError, GeometryError, TimeoutError, BrokenProcessPool) as exc:
                    record['fallback_reason'] = f'{type(exc).__name__}: {exc}'
                    if not record['stages']:
                        record['status'] = type(exc).__name__
                    if isinstance(exc, BrokenProcessPool):
                        self.close()
                        self.completion_mode = True
        record.update(selected=asdict(chosen), selected_move_m=distance(b.position, chosen.position),
                      changed=chosen.actions != base.actions, planning_s=time.monotonic()-start)
        self.records.append(record)
        return chosen
