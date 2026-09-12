"""Exact full-task continuations, with per-world suffix reuse and process workers.

Only MovementPolicy is supported: its choices do not depend on elapsed time,
request IDs or completed-channel histories. The cache is NEVER shared across
worlds or real decisions. Geometry and public observation histories are retained
in the key. No hidden coordinates are used to choose an action.
"""
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import math
import multiprocessing
import time

from .core import point_key
from .movement import MovementPolicy
from .sampling import SamplingError
from .simulator import LocalSimulator


@dataclass(frozen=True)
class Trial:
    cost: float
    steps: int
    executed_steps: int
    suffix_hits: int


def continuation_key(b):
    channels = []
    for c, p in b.channels.items():
        if p.status in ('cleared', 'absent_certified'):
            channels.append((c, p.status))
            continue
        cached = getattr(p, '_continuation_key', None)
        if cached is None or cached[0] != p.revision:
            value = (p.status, p.region.wkb, tuple(p.history), frozenset(p.measured))
            cached = (p.revision, value)
            p._continuation_key = cached
        channels.append((c, cached[1]))
    return point_key(b.position), b.receiver, tuple(channels)


def cached_rollout(belief, world, first, policy, cache, deadline=math.inf,
                   max_steps=4000):
    observer = belief.clone()
    env = LocalSimulator(world.clone(), observer.position, observer.receiver, observer.virtual_time)
    start = observer.virtual_time
    path = []
    executed = 0
    hit = 0
    total_steps = 0
    end = start
    for i in range(max_steps+1):
        if time.monotonic() >= deadline:
            raise TimeoutError('Rollout deadline')
        if observer.done():
            total_steps, end = i, observer.virtual_time
            break
        if i == max_steps:
            raise SamplingError('Rollout action limit')
        # The root action is imposed; all later actions use the same pi0.
        if i and cache is not None:
            key = continuation_key(observer)
            if key in cache:
                remaining, steps = cache[key]
                if i+steps > max_steps or observer.virtual_time+remaining >= observer.virtual_limit:
                    raise SamplingError('Cached continuation exceeds task budget')
                total_steps, end = i+steps, observer.virtual_time+remaining
                hit = 1
                break
            path.append((key, observer.virtual_time, i))
        action = first if i == 0 else policy.choose(observer)
        request_id = f'rollout-{observer.steps}-{i}'
        observer.apply(action, env.execute(action, request_id), request_id)
        executed += 1
        if observer.virtual_time >= observer.virtual_limit:
            raise SamplingError('Rollout virtual budget')
    # Store only suffixes of fully certified completions, never failed prefixes.
    if cache is not None:
        for key, clock, step in reversed(path):
            if len(cache) >= 2000:
                break
            cache[key] = (end-clock, total_steps-step)
    return Trial(end-start, total_steps, executed, hit)


def evaluate_world(payload):
    belief, world, actions, ring, speculative, deadline, reuse = payload
    policy = MovementPolicy(ring, speculative_clear=speculative)
    cache = {} if reuse else None
    return [cached_rollout(belief, world, a, policy, cache, deadline) for a in actions]


class RolloutEngine:
    def __init__(self, workers=1, ring=1500., speculative=True, reuse=True):
        if workers < 1:
            raise ValueError('workers must be positive')
        self.workers, self.ring, self.speculative, self.reuse = workers, ring, speculative, reuse
        self.pool = None

    def evaluate(self, belief, worlds, actions, deadline=math.inf):
        payloads = [(belief, w, actions, self.ring, self.speculative, deadline, self.reuse)
                    for w in worlds]
        if self.workers == 1:
            return [evaluate_world(p) for p in payloads]
        if self.pool is None:
            self.pool = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context('spawn'))
        futures = []
        try:
            for p in payloads:
                futures.append(self.pool.submit(evaluate_world, p))
            # Preserve world order. The parent commits only a whole paired batch.
            return [f.result(timeout=max(0., deadline-time.monotonic()) if math.isfinite(deadline) else None)
                    for f in futures]
        except BrokenProcessPool as exc:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None
            raise SamplingError('Rollout worker failed; discard this comparison') from exc
        finally:
            for f in futures:
                f.cancel()  # Running trials obey the same monotonic deadline.

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
