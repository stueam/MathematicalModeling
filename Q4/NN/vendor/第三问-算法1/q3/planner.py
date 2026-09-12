"""Cost-triggered root Monte Carlo planning with paired adaptive batches.

No modulo-step gate: every departure >= 50 m is reviewed, unless a documented
completion lock or real-deadline reserve forces the completion policy.
"""
import math
import time
from dataclasses import dataclass

import numpy as np

from .core import GeometryError, distance
from .movement import MovementPolicy
from .sampling import WorldSampler, SamplingError
from .simulator import LocalSimulator


@dataclass(frozen=True)
class RolloutResult:
    cost: float
    steps: int


def rollout_result(belief, world, first, baseline, deadline=math.inf, max_steps=4000):
    observer = belief.clone()
    env = LocalSimulator(world.clone(), observer.position, observer.receiver, observer.virtual_time)
    start_time = observer.virtual_time
    for i in range(max_steps):
        if time.monotonic() >= deadline:
            raise TimeoutError('Rollout deadline')
        if observer.done():
            return RolloutResult(observer.virtual_time-start_time, i)
        action = first if i == 0 else baseline.choose(observer)
        request_id = f'rollout-{observer.steps}-{i}'
        response = env.execute(action, request_id)
        observer.apply(action, response, request_id)
        if observer.virtual_time >= observer.virtual_limit:
            raise SamplingError('Rollout failed to finish within virtual budget')
    if observer.done():
        return RolloutResult(observer.virtual_time-start_time, max_steps)
    raise SamplingError('Rollout action limit; unfinished return must not be scored as zero')


def rollout_cost(belief, world, first, baseline, deadline=math.inf, max_steps=4000):
    return rollout_result(belief, world, first, baseline, deadline, max_steps).cost


def paired_statistics(costs):
    values = np.asarray(costs)
    means = values.mean(axis=0)
    order = np.argsort(means)
    best = int(order[0])
    second = int(order[1]) if len(order) > 1 else best
    delta = values[:, best]-values[:, second]
    se = float(delta.std(ddof=1)/math.sqrt(len(delta))) if len(delta) > 1 else math.inf
    # A heuristic stopping diagnostic, not a multiple-comparison confidence bound.
    ambiguous = best != second and float(delta.mean())+1.5*se > -2.
    return means, best, se, ambiguous


class RolloutPlanner:
    def __init__(self, seed=2026, worlds=8, candidates=10, budget_s=8., interval=None,
                 ring_radius=1500., speculative_clear=True, max_worlds=32):
        if worlds < 1 or candidates < 2 or budget_s <= 0 or max_worlds < worlds:
            raise ValueError('Invalid planner budget')
        # interval is accepted for compatibility but never suppresses reviews.
        self.baseline = MovementPolicy(ring_radius, speculative_clear=speculative_clear)
        self.sampler = WorldSampler(seed)
        self.worlds, self.max_worlds, self.candidates = worlds, max_worlds, candidates
        self.budget_s = budget_s
        self.speculative_clear = speculative_clear
        self.records = []
        self._last_status = None
        self._last_area = math.inf
        self._stagnation = 0
        self._lock = None

    def _forced(self, b, base):
        status = tuple((c, p.status) for c, p in b.channels.items())
        area = sum(p.region.area for p in b.channels.values() if p.status not in ('cleared', 'absent_certified'))
        progressed = status != self._last_status or area < self._last_area-1e-3
        self._stagnation = 0 if progressed else self._stagnation+1
        self._last_status, self._last_area = status, area
        if self._lock is None and self._stagnation >= 12:
            self._lock = ('channel', base.channel) if b.channels[base.channel].status == 'detected' else ('station', base.position)
        if self._lock:
            kind, value = self._lock
            if kind == 'channel' and b.channels[value].status == 'detected':
                return self.baseline.for_channel(b, value)
            if kind == 'station':
                action = self.baseline.scan_action(b, value)
                if action is not None:
                    return action
            self._lock, self._stagnation = None, 0
        return None

    def choose(self, b):
        started = time.monotonic()
        base = self.baseline.choose(b)
        move = distance(b.position, base.position)
        record = {'step': b.steps, 'worlds_completed': 0, 'batches_completed': [],
                  'base_move_m': move, 'departure_review_required': move >= 50,
                  'departure_review_attempted': False}

        def finish(action, reason):
            record.update(status=reason, elapsed_s=time.monotonic()-started,
                          selected={'kind': action.kind, **action.payload()}, changed=action != base,
                          selected_move_m=distance(b.position, action.position))
            self.records.append(record)
            return action

        forced = self._forced(b, base)
        if forced is not None:
            return finish(forced, 'forced_completion_task')
        if base.kind == 'clear' and move < 1e-6:
            return finish(base, 'immediate_clear')
        if b.deadline-started < max(10., self.budget_s+5):
            return finish(base, 'remaining_time_reserve')
        # Cheap on-site measurements may proceed. Never exempts a departure.
        if move < 10 and base.kind == 'measure':
            return finish(base, 'useful_inplace_measurement')
        actions = self.baseline.proposals(b, self.candidates, self.speculative_clear)
        if len(actions) < 2:
            return finish(base, 'single_candidate')
        record['departure_review_attempted'] = True
        record['candidates'] = [{'kind': a.kind, **a.payload()} for a in actions]
        # The configured budget remains a hard cap, including proposal work.
        budget = min(self.budget_s, max(4., move/75.))
        record['budget_s'] = budget
        deadline = min(started+budget, b.deadline-5)
        try:
            selected = self._select(b, actions, deadline, record)
        except (TimeoutError, SamplingError, GeometryError) as exc:
            record['detail'] = str(exc)
            return finish(base, type(exc).__name__)
        return finish(selected, 'monte_carlo')

    def _select(self, b, actions, deadline, record):
        """Legacy V2 allocation, retained as a reproducible control."""
        move = record['base_move_m']
        costs = []
        target = self.worlds
        try:
            while True:
                additional = target-len(costs)
                batch_started = time.monotonic()
                worlds = self.sampler.sample(b, additional, deadline)
                batch = []
                for world in worlds:
                    batch.append([rollout_cost(b, world, a, self.baseline, deadline) for a in actions])
                # Commit only an entire predeclared paired batch. No partial
                # rows and no filtering fast successful trials.
                costs.extend(batch)
                batch_wall = time.monotonic()-batch_started
                record['worlds_completed'] = len(costs)
                record['batches_completed'].append(len(costs))
                means, best, se, ambiguous = paired_statistics(costs)
                if not ambiguous or target >= self.max_worlds:
                    break
                next_target = min(self.max_worlds, target*2)
                estimate = batch_wall*(next_target-target)/additional
                if time.monotonic()+estimate*1.15 >= deadline:
                    record['refinement_stopped'] = 'insufficient_budget_for_next_complete_batch'
                    break
                target = next_target
        except TimeoutError as exc:
            record['detail'] = str(exc)
            if not costs:
                raise
            record['refinement_stopped'] = 'incomplete_later_batch_discarded'
        means, best, se, ambiguous = paired_statistics(costs)
        values = np.asarray(costs)
        difference = values[:, best]-values[:, 0]
        base_se = float(difference.std(ddof=1)/math.sqrt(len(difference))) if len(difference) > 1 else math.inf
        record.update(mean_cost_s=means.tolist(), paired_delta_s=float(difference.mean()),
                      paired_se_s=base_se if math.isfinite(base_se) else None,
                      ambiguous=ambiguous, best_vs_second_se_s=se if math.isfinite(se) else None)
        # Do not switch to a longer physical leg on a tiny noisy mean gain.
        chosen = best
        if distance(b.position, actions[best].position) > move+25 and difference.mean()+base_se > -2:
            chosen = 0
            record['selection_guard'] = 'uncertain_gain_does_not_justify_longer_departure'
        return actions[chosen]
