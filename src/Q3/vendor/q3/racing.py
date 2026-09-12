"""Paired candidate racing: allocate worlds to consequential close decisions.

The error bars are diagnostics, NOT sequential/multiple-testing guarantees.
Full geometric completion and compatible-world sampling remain unchanged.
"""
import math
import time

import numpy as np

from .core import distance
from .planner import RolloutPlanner, paired_statistics
from .rollout_engine import RolloutEngine


def race_statistics(values, active, tolerance=5., z=2.5):
    values = np.asarray(values, dtype=float)
    means = values.mean(axis=0)
    best = int(np.argmin(means))
    differences = values-values[:, [best]]
    se = (differences.std(axis=0, ddof=1)/math.sqrt(len(values))
          if len(values) > 1 else np.full(len(active), math.inf))
    if len(values) <= 1:
        se[best] = 0.
    # Positive lower gap => this candidate is distinctly worse than the leader.
    lower = differences.mean(axis=0)-z*se
    keep = [j for j, a in enumerate(active) if a == 0 or lower[j] <= tolerance]
    # Upper bound on the leader's possible loss, under the working SE heuristic.
    regret = float(np.max(-differences.mean(axis=0)+z*se))
    return means, best, keep, regret


class RacingPlanner(RolloutPlanner):
    def __init__(self, *args, workers=1, tolerance_s=5., reuse=True, **kwargs):
        super().__init__(*args, **kwargs)
        if tolerance_s < 0:
            raise ValueError('tolerance must be nonnegative')
        self.tolerance = tolerance_s
        self.engine = RolloutEngine(workers, distance(self.baseline.stations[1], (0., 0.)),
                                    self.speculative_clear, reuse)

    def close(self):
        self.engine.close()

    def _select(self, b, actions, deadline, record):
        active = list(range(len(actions)))
        columns = [[] for _ in actions]
        target, completed = self.worlds, 0
        record.update(workers=self.engine.workers, tolerance_s=self.tolerance,
                      rollout_count=0, rollout_steps=0, executed_rollout_steps=0,
                      suffix_hits=0, racing_stages=[], sampling_s=0., simulation_s=0.)
        while True:
            additional = target-completed
            try:
                tick = time.monotonic()
                worlds = self.sampler.sample(b, additional, deadline)
                sample_wall = time.monotonic()-tick
                tick = time.monotonic()
                trials = self.engine.evaluate(b, worlds, [actions[a] for a in active], deadline)
                simulation_wall = time.monotonic()-tick
            except TimeoutError:
                if completed == 0:
                    raise
                record['refinement_stopped'] = 'incomplete_later_batch_discarded'
                break
            # Atomic commit: no fast-world filtering and no partial candidate rows.
            for row in trials:
                for a, trial in zip(active, row):
                    columns[a].append(trial.cost)
                    record['rollout_count'] += 1
                    record['rollout_steps'] += trial.steps
                    record['executed_rollout_steps'] += trial.executed_steps
                    record['suffix_hits'] += trial.suffix_hits
            record['sampling_s'] += sample_wall
            record['simulation_s'] += simulation_wall
            completed = target
            record['worlds_completed'] = completed
            record['batches_completed'].append(completed)
            values = np.array([columns[a] for a in active]).T
            means, best, keep, regret = race_statistics(values, active, self.tolerance)
            survivors = [active[j] for j in keep]
            record['racing_stages'].append({'worlds': completed, 'active': active.copy(),
                                            'survivors': survivors,
                                            'regret_diagnostic_s': regret if math.isfinite(regret) else None})
            if regret <= self.tolerance or completed >= self.max_worlds:
                record['refinement_stopped'] = 'practically_resolved' if regret <= self.tolerance else 'max_worlds'
                break
            next_target = min(self.max_worlds, target*2)
            estimate = (sample_wall+simulation_wall*len(survivors)/len(active))*(next_target-target)/additional
            if time.monotonic()+estimate*1.2 >= deadline:
                record['refinement_stopped'] = 'insufficient_budget_for_next_complete_batch'
                break
            active = survivors
            target = next_target
        values = np.array([columns[a] for a in active]).T
        means, best, se, ambiguous = paired_statistics(values)
        base = active.index(0)
        difference = values[:, best]-values[:, base]
        base_se = float(difference.std(ddof=1)/math.sqrt(completed)) if completed > 1 else math.inf
        chosen = active[best]
        if means[base]-means[best] <= self.tolerance:
            chosen = 0
            record['selection_guard'] = 'baseline_within_practical_tolerance'
        elif distance(b.position, actions[chosen].position) > record['base_move_m']+25 and difference.mean()+base_se > -2:
            chosen = 0
            record['selection_guard'] = 'uncertain_gain_does_not_justify_longer_departure'
        record.update(mean_cost_s=[float(np.mean(c)) if c else None for c in columns],
                      candidate_sample_counts=[len(c) for c in columns], final_active=active,
                      paired_costs_s=values.tolist(), paired_delta_s=float(difference.mean()),
                      paired_se_s=base_se if math.isfinite(base_se) else None,
                      ambiguous=ambiguous, best_vs_second_se_s=se if math.isfinite(se) else None,
                      average_rollout_steps=record['rollout_steps']/record['rollout_count'])
        return actions[chosen]
