"""Discovery-first sector batches, movement-aware probes and certified backstop.

Scores below are explicit heuristics, not full-mission rollout estimates. Only
actual per-channel observations can certify completion. No simulator import.
"""

from functools import lru_cache
import math

import numpy as np

from .compact import CoverPolicy
from .core import DOMAIN
from .coverage import certifies
from .localization import guaranteed_clear, local_attempts
from .posterior import QuadratureError, expected_measure
from .shared import Action, distance


@lru_cache(maxsize=1)
def ring22_stations():
    """An inner ring and circumscribed outer landmarks, including domain slack."""
    outer = 1805 / math.cos(math.pi / 14)
    points = [(0.0, 0.0)]
    points += [(975 * math.cos(k * math.tau / 7), 975 * math.sin(k * math.tau / 7)) for k in range(7)]
    points += [(outer * math.cos(k * math.tau / 14), outer * math.sin(k * math.tau / 14)) for k in range(14)]
    points = tuple(points)
    if not certifies(DOMAIN, points):
        raise ValueError('Ring landmarks lack a full Q4 coverage certificate')
    return points


@lru_cache(maxsize=1)
def ring_stations():
    """Ultra defaults to the integer-certified S21 layout."""
    from s21_layout import stations

    return stations()


class RingPolicy(CoverPolicy):
    def __init__(self, config=None):
        super().__init__('trim', config)
        self.points = {-i - 1: p for i, p in enumerate(ring_stations())}
        self.implementation = 'ultra_s21_cover_v1'
        self.counters['initial_search_stations'] = len(self.points)


class ProbePolicy(RingPolicy):
    """Joint route retains every obligation; only probe locations are enlarged."""

    def __init__(self, config=None):
        super().__init__(config)
        self.implementation = 'ultra_s21_route_probes_v1'
        self.counters['movement_probe_candidates'] = 0

    def probe(self, b, c, position, bins):
        p = b.channels[c]
        post = self.model.posterior(p)
        key = (c, p.revision, position, bins)
        if key not in self.local_cache:
            if len(self.local_cache) > 1024:
                self.local_cache.clear()
            self.local_cache[key] = expected_measure(post, position, bins)
        future, detail = self.local_cache[key]
        move = distance(b.position, position) / 5
        return (
            Action('measure', position, c),
            move + 5 + int(c != b.receiver) + future,
            {
                **detail,
                'move_to_probe_s': move,
                'expected_finish_proxy_s': future,
                'probe_to_target_center_m': distance(position, tuple(post.mean)),
                'cost_is_proxy': True,
            },
        )

    def movement_choices(self, b, c, goals):
        choices, _mean = RingPolicy.local_choices(self, b, c)
        p = b.channels[c]
        if guaranteed_clear(b, c) or local_attempts(p) >= self.config.local_limit:
            return choices
        try:
            post = self.model.posterior(p)
            first = np.asarray(p.directions[-1].action.position)
            center = np.asarray(post.mean)
            vector = center - first
            normal = np.asarray([-vector[1], vector[0]]) / max(float(np.linalg.norm(vector)), 1.0)
            candidates = [
                tuple(center + side * offset * normal) for side in (-1, 1) for offset in (20.0, 100.0, 200.0)
            ]
            candidates += [
                tuple(first + fraction * vector + side * 100 * normal)
                for fraction in (0.35, 0.65)
                for side in (-1, 1)
            ]
            other = sorted((j for j in goals if j != c), key=lambda j: distance(center, goals[j]))[:3]
            for j in other:
                goal = np.asarray(goals[j])
                candidates += [tuple(goal), tuple((goal + center) / 2)]
                delta = goal - b.position
                t = np.clip(float((center - b.position) @ delta) / max(float(delta @ delta), 1e-12), 0, 1)
                candidates.append(tuple(np.asarray(b.position) + t * delta))
            existing = {a.position for a, _, _ in choices}
            candidates = [
                tuple(map(float, x))
                for x in dict.fromkeys(candidates)
                if x not in existing
                and all(distance(x, o.action.position) >= 10 for o in p.history if o.action.kind == 'measure')
            ]
            self.counters['movement_probe_candidates'] += len(candidates)
            coarse = sorted((self.probe(b, c, x, 10.0)[1], i) for i, x in enumerate(candidates))
            choices += [self.probe(b, c, candidates[i], self.config.bearing_bin) for _, i in coarse[:3]]
        except QuadratureError:
            pass  # Unmodified geometry/finite baseline choices remain available.
        return choices

    def local_choices(self, b, c):
        goals = {j: p.summary()[0] for j, p in b.channels.items() if p.status == 'detected'}
        nearest = sorted(self.points, key=lambda k: distance(self.points[k], goals[c]))[:2]
        goals.update({k: self.points[k] for k in nearest if self.needed(b, self.points[k])})
        choices = self.movement_choices(b, c, goals)
        try:
            mean = tuple(map(float, self.model.posterior(b.channels[c]).mean))
        except QuadratureError:
            mean = goals[c]
        return choices, mean
