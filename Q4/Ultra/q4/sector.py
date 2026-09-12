"""Discovery-first sector batches, movement-aware probes and certified backstop.

Scores below are explicit heuristics, not full-mission rollout estimates. Only
actual per-channel observations can certify completion. No simulator import.
"""
from functools import lru_cache
import math
import time

import numpy as np

from .compact import CoverPolicy
from .core import DOMAIN
from .coverage import certifies
from .localization import guaranteed_clear, local_attempts
from .posterior import QuadratureError, expected_measure
from .shared import Action, distance, point_key


@lru_cache(maxsize=1)
def ring22_stations():
    """An inner ring and circumscribed outer landmarks, including domain slack."""
    outer = 1805/math.cos(math.pi/14)
    points = [(0., 0.)]
    points += [(975*math.cos(k*math.tau/7), 975*math.sin(k*math.tau/7)) for k in range(7)]
    points += [(outer*math.cos(k*math.tau/14), outer*math.sin(k*math.tau/14)) for k in range(14)]
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
        self.points = {-i-1: p for i, p in enumerate(ring_stations())}
        self.implementation = 'ultra_s21_cover_v1'
        self.counters['initial_search_stations'] = len(self.points)


class SectorPolicy(RingPolicy):
    def __init__(self, config=None):
        super().__init__(config)
        self.implementation = 'q4_v4_sector_movement'
        self.focus = None
        self.last_opportunistic_review = -1
        self.last_shared_position = None
        self.counters.update(shared_known_measures=0, opportunistic_scans=0,
                             movement_probe_candidates=0, sector_changes=0)

    def sector(self, point):
        return int((math.atan2(point[1], point[0])+math.pi/6) % math.tau/(math.pi/3))

    def probe(self, b, c, position, bins):
        p = b.channels[c]
        post = self.model.posterior(p)
        key = (c, p.revision, position, bins)
        if key not in self.local_cache:
            if len(self.local_cache) > 1024:
                self.local_cache.clear()
            self.local_cache[key] = expected_measure(post, position, bins)
        future, detail = self.local_cache[key]
        move = distance(b.position, position)/5
        return (Action('measure', position, c), move+5+int(c != b.receiver)+future,
                {**detail, 'move_to_probe_s': move, 'expected_finish_proxy_s': future,
                 'probe_to_target_center_m': distance(position, tuple(post.mean)),
                 'cost_is_proxy': True})

    def movement_choices(self, b, c, goals):
        choices, mean = RingPolicy.local_choices(self, b, c)
        p = b.channels[c]
        if guaranteed_clear(b, c) or local_attempts(p) >= self.config.local_limit:
            return choices
        try:
            post = self.model.posterior(p)
            first = np.asarray(p.directions[-1].action.position)
            center = np.asarray(post.mean)
            vector = center-first
            normal = np.asarray([-vector[1], vector[0]])/max(float(np.linalg.norm(vector)), 1.)
            candidates = [tuple(center+side*offset*normal)
                          for side in (-1, 1) for offset in (20., 100., 200.)]
            candidates += [tuple(first+fraction*vector+side*100*normal)
                           for fraction in (.35, .65) for side in (-1, 1)]
            other = sorted((j for j in goals if j != c), key=lambda j: distance(center, goals[j]))[:3]
            for j in other:
                goal = np.asarray(goals[j])
                candidates += [tuple(goal), tuple((goal+center)/2)]
                delta = goal-b.position
                t = np.clip(float((center-b.position) @ delta)/max(float(delta @ delta), 1e-12), 0, 1)
                candidates.append(tuple(np.asarray(b.position)+t*delta))
            existing = {a.position for a, _, _ in choices}
            candidates = [tuple(map(float, x)) for x in dict.fromkeys(candidates)
                          if x not in existing and all(distance(x, o.action.position) >= 10
                          for o in p.history if o.action.kind == 'measure')]
            self.counters['movement_probe_candidates'] += len(candidates)
            coarse = sorted((self.probe(b, c, x, 10.)[1], i) for i, x in enumerate(candidates))
            choices += [self.probe(b, c, candidates[i], self.config.bearing_bin) for _, i in coarse[:3]]
        except QuadratureError:
            pass  # Unmodified geometry/finite baseline choices remain available.
        return choices

    def opportunistic_scan(self, b):
        if self.last_opportunistic_review == b.steps:
            return None
        self.last_opportunistic_review = b.steps
        channels = self.needed(b, b.position)
        if not channels or not b.applied:
            return None
        _, response = b.applied[next(reversed(b.applied))]
        if not (response.get('clear_result') == 'success'
                or response.get('measure_result') in ('direction', 'near')):
            return None
        if self.last_shared_position is not None and distance(b.position, self.last_shared_position) < 400:
            return None
        try:
            existence = self.model.existence(b)
            expected_discoveries = sum(existence[c]*float(self.model.posterior(b.channels[c])
                                      .reception(b.position)[1].sum()) for c in channels)
        except QuadratureError:
            return None
        # A discovered source can avoid a later detour. This value is an
        # explicit selection heuristic, never an absence or completion rule.
        scanning_s = 6*len(channels)
        if 600*expected_discoveries <= scanning_s+30:
            return None
        self.counters['opportunistic_scans'] += 1
        self.last_shared_position = b.position
        return self.scan(b, b.position, 'sector_opportunistic_scan',
                         expected_discoveries=expected_discoveries, scanning_proxy_s=scanning_s,
                         avoided_detour_value_per_discovery_s=600., cost_is_proxy=True)

    def choose(self, b):
        if b.done():
            raise ValueError('No action after completion')
        if (self.completion_mode or b.steps >= self.config.completion_after
                or b.deadline-time.monotonic() < self.config.reserve_s):
            return super().choose(b)
        detected = [c for c, p in b.channels.items() if p.status == 'detected']
        for c in detected:
            safe = guaranteed_clear(b, c)
            if safe and distance(b.position, safe.position) < 1e-6:
                return self.record(b, safe, 'guaranteed_clear')
        if self.batch is not None:
            if distance(self.batch, b.position) < 1e-6:
                action = self.scan(b, self.batch, 'station_batch')
                if action:
                    return action
            self.batch = None
        if b.steps == 0:
            self.last_shared_position = b.position
            return self.scan(b, b.position, 'origin_scan')
        action = self.opportunistic_scan(b)
        if action:
            return action
        if detected:
            goals = {}
            for c in detected:
                try:
                    goals[c] = tuple(map(float, self.model.posterior(b.channels[c]).mean))
                except QuadratureError:
                    goals[c] = b.channels[c].summary()[0]
            route, route_m = self.route(b.position, goals)
            focus_channels = [c for c in detected if self.sector(goals[c]) == self.focus]
            if not focus_channels:
                self.focus = self.sector(goals[route[0]])
                self.counters['sector_changes'] += 1
                focus_channels = [c for c in detected if self.sector(goals[c]) == self.focus]
            local = {c: self.movement_choices(b, c, goals) for c in detected}
            shared = []
            for c in detected:
                p = b.channels[c]
                if (local_attempts(p) >= self.config.local_limit
                        or any(distance(b.position, o.action.position) < 10
                               for o in p.history if o.action.kind == 'measure')):
                    continue
                try:
                    candidate = self.probe(b, c, b.position, self.config.bearing_bin)
                except QuadratureError:
                    continue
                saving = min(cost for _, cost, _ in local[c])-candidate[1]
                if saving > 2:
                    shared.append((saving, c, candidate))
            if shared:
                saving, c, (action, cost, detail) = max(shared, key=lambda t: t[0])
                self.counters['shared_known_measures'] += 1
                return self.record(b, action, 'sector_shared_known_measure', target=c,
                                   predicted_local_saved_s=saving, **detail)
            candidates = []
            for c in focus_channels:
                tail, tail_m = self.route(goals[c], goals, remaining=[j for j in route if j != c])
                for action, cost, detail in local[c]:
                    candidates.append({'action': action, 'target': c, 'score_s': cost+tail_m/5,
                                       'local_proxy_s': cost, 'known_tail_m': tail_m,
                                       'known_tail': tail, **detail})
            chosen = min(candidates, key=lambda row: row['score_s'])
            from dataclasses import asdict
            return self.record(b, chosen['action'], 'sector_known_route', focus=self.focus,
                               candidates=[{**row, 'action': asdict(row['action'])} for row in candidates],
                               known_route_m=route_m, deferred_search_stations=len(self.points),
                               score_kind='known-route-and-local-work-proxy-with-separate-finite-search')
        # Only public feedback can empty a channel. Ring landmarks provide
        # the complete backstop; inherited certified movement/pruning can reuse
        # all actual measurements obtained while processing known sectors.
        self.focus = None
        return super().choose(b)


class ProbePolicy(RingPolicy):
    """Joint route retains every obligation; only probe locations are enlarged."""
    probe = SectorPolicy.probe

    def __init__(self, config=None):
        super().__init__(config)
        self.implementation = 'ultra_s21_route_probes_v1'
        self.counters['movement_probe_candidates'] = 0

    def local_choices(self, b, c):
        goals = {j: p.summary()[0] for j, p in b.channels.items() if p.status == 'detected'}
        nearest = sorted(self.points, key=lambda k: distance(self.points[k], goals[c]))[:2]
        goals.update({k: self.points[k] for k in nearest if self.needed(b, self.points[k])})
        choices = SectorPolicy.movement_choices(self, b, c, goals)
        try:
            mean = tuple(map(float, self.model.posterior(b.channels[c]).mean))
        except QuadratureError:
            mean = goals[c]
        return choices, mean


class ReusePolicy(ProbePolicy):
    """Keep the joint route while accumulating useful scans at actual stops."""
    opportunistic_scan = SectorPolicy.opportunistic_scan

    def __init__(self, config=None):
        super().__init__(config)
        self.implementation = 'q4_v4_ring22_route_probes_reuse'
        self.last_opportunistic_review = -1
        self.last_shared_position = (0., 0.)
        self.counters['opportunistic_scans'] = 0

    def choose(self, b):
        fallback = (self.completion_mode or b.steps >= self.config.completion_after
                    or b.deadline-time.monotonic() < self.config.reserve_s)
        pending = (self.batch is not None and distance(self.batch, b.position) < 1e-6
                   and bool(self.needed(b, self.batch)))
        if not fallback and not pending and not b.done():
            for c, channel in b.channels.items():
                if channel.status == 'detected':
                    safe = guaranteed_clear(b, c)
                    if safe and distance(b.position, safe.position) < 1e-6:
                        return self.record(b, safe, 'guaranteed_clear')
            action = self.opportunistic_scan(b)
            if action:
                return action
        return super().choose(b)


class AlignedPolicy(ProbePolicy):
    """Choose landmark orientation once from the completed origin scan."""
    def __init__(self, config=None):
        super().__init__(config)
        self.implementation = 'q4_v4_ring22_aligned_probes'
        self.oriented = False

    def choose(self, b):
        pending = (self.batch is not None and distance(self.batch, b.position) < 1e-6
                   and bool(self.needed(b, self.batch)))
        if not self.oriented and b.steps > 0 and not pending and not self.completion_mode and not b.done():
            self.oriented = True
            known = {}
            for c, p in b.channels.items():
                if p.status == 'detected':
                    try:
                        known[c] = tuple(map(float, self.model.posterior(p).mean))
                    except QuadratureError:
                        known[c] = p.summary()[0]
            trials = []
            for i in range(14):
                angle = math.tau/7*i/14
                ca, sa = math.cos(angle), math.sin(angle)
                points = {k: (ca*x-sa*y, sa*x+ca*y) for k, (x, y) in self.points.items()}
                goals = {**known, **{k: p for k, p in points.items() if self.needed(b, p)}}
                _, route_m = self.route(b.position, goals)
                trials.append((route_m, angle, points))
            for route_m, angle, points in sorted(trials, key=lambda row: row[0]):
                valid = self.valid_future(b, points)
                self.pending_cover_audit.append({'operation': 'orient_landmarks',
                    'rotation_deg': math.degrees(angle), 'planned_route_m': route_m,
                    'full_certificate_valid': valid, 'public_detected_count': len(known),
                    'probability_used_for_proof': False})
                if valid:
                    self.points = points
                    break
        return super().choose(b)
