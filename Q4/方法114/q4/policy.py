"""Finite Q4 baseline and Bayesian joint routing; never imports simulator truth."""
from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from .coverage import certifies, relevant, square_stations
from .localization import finite_clear, guaranteed_clear, local_attempts, measure_points
from .posterior import Model, QuadratureError, expected_measure, finish_proxy
from .routing import length, open_route
from .shared import Action, GeometryError, distance, point_key


@dataclass(frozen=True)
class Config:
    resolution: int = 16
    bearing_bin: float = 2.
    directional_prior: float = .5
    existence_prior: float = .65
    local_limit: int = 12
    completion_after: int = 1600
    reserve_s: float = 30.
    exact_limit: int = 16


class Policy:
    implementation = 'q4_v2_heading_joint_exact16'

    def __init__(self, mode='joint', config=None):
        if mode not in ('baseline', 'bayes', 'joint', 'mobile'):
            raise ValueError('Unknown Q4 policy')
        self.mode, self.config = mode, config or Config()
        if self.config.local_limit < 2 or self.config.completion_after < 1 or not 0 <= self.config.exact_limit <= 16:
            raise ValueError('Invalid completion guard')
        self.model = Model(self.config.resolution, self.config.directional_prior, self.config.existence_prior)
        self.fixed = {-i-1: p for i, p in enumerate(square_stations())}
        self.points = dict(self.fixed)
        self.records = []
        self.batch = None
        self.completion_mode = False
        self.shared_checked = -1
        self.local_cache = {}
        self.geometry_cache = {}
        self.counters = {'quadrature_fallbacks': 0, 'finite_clears': 0, 'shared_replacements': 0,
                         'station_moves': 0, 'certificate_reviews': 0,
                         'exact_route_calls': 0, 'heuristic_route_calls': 0}

    def route(self, position, goals, remaining=None):
        count = len(goals) if remaining is None else len(remaining)
        key = 'exact_route_calls' if count <= self.config.exact_limit else 'heuristic_route_calls'
        self.counters[key] += 1
        return open_route(position, goals, exact_limit=self.config.exact_limit, remaining=remaining)

    def record(self, b, action, reason, **extra):
        move = distance(b.position, action.position)
        self.records.append({'step': b.steps, 'reason': reason, 'action': asdict(action),
                             'move_m': move, 'long_move_review': move >= 200,
                             'completion_mode': self.completion_mode, **extra})
        return action

    def needed(self, b, p):
        if len(b.known) == 16:
            return []
        key = point_key(p)
        result = []
        for c, channel in b.channels.items():
            if channel.status != 'unresolved' or key in channel.measured:
                continue
            cache_key = (channel.region.wkb, p)
            if cache_key not in self.geometry_cache:
                if len(self.geometry_cache) >= 2048:
                    self.geometry_cache.clear()
                self.geometry_cache[cache_key] = relevant(channel.region, p)
            if self.geometry_cache[cache_key]:
                result.append(c)
        return result

    def scan(self, b, position, reason, **extra):
        channels = self.needed(b, position)
        if not channels:
            return None
        c = b.receiver if b.receiver in channels else channels[0]
        self.batch = position
        return self.record(b, Action('measure', position, c), reason, channels=channels, **extra)

    def basic_local(self, b, c):
        safe = guaranteed_clear(b, c)
        if safe:
            return safe
        p = b.channels[c]
        if self.completion_mode or local_attempts(p) >= self.config.local_limit:
            self.counters['finite_clears'] += 1
            return finite_clear(b, c)
        mean, radius = p.summary()
        points = measure_points(b, c, mean)
        if not points:
            self.counters['finite_clears'] += 1
            return finite_clear(b, c)
        # Geometry-only baseline seeks a useful cross-angle; no heading prior.
        first = p.directions[-1]
        u = np.asarray(mean)-first.action.position
        def score(q):
            v = np.asarray(mean)-q
            cross = abs(u[0]*v[1]-u[1]*v[0])/max(np.linalg.norm(u)*np.linalg.norm(v), 1e-9)
            return distance(b.position, q)/5+radius/(5*max(cross, .08))
        return Action('measure', min(points, key=score), c)

    def local_choices(self, b, c):
        p = b.channels[c]
        safe = guaranteed_clear(b, c)
        if safe:
            return [(safe, distance(b.position, safe.position)/5+5, {'guaranteed': True})], safe.position
        if self.completion_mode or local_attempts(p) >= self.config.local_limit:
            a = finite_clear(b, c)
            self.counters['finite_clears'] += 1
            return [(a, distance(b.position, a.position)/5+3+p.region.area/400*3,
                     {'finite_grid': True, 'cost_is_proxy': True})], p.summary()[0]
        try:
            post = self.model.posterior(p)
        except QuadratureError as exc:
            self.counters['quadrature_fallbacks'] += 1
            a = finite_clear(b, c)
            return [(a, distance(b.position, a.position)/5+3+p.region.area/400*3,
                     {'quadrature_error': str(exc), 'finite_grid': True})], p.summary()[0]
        options = []
        # Coarse 10-degree integration preselects; the configured bins refine.
        candidates = measure_points(b, c, post.mean)
        def evaluate(pos, bins):
            cache_key = (c, p.revision, pos, bins)
            if cache_key not in self.local_cache:
                if len(self.local_cache) > 1024:
                    self.local_cache.clear()
                self.local_cache[cache_key] = expected_measure(post, pos, bins)
            future, detail = self.local_cache[cache_key]
            return distance(b.position, pos)/5+5+int(c != b.receiver)+future, detail
        coarse = sorted((evaluate(pos, 10.)[0], i) for i, pos in enumerate(candidates))
        finalists = [i for _, i in coarse[:3]]
        base = self.basic_local(b, c)
        if base.kind == 'measure' and base.position in candidates:
            i = candidates.index(base.position)
            if i not in finalists:
                finalists.append(i)
        for i in finalists:
            pos = candidates[i]
            cost, detail = evaluate(pos, self.config.bearing_bin)
            options.append((Action('measure', pos, c), cost, detail))
        failed = {point_key(o.action.position) for o in p.history if o.result == 'no_target_in_range'}
        failures = sum(o.result == 'no_target_in_range' for o in p.history)
        for pos in dict.fromkeys((b.position, tuple(map(float, post.mean)), post.clear_point())):
            chance = post.clear_probability(pos)
            if chance < .35 or failures >= 3 or point_key(pos) in failed:
                continue
            missed = post.weights*(np.linalg.norm(post.xy-pos, axis=1) > 20)
            mass = float(missed.sum())
            recovery = finish_proxy(pos, post.xy, missed/mass) if mass > 1e-15 else 0.
            # Failure does not teleport to the next point or erase remaining work.
            cost = distance(b.position, pos)/5+3+2*chance+mass*recovery
            options.append((Action('clear', pos, c), cost, {'clear_probability': chance,
                                                          'failure_recovery_proxy_s': recovery}))
        if not options:
            a = finite_clear(b, c)
            return [(a, distance(b.position, a.position)/5+3+p.region.area/400*3,
                     {'finite_grid': True})], p.summary()[0]
        return options, tuple(map(float, post.mean))

    def valid_future(self, b, points):
        """No truth and no belief mutation: verify the all-negative continuation."""
        self.counters['certificate_reviews'] += 1
        seen = set()
        for channel in b.channels.values():
            if channel.status != 'unresolved':
                continue
            key = (channel.region.wkb, tuple(sorted(channel.negatives)))
            if key in seen:
                continue
            seen.add(key)
            if not certifies(channel.region, channel.negatives+tuple(points.values())):
                return False
        return True

    def shared_replacement(self, b):
        if not b.applied or self.shared_checked == b.steps:
            return None
        self.shared_checked = b.steps
        _, response = b.applied[next(reversed(b.applied))]
        if response.get('clear_result') != 'success' or not self.needed(b, b.position):
            return None
        remaining = {k: p for k, p in self.points.items() if self.needed(b, p)}
        route, before = self.route(b.position, remaining)
        trials = sorted(remaining, key=lambda k: distance(b.position, remaining[k]))[:3]
        audit = []
        for key in trials:
            proposed = dict(self.points)
            proposed[key] = b.position
            adjusted = dict(remaining)
            adjusted[key] = b.position
            _, after = self.route(b.position, adjusted)
            extra_scans = 6*(len(self.needed(b, b.position))-len(self.needed(b, remaining[key])))
            saving = (before-after)/5-extra_scans
            valid = saving > 1 and self.valid_future(b, proposed)
            audit.append({'station': key, 'predicted_saving_s': saving, 'certificate_valid': valid})
            if valid:
                self.points = proposed
                self.counters['shared_replacements'] += 1
                return self.scan(b, b.position, 'shared_station_replacement', certificate_audit=audit)
        return None

    def move_station(self, b, route, goals):
        if not route or route[0] >= 0:
            return []
        key = route[0]
        old = np.asarray(goals[key])
        previous = np.asarray(b.position)
        following = np.asarray(goals[route[1]]) if len(route) > 1 else previous
        v = following-previous
        t = np.clip(float((old-previous) @ v)/max(float(v @ v), 1e-12), 0, 1)
        target = previous+t*v
        before = distance(previous, old)+distance(old, following)
        audit = []
        for fraction in (1., .5, .25, .125):
            p = tuple(map(float, old+fraction*(target-old)))
            saving = before-distance(previous, p)-distance(p, following)
            if saving <= 1:
                continue
            proposed = dict(self.points)
            proposed[key] = p
            valid = self.valid_future(b, proposed)
            audit.append({'station': key, 'fraction': fraction, 'saved_leg_m': saving,
                          'certificate_valid': valid})
            if valid:
                self.points = proposed
                self.counters['station_moves'] += 1
                goals[key] = p
                break
        return audit

    def choose(self, b):
        if b.done():
            raise ValueError('No action after completion')
        if b.steps >= self.config.completion_after or b.deadline-time.monotonic() < self.config.reserve_s:
            self.completion_mode = True
        if self.completion_mode:
            self.points = dict(self.fixed)  # Restore finite obligations, keeping actual evidence.
        detected = [c for c, p in b.channels.items() if p.status == 'detected']
        for c in detected:
            safe = guaranteed_clear(b, c)
            if safe and distance(safe.position, b.position) < 1e-6:
                return self.record(b, safe, 'guaranteed_clear')
        if self.batch is not None:
            if distance(b.position, self.batch) < 1e-6:
                action = self.scan(b, self.batch, 'station_batch')
                if action:
                    return action
            self.batch = None
        if self.mode in ('joint', 'mobile') and not self.completion_mode:
            action = self.shared_replacement(b)
            if action:
                return action
        at_station = next((p for p in self.points.values() if distance(p, b.position) < 1e-6), None)
        if at_station is not None:
            action = self.scan(b, at_station, 'station_batch')
            if action:
                return action
        if self.mode == 'baseline' or self.completion_mode:
            if detected:
                c = min(detected, key=lambda c: distance(b.position, b.channels[c].summary()[0]))
                a = self.basic_local(b, c)
                return self.record(b, a, 'baseline_local', target=c)
            stations = [p for p in self.points.values() if self.needed(b, p)]
            if stations:
                return self.scan(b, min(stations, key=lambda p: distance(p, b.position)), 'baseline_survey')
            raise GeometryError('Fixed grid exhausted without a completion certificate')
        local, goals, overhead = {}, {}, {}
        for c in detected:
            choices, mean = self.local_choices(b, c)
            local[c], goals[c] = choices, mean
            overhead[c] = max(0., min(cost-distance(b.position, a.position)/5 for a, cost, _ in choices))
        if self.mode != 'bayes' or not detected:
            for k, p in self.points.items():
                channels = self.needed(b, p)
                if channels:
                    goals[k], overhead[k] = p, 6*len(channels)
        if not goals:
            raise GeometryError('No tasks but completion is not proved')
        route, route_m = self.route(b.position, goals)
        mobile_audit = self.move_station(b, route, goals) if self.mode == 'mobile' else []
        if mobile_audit:
            for k in list(goals):
                if k < 0 and not self.needed(b, goals[k]):
                    del goals[k]
                    overhead.pop(k, None)
            route, route_m = self.route(b.position, goals)
            for k in goals:
                if k < 0:
                    overhead[k] = 6*len(self.needed(b, goals[k]))
        shortlist = list(dict.fromkeys(route[:3]+sorted(detected, key=lambda c: distance(b.position, goals[c]))[:2]))
        candidates = []
        for k in shortlist:
            rest = [j for j in route if j != k]
            other = sum(overhead[j] for j in rest)
            # Recompute the entire conditional remaining tour for this task.
            # Exact whenever its total node count is within the configured cap.
            tail_order, tail_m = self.route(goals[k], goals, remaining=rest)
            if k < 0:
                channels = self.needed(b, goals[k])
                if not channels:
                    continue
                c = b.receiver if b.receiver in channels else channels[0]
                options = [(Action('measure', goals[k], c), distance(b.position, goals[k])/5+overhead[k], {})]
            else:
                options = local[k]
            for a, cost, detail in options:
                # Known source completion is represented by its posterior goal;
                # a scan completes at the actual station. Tail order is heuristic.
                tail = tail_m/5
                candidates.append({'action': a, 'task': k, 'score_s': cost+tail+other,
                                   'local_proxy_s': cost, 'tail_route_s': tail,
                                   'other_work_proxy_s': other,
                                   'tail_order': tail_order,
                                   'tail_exact': len(rest) <= self.config.exact_limit, **detail})
        selected = min(candidates, key=lambda r: r['score_s'])
        if selected['task'] < 0:
            self.batch = selected['action'].position
        audit = [{**r, 'action': asdict(r['action'])} for r in candidates]
        return self.record(b, selected['action'], 'joint_route' if self.mode != 'bayes' else 'bayes_route',
                           candidates=audit, planned_route=route, planned_route_m=route_m,
                           route_exact=len(goals) <= self.config.exact_limit,
                           station_adjustments=mobile_audit,
                           score_kind='remaining-route-and-local-work-proxy',
                           unknown_source_localization_not_predicted=True)
