"""Q4 paper algorithm: S21 coverage certificates and movement-aware Bayesian probes."""

from dataclasses import asdict, dataclass
import time
import numpy as np
from s21_layout import stations
from .coverage import certifies, relevant, square_stations
from .localization import finite_clear, guaranteed_clear, local_attempts, measure_points
from .posterior import Model, QuadratureError, expected_measure, finish_proxy
from .routing import open_route
from .shared import Action, GeometryError, distance, point_key


@dataclass(frozen=True)
class Config:
    resolution: int = 16
    bearing_bin: float = 2.0
    directional_prior: float = 0.5
    existence_prior: float = 0.65
    local_limit: int = 12
    completion_after: int = 1600
    reserve_s: float = 30.0
    exact_limit: int = 16


class Policy:
    implementation = 'q4_paper_s21_probes'

    def __init__(self, config=None):
        self.config = config or Config()
        if (
            self.config.local_limit < 2
            or self.config.completion_after < 1
            or (not 0 <= self.config.exact_limit <= 16)
        ):
            raise ValueError('Invalid completion guard')
        self.model = Model(self.config.resolution, self.config.directional_prior, self.config.existence_prior)
        self.fixed = {-i - 1: p for i, p in enumerate(square_stations())}
        self.points = {-i - 1: p for i, p in enumerate(stations())}
        self.records = []
        self.batch = None
        self.completion_mode = False
        self.shared_checked = -1
        self.local_cache = {}
        self.geometry_cache = {}
        self.counters = {
            'quadrature_fallbacks': 0,
            'finite_clears': 0,
            'shared_replacements': 0,
            'station_moves': 0,
            'certificate_reviews': 0,
            'exact_route_calls': 0,
            'heuristic_route_calls': 0,
        }
        self.prune_signature = None
        self.prune_failed = set()
        self.pending_cover_audit = []
        self.counters.update(
            cover_prune_reviews=0,
            cover_pruned_stations=0,
            cover_prune_cache_hits=0,
            cover_exhaustion_fallbacks=0,
            initial_search_stations=len(self.points),
        )
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
        choices, _mean = self._local_menu(b, c)
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
                and all(
                    (distance(x, o.action.position) >= 10 for o in p.history if o.action.kind == 'measure')
                )
            ]
            self.counters['movement_probe_candidates'] += len(candidates)
            coarse = sorted(((self.probe(b, c, x, 10.0)[1], i) for i, x in enumerate(candidates)))
            choices += [self.probe(b, c, candidates[i], self.config.bearing_bin) for _, i in coarse[:3]]
        except QuadratureError:
            pass
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
        return (choices, mean)

    def record(self, b, action, reason, **extra):
        if self.pending_cover_audit:
            extra['coverage_plan_audit'] = self.pending_cover_audit
            self.pending_cover_audit = []
        extra['planned_search_stations'] = len(self.points)
        return self._append_record(b, action, reason, **extra)

    def prune_stations(self, b):
        active = {k: p for k, p in self.points.items() if self.needed(b, p)}
        if len(active) != len(self.points) and self.valid_future(b, active):
            self.points = active
        if len(active) < 2:
            return
        signature = (
            tuple(sorted(self.points.items())),
            tuple(
                sorted(
                    {
                        (c.region.wkb, tuple(sorted(c.negatives)))
                        for c in b.channels.values()
                        if c.status == 'unresolved'
                    }
                )
            ),
        )
        if signature != self.prune_signature:
            self.prune_signature, self.prune_failed = (signature, set())
        route, _ = self.route(b.position, active)
        trials = []
        for i, k in enumerate(route):
            if distance(b.position, active[k]) < 1e-06:
                continue
            previous = b.position if i == 0 else active[route[i - 1]]
            saving_m = distance(previous, active[k])
            if i + 1 < len(route):
                following = active[route[i + 1]]
                saving_m += distance(active[k], following) - distance(previous, following)
            predicted_saving = saving_m / 5 + 6 * len(self.needed(b, active[k]))
            trials.append((predicted_saving, k, saving_m))
        reviews = 0
        for saving, key, saving_m in sorted(trials, reverse=True):
            if key in self.prune_failed:
                self.counters['cover_prune_cache_hits'] += 1
                continue
            if reviews >= 4:
                break
            reviews += 1
            proposed = {k: p for k, p in self.points.items() if k != key}
            valid = self.valid_future(b, proposed)
            self.counters['cover_prune_reviews'] += 1
            self.pending_cover_audit.append(
                {
                    'operation': 'remove_station',
                    'station': key,
                    'position': self.points[key],
                    'route_shortcut_m': saving_m,
                    'predicted_saved_s': saving,
                    'full_certificate_valid': valid,
                    'sites_before': len(self.points),
                    'sites_after': len(proposed),
                    'probability_used_for_proof': False,
                }
            )
            if valid:
                self.points = proposed
                self.counters['cover_pruned_stations'] += 1
                self.prune_signature = None
                break
            self.prune_failed.add(key)

    def choose(self, b):
        fallback = (
            self.completion_mode
            or b.steps >= self.config.completion_after
            or b.deadline - time.monotonic() < self.config.reserve_s
        )
        batch_pending = (
            self.batch is not None
            and distance(self.batch, b.position) < 1e-06
            and bool(self.needed(b, self.batch))
        )
        if not fallback and (not b.done()) and (not batch_pending):
            self.prune_stations(b)
        if (
            not fallback
            and (not b.done())
            and (not any((c.status == 'detected' for c in b.channels.values())))
            and (not any((self.needed(b, p) for p in self.points.values())))
        ):
            self.completion_mode = True
            self.counters['cover_exhaustion_fallbacks'] += 1
            self.pending_cover_audit.append(
                {'operation': 'restore_finite_grid', 'reason': 'plan_exhausted_without_actual_certificate'}
            )
        return self._choose_observed(b)

    def route(self, position, goals, remaining=None):
        count = len(goals) if remaining is None else len(remaining)
        key = 'exact_route_calls' if count <= self.config.exact_limit else 'heuristic_route_calls'
        self.counters[key] += 1
        return open_route(position, goals, exact_limit=self.config.exact_limit, remaining=remaining)

    def _append_record(self, b, action, reason, **extra):
        move = distance(b.position, action.position)
        self.records.append(
            {
                'step': b.steps,
                'reason': reason,
                'action': asdict(action),
                'move_m': move,
                'long_move_review': move >= 200,
                'completion_mode': self.completion_mode,
                **extra,
            }
        )
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
        first = p.directions[-1]
        u = np.asarray(mean) - first.action.position

        def score(q):
            v = np.asarray(mean) - q
            cross = abs(u[0] * v[1] - u[1] * v[0]) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-09)
            return distance(b.position, q) / 5 + radius / (5 * max(cross, 0.08))

        return Action('measure', min(points, key=score), c)

    def _local_menu(self, b, c):
        p = b.channels[c]
        safe = guaranteed_clear(b, c)
        if safe:
            return (
                [(safe, distance(b.position, safe.position) / 5 + 5, {'guaranteed': True})],
                safe.position,
            )
        if self.completion_mode or local_attempts(p) >= self.config.local_limit:
            a = finite_clear(b, c)
            self.counters['finite_clears'] += 1
            return (
                [
                    (
                        a,
                        distance(b.position, a.position) / 5 + 3 + p.region.area / 400 * 3,
                        {'finite_grid': True, 'cost_is_proxy': True},
                    )
                ],
                p.summary()[0],
            )
        try:
            post = self.model.posterior(p)
        except QuadratureError as exc:
            self.counters['quadrature_fallbacks'] += 1
            a = finite_clear(b, c)
            return (
                [
                    (
                        a,
                        distance(b.position, a.position) / 5 + 3 + p.region.area / 400 * 3,
                        {'quadrature_error': str(exc), 'finite_grid': True},
                    )
                ],
                p.summary()[0],
            )
        options = []
        candidates = measure_points(b, c, post.mean)

        def evaluate(pos, bins):
            cache_key = (c, p.revision, pos, bins)
            if cache_key not in self.local_cache:
                if len(self.local_cache) > 1024:
                    self.local_cache.clear()
                self.local_cache[cache_key] = expected_measure(post, pos, bins)
            future, detail = self.local_cache[cache_key]
            return (distance(b.position, pos) / 5 + 5 + int(c != b.receiver) + future, detail)

        coarse = sorted(((evaluate(pos, 10.0)[0], i) for i, pos in enumerate(candidates)))
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
        failures = sum((o.result == 'no_target_in_range' for o in p.history))
        for pos in dict.fromkeys((b.position, tuple(map(float, post.mean)), post.clear_point())):
            chance = post.clear_probability(pos)
            if chance < 0.35 or failures >= 3 or point_key(pos) in failed:
                continue
            missed = post.weights * (np.linalg.norm(post.xy - pos, axis=1) > 20)
            mass = float(missed.sum())
            recovery = finish_proxy(pos, post.xy, missed / mass) if mass > 1e-15 else 0.0
            cost = distance(b.position, pos) / 5 + 3 + 2 * chance + mass * recovery
            options.append(
                (
                    Action('clear', pos, c),
                    cost,
                    {'clear_probability': chance, 'failure_recovery_proxy_s': recovery},
                )
            )
        if not options:
            a = finite_clear(b, c)
            return (
                [
                    (
                        a,
                        distance(b.position, a.position) / 5 + 3 + p.region.area / 400 * 3,
                        {'finite_grid': True},
                    )
                ],
                p.summary()[0],
            )
        return (options, tuple(map(float, post.mean)))

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
            if not certifies(channel.region, channel.negatives + tuple(points.values())):
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
        _route, before = self.route(b.position, remaining)
        trials = sorted(remaining, key=lambda k: distance(b.position, remaining[k]))[:3]
        audit = []
        for key in trials:
            proposed = dict(self.points)
            proposed[key] = b.position
            adjusted = dict(remaining)
            adjusted[key] = b.position
            _, after = self.route(b.position, adjusted)
            extra_scans = 6 * (len(self.needed(b, b.position)) - len(self.needed(b, remaining[key])))
            saving = (before - after) / 5 - extra_scans
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
        v = following - previous
        t = np.clip(float((old - previous) @ v) / max(float(v @ v), 1e-12), 0, 1)
        target = previous + t * v
        before = distance(previous, old) + distance(old, following)
        audit = []
        for fraction in (1.0, 0.5, 0.25, 0.125):
            p = tuple(map(float, old + fraction * (target - old)))
            saving = before - distance(previous, p) - distance(p, following)
            if saving <= 1:
                continue
            proposed = dict(self.points)
            proposed[key] = p
            valid = self.valid_future(b, proposed)
            audit.append(
                {'station': key, 'fraction': fraction, 'saved_leg_m': saving, 'certificate_valid': valid}
            )
            if valid:
                self.points = proposed
                self.counters['station_moves'] += 1
                goals[key] = p
                break
        return audit

    def _choose_observed(self, b):
        if b.done():
            raise ValueError('No action after completion')
        if b.steps >= self.config.completion_after or b.deadline - time.monotonic() < self.config.reserve_s:
            self.completion_mode = True
        if self.completion_mode:
            self.points = dict(self.fixed)
        detected = [c for c, p in b.channels.items() if p.status == 'detected']
        for c in detected:
            safe = guaranteed_clear(b, c)
            if safe and distance(safe.position, b.position) < 1e-06:
                return self.record(b, safe, 'guaranteed_clear')
        if self.batch is not None:
            if distance(b.position, self.batch) < 1e-06:
                action = self.scan(b, self.batch, 'station_batch')
                if action:
                    return action
            self.batch = None
        if not self.completion_mode:
            action = self.shared_replacement(b)
            if action:
                return action
        at_station = next((p for p in self.points.values() if distance(p, b.position) < 1e-06), None)
        if at_station is not None:
            action = self.scan(b, at_station, 'station_batch')
            if action:
                return action
        if self.completion_mode:
            if detected:
                c = min(detected, key=lambda c: distance(b.position, b.channels[c].summary()[0]))
                a = self.basic_local(b, c)
                return self.record(b, a, 'finite_local', target=c)
            stations = [p for p in self.points.values() if self.needed(b, p)]
            if stations:
                return self.scan(b, min(stations, key=lambda p: distance(p, b.position)), 'finite_survey')
            raise GeometryError('Fixed grid exhausted without a completion certificate')
        local, goals, overhead = ({}, {}, {})
        for c in detected:
            choices, mean = self.local_choices(b, c)
            local[c], goals[c] = (choices, mean)
            overhead[c] = max(
                0.0, min((cost - distance(b.position, a.position) / 5 for a, cost, _ in choices))
            )
        for k, p in self.points.items():
            channels = self.needed(b, p)
            if channels:
                goals[k], overhead[k] = (p, 6 * len(channels))
        if not goals:
            raise GeometryError('No tasks but completion is not proved')
        route, route_m = self.route(b.position, goals)
        mobile_audit = self.move_station(b, route, goals)
        if mobile_audit:
            for k in list(goals):
                if k < 0 and (not self.needed(b, goals[k])):
                    del goals[k]
                    overhead.pop(k, None)
            route, route_m = self.route(b.position, goals)
            for k in goals:
                if k < 0:
                    overhead[k] = 6 * len(self.needed(b, goals[k]))
        shortlist = list(
            dict.fromkeys(route[:3] + sorted(detected, key=lambda c: distance(b.position, goals[c]))[:2])
        )
        candidates = []
        for k in shortlist:
            rest = [j for j in route if j != k]
            other = sum((overhead[j] for j in rest))
            tail_order, tail_m = self.route(goals[k], goals, remaining=rest)
            if k < 0:
                channels = self.needed(b, goals[k])
                if not channels:
                    continue
                c = b.receiver if b.receiver in channels else channels[0]
                options = [
                    (Action('measure', goals[k], c), distance(b.position, goals[k]) / 5 + overhead[k], {})
                ]
            else:
                options = local[k]
            for a, cost, detail in options:
                tail = tail_m / 5
                candidates.append(
                    {
                        'action': a,
                        'task': k,
                        'score_s': cost + tail + other,
                        'local_proxy_s': cost,
                        'tail_route_s': tail,
                        'other_work_proxy_s': other,
                        'tail_order': tail_order,
                        'tail_exact': len(rest) <= self.config.exact_limit,
                        **detail,
                    }
                )
        selected = min(candidates, key=lambda r: r['score_s'])
        if selected['task'] < 0:
            self.batch = selected['action'].position
        audit = [{**r, 'action': asdict(r['action'])} for r in candidates]
        return self.record(
            b,
            selected['action'],
            'joint_route',
            candidates=audit,
            planned_route=route,
            planned_route_m=route_m,
            route_exact=len(goals) <= self.config.exact_limit,
            station_adjustments=mobile_audit,
            score_kind='remaining-route-and-local-work-proxy',
            unknown_source_localization_not_predicted=True,
        )
