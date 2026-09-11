"""V3: route-aware measurements, stable target scheduling, complete cover tours.

All decisions use observed beliefs and deterministic position/angle quadrature.
The old V2 policy remains an executable paired control in policy.py.
"""
from dataclasses import asdict, dataclass

import numpy as np
import shapely

from .policy import BayesTSPPolicy, Config
from .posterior import QuadratureError, expected_after_measure, finish_cost
from .routing import route_length
from .scheduling import CoverTour, insertion_costs, scan_route_value
from .shared import Action, disk, distance, point_key


@dataclass(frozen=True)
class EfficientConfig(Config):
    route_measurements: bool = True
    stable_routing: bool = True
    coverage_routing: bool = True
    stable_radius_m: float = 100.
    switch_gain_s: float = 20.
    future_stops: int = 3
    pending_detours: bool = True


class EfficientPolicy(BayesTSPPolicy):
    implementation = 'bayes_tsp_v3_route_scheduling'

    def __init__(self, config=None):
        super().__init__(config or EfficientConfig())
        if (self.config.stable_radius_m <= 0 or self.config.switch_gain_s < 0 or
                self.config.future_stops < 1):
            raise ValueError('Invalid route scheduler parameters')
        self.active_target = None
        self.cover_tour = CoverTour(self.stations, self.station_disks)
        self._decision_context = {}

    def _record(self, b, action, status, **details):
        return super()._record(b, action, status, **{**self._decision_context, **details})

    def choose(self, b):
        self._decision_context = {}
        return super().choose(b)

    def _local_candidates(self, b, c, post):
        choices = super()._local_candidates(b, c, post)
        if (self.guaranteed_clear(b, c) is None and self._fresh(b.channels[c], b.position)
                and not any(detail.get('finite_grid') for _, _, detail in choices)):
            remaining, probabilities = expected_after_measure(post, b.position, self.config.bearing_bin)
            choices.append((Action('measure', b.position, c), 5+int(c != b.receiver)+remaining, probabilities))
        return choices

    def _tasks(self, b, posts):
        stable, pending, uncertainty = [], [], {}
        points = {}
        for c, post in posts.items():
            d = np.linalg.norm(post.xy-post.mean, axis=1)
            order = np.argsort(d)
            j = min(len(order)-1, int(np.searchsorted(np.cumsum(post.weights[order]), .9)))
            radius = float(d[order[j]])
            safe = self.guaranteed_clear(b, c)
            is_stable = safe is not None or (len(b.channels[c].directions) >= 2 and radius <= self.config.stable_radius_m)
            (stable if is_stable else pending).append(c)
            # A safe clear endpoint already accounts for the 20 m neighborhood.
            points[c] = safe.position if safe else tuple(map(float, post.mean))
            uncertainty[c] = {'radius90_m': radius, 'stable': is_stable,
                              'guaranteed_endpoint': safe is not None}
        return stable, pending, points, uncertainty

    def _select_task(self, b, posts, stable, pending, points):
        if self.active_target not in posts:
            self.active_target = None
        routable = stable if self.config.stable_routing else list(posts)
        if not routable:
            routable = []
        route_points = {f'source:{c}': points[c] for c in routable}
        route_keys = self.tour.update(b.position, route_points)
        route = [int(k.split(':')[1]) for k in route_keys]
        # Stable tasks first. If none are available, explicitly promote one
        # pending localizer; pending tasks are never silently dropped.
        considered = route[:3] if route else pending
        if route and self.config.stable_routing and self.config.pending_detours:
            # Stability controls which coordinates constrain TSP, not a ban on
            # nearby unfinished localizers. Price two pending detours against
            # the complete remaining stable route rather than driving past them.
            nearby = sorted(pending, key=lambda c: distance(b.position, points[c]))[:2]
            considered = considered+[c for c in nearby if c not in considered]
        if self.active_target is not None and self.active_target not in considered:
            considered = considered+[self.active_target]
        candidates = []
        for c in considered:
            rest = [k for k in route if k != c]
            for action, local, detail in self._local_candidates(b, c, posts[c]):
                endpoint = action.position if action.kind == 'clear' else points[c]
                tail = route_length(endpoint, rest, points)/5
                other = sum(finish_cost(posts[k].mean, posts[k].xy, posts[k].weights)
                            for k in posts if k != c)
                pending_travel = 0.
                if self.config.stable_routing and self.config.pending_detours:
                    # Pending targets do not become fixed TSP nodes, but their
                    # future travel is NOT free. Integrate each target's minimum
                    # insertion detour along the stable route. Summing separately
                    # is a heuristic and does not model interactions of insertions.
                    for k in pending:
                        if k != c:
                            extra = insertion_costs(endpoint, [points[j] for j in rest], posts[k].xy)
                            pending_travel += float(posts[k].weights @ extra.min(axis=0))/5
                switching = (self.config.switch_gain_s if self.config.stable_routing and
                             self.active_target is not None and c != self.active_target else 0.)
                candidates.append({'action': action, 'task': f'source:{c}', 'local_s': local,
                    'tail_route_s': tail, 'other_work_s': other, 'switch_hysteresis_s': switching,
                    'pending_travel_s': pending_travel,
                    'score_s': local+tail+other+pending_travel+switching, **detail})
        selected = min(candidates, key=lambda item: item['score_s'])
        target = int(selected['task'].split(':')[1])
        ordered = [target]+[k for k in route if k != target]
        # Include the impending physical observation stop before predicted clear
        # stops. These are forecasts only, never actual coverage certificates.
        future = []
        for q in [selected['action'].position]+[points[c] for c in ordered]:
            if distance(b.position, q) >= 20 and all(distance(q, old) >= 20 for old in future):
                future.append(q)
        return selected, candidates, ordered, future[:self.config.future_stops]

    def _known_sensing(self, b, posts, future, target):
        choices, reviews = [], []
        for c, post in posts.items():
            p = b.channels[c]
            if (self.guaranteed_clear(b, c) is not None or len(p.directions) >= self.max_bearings
                    or not self._fresh(p, b.position)):
                continue
            before = finish_cost(b.position, post.xy, post.weights)
            after, probabilities = expected_after_measure(post, b.position, self.config.bearing_bin)
            gain = before-after-5-int(c != b.receiver)
            future_gain = 0.
            # The active localizer has its actual current-position measurement
            # in the departure menu; do not defer its necessary measurement.
            if c != target:
                for pos in future:
                    if not self._fresh(p, pos):
                        continue
                    later, _ = expected_after_measure(post, pos, self.config.bearing_bin)
                    future_gain = max(future_gain, finish_cost(pos, post.xy, post.weights)-later-6)
            execute = gain > 2 and gain > future_gain+2
            reviews.append({'channel': c, 'now_gain_s': gain, 'future_gain_s': future_gain,
                            'execute_candidate': execute})
            if execute:
                choices.append((gain-future_gain, Action('measure', b.position, c), probabilities))
        self._decision_context['known_sensing_review'] = reviews
        return max(choices, key=lambda item: item[0]) if choices else None

    def _unknown_sensing(self, b, existence, future):
        reviews, choices = [], []
        remaining = [s for s in self.stations if self.unknown_at(b, s)]
        recovery = min(120., min((distance(b.position, s)/5 for s in remaining), default=0.))
        for c in self.unknown_at(b, b.position):
            if existence.get(c, 0.) <= 0:
                continue
            post = self.model.posterior(b.channels[c])
            review = scan_route_value(post, existence[c], b.position, future, recovery)
            expense = 5+int(c != b.receiver)
            net = review['gross_saving_s']-expense
            reviews.append({'channel': c, 'net_saving_s': net, 'measure_cost_s': expense, **review})
            if net > 2:
                choices.append((net, -c, Action('measure', b.position, c)))
        self._decision_context['unknown_sensing_review'] = reviews
        return max(choices, key=lambda item: item[:2]) if choices else None

    def _search_route(self, b, existence):
        remaining = shapely.union_all([p.region for p in b.channels.values() if p.status == 'unresolved'])
        scan_costs = [6*len(self.unknown_at(b, s)) for s in self.stations]
        cost, route = self.cover_tour.plan(b.position, remaining, scan_costs)
        if not route:
            raise QuadratureError('No coverage route while public state remains unfinished')
        first = self.stations[route[0]]
        scan = self._scan_choice(b, first, existence, forced=True)
        if scan is None:
            raise QuadratureError('Coverage route has no executable first scan')
        candidates = [{'action': scan[2], 'task': f'survey:{route[0]}', 'score_s': cost,
                       'cover_route': route, 'score_kind': 'full_no_signal_cover_route'}]
        for pos in self._blind_points(b):
            scan = self._scan_choice(b, pos, existence, forced=True)
            if scan is None or distance(pos, b.position) < 1e-6:
                continue
            # Hypothetical no-signal coverage for planning only. Never write this
            # projected geometry back to the actual belief or stop condition.
            rest = remaining.difference(disk(pos, 1000))
            tail_scan = [6*sum(p.status == 'unresolved' and
                not p.region.difference(disk(pos, 1000)).intersection(cover).is_empty
                for p in b.channels.values()) for cover in self.station_disks]
            tail, path = self.cover_tour.plan(pos, rest, tail_scan)
            total = distance(b.position, pos)/5+6*len(self.unknown_at(b, pos))+tail
            candidates.append({'action': scan[2], 'task': 'blind:projected', 'score_s': total,
                               'cover_route': path, 'score_kind': 'full_no_signal_cover_route'})
        selected = min(candidates, key=lambda item: item['score_s'])
        action = selected['action']
        self.batch_position = action.position
        self.batch_channels = self.unknown_at(b, action.position).copy()
        self.batch_kind = 'blind_batch_measure'
        return self._record(b, action, 'route_decision', selected_score_s=selected['score_s'],
            candidates=[{**r, 'action': asdict(r['action'])} for r in candidates],
            coverage_phase=True, active_target=None)

    def _choose_bayesian(self, b):
        # Keep already-paid-for coverage batches, reacting to near via choose().
        if self.batch_channels:
            if distance(b.position, self.batch_position) < 1e-6:
                self.batch_channels = [c for c in self.batch_channels if b.channels[c].status in ('detected', 'unresolved')
                    and point_key(self.batch_position) not in b.channels[c].measured]
                if self.batch_channels:
                    c = b.receiver if b.receiver in self.batch_channels else self.batch_channels[0]
                    return self._record(b, Action('measure', self.batch_position, c), self.batch_kind)
            self.batch_channels = []
        existence = self.model.existence(b)
        posts = {c: self.model.posterior(p) for c, p in b.channels.items() if p.status == 'detected'}
        if distance(b.position, (0., 0.)) < 1e-6:
            scan = self._scan_choice(b, b.position, existence, forced=True)
            if scan:
                return self._record(b, scan[2], 'station_scan', existence=existence)
        if not posts:
            self.active_target = None
            if self.config.coverage_routing:
                return self._search_route(b, existence)
            return super()._choose_bayesian(b)
        stable, pending, points, uncertainty = self._tasks(b, posts)
        selected, candidates, route, future = self._select_task(b, posts, stable, pending, points)
        target = int(selected['task'].split(':')[1])
        self._decision_context = {'stable_targets': stable, 'pending_targets': pending,
            'uncertainty': uncertainty, 'forecast_stops': future, 'route': route,
            'active_target_before': self.active_target}
        if self.config.shared:
            if self.config.route_measurements:
                shared = self._known_sensing(b, posts, future, target)
                scan = self._unknown_sensing(b, existence, future)
            else:
                shared = self._shared_choice(b, posts)
                at_clear = any(o.result == 'success' and distance(o.action.position, b.position) < 1e-6
                               for p in b.channels.values() for o in p.history)
                old_scan = self._scan_choice(b, b.position, existence) if at_clear else None
                scan = (old_scan[3], old_scan[1], old_scan[2]) if old_scan else None
            if shared and (scan is None or shared[0] >= scan[0]):
                return self._record(b, shared[1], 'shared_measure', predicted_saving_s=shared[0],
                                    feedback_probability=shared[2])
            if scan:
                return self._record(b, scan[2], 'incidental_scan', predicted_saving_s=scan[0], existence=existence)
        # A shared measurement is now scheduled at an upcoming physical stop,
        # rather than committing to a new 300--600 m joint tour before comparing
        # the free stops already available on the selected localization leg.
        previous = self.active_target
        self.active_target = target
        return self._record(b, selected['action'], 'route_decision', active_target=target,
            target_switched=previous is not None and previous != target,
            selected_score_s=selected['score_s'], route_points=points,
            candidates=[{**r, 'action': asdict(r['action'])} for r in candidates],
            coverage_phase=False)
