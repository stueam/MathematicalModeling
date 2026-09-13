"""Q3 paper algorithm: fixed-radius posterior, joint work routing and finite completion."""

from dataclasses import asdict, dataclass
import math
import time
import numpy as np
import shapely
from shapely.geometry import Polygon, box
from .posterior import BayesModel, QuadratureError, clear_finish_cost, expected_after_measure, finish_cost
from .routing import DynamicTour, route_length
from .open_routes import OpenRoutes
from .scheduling import scan_route_value
from .stations import covers_vertices, adjust_station, repaired_station
from .shared import Action, GeometryError, core, disk, distance, point_key


@dataclass(frozen=True)
class Config:
    ring: float = 1200.0
    resolution: int = 24
    bearing_bin: float = 1.0
    p0: float = 0.65
    max_bearings: int = 8
    completion_after: int = 600
    reserve_s: float = 30.0
    future_stops: int = 3
    exact_limit: int = 12
    sectors: int = 7
    station_sweeps: int = 2
    cover_radius_m: float = 997.0
    recovery_sine: float = 0.15


class Policy:
    implementation = 'q3_paper_bayes_fast'

    def __init__(self, config=None):
        self.config = config or Config()
        worst = max(
            (
                math.sqrt(r * r + self.config.ring**2 - 2 * r * self.config.ring * math.cos(math.pi / 6))
                for r in (1000, 1800)
            )
        )
        if worst >= 995:
            raise ValueError('Patrol ring lacks sufficient coverage margin')
        self.stations = [(0.0, 0.0)] + [
            (self.config.ring * math.cos(k * math.pi / 3), self.config.ring * math.sin(k * math.pi / 3))
            for k in range(6)
        ]
        self.max_bearings = self.config.max_bearings
        if (
            self.config.bearing_bin <= 0
            or abs(round(360 / self.config.bearing_bin) * self.config.bearing_bin - 360) > 1e-09
        ):
            raise ValueError('bearing_bin must divide 360')
        if self.config.max_bearings < 2 or self.config.completion_after < 1:
            raise ValueError('Invalid completion limits')
        self.model = BayesModel(self.config.resolution, self.config.p0)
        self.tour = DynamicTour()
        self.records = []
        self.completion_mode = False
        self._unknown_revision = None
        self._unknown_cache = {}
        self.batch_position, self.batch_channels = (None, [])
        self.batch_kind = 'joint_batch_measure'
        if self.config.future_stops < 1:
            raise ValueError('Invalid route scheduler parameters')
        self.active_target = None
        self._decision_context = {}
        if not 1 <= self.config.exact_limit <= 16:
            raise ValueError('exact_limit must be 1..16')
        self.after_clear = None
        self._joint_details = {}
        self._tables = {}
        if not 6 <= self.config.sectors <= 12 or not 0 <= self.config.station_sweeps <= 4:
            raise ValueError('sectors must be 6..12 and station_sweeps 0..4')
        if not 900 <= self.config.cover_radius_m <= 997:
            raise ValueError('cover radius must leave margin inside the exclusion polygon')
        self.sector_shapes = []
        self.fixed_centers = []
        for j in range(self.config.sectors):
            angle = j * math.tau / self.config.sectors
            endpoints = [
                angle - math.pi / self.config.sectors - 1e-09,
                angle + math.pi / self.config.sectors + 1e-09,
            ]
            self.sector_shapes.append(
                Polygon([(0.0, 0.0)] + [(5000 * math.cos(a), 5000 * math.sin(a)) for a in endpoints])
            )
            self.fixed_centers.append((1100 * math.cos(angle), 1100 * math.sin(angle)))
        self._task_key, self._tasks_cache = (None, None)
        self._last_task_log = {}
        self.station_adjustments = 0
        self.station_adjustments_accepted = 0
        self.station_leg_saving_m = 0.0
        self.clear_sector_services = 0
        self.repair_accepts = 0
        self.recovery_filtered = 0
        self._decision_menus = None
        self._decision_local_costs = None

    def choose(self, b):
        """Choose from public feedback; discard numerical menu caches after every action."""
        self._decision_menus, self._decision_local_costs = ({}, {})
        self._joint_details, self._decision_context = ({}, {})
        try:
            return self._choose_observed(b)
        finally:
            self._decision_menus = self._decision_local_costs = None

    def _table(self, points):
        key = tuple(((k, tuple(map(float, points[k]))) for k in sorted(points)))
        if key not in self._tables:
            if len(self._tables) >= 8:
                self._tables.pop(next(iter(self._tables)))
            self._tables[key] = OpenRoutes(points, self.config.exact_limit)
        return self._tables[key]

    @staticmethod
    def _copy_menu(menu):
        return [(action, cost, dict(detail)) for action, cost, detail in menu]

    def _local_candidates(self, b, c, post):
        if self._decision_menus is None:
            return self._recovery_candidates(b, c, post)
        key = (c, id(post))
        if key in self._decision_menus:
            menu, recovery_count = self._decision_menus[key]
            self.recovery_filtered += recovery_count
            return self._copy_menu(menu)
        before = self.recovery_filtered
        menu = self._recovery_candidates(b, c, post)
        self._decision_menus[key] = (self._copy_menu(menu), self.recovery_filtered - before)
        for action, local, detail in menu:
            cost_key = (c, id(post), action.kind, tuple(action.position), bool(detail.get('guaranteed')))
            probabilities = (
                {name: detail[name] for name in ('near', 'no_signal', 'direction', 'positive_bearing_bins')}
                if action.kind == 'measure'
                else {}
            )
            self._decision_local_costs[cost_key] = (local, probabilities)
        return menu

    def _score(self, b, action, c, plan, posts, detail=None):
        detail = dict(detail or {})
        key = (
            (c, id(posts[c]), action.kind, tuple(action.position), bool(detail.get('guaranteed')))
            if c >= 0
            else None
        )
        cached = self._decision_local_costs.get(key) if self._decision_local_costs is not None else None
        if cached is None:
            return self._score_uncached(b, action, c, plan, posts, detail=detail)
        local_cost, probabilities = cached
        detail.update(probabilities)
        local_cost += plan['scans'][c]
        points = plan['points']
        endpoint = action.position if action.kind == 'clear' else points[c]
        rest = [k for k in points if k != c]
        length, route = plan['table'].plan(endpoint, rest)
        other = sum((plan['local'][k] + plan['scans'][k] for k in rest))
        return {
            'action': action,
            'task': f'source:{c}',
            'local_s': local_cost,
            'tail_route_s': length / 5,
            'other_work_s': other,
            'score_s': local_cost + length / 5 + other,
            'continuation': route,
            'mask': plan['mask'],
            **detail,
        }

    def _plans(self, b, posts, source_points):
        plans = self._initial_plan(b, posts, source_points)
        tasks = self._coverage_tasks(b)
        for plan in plans:
            points, route = (dict(plan['points']), plan['route'])
            records = []
            for _ in range(self.config.station_sweeps):
                for j, key in enumerate(route):
                    if key >= 0:
                        continue
                    previous = b.position if j == 0 else points[route[j - 1]]
                    following = points[route[j + 1]] if j + 1 < len(route) else None
                    points[key], record = repaired_station(
                        tasks[key]['vertices'], points[key], previous, following, self.config.cover_radius_m
                    )
                    records.append({'task': key, **record})
                table = self._table(points)
                _, route = table.plan(b.position)
            table = self._table(points)
            length, route = table.plan(b.position)
            if length <= plan['length'] + 1e-06:
                plan.update(
                    points=points,
                    route=route,
                    table=table,
                    length=length,
                    proxy=length / 5 + sum(plan['local'].values()) + sum(plan['scans'].values()),
                )
            self.repair_accepts += sum((r['accepted'] for r in records))
            self._last_task_log.update(repaired_station_pass=records, repaired_route_m=plan['length'])
        return plans

    def _recovery_candidates(self, b, c, post):
        choices = self._include_current_measurement(b, c, post)
        if any((d.get('guaranteed') or d.get('finite_grid') for _, _, d in choices)):
            return choices
        failed_since_direction = False
        for obs in reversed(b.channels[c].history):
            if obs.result in ('direction', 'near'):
                break
            if obs.result == 'no_target_in_range':
                failed_since_direction = True
        if not failed_since_direction:
            return choices
        probes = [item for item in choices if item[0].kind == 'measure']
        if not probes:
            return choices
        useful = []
        for item in probes:
            v = post.mean - item[0].position
            crosses = []
            for obs in b.channels[c].directions:
                u = post.mean - obs.action.position
                crosses.append(
                    abs(u[0] * v[1] - u[1] * v[0]) / max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-09)
                )
            if max(crosses, default=0.0) >= self.config.recovery_sine:
                useful.append(item)
        self.recovery_filtered += sum((item[0].kind == 'clear' for item in choices))
        return useful or probes

    def _coverage_tasks(self, b):
        unknown = [(c, p) for c, p in b.channels.items() if p.status == 'unresolved']
        if len(b.cleared) + sum((p.status == 'detected' for p in b.channels.values())) == 16:
            return {}
        key = tuple(((c, p.revision) for c, p in unknown))
        if key == self._task_key:
            return self._tasks_cache
        tasks = {}
        for j, sector in enumerate(self.sector_shapes):
            members = {}
            for c, channel in unknown:
                part = channel.region.intersection(sector)
                if not part.is_empty:
                    members[c] = part
            if members:
                region = shapely.union_all(list(members.values()))
                tasks[-j - 1] = {
                    'region': region,
                    'members': members,
                    'vertices': shapely.get_coordinates(region.convex_hull),
                }
        self._task_key, self._tasks_cache = (key, tasks)
        return tasks

    def _clear_service(self, b):
        tasks = self._coverage_tasks(b)
        covered = [
            k
            for k, task in tasks.items()
            if covers_vertices(task['vertices'], b.position, self.config.cover_radius_m)
        ]
        channels = sorted(
            {
                c
                for k in covered
                for c in tasks[k]['members']
                if point_key(b.position) not in b.channels[c].measured
            }
        )
        return (channels, covered)

    def _choose_bayesian(self, b):
        if self.after_clear is not None:
            position, channel = self.after_clear
            self.after_clear = None
            if b.channels[channel].status == 'cleared' and distance(position, b.position) < 1e-06:
                if distance(b.position, (0.0, 0.0)) < 1e-06:
                    channels, tasks = (self.unknown_at(b, b.position), [])
                else:
                    channels, tasks = self._clear_service(b)
                if channels:
                    self.batch_position, self.batch_channels = (b.position, list(channels))
                    self.batch_kind = 'task_clear_search'
                    self.clear_sector_services += len(tasks)
                    self._decision_context['covered_sector_services'] = tasks
        return self._decide_tasks(b)

    def _initial_center(self, key, task):
        p = self.fixed_centers[-key - 1]
        if covers_vertices(task['vertices'], p, self.config.cover_radius_m):
            return p
        center, radius = core.region_summary(task['region'])
        if radius <= self.config.cover_radius_m and covers_vertices(
            task['vertices'], center, self.config.cover_radius_m
        ):
            return center
        raise QuadratureError('Sector too wide for one guaranteed stop; retain finite seven-station fallback')

    def _initial_plan(self, b, posts, source_points):
        tasks = self._coverage_tasks(b)
        _, source_order = self._table(source_points).plan(b.position)
        vertices = {c: shapely.get_coordinates(b.channels[c].region.convex_hull) for c in posts}
        spread = {c: float(np.linalg.norm(vertices[c] - source_points[c], axis=1).max()) for c in posts}
        assigned = {c: set() for c in posts}
        assignments = {}
        explicit = {}
        for key, task in tasks.items():
            eligible = [
                c
                for c in source_order
                if float(np.linalg.norm(task['vertices'] - source_points[c], axis=1).max()) + spread[c] + 20
                <= self.config.cover_radius_m
            ]
            if eligible:
                c = min(
                    eligible, key=lambda c: (len(set(task['members']) - assigned[c]), source_order.index(c))
                )
                assigned[c].update(task['members'])
                assignments[key] = c
            else:
                explicit[key] = task
        points = dict(source_points)
        channels = {c: sorted(members) for c, members in assigned.items()}
        local = {c: finish_cost(p.mean, p.xy, p.weights) for c, p in posts.items()}
        scans = {c: 6 * len(channels[c]) for c in posts}
        for key, task in explicit.items():
            points[key] = self._initial_center(key, task)
            channels[key] = sorted(task['members'])
            local[key], scans[key] = (0.0, 6 * len(channels[key]))
        table = self._table(points)
        before, route = table.plan(b.position)
        adjustments = []
        for _ in range(self.config.station_sweeps):
            for index, key in enumerate(route):
                if key >= 0:
                    continue
                previous = b.position if index == 0 else points[route[index - 1]]
                following = points[route[index + 1]] if index + 1 < len(route) else None
                points[key], detail = adjust_station(
                    explicit[key]['vertices'], points[key], previous, following, self.config.cover_radius_m
                )
                adjustments.append({'task': key, **detail})
            table = self._table(points)
            _, route = table.plan(b.position)
        length, route = table.plan(b.position)
        self.station_adjustments += len(adjustments)
        self.station_adjustments_accepted += sum((row['accepted'] for row in adjustments))
        self.station_leg_saving_m += sum((row['saved_leg_m'] for row in adjustments))
        self._last_task_log = {
            'sector_assignments': assignments,
            'sector_members': {key: sorted(task['members']) for key, task in tasks.items()},
            'station_adjustments': adjustments,
            'fixed_coordinate_route_m': before,
            'adjusted_coordinate_route_m': length,
        }
        mask = sum((1 << -key - 1 for key in explicit))
        return [
            {
                'points': points,
                'local': local,
                'scans': scans,
                'channels': channels,
                'table': table,
                'route': route,
                'length': length,
                'mask': mask,
                'proxy': length / 5 + sum(local.values()) + sum(scans.values()),
            }
        ]

    def _search_route(self, b):
        plan = self._plans(b, {}, {})[0]
        if not plan['route']:
            raise QuadratureError('Empty task route without a public completion certificate')
        candidates = []
        for key in plan['route'][:3]:
            members = plan['channels'][key]
            c = b.receiver if b.receiver in members else min(members)
            candidates.append(self._score(b, Action('measure', plan['points'][key], c), key, plan, {}))
        selected = min(candidates, key=lambda item: item['score_s'])
        key = int(selected['task'].split(':')[1])
        self._joint_details = {
            **self._last_task_log,
            'joint_mask': plan['mask'],
            'selected_task': selected['task'],
            'joint_route': [key] + selected['continuation'],
            'service_channels': plan['channels'][key],
        }
        from dataclasses import asdict

        return self._record(
            b,
            selected['action'],
            'route_decision',
            selected_score_s=selected['score_s'],
            candidates=[{**r, 'action': asdict(r['action'])} for r in candidates],
            coverage_phase=True,
        )

    def _record(self, b, action, status, **details):
        if action.kind == 'clear':
            self.after_clear = (action.position, action.channel)
        if status == 'route_decision':
            details = {**self._joint_details, **details}
            details['selected_task'] = self._joint_details.get('selected_task', details.get('selected_task'))
            if self._joint_details.get('service_channels'):
                self.batch_position = action.position
                self.batch_channels = self._joint_details['service_channels'].copy()
                self.batch_kind = 'joint_search_batch'
        details = {**self._decision_context, **details}
        self.records.append(
            {
                'step': b.steps,
                'status': status,
                'selected': asdict(action),
                'selected_move_m': distance(b.position, action.position),
                'departure_review': 'completion_override'
                if status.endswith('fallback')
                else 'deterministic_cost_comparison'
                if status == 'route_decision'
                else 'in_place_rule',
                **details,
            }
        )
        return action

    def _score_uncached(self, b, action, c, plan, posts, detail=None):
        points, table = (plan['points'], plan['table'])
        detail = dict(detail or {})
        travel = distance(b.position, action.position) / 5
        if c < 0:
            local_cost = travel + plan['scans'][c]
            endpoint = action.position
        else:
            if action.kind == 'clear':
                local_cost = travel + (
                    5 if detail.get('guaranteed') else clear_finish_cost(posts[c], action.position)
                )
            else:
                after, probabilities = expected_after_measure(
                    posts[c], action.position, self.config.bearing_bin
                )
                local_cost = travel + 5 + int(c != b.receiver) + after
                detail.update(probabilities)
            endpoint = action.position if action.kind == 'clear' else points[c]
            local_cost += plan['scans'][c]
        rest = [k for k in points if k != c]
        length, route = table.plan(endpoint, rest)
        other = sum((plan['local'][k] + plan['scans'][k] for k in rest))
        return {
            'action': action,
            'task': f'{("survey" if c < 0 else "source")}:{c}',
            'local_s': local_cost,
            'tail_route_s': length / 5,
            'other_work_s': other,
            'score_s': local_cost + length / 5 + other,
            'continuation': route,
            'mask': plan['mask'],
            **detail,
        }

    def _select_task(self, b, posts, points):
        reference = self._reference_candidate(b, posts, points)
        ref_c = reference['action'].channel
        plans = self._plans(b, posts, points)
        candidates = []
        menus = {}
        for plan in plans:
            route = plan['route']
            considered = list(dict.fromkeys(route[:4] + [ref_c]))
            for c in considered:
                if c < 0:
                    channels = plan['channels'][c]
                    channel = b.receiver if b.receiver in channels else min(channels)
                    action = Action('measure', plan['points'][c], channel)
                    candidates.append(self._score(b, action, c, plan, posts))
                else:
                    if c not in menus:
                        menus[c] = self._local_candidates(b, c, posts[c])
                    for action, _, detail in menus[c]:
                        candidates.append(self._score(b, action, c, plan, posts, detail=detail))
            candidates.append(
                self._score(
                    b,
                    reference['action'],
                    ref_c,
                    plan,
                    posts,
                    detail={'reference_candidate': True, 'guaranteed': reference.get('guaranteed', False)},
                )
            )
        chosen = min(candidates, key=lambda item: item['score_s'])
        plan = next((p for p in plans if p['mask'] == chosen['mask']))
        c = int(chosen['task'].split(':')[1])
        route = [c] + chosen['continuation']
        points.update(plan['points'])
        future = []
        for q in [chosen['action'].position] + [points[k] for k in route]:
            if distance(b.position, q) >= 20 and all((distance(q, old) >= 20 for old in future)):
                future.append(q)
        self._joint_details = {
            'joint_mask': plan['mask'],
            'selected_task': chosen['task'],
            'joint_route': route,
            'route_solver_exact': plan['table'].exact,
            'reference_candidate': asdict(reference['action']),
            'service_channels': plan['channels'].get(c, []) if c < 0 else [],
        }
        self._joint_details.update(self._last_task_log)
        return (chosen, candidates, route, future[: self.config.future_stops])

    def _include_current_measurement(self, b, c, post):
        choices = self._build_local_candidates(b, c, post)
        if (
            self.guaranteed_clear(b, c) is None
            and self._fresh(b.channels[c], b.position)
            and (not any((detail.get('finite_grid') for _, _, detail in choices)))
        ):
            remaining, probabilities = expected_after_measure(post, b.position, self.config.bearing_bin)
            choices.append(
                (Action('measure', b.position, c), 5 + int(c != b.receiver) + remaining, probabilities)
            )
        return choices

    def _source_points(self, b, posts):
        """Represent each known task by a safe clear endpoint or its posterior mean."""
        points = {}
        for c, post in posts.items():
            safe = self.guaranteed_clear(b, c)
            points[c] = safe.position if safe else tuple(map(float, post.mean))
        return points

    def _reference_candidate(self, b, posts, points):
        if self.active_target not in posts:
            self.active_target = None
        routable = list(posts)
        route_points = {f'source:{c}': points[c] for c in routable}
        route_keys = self.tour.update(b.position, route_points)
        route = [int(k.split(':')[1]) for k in route_keys]
        considered = route[:3]
        if self.active_target is not None and self.active_target not in considered:
            considered = considered + [self.active_target]
        candidates = []
        for c in considered:
            rest = [k for k in route if k != c]
            for action, local, detail in self._local_candidates(b, c, posts[c]):
                endpoint = action.position if action.kind == 'clear' else points[c]
                tail = route_length(endpoint, rest, points) / 5
                other = sum(
                    (finish_cost(posts[k].mean, posts[k].xy, posts[k].weights) for k in posts if k != c)
                )
                candidates.append(
                    {
                        'action': action,
                        'task': f'source:{c}',
                        'local_s': local,
                        'tail_route_s': tail,
                        'other_work_s': other,
                        'score_s': local + tail + other,
                        **detail,
                    }
                )
        selected = min(candidates, key=lambda item: item['score_s'])
        return selected

    def _known_sensing(self, b, posts, future, target):
        choices, reviews = ([], [])
        for c, post in posts.items():
            p = b.channels[c]
            if (
                self.guaranteed_clear(b, c) is not None
                or len(p.directions) >= self.max_bearings
                or (not self._fresh(p, b.position))
            ):
                continue
            before = finish_cost(b.position, post.xy, post.weights)
            after, probabilities = expected_after_measure(post, b.position, self.config.bearing_bin)
            gain = before - after - 5 - int(c != b.receiver)
            future_gain = 0.0
            if c != target:
                for pos in future:
                    if not self._fresh(p, pos):
                        continue
                    later, _ = expected_after_measure(post, pos, self.config.bearing_bin)
                    future_gain = max(future_gain, finish_cost(pos, post.xy, post.weights) - later - 6)
            execute = gain > 2 and gain > future_gain + 2
            reviews.append(
                {'channel': c, 'now_gain_s': gain, 'future_gain_s': future_gain, 'execute_candidate': execute}
            )
            if execute:
                choices.append((gain - future_gain, Action('measure', b.position, c), probabilities))
        self._decision_context['known_sensing_review'] = reviews
        return max(choices, key=lambda item: item[0]) if choices else None

    def _unknown_sensing(self, b, existence, future):
        reviews, choices = ([], [])
        remaining = [s for s in self.stations if self.unknown_at(b, s)]
        recovery = min(120.0, min((distance(b.position, s) / 5 for s in remaining), default=0.0))
        for c in self.unknown_at(b, b.position):
            if existence.get(c, 0.0) <= 0:
                continue
            post = self.model.posterior(b.channels[c])
            review = scan_route_value(post, existence[c], b.position, future, recovery)
            expense = 5 + int(c != b.receiver)
            net = review['gross_saving_s'] - expense
            reviews.append({'channel': c, 'net_saving_s': net, 'measure_cost_s': expense, **review})
            if net > 2:
                choices.append((net, -c, Action('measure', b.position, c)))
        self._decision_context['unknown_sensing_review'] = reviews
        return max(choices, key=lambda item: item[:2]) if choices else None

    def _decide_tasks(self, b):
        if self.batch_channels:
            if distance(b.position, self.batch_position) < 1e-06:
                self.batch_channels = [
                    c
                    for c in self.batch_channels
                    if b.channels[c].status in ('detected', 'unresolved')
                    and point_key(self.batch_position) not in b.channels[c].measured
                ]
                if self.batch_channels:
                    c = b.receiver if b.receiver in self.batch_channels else self.batch_channels[0]
                    return self._record(b, Action('measure', self.batch_position, c), self.batch_kind)
            self.batch_channels = []
        existence = self.model.existence(b)
        posts = {c: self.model.posterior(p) for c, p in b.channels.items() if p.status == 'detected'}
        if distance(b.position, (0.0, 0.0)) < 1e-06:
            scan = self._initial_scan(b, existence)
            if scan:
                return self._record(b, scan, 'station_scan', existence=existence)
        if not posts:
            self.active_target = None
            return self._search_route(b)
        points = self._source_points(b, posts)
        selected, candidates, route, future = self._select_task(b, posts, points)
        target = int(selected['task'].split(':')[1])
        self._decision_context = {
            'forecast_stops': future,
            'route': route,
            'active_target_before': self.active_target,
        }
        shared = self._known_sensing(b, posts, future, target)
        scan = self._unknown_sensing(b, existence, future)
        if shared and (scan is None or shared[0] >= scan[0]):
            return self._record(
                b, shared[1], 'shared_measure', predicted_saving_s=shared[0], feedback_probability=shared[2]
            )
        if scan:
            return self._record(
                b, scan[2], 'incidental_scan', predicted_saving_s=scan[0], existence=existence
            )
        previous = self.active_target
        self.active_target = target
        return self._record(
            b,
            selected['action'],
            'route_decision',
            active_target=target,
            target_switched=previous is not None and previous != target,
            selected_score_s=selected['score_s'],
            route_points=points,
            candidates=[{**r, 'action': asdict(r['action'])} for r in candidates],
            coverage_phase=False,
        )

    def unknown_at(self, b, position):
        revision = tuple(((c, p.revision) for c, p in b.channels.items() if p.status == 'unresolved'))
        if revision != self._unknown_revision:
            self._unknown_revision, self._unknown_cache = (revision, {})
        key = point_key(position)
        if key in self._unknown_cache:
            return self._unknown_cache[key]
        cover = disk(position, 1000)
        result = [
            c
            for c, channel in b.channels.items()
            if channel.status == 'unresolved'
            and key not in channel.measured
            and not channel.region.intersection(cover).is_empty
        ]
        self._unknown_cache[key] = result
        return result

    def _initial_scan(self, b, existence):
        """Complete the initial census at the origin, ranking channels by expected gain."""
        ranked = []
        for c in self.unknown_at(b, b.position):
            if existence.get(c, 0.0) == 0:
                continue
            post = self.model.posterior(b.channels[c])
            receive, cover = post.scan_stats(b.position)
            expense = 5 + int(c != b.receiver)
            merit = (existence[c] * receive + 0.5 * cover) / expense
            ranked.append((merit, -c, Action('measure', b.position, c)))
        return max(ranked, key=lambda item: item[:2])[2] if ranked else None

    @staticmethod
    def _fresh(channel, point):
        return all(
            (distance(point, o.action.position) >= 20 for o in channel.history if o.action.kind == 'measure')
        )

    def _build_local_candidates(self, b, c, post):
        channel = b.channels[c]
        safe = self.guaranteed_clear(b, c)
        if safe is not None:
            return [(safe, distance(b.position, safe.position) / 5 + 5, {'guaranteed': True})]
        first = next((i for i, o in enumerate(channel.history) if o.result == 'direction'))
        attempts = sum((o.action.kind == 'measure' for o in channel.history[first:]))
        failures = sum((o.result == 'no_target_in_range' for o in channel.history))
        if len(channel.directions) >= self.max_bearings or attempts >= 12 or failures >= 6:
            action = self.grid_clear(b, c)
            return [
                (
                    action,
                    distance(b.position, action.position) / 5 + clear_finish_cost(post, action.position),
                    {'finite_grid': True},
                )
            ]
        origin, mean = (np.asarray(b.position), post.mean)
        toward = mean - origin
        length = np.linalg.norm(toward)
        u = toward / max(length, 1e-09)
        if length < 1:
            th = math.radians(channel.directions[-1].bearing)
            u = np.array([math.cos(th), math.sin(th)])
        v = np.array([-u[1], u[0]])
        points = [mean]
        points += [mean - standoff * u for standoff in (100.0, 150.0, 200.0) if length > standoff + 20]
        for sign in (-1, 1):
            points += [origin + sign * s * v for s in (30.0, 75.0, 150.0)]
            points += [origin + fraction * toward + sign * 35 * v for fraction in (0.4, 0.75)]
            points += [mean + sign * s * v for s in (30.0, 75.0)]
        obs = channel.directions[0]
        th = math.radians(obs.bearing)
        fu, fv = (np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)]))
        points += [np.asarray(obs.action.position) + 750 * fu + sign * 500 * fv for sign in (-1, 1)]
        result = []
        for pos in np.unique(np.round(points, 8), axis=0):
            position = tuple(map(float, pos))
            if not self._fresh(channel, position):
                continue
            remaining, probabilities = expected_after_measure(post, position, self.config.bearing_bin)
            cost = distance(b.position, position) / 5 + 5 + int(c != b.receiver) + remaining
            result.append((Action('measure', position, c), cost, probabilities))
        failed = {point_key(o.action.position) for o in channel.history if o.result == 'no_target_in_range'}
        for pos in dict.fromkeys((b.position, tuple(map(float, mean)), post.best_clear_point)):
            probability = post.clear_probability(pos)
            if probability >= 0.2 and point_key(pos) not in failed:
                cost = distance(b.position, pos) / 5 + clear_finish_cost(post, pos)
                result.append((Action('clear', pos, c), cost, {'clear_probability': probability}))
        if not result:
            action = self.grid_clear(b, c)
            result.append(
                (
                    action,
                    distance(b.position, action.position) / 5 + clear_finish_cost(post, action.position),
                    {'finite_grid': True},
                )
            )
        return result

    def _choose_observed(self, b):
        if b.done():
            raise ValueError('No next action after certified completion')
        for c, p in b.channels.items():
            if p.status != 'detected':
                continue
            near = next((o for o in reversed(p.history) if o.result == 'near'), None)
            if near:
                return self._record(b, Action('clear', near.action.position, c), 'near_clear')
            safe = self.guaranteed_clear(b, c)
            if safe and distance(b.position, safe.position) < 1e-06:
                return self._record(b, safe, 'safe_clear')
        if b.steps >= self.config.completion_after or b.deadline - time.monotonic() < self.config.reserve_s:
            self.completion_mode = True
        if self.completion_mode:
            return self._record(b, self._finite_completion(b), 'completion_fallback')
        try:
            return self._choose_bayesian(b)
        except QuadratureError as exc:
            self.completion_mode = True
            return self._record(b, self._finite_completion(b), 'quadrature_fallback', error=str(exc))

    def scan_action(self, b, position):
        channels = self.unknown_at(b, position)
        if not channels:
            return None
        c = b.receiver if b.receiver in channels else min(channels)
        return Action('measure', position, c)

    def guaranteed_clear(self, b, c):
        p = b.channels[c]
        if p.status != 'detected':
            return None
        center, radius = p.summary()
        vertices = shapely.get_coordinates(p.region.convex_hull)
        if np.linalg.norm(vertices - b.position, axis=1).max() <= 19.999:
            return Action('clear', b.position, c)
        if radius > 19.999:
            return None
        d = distance(center, b.position)
        step = min(d, 19.999 - radius)
        pos = (
            center
            if d == 0
            else tuple((center[i] + step * (b.position[i] - center[i]) / d for i in range(2)))
        )
        return Action('clear', pos, c)

    def measurement_candidates(self, b, c):
        p = b.channels[c]
        history = p.directions
        if not history:
            return []
        if len(history) == 1:
            first = history[0]
            theta = math.radians(first.bearing)
            u, v = (
                np.array([math.cos(theta), math.sin(theta)]),
                np.array([-math.sin(theta), math.cos(theta)]),
            )
            points = [np.asarray(first.action.position) + 750 * u + side * 500 * v for side in (-1, 1)]
        else:
            center, radius = p.summary()
            last = history[-1]
            theta = math.radians(last.bearing)
            v = np.array([-math.sin(theta), math.cos(theta)])
            offset = min(200.0, max(35.0, 1.3 * radius))
            points = [np.asarray(center) + sign * offset * v for sign in (-1, 1)]
        actions = [
            Action('measure', tuple(map(float, q)), c) for q in points if point_key(q) not in p.measured
        ]
        return sorted(actions, key=lambda a: distance(b.position, a.position))

    def grid_clear(self, b, c):
        """Finite fallback: include every intersecting 20 m square, even slivers."""
        p = b.channels[c]
        x0, y0, x1, y1 = p.region.bounds
        failed = {point_key(o.action.position) for o in p.history if o.result == 'no_target_in_range'}
        candidates = []
        for ix in range(math.floor(x0 / 20), math.floor(x1 / 20) + 1):
            for iy in range(math.floor(y0 / 20), math.floor(y1 / 20) + 1):
                pos = (ix * 20 + 10.0, iy * 20 + 10.0)
                if point_key(pos) not in failed:
                    candidates.append(pos)
        for pos in sorted(candidates, key=lambda q: distance(b.position, q)):
            if p.region.intersects(box(pos[0] - 10, pos[1] - 10, pos[0] + 10, pos[1] + 10)):
                return Action('clear', pos, c)
        raise GeometryError('No untried fallback cell covers the nonempty region')

    def for_channel(self, b, c):
        p = b.channels[c]
        if p.status != 'detected':
            raise ValueError('Localizer requires a detected source')
        for o in reversed(p.history):
            if o.result == 'near':
                return Action('clear', o.action.position, c)
        clear = self.guaranteed_clear(b, c)
        if clear is not None:
            return clear
        measurements = sum((o.action.kind == 'measure' for o in p.history))
        if len(p.directions) >= self.max_bearings or measurements >= 10:
            return self.grid_clear(b, c)
        candidates = self.measurement_candidates(b, c)
        return candidates[0] if candidates else self.grid_clear(b, c)

    def _finite_completion(self, b):
        if b.done():
            raise ValueError('No next action after completion')
        detected = [c for c, p in b.channels.items() if p.status == 'detected']
        for c in detected:
            clear = self.guaranteed_clear(b, c)
            if clear is not None and distance(clear.position, b.position) < 1e-05:
                return clear
            if any((o.result == 'near' for o in b.channels[c].history)):
                return self.for_channel(b, c)
        for station in self.stations:
            if distance(station, b.position) < 1e-07:
                scan = self.scan_action(b, station)
                if scan is not None:
                    return scan
        if detected:
            candidates = [self.for_channel(b, c) for c in detected]
            return min(
                candidates,
                key=lambda a: (
                    distance(b.position, a.position) / 5
                    + (0 if a.kind == 'clear' else 10 + b.channels[a.channel].summary()[1] / 5)
                ),
            )
        for station in sorted(self.stations, key=lambda s: distance(b.position, s)):
            scan = self.scan_action(b, station)
            if scan is not None:
                return scan
        raise GeometryError('Patrol exhausted but completion not certified')
