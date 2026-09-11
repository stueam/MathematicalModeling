"""Observation-only Bayesian local decisions with an evolving open TSP tour."""
from dataclasses import asdict, dataclass
import math
import time

import numpy as np
import shapely
from shapely.geometry import Polygon

from .posterior import (BayesModel, QuadratureError, clear_finish_cost,
                        expected_after_measure, finish_cost)
from .routing import DynamicTour, route_length
from .shared import Action, Baseline, core, disk, distance, point_key


@dataclass(frozen=True)
class Config:
    ring: float = 1200.
    resolution: int = 24
    bearing_bin: float = 1.
    p0: float = .65
    shared: bool = True
    dynamic_tsp: bool = True
    speculative_clear: bool = True
    max_bearings: int = 8
    completion_after: int = 600
    reserve_s: float = 30.


class BayesTSPPolicy(Baseline):
    def __init__(self, config=None):
        self.config = config or Config()
        super().__init__(self.config.ring, self.config.max_bearings)
        if self.config.bearing_bin <= 0 or abs(round(360/self.config.bearing_bin)*self.config.bearing_bin-360) > 1e-9:
            raise ValueError('bearing_bin must divide 360')
        if self.config.max_bearings < 2 or self.config.completion_after < 1:
            raise ValueError('Invalid completion limits')
        self.model = BayesModel(self.config.resolution, self.config.p0)
        self.tour = DynamicTour()
        self.records = []
        self.completion_mode = False
        self.station_disks = [disk(s, 1000) for s in self.stations]
        self._unknown_revision = None
        self._unknown_cache = {}
        self.batch_position, self.batch_channels = None, []
        self.batch_kind = 'joint_batch_measure'

    def unknown_at(self, b, position):
        revision = tuple((c, p.revision) for c, p in b.channels.items() if p.status == 'unresolved')
        if revision != self._unknown_revision:
            self._unknown_revision, self._unknown_cache = revision, {}
        key = point_key(position)
        if key in self._unknown_cache:
            return self._unknown_cache[key]
        cover = disk(position, 1000)
        result = [c for c in super().unknown_at(b, position)
                  if not b.channels[c].region.intersection(cover).is_empty]
        self._unknown_cache[key] = result
        return result

    def _complete(self, b):
        # Use the audited finite grid fallback. This object stays in completion
        # mode permanently once the global action/deadline guard is triggered.
        return super().choose(b)

    def _record(self, b, action, status, **details):
        self.records.append({'step': b.steps, 'status': status,
            'selected': asdict(action), 'selected_move_m': distance(b.position, action.position),
            'departure_review': 'completion_override' if status.endswith('fallback') else
                                'deterministic_cost_comparison' if status == 'route_decision' else 'in_place_rule',
            **details})
        return action

    def _scan_choice(self, b, position, existence, forced=False):
        ranked = []
        remaining = [s for s in self.stations if self.unknown_at(b, s)]
        # Approximate value of avoiding a return to a required survey stop.
        detour_s = min(120., min((distance(position, s)/5 for s in remaining), default=0.))
        for c in self.unknown_at(b, position):
            if existence.get(c, 0.) == 0:
                continue
            post = self.model.posterior(b.channels[c])
            receive, cover = post.scan_stats(position)
            expense = 5+int(c != b.receiver)
            benefit = detour_s*(existence[c]*receive+.5*cover)
            merit = (existence[c]*receive+.5*cover)/expense
            if forced or benefit > expense+2:
                ranked.append((merit, -c, Action('measure', position, c), benefit-expense))
        return max(ranked, key=lambda item: item[:2]) if ranked else None

    def _shared_choice(self, b, posts):
        choices = []
        for c, post in posts.items():
            channel = b.channels[c]
            if self.guaranteed_clear(b, c) is not None or len(channel.directions) >= self.max_bearings:
                continue
            if not self._fresh(channel, b.position):
                continue
            before = finish_cost(b.position, post.xy, post.weights)
            after, probabilities = expected_after_measure(post, b.position, self.config.bearing_bin)
            saving = before-after-5-int(c != b.receiver)
            if saving > 2:
                choices.append((saving, Action('measure', b.position, c), probabilities))
        return max(choices, key=lambda item: item[0]) if choices else None

    @staticmethod
    def _fresh(channel, point):
        return all(distance(point, o.action.position) >= 20
                   for o in channel.history if o.action.kind == 'measure')

    def _local_candidates(self, b, c, post):
        channel = b.channels[c]
        safe = self.guaranteed_clear(b, c)
        if safe is not None:
            return [(safe, distance(b.position, safe.position)/5+5, {'guaranteed': True})]
        first = next(i for i, o in enumerate(channel.history) if o.result == 'direction')
        attempts = sum(o.action.kind == 'measure' for o in channel.history[first:])
        failures = sum(o.result == 'no_target_in_range' for o in channel.history)
        if len(channel.directions) >= self.max_bearings or attempts >= 12 or failures >= 6:
            action = self.grid_clear(b, c)
            # Rank the fallback task conservatively, but execute one actual cell
            # at a time. Its predicted cost is not a finite completion bound.
            return [(action, distance(b.position, action.position)/5+
                     clear_finish_cost(post, action.position), {'finite_grid': True})]
        origin, mean = np.asarray(b.position), post.mean
        toward = mean-origin
        length = np.linalg.norm(toward)
        u = toward/max(length, 1e-9)
        if length < 1:
            th = math.radians(channel.directions[-1].bearing)
            u = np.array([math.cos(th), math.sin(th)])
        v = np.array([-u[1], u[0]])
        points = [mean]
        # Guide section 6: measurements on the approach, 100--200 m before the
        # estimated target, with zero geometric detour on that estimated leg.
        points += [mean-standoff*u for standoff in (100., 150., 200.) if length > standoff+20]
        for sign in (-1, 1):
            points += [origin+sign*s*v for s in (30., 75., 150.)]
            points += [origin+fraction*toward+sign*35*v for fraction in (.4, .75)]
            points += [mean+sign*s*v for s in (30., 75.)]
        # Guaranteed-reception backups remain in the menu, with full travel cost.
        obs = channel.directions[0]
        th = math.radians(obs.bearing)
        fu, fv = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
        points += [np.asarray(obs.action.position)+750*fu+sign*500*fv for sign in (-1, 1)]
        result = []
        for pos in np.unique(np.round(points, 8), axis=0):
            position = tuple(map(float, pos))
            if not self._fresh(channel, position):
                continue
            remaining, probabilities = expected_after_measure(post, position, self.config.bearing_bin)
            cost = distance(b.position, position)/5+5+int(c != b.receiver)+remaining
            result.append((Action('measure', position, c), cost, probabilities))
        if self.config.speculative_clear:
            failed = {point_key(o.action.position) for o in channel.history if o.result == 'no_target_in_range'}
            for pos in dict.fromkeys((b.position, tuple(map(float, mean)), post.best_clear_point)):
                probability = post.clear_probability(pos)
                if probability >= .2 and point_key(pos) not in failed:
                    cost = distance(b.position, pos)/5+clear_finish_cost(post, pos)
                    result.append((Action('clear', pos, c), cost, {'clear_probability': probability}))
        if not result:
            action = self.grid_clear(b, c)
            result.append((action, distance(b.position, action.position)/5+
                           clear_finish_cost(post, action.position), {'finite_grid': True}))
        return result

    def choose(self, b):
        if b.done():
            raise ValueError('No next action after certified completion')
        for c, p in b.channels.items():
            if p.status != 'detected':
                continue
            near = next((o for o in reversed(p.history) if o.result == 'near'), None)
            if near:
                return self._record(b, Action('clear', near.action.position, c), 'near_clear')
            safe = self.guaranteed_clear(b, c)
            if safe and distance(b.position, safe.position) < 1e-6:
                return self._record(b, safe, 'safe_clear')
        if b.steps >= self.config.completion_after or b.deadline-time.monotonic() < self.config.reserve_s:
            self.completion_mode = True
        if self.completion_mode:
            return self._record(b, self._complete(b), 'completion_fallback')
        try:
            return self._choose_bayesian(b)
        except QuadratureError as exc:
            # Never mark ABSENT or discard hard support because integration failed.
            self.completion_mode = True
            return self._record(b, self._complete(b), 'quadrature_fallback', error=str(exc))

    def _choose_bayesian(self, b):
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
        # Complete the necessary census at an actual survey stop. Known=16 can
        # omit discovery, but the 16 sources still all need successful clears.
        at_station = any(distance(b.position, s) < 1e-6 for s in self.stations)
        if at_station:
            scan = self._scan_choice(b, b.position, existence, forced=True)
            if scan:
                return self._record(b, scan[2], 'station_scan', existence=existence)
        if self.config.shared:
            shared = self._shared_choice(b, posts)
            if shared:
                return self._record(b, shared[1], 'shared_measure', predicted_saving_s=shared[0],
                                    feedback_probability=shared[2])
            # Guide section 8: incidental UNKNOWN scans are attached to actual
            # successful clear stops, not to every temporary localization stop.
            at_clear = any(o.result == 'success' and distance(o.action.position, b.position) < 1e-6
                           for p in b.channels.values() for o in p.history)
            scan = self._scan_choice(b, b.position, existence) if at_clear else None
            if scan:
                return self._record(b, scan[2], 'incidental_scan', predicted_saving_s=scan[3], existence=existence)

        points = {f'source:{c}': tuple(map(float, post.mean)) for c, post in posts.items()}
        overhead = {f'source:{c}': finish_cost(post.mean, post.xy, post.weights) for c, post in posts.items()}
        scans = {}
        # Per the algorithm-3 guide, discovered targets supply search stops.
        # Dedicated blind-region tasks enter the tour when known work is done.
        if not posts and any(existence.values()):
            for index, station in enumerate(self.stations):
                scan = self._scan_choice(b, station, existence, forced=True)
                if scan is not None:
                    key = f'survey:{index}'
                    points[key], scans[key] = station, scan[2]
                    overhead[key] = 6*len(self.unknown_at(b, station))
            for index, pos in enumerate(self._blind_points(b)):
                scan = self._scan_choice(b, pos, existence, forced=True)
                if scan:
                    key = f'blind:{index}'
                    # Blind alternatives are compared separately below; they
                    # are not all mandatory visits in the open TSP.
                    scans[key] = scan[2]
        if not points:
            raise QuadratureError('No probabilistic route tasks but geometric work remains')
        if self.config.dynamic_tsp:
            route = self.tour.update(b.position, points)
        else:
            route, remaining = [], dict(points)
            pos = b.position
            while remaining:
                key = min(remaining, key=lambda k: (distance(pos, remaining[k]), k))
                route.append(key)
                pos = remaining.pop(key)
        # Compare departures that visit nearby entries of the current open tour.
        # Remaining route + remaining known local/scan work are included; unknown
        # source detours are not predicted, so this is explicitly a surrogate.
        shortlist = route[:3] if posts else route
        survey = next((k for k in route if k.startswith('survey:')), None)
        if survey and survey not in shortlist:
            shortlist.append(survey)
        candidates = []
        for key in shortlist:
            rest = [k for k in route if k != key]
            tail = route_length(points[key], rest, points)/5
            others = sum(overhead[k] for k in rest)
            if key.startswith('source:'):
                c = int(key.split(':')[1])
                for action, local_cost, detail in self._local_candidates(b, c, posts[c]):
                    candidates.append({'action': action, 'task': key, 'local_s': local_cost,
                        'tail_route_s': tail, 'other_work_s': others,
                        'score_s': local_cost+tail+others, **detail})
            else:
                local = distance(b.position, points[key])/5+overhead[key]
                candidates.append({'action': scans[key], 'task': key, 'local_s': local,
                    'tail_route_s': tail, 'other_work_s': others, 'score_s': local+tail+others})
        if self.config.shared and len(posts) >= 2 and distance(b.position, (0., 0.)) < 1e-6:
            # The guide's dedicated common second station is an INITIAL task.
            # Later discoveries use future route stops, avoiding repeated
            # 300--600 m dedicated detours each time two channels are discovered.
            candidates.extend(self._joint_candidates(b, posts, route, points, overhead))
        if not posts:
            # Choose a useful uncovered sector by covered posterior mass per
            # physical second. Fixed stations remain the finite fallback.
            for key, action in scans.items():
                if not key.startswith('blind:'):
                    continue
                total, count = 0., 0
                for c in self.unknown_at(b, action.position):
                    post = self.model.posterior(b.channels[c])
                    _, cover = post.scan_stats(action.position)
                    total += cover
                    count += 1
                if total > 0:
                    # Only used against corresponding fixed-site ratios below.
                    candidates.append({'action': action, 'task': key, 'score_s': 0.,
                        'blind_mass': total, 'local_s': distance(b.position, action.position)/5+6*count})
            for item in candidates:
                action = item['action']
                total = sum(self.model.posterior(b.channels[c]).scan_stats(action.position)[1]
                            for c in self.unknown_at(b, action.position))
                item['score_s'] = item['local_s']/max(total, 1e-12)
                item['score_kind'] = 'seconds_per_channel_coverage_mass'
        selected = min(candidates, key=lambda item: item['score_s'])
        if 'batch_channels' in selected:
            self.batch_position = selected['action'].position
            self.batch_channels = selected['batch_channels'].copy()
            self.batch_kind = 'joint_batch_measure'
        elif not posts:
            self.batch_position = selected['action'].position
            self.batch_channels = self.unknown_at(b, self.batch_position).copy()
            self.batch_kind = 'blind_batch_measure'
        records = [{**item, 'action': asdict(item['action'])} for item in candidates]
        return self._record(b, selected['action'], 'route_decision', route=route,
            route_points=points, route_length_m=route_length(b.position, route, points),
            candidates=records, selected_score_s=selected['score_s'], existence=existence,
            posterior_cells={c: p.cells for c, p in posts.items()},
            estimates={c: {'mean': p.mean.tolist(), 'best_clear_point': p.best_clear_point,
                'clear_probability': p.clear_probability(p.best_clear_point),
                'expected_error_m': float(p.weights @ np.linalg.norm(p.xy-p.mean, axis=1))}
                       for c, p in posts.items()})

    def _blind_points(self, b):
        regions = [p.region for p in b.channels.values() if p.status == 'unresolved']
        if not regions:
            return []
        remaining = shapely.union_all(regions)
        points = []
        for k in range(6):
            angles = (k*math.pi/3-math.pi/6, k*math.pi/3+math.pi/6)
            sector = Polygon([(0, 0)]+[(5000*math.cos(a), 5000*math.sin(a)) for a in angles])
            part = remaining.intersection(sector)
            if part.is_empty:
                continue
            center, radius = core.region_summary(part)
            if radius <= 995:
                d = distance(center, b.position)
                shift = min(d, 995-radius)
                point = center if d == 0 else tuple(center[i]+shift*(b.position[i]-center[i])/d for i in (0, 1))
                points.append(point)
        return points

    def _joint_candidates(self, b, posts, route, points, overhead):
        singles = [c for c in posts if len(b.channels[c].directions) == 1
                   and sum(o.action.kind == 'measure' for o in b.channels[c].history) < 10]
        if len(singles) < 2:
            return []
        # The guide's finite 300/400/500/600 m x 15 degree search, now centered
        # on the current departure. All costs have units of seconds, not E+lambda*T.
        result = []
        for radius in (300., 400., 500., 600.):
            for angle in range(0, 360, 15):
                th = math.radians(angle)
                pos = (b.position[0]+radius*math.cos(th), b.position[1]+radius*math.sin(th))
                batch = [c for c in singles if self._fresh(b.channels[c], pos)]
                if len(batch) < 2:
                    continue
                travel = radius/5
                local = sum(overhead.values())
                for c in batch:
                    expected, _ = expected_after_measure(posts[c], pos, self.config.bearing_bin)
                    # Full open route already pays approach to each prior mean;
                    # use only residual local cost here to avoid double travel.
                    residual = max(5., expected-distance(pos, posts[c].mean)/5)
                    local += 6+residual-overhead[f'source:{c}']
                batch.sort(key=lambda c: (c != b.receiver, c))
                if batch[0] == b.receiver:
                    local -= 1
                length = route_length(pos, route, points)/5
                result.append({'action': Action('measure', pos, batch[0]), 'task': 'shared_second',
                    'local_s': travel+local, 'tail_route_s': length,
                    'score_s': travel+local+length, 'batch_channels': batch})
        return result
