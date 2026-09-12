"""V4 joint localization/clearance/coverage routes, without Monte Carlo."""
from dataclasses import asdict, dataclass

import numpy as np
import shapely
from shapely.geometry import GeometryCollection

from .efficient import EfficientConfig, EfficientPolicy
from .open_routes import OpenRoutes
from .posterior import BayesModel, QuadratureError, clear_finish_cost, expected_after_measure, finish_cost
from .scheduling import insertion_costs
from .shared import Action, disk, distance


@dataclass(frozen=True)
class JointConfig(EfficientConfig):
    stable_routing: bool = False
    exact_routes: bool = True
    joint_coverage: bool = True
    service_scans: bool = True
    route_probes: bool = False
    review_departures: bool = False
    exact_limit: int = 12


class JointPolicy(EfficientPolicy):
    implementation = 'bayes_tsp_v4_joint_routes'

    def __init__(self, config=None):
        super().__init__(config or JointConfig())
        if not 1 <= self.config.exact_limit <= 16:
            raise ValueError('exact_limit must be 1..16')
        self.after_clear = None
        self._joint_details = {}
        self._tables = {}
        self.fine_model = BayesModel(min(96, 2*self.config.resolution), self.config.p0)

    def choose(self, b):
        self._joint_details = {}
        return super().choose(b)

    def _record(self, b, action, status, **details):
        if self.config.service_scans and action.kind == 'clear':
            self.after_clear = (action.position, action.channel)
        if status == 'route_decision':
            details = {**self._joint_details, **details}
            details['selected_task'] = self._joint_details.get('selected_task', details.get('selected_task'))
            if self._joint_details.get('service_channels'):
                self.batch_position = action.position
                self.batch_channels = self._joint_details['service_channels'].copy()
                self.batch_kind = 'joint_search_batch'
        return super()._record(b, action, status, **details)

    def _choose_bayesian(self, b):
        if self.after_clear is not None:
            position, c = self.after_clear
            self.after_clear = None
            if b.channels[c].status == 'cleared' and distance(position, b.position) < 1e-6:
                # Fulfil the route's declared discovery service at a successful
                # clear stop. Only actual feedback updates coverage evidence.
                self.batch_position = b.position
                self.batch_channels = self.unknown_at(b, b.position).copy()
                if len(b.cleared)+sum(p.status == 'detected' for p in b.channels.values()) == 16:
                    self.batch_channels = []
                self.batch_kind = 'clear_stop_search'
        return super()._choose_bayesian(b)

    def _table(self, points):
        key = tuple((k, tuple(map(float, points[k]))) for k in sorted(points))
        if key not in self._tables:
            if len(self._tables) >= 8:
                self._tables.pop(next(iter(self._tables)))
            self._tables[key] = OpenRoutes(points, self.config.exact_limit)
        return self._tables[key]

    def _coverage_forecast(self, b, posts, points):
        projected = []
        if self.config.service_scans:
            for c in posts:
                vertices = shapely.get_coordinates(b.channels[c].region.convex_hull)
                radius = float(np.linalg.norm(vertices-points[c], axis=1).max())
                # A successful clear point is at most radius+20 from this
                # representative. Its 1000 m reception disk therefore covers
                # this smaller disk, conditional on actually doing that scan.
                inner = 1000-radius-20
                if inner > 0:
                    projected.append(disk(points[c], inner))
        cover = shapely.union_all(projected) if projected else GeometryCollection()
        unresolved = {c: p.region.difference(cover) for c, p in b.channels.items() if p.status == 'unresolved'}
        if len(posts)+len(b.cleared) == 16:
            unresolved = {}
        remaining = shapely.union_all(list(unresolved.values())) if unresolved else GeometryCollection()
        return remaining, unresolved

    def _plans(self, b, posts, source_points):
        remaining, residual = self._coverage_forecast(b, posts, source_points)
        masks = [0]
        if self.config.joint_coverage and not remaining.is_empty:
            possible = self.cover_tour.masks(remaining)
            masks = [mask for mask in range(len(possible)) if possible[mask] and
                     all(not possible[mask ^ (1 << j)] for j in range(len(self.stations)) if mask & (1 << j))]
            # Cheap insertion ranking before full route optimization. All kept
            # subsets cover the complete conservative projected region.
            _, base_route = self._table(source_points).plan(b.position)
            future = [source_points[c] for c in base_route]
            def estimate(mask):
                extra = 0.
                for j, pos in enumerate(self.stations):
                    if mask & (1 << j):
                        extra += float(insertion_costs(b.position, future, np.asarray([pos])).min())/5
                        extra += 6*sum(not r.intersection(self.station_disks[j]).is_empty for r in residual.values())
                return extra, mask
            masks = sorted(masks, key=estimate)[:3]
        plans = []
        for mask in masks:
            points = dict(source_points)
            local = {c: finish_cost(posts[c].mean, posts[c].xy, posts[c].weights) for c in posts}
            scans = {c: (6*len(self.unknown_at(b, points[c])) if self.config.service_scans and residual else 0.) for c in posts}
            channels = {}
            for j, station in enumerate(self.stations):
                if mask & (1 << j):
                    key = -j-1
                    required = [c for c, region in residual.items() if not region.intersection(self.station_disks[j]).is_empty]
                    if not required:
                        continue
                    points[key], channels[key] = station, required
                    local[key], scans[key] = 0., 6*len(required)
            table = self._table(points)
            length, route = table.plan(b.position)
            plans.append({'points': points, 'local': local, 'scans': scans, 'channels': channels,
                          'table': table, 'route': route, 'length': length, 'mask': mask,
                          'proxy': length/5+sum(local.values())+sum(scans.values())})
        return plans

    def _score(self, b, action, c, plan, posts, kind='finish', detail=None):
        points, table = plan['points'], plan['table']
        detail = dict(detail or {})
        travel = distance(b.position, action.position)/5
        if c < 0:
            local_cost = travel+plan['scans'][c]
            endpoint = action.position
        else:
            if action.kind == 'clear':
                local_cost = travel+(5 if detail.get('guaranteed') else clear_finish_cost(posts[c], action.position))
            else:
                after, probabilities = expected_after_measure(posts[c], action.position, self.config.bearing_bin)
                local_cost = travel+5+int(c != b.receiver)+after
                detail.update(probabilities)
            endpoint = action.position if action.kind == 'clear' else points[c]
            local_cost += plan['scans'][c]
        rest = [k for k in points if k != c]
        length, route = table.plan(endpoint, rest)
        other = sum(plan['local'][k]+plan['scans'][k] for k in rest)
        if kind == 'probe':
            # Arrive and measure, then continue the full mixed route. Do not
            # assume immediate return to clear this same source after the probe.
            after, probabilities = expected_after_measure(posts[c], action.position, self.config.bearing_bin)
            residual = max(5., after-distance(action.position, posts[c].mean)/5)
            length, route = table.plan(action.position)
            other = sum(plan['local'].values())-plan['local'][c]+sum(plan['scans'].values())
            local_cost = travel+5+int(c != b.receiver)+residual
            detail.update(probabilities)
        return {'action': action, 'task': f'{"survey" if c < 0 else "source"}:{c}',
                'kind': kind, 'local_s': local_cost, 'tail_route_s': length/5,
                'other_work_s': other, 'score_s': local_cost+length/5+other,
                'continuation': route, 'mask': plan['mask'], **detail}

    def _select_task(self, b, posts, stable, pending, points):
        if not self.config.exact_routes and not self.config.joint_coverage and not self.config.route_probes:
            return super()._select_task(b, posts, stable, pending, points)
        # V3's next physical action stays in the comparison, repriced with the
        # same mixed workload rather than its old, incompatible cost scale.
        reference, _, _, _ = super()._select_task(b, posts, stable, pending, points)
        ref_c = reference['action'].channel
        plans = self._plans(b, posts, points)
        candidates = []
        menus = {}
        for plan in plans:
            route = plan['route']
            considered = list(dict.fromkeys(route[:4]+[ref_c]))
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
            candidates.append(self._score(b, reference['action'], ref_c, plan, posts,
                                          detail={'v3_reference': True, 'guaranteed': reference.get('guaranteed', False)}))
            if self.config.route_probes:
                for key in route[:3]:
                    pos = plan['points'][key]
                    for c in sorted(posts, key=lambda c: distance(pos, posts[c].mean))[:3]:
                        p = b.channels[c]
                        if (self.guaranteed_clear(b, c) is not None or len(p.directions) >= self.max_bearings
                                or not self._fresh(p, pos)):
                            continue
                        candidates.append(self._score(b, Action('measure', pos, c), c, plan, posts, kind='probe'))
        chosen = min(candidates, key=lambda item: item['score_s'])
        plan = next(p for p in plans if p['mask'] == chosen['mask'])
        reference_row = next(r for r in candidates if r.get('v3_reference') and r['mask'] == chosen['mask'])
        review = {'status': 'not_required'}
        extra_move = distance(b.position, chosen['action'].position)-distance(b.position, reference['action'].position)
        saving = reference_row['score_s']-chosen['score_s']
        if self.config.review_departures and extra_move > 50 and saving < 30:
            # A second deterministic quadrature resolution, not independent MC
            # samples or a statistical confidence guarantee.
            try:
                fine = {c: self.fine_model.posterior(b.channels[c]) for c in posts}
                fine_plan = {**plan, 'local': {c: finish_cost(p.mean, p.xy, p.weights) for c, p in fine.items()}}
                fine_plan['local'].update({c: 0. for c in plan['points'] if c < 0})
                c = int(chosen['task'].split(':')[1])
                fine_new = self._score(b, chosen['action'], c, fine_plan, fine, kind=chosen['kind'],
                                       detail={'guaranteed': chosen.get('guaranteed', False)})
                fine_ref = self._score(b, reference['action'], ref_c, fine_plan, fine, detail={'guaranteed': reference.get('guaranteed', False)})
                accepted = fine_ref['score_s']-fine_new['score_s'] > 3
                review = {'status': 'fine_quadrature', 'accepted': accepted,
                          'fine_gain_s': fine_ref['score_s']-fine_new['score_s']}
            except QuadratureError as exc:
                accepted = False
                review = {'status': 'quadrature_fallback', 'accepted': False, 'error': str(exc)}
            if not accepted:
                chosen = reference_row
        c = int(chosen['task'].split(':')[1])
        route = ([c]+chosen['continuation']) if chosen['kind'] != 'probe' else chosen['continuation']
        points.update(plan['points'])
        future = []
        for q in [chosen['action'].position]+[points[k] for k in route]:
            if distance(b.position, q) >= 20 and all(distance(q, old) >= 20 for old in future):
                future.append(q)
        self._joint_details = {'joint_mask': plan['mask'], 'selected_task': chosen['task'],
            'joint_route': route, 'route_solver_exact': plan['table'].exact,
            'v3_reference': asdict(reference['action']), 'departure_check': review,
            'service_channels': plan['channels'].get(c, []) if c < 0 else []}
        return chosen, candidates, route, future[:self.config.future_stops]
