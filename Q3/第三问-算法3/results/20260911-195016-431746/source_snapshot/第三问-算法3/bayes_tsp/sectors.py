"""V5: channel-specific coverage tasks and movable guaranteed survey stops.

Inspired by GitHub Q3/B2 (02d2cd3), reimplemented against this project's full
outer geometry, Bayesian posterior, legal atomic actions and completion guard.
"""
from dataclasses import dataclass
import math

import numpy as np
import shapely
from shapely.geometry import Polygon
from scipy.optimize import minimize

from .joint import JointConfig, JointPolicy
from .posterior import QuadratureError, finish_cost
from .shared import Action, core, distance, point_key


@dataclass(frozen=True)
class SectorConfig(JointConfig):
    radius_coupling: bool = True
    movable_stations: bool = True
    sectors: int = 7
    station_sweeps: int = 2
    cover_radius_m: float = 997.


def covers_vertices(vertices, position, radius=997.):
    return bool(np.linalg.norm(vertices-position, axis=1).max() <= radius)


def adjust_station(vertices, current, previous, following, radius=997.):
    """Constrained two-edge improvement, accepted only after full vertex check."""
    current, previous = np.asarray(current, dtype=float), np.asarray(previous, dtype=float)
    if not covers_vertices(vertices, current, radius):
        raise ValueError('Station optimizer needs an initially feasible point')
    def objective(x):
        return float(np.linalg.norm(x-previous)+(np.linalg.norm(x-following) if following is not None else 0.))
    def jac(x):
        d = x-previous
        out = d/max(float(np.linalg.norm(d)), 1e-12)
        if following is not None:
            d = x-following
            out += d/max(float(np.linalg.norm(d)), 1e-12)
        return out
    def constraint(x):
        return radius-np.linalg.norm(vertices-x, axis=1)
    def constraint_jac(x):
        d = vertices-x
        return d/np.maximum(np.linalg.norm(d, axis=1)[:, None], 1e-12)
    result = minimize(objective, current, jac=jac, method='SLSQP',
        constraints=[{'type': 'ineq', 'fun': constraint, 'jac': constraint_jac}],
        options={'maxiter': 30, 'ftol': 1e-5})
    accepted = (result.success and np.isfinite(result.x).all() and
                covers_vertices(vertices, result.x, radius) and objective(result.x) < objective(current)-1e-6)
    point = result.x if accepted else current
    return tuple(map(float, point)), {'accepted': bool(accepted), 'optimizer_success': bool(result.success),
        'saved_leg_m': objective(current)-objective(point), 'iterations': int(result.nit)}


class SectorPolicy(JointPolicy):
    implementation = 'bayes_tsp_v5_sector_services'

    def __init__(self, config=None):
        super().__init__(config or SectorConfig())
        if not 6 <= self.config.sectors <= 12 or not 0 <= self.config.station_sweeps <= 4:
            raise ValueError('sectors must be 6..12 and station_sweeps 0..4')
        # The 64-gon used to exclude no-signal disks has inradius about 998.8m.
        # Using 997m lets an actual no-signal remove the entire assigned task in
        # that conservative geometry, including edges, not just in an ideal disk.
        if not 900 <= self.config.cover_radius_m <= 997:
            raise ValueError('cover radius must leave margin inside the exclusion polygon')
        self.sector_shapes = []
        self.fixed_centers = []
        for j in range(self.config.sectors):
            angle = j*math.tau/self.config.sectors
            # Slight overlap prevents floating-point trigonometry from leaving
            # a crack between adjacent sector boundaries. It never deletes area.
            endpoints = [angle-math.pi/self.config.sectors-1e-9, angle+math.pi/self.config.sectors+1e-9]
            self.sector_shapes.append(Polygon([(0., 0.)]+[(5000*math.cos(a), 5000*math.sin(a)) for a in endpoints]))
            self.fixed_centers.append((1100*math.cos(angle), 1100*math.sin(angle)))
        self._task_key, self._tasks_cache = None, None
        self._last_task_log = {}
        self.station_adjustments = 0
        self.station_adjustments_accepted = 0
        self.station_leg_saving_m = 0.
        self.clear_sector_services = 0

    def _coverage_tasks(self, b):
        unknown = [(c, p) for c, p in b.channels.items() if p.status == 'unresolved']
        if len(b.cleared)+sum(p.status == 'detected' for p in b.channels.values()) == 16:
            return {}
        key = tuple((c, p.revision) for c, p in unknown)
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
                tasks[-j-1] = {'region': region, 'members': members,
                    'vertices': shapely.get_coordinates(region.convex_hull)}
        self._task_key, self._tasks_cache = key, tasks
        return tasks

    def _clear_service(self, b):
        tasks = self._coverage_tasks(b)
        covered = [k for k, task in tasks.items() if covers_vertices(task['vertices'], b.position, self.config.cover_radius_m)]
        channels = sorted({c for k in covered for c in tasks[k]['members']
                           if point_key(b.position) not in b.channels[c].measured})
        return channels, covered

    def _choose_bayesian(self, b):
        if self.after_clear is not None:
            position, channel = self.after_clear
            self.after_clear = None  # V4 must not turn this into blanket scanning.
            if b.channels[channel].status == 'cleared' and distance(position, b.position) < 1e-6:
                if distance(b.position, (0., 0.)) < 1e-6:
                    channels, tasks = self.unknown_at(b, b.position), []
                else:
                    channels, tasks = self._clear_service(b)
                if channels:
                    self.batch_position, self.batch_channels = b.position, list(channels)
                    self.batch_kind = 'task_clear_search'
                    self.clear_sector_services += len(tasks)
                    self._decision_context['covered_sector_services'] = tasks
        return super()._choose_bayesian(b)

    def _initial_center(self, key, task):
        p = self.fixed_centers[-key-1]
        if covers_vertices(task['vertices'], p, self.config.cover_radius_m):
            return p
        center, radius = core.region_summary(task['region'])
        if radius <= self.config.cover_radius_m and covers_vertices(task['vertices'], center, self.config.cover_radius_m):
            return center
        raise QuadratureError('Sector too wide for one guaranteed stop; retain finite seven-station fallback')

    def _plans(self, b, posts, source_points):
        tasks = self._coverage_tasks(b)
        _, source_order = self._table(source_points).plan(b.position)
        vertices = {c: shapely.get_coordinates(b.channels[c].region.convex_hull) for c in posts}
        spread = {c: float(np.linalg.norm(vertices[c]-source_points[c], axis=1).max()) for c in posts}
        assigned = {c: set() for c in posts}
        assignments = {}
        explicit = {}
        for key, task in tasks.items():
            # One future clear stop must cover the WHOLE task, for every feasible
            # source position and every successful <=20m clear position.
            eligible = [c for c in source_order if
                float(np.linalg.norm(task['vertices']-source_points[c], axis=1).max())+spread[c]+20 <= self.config.cover_radius_m]
            if eligible:
                c = min(eligible, key=lambda c: (len(set(task['members'])-assigned[c]), source_order.index(c)))
                assigned[c].update(task['members'])
                assignments[key] = c
            else:
                explicit[key] = task
        points = dict(source_points)
        channels = {c: sorted(members) for c, members in assigned.items()}
        local = {c: finish_cost(p.mean, p.xy, p.weights) for c, p in posts.items()}
        scans = {c: 6*len(channels[c]) for c in posts}
        for key, task in explicit.items():
            points[key] = self._initial_center(key, task)
            channels[key] = sorted(task['members'])
            local[key], scans[key] = 0., 6*len(channels[key])
        table = self._table(points)
        before, route = table.plan(b.position)
        adjustments = []
        if self.config.movable_stations:
            for _ in range(self.config.station_sweeps):
                for index, key in enumerate(route):
                    if key >= 0:
                        continue
                    previous = b.position if index == 0 else points[route[index-1]]
                    following = points[route[index+1]] if index+1 < len(route) else None
                    points[key], detail = adjust_station(explicit[key]['vertices'], points[key], previous, following,
                                                         self.config.cover_radius_m)
                    adjustments.append({'task': key, **detail})
                table = self._table(points)
                _, route = table.plan(b.position)
        length, route = table.plan(b.position)
        self.station_adjustments += len(adjustments)
        self.station_adjustments_accepted += sum(row['accepted'] for row in adjustments)
        self.station_leg_saving_m += sum(row['saved_leg_m'] for row in adjustments)
        self._last_task_log = {'sector_assignments': assignments,
            'sector_members': {key: sorted(task['members']) for key, task in tasks.items()},
            'station_adjustments': adjustments, 'fixed_coordinate_route_m': before,
            'adjusted_coordinate_route_m': length}
        # mask remains a pure task identifier here. Actual completion only uses
        # channel geometry; no visited-sector flag can certify absence.
        mask = sum(1 << (-key-1) for key in explicit)
        return [{'points': points, 'local': local, 'scans': scans, 'channels': channels,
                 'table': table, 'route': route, 'length': length, 'mask': mask,
                 'proxy': length/5+sum(local.values())+sum(scans.values())}]

    def _select_task(self, b, posts, stable, pending, points):
        selected, candidates, route, future = super()._select_task(b, posts, stable, pending, points)
        self._joint_details.update(self._last_task_log)
        return selected, candidates, route, future

    def _search_route(self, b, existence):
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
        self._joint_details = {**self._last_task_log, 'joint_mask': plan['mask'],
            'selected_task': selected['task'], 'joint_route': [key]+selected['continuation'],
            'service_channels': plan['channels'][key]}
        from dataclasses import asdict
        return self._record(b, selected['action'], 'route_decision', selected_score_s=selected['score_s'],
            candidates=[{**r, 'action': asdict(r['action'])} for r in candidates], coverage_phase=True)
