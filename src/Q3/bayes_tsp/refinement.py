"""Local experimental improvements motivated by the ten official PRACTICE traces.

V5 remains intact. All feedback during planning is Bayesian deterministic
quadrature; this policy never reuses future official feedback at altered points.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .repair import repair_segment
from .sectors import SectorConfig, SectorPolicy, covers_vertices


@dataclass(frozen=True)
class RefinementConfig(SectorConfig):
    repair_stations: bool = True
    recovery_measure: bool = True
    recovery_sine: float = .15


def repaired_station(vertices, current, previous, following, radius):
    current, previous = np.asarray(current), np.asarray(previous)
    def objective(x):
        return float(np.linalg.norm(x-previous)+(np.linalg.norm(x-following) if following is not None else 0.))
    def jac(x):
        d = x-previous
        g = d/max(float(np.linalg.norm(d)), 1e-12)
        if following is not None:
            d = x-following
            g += d/max(float(np.linalg.norm(d)), 1e-12)
        return g
    def constraint(x):
        return radius-np.linalg.norm(vertices-x, axis=1)
    def constraint_jac(x):
        d = vertices-x
        return d/np.maximum(np.linalg.norm(d, axis=1)[:, None], 1e-12)
    result = minimize(objective, current, jac=jac, method='SLSQP',
        constraints=[{'type': 'ineq', 'fun': constraint, 'jac': constraint_jac}],
        options={'maxiter': 30, 'ftol': 1e-5})
    point = repair_segment(current, result.x, constraint) if result.success else None
    accepted = (point is not None and covers_vertices(vertices, point, radius) and
                objective(point) < objective(current)-1e-6)
    return (tuple(map(float, point if accepted else current)),
            {'accepted': bool(accepted), 'optimizer_success': bool(result.success),
             'raw_margin_m': float(np.min(constraint(result.x))),
             'saved_leg_m': objective(current)-objective(point) if accepted else 0.})


class RefinementPolicy(SectorPolicy):
    implementation = 'bayes_tsp_v6_practice_diagnostics'

    def __init__(self, config=None):
        super().__init__(config or RefinementConfig())
        self.repair_accepts = 0
        self.recovery_filtered = 0

    def _plans(self, b, posts, source_points):
        plans = super()._plans(b, posts, source_points)
        if not self.config.repair_stations:
            return plans
        tasks = self._coverage_tasks(b)
        for plan in plans:
            points, route = dict(plan['points']), plan['route']
            records = []
            # An additional two-edge optimization pass after the unchanged V5
            # pass. Fixed-node route cost and hard coverage are checked again.
            for _ in range(self.config.station_sweeps):
                for j, key in enumerate(route):
                    if key >= 0:
                        continue
                    previous = b.position if j == 0 else points[route[j-1]]
                    following = points[route[j+1]] if j+1 < len(route) else None
                    points[key], record = repaired_station(tasks[key]['vertices'], points[key], previous,
                                                          following, self.config.cover_radius_m)
                    records.append({'task': key, **record})
                table = self._table(points)
                _, route = table.plan(b.position)
            table = self._table(points)
            length, route = table.plan(b.position)
            if length <= plan['length']+1e-6:
                plan.update(points=points, route=route, table=table, length=length,
                            proxy=length/5+sum(plan['local'].values())+sum(plan['scans'].values()))
            self.repair_accepts += sum(r['accepted'] for r in records)
            self._last_task_log.update(repaired_station_pass=records, repaired_route_m=plan['length'])
        return plans

    def _local_candidates(self, b, c, post):
        choices = super()._local_candidates(b, c, post)
        if not self.config.recovery_measure or any(d.get('guaranteed') or d.get('finite_grid') for _, _, d in choices):
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
            return choices  # The shared finite completion guard remains available.
        useful = []
        for item in probes:
            v = post.mean-item[0].position
            crosses = []
            for obs in b.channels[c].directions:
                u = post.mean-obs.action.position
                crosses.append(abs(u[0]*v[1]-u[1]*v[0])/max(float(np.linalg.norm(u)*np.linalg.norm(v)), 1e-9))
            if max(crosses, default=0.) >= self.config.recovery_sine:
                useful.append(item)
        self.recovery_filtered += sum(item[0].kind == 'clear' for item in choices)
        return useful or probes
