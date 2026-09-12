"""Q4 adaptation of Q3's continuous survey/clear service-point design."""
import numpy as np
from compact_strategy import CompactB2
from service_geometry import station_constraints, optimize_service
from service_routes import exact_open_route


class ServiceB2(CompactB2):
    def __init__(self, action):
        super().__init__(action)
        self.mode = 'q4_service'
        self.service_attempts = 0
        self.service_accepts = 0
        self.clear_adjustments = 0
        self.local_leg_saving_m = 0.
        self.exact_route_calls = 0

    def plan_route(self, start, points):
        if len(points) <= 12:
            self.exact_route_calls += 1
            return exact_open_route(start, points)
        return super().plan_route(start, points)

    def move_scan_site(self, key, remaining, following):
        original = self.sites[key].copy()
        others = self.planned_points(remaining, omit=key)
        self.service_attempts += 1
        constraints = station_constraints(self.certificate, others)
        if constraints is None:
            return super().move_scan_site(key, remaining, following)
        vertices, normals, bounds = constraints
        point, info = optimize_service(original, self.position, following, vertices, 1000.-2e-6, normals, bounds)
        if info['accepted'] and self.certify(others+[point]):
            self.sites[key] = point
            self.moved_sites += 1
            self.service_accepts += 1
            self.local_leg_saving_m += info['saving_m']
            return point
        return super().move_scan_site(key, remaining, following)

    def clear(self, point, channel, must_succeed=False):
        # Only optimize inside a region proven to clear ALL feasible positions.
        poly = self.polys.get(channel)
        if must_succeed and poly is not None and np.max(np.linalg.norm(poly-point, axis=1)) <= 19.8:
            q, info = optimize_service(point, self.position, self.following, poly, 19.8)
            if info['accepted'] and np.max(np.linalg.norm(poly-q, axis=1)) <= 19.8:
                point = q
                self.clear_adjustments += 1
                self.local_leg_saving_m += info['saving_m']
        return super().clear(point, channel, must_succeed)

    def run(self):
        result = super().run()
        result.update(service_attempts=self.service_attempts, service_accepts=self.service_accepts,
                      clear_adjustments=self.clear_adjustments, local_leg_saving_m=self.local_leg_saving_m,
                      exact_route_calls=self.exact_route_calls)
        return result
