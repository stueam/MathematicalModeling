"""Third B2 iteration: reuse scans with net cost accounting and route-aware tasks.

All feasible-region updates and continuous completion checks are inherited
unchanged. No truth, probabilities, or simulator transport enter this policy.
"""
import time
import math
from functools import lru_cache
import numpy as np
from adaptive_strategy import AdaptiveB2, open_tour
from adaptive_coverage import CoverageCertificate


@lru_cache(maxsize=1)
def compact_template():
    def ring(n, radius, phase=0.):
        a = np.arange(n)*math.tau/n+phase
        return np.column_stack((radius*np.cos(a), radius*np.sin(a)))
    sites = np.vstack([[[0., 0.]], ring(9, 999.), ring(12, 1875., math.pi/12)])
    certificate = CoverageCertificate(80.)
    # Near the 1000 m range boundary small cells are necessary. Refinement
    # certifies every cell; it never declares success from sampled positions.
    if not certificate.refine_to_certify(sites, 9):
        raise ArithmeticError('Compact initial coverage is not certified')
    return sites, certificate


class CompactB2(AdaptiveB2):
    @staticmethod
    def plan_route(start, points):
        return open_tour(start, points)

    def __init__(self, action, route_aware=True):
        super().__init__(action)
        self.mode = 'q4_compact'
        sites, certificate = compact_template()
        self.sites = sites.copy()
        self.certificate = certificate
        self.certified_cells = np.zeros(len(certificate.areas), dtype=bool)
        self.coverage_residual = np.ones(len(certificate.areas))
        self.route_aware = route_aware
        self.following = None

    def reuse_current_point(self, remaining):
        if not remaining or any(np.linalg.norm(self.position-p) < 30 for p in self.scan_history):
            return
        unknown = 20-len(set(self.polys)|self.cleared|self.absent)
        if not unknown:
            return
        ids = sorted(remaining)
        route = [ids[i] for i in self.plan_route(self.position, self.sites[ids])]
        candidates = sorted(remaining, key=lambda k: np.linalg.norm(self.sites[k]-self.position))[:6]
        savings = {}
        for k in candidates:
            j = route.index(k)
            before = self.position if j == 0 else self.sites[route[j-1]]
            after = None if j+1 == len(route) else self.sites[route[j+1]]
            gain = np.linalg.norm(self.sites[k]-before)
            if after is not None:
                gain += np.linalg.norm(self.sites[k]-after)-np.linalg.norm(before-after)
            savings[k] = gain
        for k in sorted(candidates, key=lambda k: savings[k], reverse=True):
            # This scan REPLACES one future scan of the same unknown channels.
            # Count the geometric route gain; do not charge a full extra scan.
            # This is a scheduling estimate, not a proof of realized time gain.
            if savings[k] > 30. and self.certify(self.planned_points(remaining, extra=self.position, omit=k)):
                self.scan(self.position.copy(), unknown_only=True)
                remaining.remove(k)
                self.retired_sites += 1
                self.reused_scans += 1
                return

    def viewpoints(self, channel):
        poly = self.polys[channel]
        center, radius = self.region_circle(poly)
        _, _, basis = np.linalg.svd(poly-center, full_matrices=False)
        major, normal = basis
        lateral = max(30., min(100., .15*radius))
        candidates = [center+sign*lateral*normal for sign in (-1, 1)]
        candidates += [center+sign*.65*radius*major+side*30*normal
                       for sign in (-1, 1) for side in (-1, 1)]
        return [p for p in candidates if self.observation_key(p, channel) not in self.attempted]

    def task_point(self, channel):
        center, radius = self.region_circle(self.polys[channel])
        if radius <= 80 or not self.route_aware:
            return center
        candidates = self.viewpoints(channel)
        return min(candidates, key=lambda p: np.linalg.norm(p-self.position)+.15*np.linalg.norm(p-center)) if candidates else center

    def localize(self, channel, may_defer=False):
        if not self.route_aware:
            return super().localize(channel, may_defer)
        tried = set()
        for _ in range(2 if may_defer else 8):
            center, radius = self.region_circle(self.polys[channel])
            if radius <= 19.8:
                self.clear(center, channel, must_succeed=True)
                return
            key = tuple(center)
            if radius <= 80 and key not in tried:
                tried.add(key)
                if self.clear(center, channel):
                    return
            candidates = self.viewpoints(channel)
            if not candidates:
                break
            def cost(p):
                onward = 0. if self.following is None else .5*np.linalg.norm(p-self.following)
                return np.linalg.norm(p-self.position)+.15*np.linalg.norm(p-center)+onward
            point = min(candidates, key=cost)
            self.measure(point, channel)
            if channel in self.cleared:
                return
        if may_defer:
            self.defer_until[channel] = len(self.visited)+2
            self.deferred_localizations += 1
        else:
            self.optical_cover(channel)

    def run(self):
        entered = self.send('/enter')
        self.deadline = time.monotonic()+max(0, entered.get('remaining_real_duration_s', 1200)-5)
        self.scan(np.zeros(2))
        remaining = set(range(1, len(self.sites)))
        while remaining or set(self.polys)-self.cleared:
            if self.update_completion(): remaining.clear()
            self.shared_observations()
            if self.update_completion(): remaining.clear()
            self.reuse_current_point(remaining)
            if self.update_completion(): remaining.clear()
            tasks = [('scan', k, self.sites[k]) for k in sorted(remaining)]
            tasks += [('clear', c, self.task_point(c))
                      for c in sorted(set(self.polys)-self.cleared)
                      if not remaining or
                      ((len(self.visited) >= self.defer_until.get(c, 0) or self.region_circle(self.polys[c])[1] <= 19.8)
                       and (len(self.measurements[c]) >= 2 or self.region_circle(self.polys[c])[1] <= 150.))]
            if not tasks: break
            route = self.plan_route(self.position, [t[2] for t in tasks])
            self.routing_calls += 1
            kind, key, point = tasks[route[0]]
            self.following = tasks[route[1]][2] if len(route) > 1 else None
            if kind == 'scan':
                point = self.move_scan_site(key, remaining, self.following)
                self.scan(point)
                remaining.remove(key)
            else:
                self.localize(key, may_defer=bool(remaining))
        if not self.update_completion() or self.cleared & self.absent or len(self.cleared|self.absent) != 20:
            raise ArithmeticError('Cannot certify task completion')
        result = self.send('/exit')
        return {'method': self.mode, 'virtual_time_s': result['virtual_time_s'],
                'cleared_channels': sorted(self.cleared), 'absent_channels': sorted(self.absent),
                'actions': len(self.trace), 'optical_cover_count': self.fallback_count,
                'scanned_sites': len(self.visited), 'required_sites': len(self.sites),
                'routing_calls': self.routing_calls, 'shared_measures': self.shared_measures,
                'completion_reason': self.completion_reason, 'coverage_cells': len(self.certified_cells),
                'certified_cells': int(self.certified_cells.sum()), 'retired_sites': self.retired_sites,
                'reused_scans': self.reused_scans, 'moved_sites': self.moved_sites,
                'certificate_calls': self.certificate_calls, 'negative_pair_cuts': self.negative_pair_cuts,
                'history_pair_cuts': self.history_pair_cuts}
