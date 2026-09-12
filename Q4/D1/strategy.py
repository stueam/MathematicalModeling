"""Q4 adaptation of B2. No simulator truth or network access in this module."""
from pathlib import Path
import sys
import time
import math
import numpy as np

# Keep Q4 modules ahead of similarly named historical B2 modules (benchmark.py).
sys.path.append(str(Path(__file__).resolve().parent/'vendor_b2'))
from q3_strategy import Strategy
from joint_strategy import open_tour
from coverage import search_mesh


class DirectionalB2(Strategy):
    def __init__(self, action, method='q4_b2', mesh_step=950.):
        if method not in ('q4_b2', 'q4_safe'):
            raise ValueError(method)
        super().__init__(action)
        self.mode = method
        self.sites, self.triangles = search_mesh(mesh_step)
        self.visited = set()
        self.attempted = set()
        self.negative_observations = {}
        self.routing_calls = 0
        self.shared_measures = 0
        self.completion_reason = None
        self.defer_until = {}
        self.deferred_localizations = 0

    @staticmethod
    def observation_key(point, channel):
        return (channel, *(round(float(x), 6) for x in point))

    def measure(self, point, channel):
        key = self.observation_key(point, channel)
        if key in self.attempted:
            return 'already_measured'
        self.attempted.add(key)
        # Deliberately use the positive-only B2 ancestor, not negative/radius cuts.
        state = super().measure(point, channel)
        if state == 'no_signal':
            self.negative_observations.setdefault(channel, []).append(np.array(point).copy())
        return state

    def scan(self, point):
        for c in range(1, 21):
            if len(set(self.polys)|self.cleared) == 16:
                break
            if c in self.cleared or c in self.absent:
                continue
            if c in self.polys and self.region_circle(self.polys[c])[1] <= 19.8:
                continue
            self.measure(point, c)

    def shared_observations(self):
        for c in sorted(set(self.polys)-self.cleared):
            center, radius = self.region_circle(self.polys[c])
            dist = np.linalg.norm(center-self.position)
            if radius <= 19.8 or dist < 30 or dist > 1000:
                continue
            v = center-self.position
            cross = [abs(np.linalg.det(np.array([center-p, v])))/max(np.linalg.norm(center-p)*dist, 1e-12)
                     for p, _ in self.measurements[c]]
            if max(cross, default=0) < .1:
                continue
            key = self.observation_key(self.position, c)
            if key not in self.attempted:
                self.measure(self.position.copy(), c)
                self.shared_measures += 1

    def optical_cover(self, channel):
        """Finite orientation-independent fallback, never a no-signal exclusion."""
        self.fallback_count += 1
        poly = self.polys[channel]
        origin = poly.mean(axis=0)
        _, _, basis = np.linalg.svd(poly-origin, full_matrices=False)
        rotated = (poly-origin)@basis.T
        lo, hi = rotated.min(axis=0), rotated.max(axis=0)
        # Grid corners cover the entire rotated bounding rectangle with <=17.68 m.
        axes = [np.linspace(lo[k], hi[k], max(1, math.ceil((hi[k]-lo[k])/25))+1) for k in range(2)]
        points = [origin+np.array([x, y])@basis for i, x in enumerate(axes[0])
                  for y in (axes[1] if i % 2 == 0 else axes[1][::-1])]
        if len(points) > 10000:
            raise RuntimeError('Finite optical cover exceeds action budget; completion not certified')
        if np.linalg.norm(points[-1]-self.position) < np.linalg.norm(points[0]-self.position):
            points.reverse()
        for point in points:
            if self.clear(point, channel):
                return
        raise ArithmeticError('Optical cover exhausted; preserve failure instead of claiming completion')

    def localize(self, channel, may_defer=False):
        tried_clear = set()
        for _ in range(2 if may_defer else 8):
            poly = self.polys[channel]
            center, radius = self.region_circle(poly)
            if radius <= 19.8:
                self.clear(center, channel, must_succeed=True)
                return
            key = tuple(np.round(center, 6))
            if radius <= 80 and key not in tried_clear:
                tried_clear.add(key)
                if self.clear(center, channel):
                    return
            _, _, basis = np.linalg.svd(poly-center, full_matrices=False)
            major, normal = basis
            lateral = max(30., min(100., .15*radius))
            candidates = [center+sign*lateral*normal for sign in (-1, 1)]
            candidates += [center+sign*.65*radius*major+side*30*normal
                           for sign in (-1, 1) for side in (-1, 1)]
            candidates = [q for q in candidates if self.observation_key(q, channel) not in self.attempted]
            if not candidates:
                break
            q = min(candidates, key=lambda q: np.linalg.norm(q-self.position)+.15*np.linalg.norm(q-center))
            self.measure(q, channel)
            if channel in self.cleared:
                return
        if may_defer:
            self.defer_until[channel] = len(self.visited)+2
            self.deferred_localizations += 1
            return
        self.optical_cover(channel)

    def update_completion(self):
        detected = set(self.polys)|self.cleared
        if len(detected) > 16:
            raise ArithmeticError('More sources than the stated upper bound')
        if len(detected) == 16:
            self.absent = set(range(1, 21))-detected
            self.completion_reason = '16_distinct_sources_identified'
            return True
        if len(self.visited) == len(self.sites):
            self.absent = set(range(1, 21))-detected
            self.completion_reason = 'all_triangle_vertices_scanned'
            return True
        return False

    def run(self):
        entered = self.send('/enter')
        self.deadline = time.monotonic()+max(0, entered.get('remaining_real_duration_s', 1200)-5)
        origin = int(np.argmin(np.linalg.norm(self.sites, axis=1)))
        assert np.linalg.norm(self.sites[origin]) < 1e-9
        self.scan(self.sites[origin])
        self.visited.add(origin)
        remaining = set(range(len(self.sites)))-self.visited
        if self.mode == 'q4_safe':
            ids = sorted(remaining)
            route = open_tour(self.position, self.sites[ids])
            for j in route:
                if self.update_completion():
                    break
                k = ids[j]
                self.scan(self.sites[k])
                self.visited.add(k)
            self.update_completion()
            while set(self.polys)-self.cleared:
                pending = sorted(set(self.polys)-self.cleared)
                c = min(pending, key=lambda c: (np.linalg.norm(self.region_circle(self.polys[c])[0]-self.position), c))
                self.localize(c)
        else:
            while remaining or set(self.polys)-self.cleared:
                if self.update_completion():
                    remaining.clear()
                self.shared_observations()
                if self.update_completion():
                    remaining.clear()
                tasks = [('scan', i, self.sites[i]) for i in sorted(remaining)]
                tasks += [('clear', c, self.region_circle(self.polys[c])[0])
                          for c in sorted(set(self.polys)-self.cleared)
                          if not remaining or len(self.visited) >= self.defer_until.get(c, 0)
                          or self.region_circle(self.polys[c])[1] <= 19.8]
                if not tasks:
                    break
                route = open_tour(self.position, [task[2] for task in tasks])
                self.routing_calls += 1
                kind, key, point = tasks[route[0]]
                if kind == 'scan':
                    self.scan(point)
                    self.visited.add(key)
                    remaining.remove(key)
                else:
                    self.localize(key, may_defer=bool(remaining))
        if not self.update_completion() or self.cleared & self.absent or len(self.cleared|self.absent) != 20:
            raise ArithmeticError('No valid completion certificate')
        result = self.send('/exit')
        return {'method': self.mode, 'virtual_time_s': result['virtual_time_s'],
                'cleared_channels': sorted(self.cleared), 'absent_channels': sorted(self.absent),
                'actions': len(self.trace), 'optical_cover_count': self.fallback_count,
                'scanned_sites': len(self.visited), 'required_sites': len(self.sites),
                'routing_calls': self.routing_calls, 'shared_measures': self.shared_measures,
                'deferred_localizations': self.deferred_localizations,
                'completion_reason': self.completion_reason}


def build_strategy(action, method='q4_b2'):
    return DirectionalB2(action, method=method)
