"""B2 search optimization with continuously certified moving/reused scan sites."""
import math
import time
from functools import lru_cache
import numpy as np
from strategy import DirectionalB2
from route_search_probe import improved_tour as open_tour
from q3_strategy import EPS, clip
from adaptive_coverage import CoverageCertificate


@lru_cache(maxsize=1)
def template():
    def ring(n, radius, phase=0.):
        a = np.arange(n)*math.tau/n+phase
        return np.column_stack((radius*np.cos(a), radius*np.sin(a)))
    sites = np.vstack([[[0., 0.]], ring(6, 950.), ring(6, 1350., math.pi/6), ring(12, 1875.)])
    certificate = CoverageCertificate(80.)
    if not certificate.refine_to_certify(sites, 5):
        raise ArithmeticError('Initial planned coverage is not certified')
    return sites, certificate


class AdaptiveB2(DirectionalB2):
    def __init__(self, action):
        super().__init__(action)
        self.mode = 'q4_adaptive'
        sites, certificate = template()
        self.sites = sites.copy()
        # Geometry is immutable during a run; only actual scan history changes.
        self.certificate = certificate
        self.scan_history = []
        self.certified_cells = np.zeros(len(certificate.areas), dtype=bool)
        self.retired_sites = 0
        self.reused_scans = 0
        self.moved_sites = 0
        self.certificate_calls = 0
        self.coverage_residual = np.ones(len(certificate.areas))
        self.negative_pair_cuts = 0
        self.history_pair_cuts = 0

    @staticmethod
    def observation_key(point, channel):
        # Distinct positions can straddle the emission boundary, even when close.
        # Reuse only the exact same floating-point location.
        return channel, float(point[0]).hex(), float(point[1]).hex()

    def measure(self, point, channel):
        state = super().measure(point, channel)
        if state in ('direction', 'no_signal') and channel in self.polys and channel not in self.cleared:
            self.cut_from_history(channel)
        return state

    def cut_from_history(self, channel):
        negatives = self.negative_observations.get(channel, [])
        if len(negatives) < 2:
            return
        center = self.region_circle(self.polys[channel])[0]
        candidates = sorted(negatives, key=lambda p: np.linalg.norm(p-center))[:12]
        anchors = [self.measurements[channel][0][0]]
        if len(self.measurements[channel]) > 1:
            anchors.append(self.measurements[channel][-1][0])
        for anchor in anchors:
            for i in range(len(candidates)):
                for j in range(i+1, len(candidates)):
                    self.cut_negative_segment(channel, anchor, candidates[i], candidates[j])

    def cut_negative_segment(self, channel, anchor, n1, n2):
        v1, v2 = n1-anchor, n2-anchor
        cross = v1[0]*v2[1]-v1[1]*v2[0]
        if abs(cross) < 1e-6 or min(np.linalg.norm(v1), np.linalg.norm(v2)) < 1e-6:
            return False
        if cross < 0:
            n1, n2, v1, v2 = n2, n1, v2, v1
        edge = n2-n1
        normal = np.array([-edge[1], edge[0]])/np.linalg.norm(edge)
        if normal@(n1-anchor) < 0:
            normal = -normal
        bound = float(normal@n1)
        poly = self.polys[channel]
        values = poly@normal-bound
        if values.max() <= 1e-4:
            return False
        far = poly if values.min() >= 0 else clip(poly, -normal, -bound+1e-7)
        rays = far-anchor
        left = (v1[0]*rays[:, 1]-v1[1]*rays[:, 0])/np.linalg.norm(v1)
        right = (rays[:, 0]*v2[1]-rays[:, 1]*v2[0])/np.linalg.norm(v2)
        if min(left.min(), right.min()) < 1e-7:
            return False
        for negative in (n1, n2):
            # Either the minimum reception radius suffices for the entire far
            # polygon, or affine squared-distance dominance over the positive
            # anchor suffices (the same unknown R applies to every observation).
            within_minimum = np.max(np.linalg.norm(far-negative, axis=1)) < 1000.-1e-6
            dominance = 2*far@(anchor-negative)+(negative@negative-anchor@anchor)
            if not within_minimum and dominance.max() > -1e-4:
                # R² >= lambda*|G-anchor|² + (1-lambda)*1000² for 0<=lambda<=1.
                # The negative-distance minus this bound is convex in G, so
                # satisfying its affine-in-lambda inequalities at ALL vertices
                # proves reception throughout the far polygon.
                aa = np.sum((far-negative)**2, axis=1)-1000.**2+1e-4
                bb = np.sum((far-anchor)**2, axis=1)-1000.**2
                lo, hi = 0., 1.
                for av, bv in zip(aa, bb):
                    if abs(bv) < 1e-8:
                        if av > 0:
                            lo, hi = 1., 0.
                            break
                    elif bv > 0:
                        lo = max(lo, av/bv)
                    else:
                        hi = min(hi, av/bv)
                if lo > hi:
                    return False
        self.polys[channel] = clip(poly, normal, bound+1e-6)
        self.history_pair_cuts += 1
        return True

    def localize(self, channel, may_defer=False):
        # Keep B2's inexpensive normal viewpoints; binary transects replace its
        # expensive optical fallback when directional loss remains unresolved.
        return super().localize(channel, may_defer)

    def optical_cover(self, channel):
        return self.paired_localize(channel)

    def paired_localize(self, channel, may_defer=False):
        tried_clear = set()
        for step in range(2 if may_defer else 12):
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
            anchor, bearing = self.measurements[channel][0]
            a = math.radians(bearing)
            u = np.array([math.cos(a), math.sin(a)])
            normal = np.array([-u[1], u[0]])
            axial = (poly-anchor)@u
            t = max(.01, (float(axial.min())+float(axial.max()))/2)
            width = max(30., min(90., .12*radius), t*math.tan(EPS)+2.)
            pair = [anchor+t*u+sign*width*normal for sign in (-1, 1)]
            # Different viewpoints are required when a prior transect was inconclusive.
            if any(self.observation_key(q, channel) in self.attempted for q in pair):
                width += 12*(step+1)
                pair = [anchor+t*u+sign*width*normal for sign in (-1, 1)]
            pair.sort(key=lambda q: np.linalg.norm(q-self.position))
            states = []
            for q in pair:
                states.append(self.measure(q, channel))
                if channel in self.cleared:
                    return
                updated_center, updated_radius = self.region_circle(self.polys[channel])
                if updated_radius <= 19.8:
                    self.clear(updated_center, channel, must_succeed=True)
                    return
                clear_key = tuple(np.round(updated_center, 6))
                if updated_radius <= 80 and clear_key not in tried_clear:
                    tried_clear.add(clear_key)
                    if self.clear(updated_center, channel):
                        return
            if states == ['no_signal', 'no_signal']:
                self.apply_negative_pair(channel, anchor, u, normal, t, width, pair)
        if may_defer:
            self.defer_until[channel] = len(self.visited)+2
            self.deferred_localizations += 1
        else:
            super().optical_cover(channel)

    def apply_negative_pair(self, channel, anchor, u, normal, t, width, pair):
        poly = self.polys[channel]
        if any(np.max(np.linalg.norm(poly-q, axis=1)) > 1000.-1e-6 for q in pair):
            return False
        axial = (poly-anchor)@u
        if np.max(axial) <= t+1e-6:
            return False
        far = clip(poly, -u, float(-anchor@u-t+1e-7))
        far_axial, transverse = (far-anchor)@u, (far-anchor)@normal
        # Whole-poly ray bracket: intersection of every forward feasible ray with
        # the transect lies between its two negative endpoints. This is linear
        # (two halfplanes) and checked at every convex-polygon vertex.
        if np.min(width*far_axial-t*np.abs(transverse)) < 1e-7:
            return False
        # If G were beyond the transect, its segment to the positive anchor would
        # intersect that negative segment. Convexity of the emission halfplane
        # makes two negative endpoints impossible. The distance test above
        # rules out range loss, for ALL feasible source positions.
        self.polys[channel] = clip(poly, u, float(anchor@u+t+1e-6))
        self.negative_pair_cuts += 1
        return True

    def certify(self, points):
        self.certificate_calls += 1
        return self.certificate.complete(points)

    def scan(self, point, unknown_only=False):
        if any(np.array_equal(point, p) for p in self.scan_history):
            return
        if unknown_only:
            for c in range(1, 21):
                if len(set(self.polys)|self.cleared) == 16:
                    break
                if c not in self.polys and c not in self.cleared and c not in self.absent:
                    self.measure(point, c)
        else:
            super().scan(point)
        self.scan_history.append(np.asarray(point).copy())
        self.visited.add(len(self.scan_history)-1)
        self.certified_cells, self.coverage_residual = self.certificate.evaluate(self.scan_history)
        self.certificate_calls += 1

    def update_completion(self):
        detected = set(self.polys)|self.cleared
        if len(detected) > 16:
            raise ArithmeticError('Too many distinct source channels')
        if len(detected) == 16:
            self.absent = set(range(1, 21))-detected
            self.completion_reason = '16_distinct_sources_identified'
            return True
        if self.certified_cells.all():
            self.absent = set(range(1, 21))-detected
            self.completion_reason = 'continuous_position_heading_coverage'
            return True
        return False

    def planned_points(self, remaining, extra=None, omit=None):
        points = list(self.scan_history)+[self.sites[k] for k in sorted(remaining) if k != omit]
        if extra is not None:
            points.append(extra)
        return points

    def reuse_current_point(self, remaining):
        if not remaining or any(np.linalg.norm(self.position-p) < 80 for p in self.scan_history):
            return
        unknown = 20-len(set(self.polys)|self.cleared|self.absent)
        if not unknown:
            return
        ids = sorted(remaining)
        route = [ids[i] for i in open_tour(self.position, self.sites[ids])]
        # Check six nearby replacements; a bounded heuristic, not a global optimum.
        candidates = sorted(remaining, key=lambda k: np.linalg.norm(self.sites[k]-self.position))[:6]
        savings = {}
        for k in candidates:
            j = route.index(k)
            prev = self.position if j == 0 else self.sites[route[j-1]]
            following = None if j+1 == len(route) else self.sites[route[j+1]]
            savings[k] = np.linalg.norm(self.sites[k]-prev)
            if following is not None:
                savings[k] += np.linalg.norm(self.sites[k]-following)-np.linalg.norm(prev-following)
        for k in sorted(candidates, key=lambda k: savings[k], reverse=True):
            # Conservative: charge the extra scan but do not credit the removed scan.
            if savings[k]/5 <= 6*unknown:
                continue
            if self.certify(self.planned_points(remaining, extra=self.position, omit=k)):
                self.scan(self.position.copy(), unknown_only=True)
                remaining.remove(k)
                self.retired_sites += 1
                self.reused_scans += 1
                return
        # The development run found extra scans without certified replacements
        # expensive. Retain the evidence-based replacement gate above.

    def move_scan_site(self, key, remaining, following):
        original = self.sites[key].copy()
        best = original
        def cost(p):
            return np.linalg.norm(p-self.position)+(0 if following is None else np.linalg.norm(p-following))
        old_cost = cost(original)
        # Search from smallest travel to largest; each accepted replacement is certified.
        for fraction in (0., .35, .65, .85):
            q = self.position+fraction*(original-self.position)
            if cost(q) >= old_cost-1e-6:
                continue
            if self.certify(self.planned_points(remaining, extra=q, omit=key)):
                best = q
                break
        if np.linalg.norm(best-original) > 1e-6:
            self.sites[key] = best
            self.moved_sites += 1
        return best

    def run(self):
        entered = self.send('/enter')
        self.deadline = time.monotonic()+max(0, entered.get('remaining_real_duration_s', 1200)-5)
        self.scan(np.zeros(2))
        remaining = set(range(1, len(self.sites)))
        while remaining or set(self.polys)-self.cleared:
            if self.update_completion():
                remaining.clear()
            self.shared_observations()
            if self.update_completion():
                remaining.clear()
            self.reuse_current_point(remaining)
            if self.update_completion():
                remaining.clear()
            tasks = [('scan', k, self.sites[k]) for k in sorted(remaining)]
            tasks += [('clear', c, self.region_circle(self.polys[c])[0])
                      for c in sorted(set(self.polys)-self.cleared)
                      if not remaining or
                      ((len(self.visited) >= self.defer_until.get(c, 0) or self.region_circle(self.polys[c])[1] <= 19.8)
                       and (len(self.measurements[c]) >= 2 or self.region_circle(self.polys[c])[1] <= 150.))]
            if not tasks:
                break
            route = open_tour(self.position, [t[2] for t in tasks])
            self.routing_calls += 1
            kind, key, point = tasks[route[0]]
            if kind == 'scan':
                following = tasks[route[1]][2] if len(route) > 1 else None
                point = self.move_scan_site(key, remaining, following)
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
