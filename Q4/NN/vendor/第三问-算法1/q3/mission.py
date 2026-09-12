"""Task-level policies: adaptive coverage stops and multi-source route order.

Plans are functions of public beliefs, never sampled or actual hidden truth.
They can be compared as complete contingent continuations in Monte Carlo.
"""
import math

import numpy as np
import shapely
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from .core import Action, disk, distance, point_key
from .movement import MovementPolicy, support, measurement_costs


def open_tour(start, points):
    """Multi-start nearest-neighbor + 2-opt, for uncertain task representatives.

    This is a route heuristic, not an exact TSP solution or a location oracle.
    """
    n = len(points)
    if n < 2:
        return list(range(n))
    xy = np.asarray([start]+list(points))
    dd = np.linalg.norm(xy[:, None, :]-xy[None, :, :], axis=2)
    best = None
    for first in np.argsort(dd[0, 1:])[:min(n, 4)]:
        remaining = set(range(1, n+1))
        route = [0, int(first)+1]
        remaining.remove(route[-1])
        while remaining:
            nxt = min(remaining, key=lambda j: (dd[route[-1], j], j))
            route.append(nxt)
            remaining.remove(nxt)
        for _ in range(4):
            improved = False
            for i in range(1, n):
                for j in range(i+1, n+1):
                    old = dd[route[i-1], route[i]]
                    new = dd[route[i-1], route[j]]
                    if j < n:
                        old += dd[route[j], route[j+1]]
                        new += dd[route[i], route[j+1]]
                    if new < old-1e-8:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
            if not improved:
                break
        length = sum(dd[x, y] for x, y in zip(route, route[1:]))
        candidate = (float(length), tuple(route[1:]))
        if best is None or candidate < best:
            best = candidate
    return [i-1 for i in best[1]]


def feasible_cover_centers(region, radius=995.):
    """An INNER approximation of positions that cover the entire outer region.

    Every point in the result is within radius of every convex-hull vertex,
    hence every point of the region. Never confuses a centroid with a cover.
    """
    vertices = shapely.get_coordinates(region.convex_hull)
    if len(vertices) > 1 and np.allclose(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    result = None
    for v in vertices:
        constraint = disk(v, radius, outer=False)
        result = constraint if result is None else result.intersection(constraint)
        if result.is_empty:
            break
    return result


class MissionPolicy(MovementPolicy):
    def __init__(self, mode='route', ring_radius=1500., speculative_clear=True):
        super().__init__(ring_radius, speculative_clear=speculative_clear)
        if mode not in ('route', 'nearest', 'clockwise', 'counterclockwise', 'survey'):
            raise ValueError('Unknown mission mode')
        self.mode = mode
        self._coverage_cache = {}
        self._tour_cache = {}

    def information_action(self, b, detected):
        """Value an incidental bearing by uncertainty reduction, not by an
        imaginary commitment to clear that source immediately after measuring.
        Its approach trip is a different task and must not hide a cheap look.
        """
        choices = []
        for c in detected:
            p = b.channels[c]
            if p.summary()[1] <= 19.999:
                continue
            if any(distance(b.position, o.action.position) < 25 for o in p.history if o.action.kind == 'measure'):
                continue
            xy, weights, _, _, _, cov = support(p)
            radius = 2*math.sqrt(float(np.linalg.eigvalsh(cov)[-1]))
            before = max(radius-12, 0)/5+8*math.log2(max(1, radius/18))
            approach = float(weights @ (np.maximum(np.linalg.norm(xy-b.position, axis=1)-18, 0)/5))
            after = float(measurement_costs(p, [b.position])[0])-5-approach
            gain = before-after-5-int(b.receiver != c)
            if gain > 7:
                choices.append((gain, Action('measure', b.position, c)))
        return max(choices, key=lambda x: x[0])[1] if choices else None

    def coverage_options(self, b):
        unresolved = [(c, p) for c, p in b.channels.items() if p.status == 'unresolved']
        if not unresolved:
            return []
        # Histories determine conservative regions. Exclude robot position from
        # the cached geometry: closest feasible stop is projected separately.
        key = tuple((c, tuple(p.history)) for c, p in unresolved)
        if key not in self._coverage_cache:
            remaining = shapely.union_all([p.region for _, p in unresolved])
            cells = []
            for k in range(6):
                a, z = k*math.pi/3-math.pi/6, k*math.pi/3+math.pi/6
                sector = Polygon([(0., 0.), (5000*math.cos(a), 5000*math.sin(a)),
                                  (5000*math.cos(z), 5000*math.sin(z))])
                part = remaining.intersection(sector)
                if part.is_empty:
                    continue
                feasible = feasible_cover_centers(part)
                if feasible is not None and not feasible.is_empty:
                    cells.append(feasible)
            # Bound memory across independent hypothetical futures.
            if len(self._coverage_cache) > 128:
                self._coverage_cache.clear()
            self._coverage_cache[key] = cells
        options = []
        for feasible in self._coverage_cache[key]:
            q = nearest_points(Point(b.position), feasible)[1]
            pos = (q.x, q.y)
            if distance(pos, b.position) < 1:
                pos = b.position
            cands = self.unknown_at(b, pos)
            if not cands:
                continue
            c = max(cands, key=lambda c: self.scan_merit(b, c, pos))
            action = Action('measure', pos, c)
            # All channels at this stop can share the same physical trip.
            merits = [self.scan_merit(b, c, pos) for c in cands]
            score = distance(b.position, pos)/(1+min(3., sum(merits)))
            options.append((score, action))
        return [a for _, a in sorted(options, key=lambda z: z[0])]

    def ordered_channels(self, b, detected):
        if len(detected) <= 1:
            return detected
        # Replan when public information changes, never using source truth.
        centers = [tuple(support(b.channels[c])[4]) for c in detected]
        if self.mode in ('clockwise', 'counterclockwise'):
            direction = 1 if self.mode == 'counterclockwise' else -1
            angle = math.atan2(b.position[1], b.position[0])
            return sorted(detected, key=lambda c: (direction*(math.atan2(support(b.channels[c])[4][1],
                                      support(b.channels[c])[4][0])-angle)) % (2*math.pi))
        if self.mode == 'nearest':
            return sorted(detected, key=lambda c: self.ranked_local(b, c, limit=1)[0][0])
        key = (point_key(b.position), tuple((c, tuple(b.channels[c].history)) for c in detected))
        if key not in self._tour_cache:
            self._tour_cache[key] = [detected[i] for i in open_tour(b.position, centers)]
            if len(self._tour_cache) > 256:
                last = self._tour_cache[key]
                self._tour_cache = {key: last}
        return self._tour_cache[key]

    def choose(self, b):
        if b.done():
            raise ValueError('No action after certified completion')
        detected = [c for c, p in b.channels.items() if p.status == 'detected']
        for c in detected:
            action = self.guaranteed_clear(b, c)
            if action is not None and distance(action.position, b.position) < 1e-6:
                return action
        information = self.information_action(b, detected)
        if information is not None:
            return information
        local = [(cost, a) for c in detected for cost, a in self.ranked_local(b, c, limit=1)]
        inplace = [(cost, a) for cost, a in local if distance(a.position, b.position) < 1e-6]
        if inplace:
            return min(inplace, key=lambda z: z[0])[1]
        unknown = self.unknown_at(b, b.position)
        if unknown:
            c = max(unknown, key=lambda c: self.scan_merit(b, c, b.position))
            # A cheap scan only if this stop has meaningful predicted value.
            # Original origin census is retained, without hard step-number gates.
            if distance(b.position, (0, 0)) < 1e-6 or self.scan_merit(b, c, b.position) > .08:
                return Action('measure', b.position, c)
        if self.mode == 'survey':
            coverage = self.coverage_options(b)
            if coverage:
                return coverage[0]
        if detected:
            c = self.ordered_channels(b, detected)[0]
            return self.ranked_local(b, c, limit=1)[0][1]
        coverage = self.coverage_options(b)
        if coverage:
            return coverage[0]
        # If a sector is too broad to cover in one stop, the proven seven-stop
        # completion strategy remains available. No numerical absence shortcut.
        return super().choose(b)
