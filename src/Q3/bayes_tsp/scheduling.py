"""Deterministic now-versus-route sensing and geometrically complete patrols."""
import numpy as np
import shapely
from shapely.geometry import GeometryCollection


def insertion_costs(start, future, xy):
    """Extra metres to insert each quadrature location along every open edge."""
    vertices = np.asarray([start]+list(future), dtype=float)
    d = np.linalg.norm(vertices[:, None, :]-xy[None, :, :], axis=2)
    if not future:
        return d
    edges = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    return np.vstack((d[:-1]+d[1:]-edges[:, None], d[-1:]))


def scan_route_value(posterior, existence, position, future, recovery_s):
    """Expected discovery-insertion regret avoided by measuring NOW.

R is fixed across all future stops. Integrate the interval for 'now receives,
first future reception is j', rather than multiplying independent hit chances.
Coverage is a planning estimate; actual no-signal disks alone certify absence.
"""
    xy, w, lo, hi = posterior.xy, posterior.weights, posterior.lo, posterior.hi
    d0 = np.linalg.norm(xy-position, axis=1)
    width = hi-lo
    receive_now = np.clip((hi-np.maximum(lo, d0))/width, 0, 1)
    best_seen = np.full(len(xy), np.inf)
    extra = insertion_costs(position, future, xy)
    best_now = extra.min(axis=0)
    gain = 0.
    for j, stop in enumerate(future):
        dj = np.linalg.norm(xy-stop, axis=1)
        # R >= max(d0,dj,lo) and R < every earlier future distance.
        probability = np.maximum(0, np.minimum(hi, best_seen)-np.maximum.reduce((lo, d0, dj)))/width
        late_extra = extra[j+1:].min(axis=0)
        gain += existence*float(w @ (probability*np.maximum(late_extra-best_now, 0)))/5
        best_seen = np.minimum(best_seen, dj)
    exclusive = np.maximum(0, np.minimum(hi, best_seen)-np.maximum(lo, d0))/width
    unique_cover = (d0 <= 1000) & (best_seen > 1000)
    exclusive_probability = float(w @ exclusive)
    coverage = float(w @ unique_cover)
    # A bounded heuristic price for work not covered by the forecast route.
    discovery = existence*exclusive_probability*recovery_s
    certification = .5*coverage*recovery_s
    return {'gross_saving_s': gain+discovery+certification,
            'early_insertion_s': gain, 'exclusive_discovery_s': discovery,
            'coverage_value_s': certification, 'reception_now': float(w @ receive_now),
            'exclusive_reception': exclusive_probability, 'unique_coverage': coverage}


class CoverTour:
    """Small exact open-route DP over all subsets of fixed safe survey disks.

Certify a subset against the entire conservative remaining geometry, not grid
points, area thresholds, or Bayesian mass. No claim of global continuous optimum.
"""
    def __init__(self, stations, disks):
        self.stations = list(stations)
        self.disks = list(disks)
        self.covers = [GeometryCollection()]
        for mask in range(1, 1 << len(stations)):
            bit = mask & -mask
            self.covers.append(self.covers[mask ^ bit].union(disks[bit.bit_length()-1]))
        self.cache = {}

    def masks(self, region):
        key = region.wkb
        if key not in self.cache:
            possible = shapely.is_empty(shapely.difference(region, np.asarray(self.covers, dtype=object)))
            if not possible.any():
                raise ValueError('Fixed station disks do not cover remaining conservative support')
            if len(self.cache) >= 32:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = possible
        return self.cache[key]

    def plan(self, start, region, scan_costs):
        possible = self.masks(region)
        if possible[0]:
            return 0., []
        xy = np.asarray(self.stations)
        distance = np.linalg.norm(xy[:, None, :]-xy[None, :, :], axis=2)/5
        initial = np.linalg.norm(xy-start, axis=1)/5
        n = len(xy)
        dp = {(1 << j, j): (float(initial[j]+scan_costs[j]), (j,))
              for j in range(n) if scan_costs[j] > 0}
        for mask in range(1, 1 << n):
            for last in range(n):
                old = dp.get((mask, last))
                if old is None:
                    continue
                for nxt in range(n):
                    if mask & (1 << nxt) or scan_costs[nxt] <= 0:
                        continue
                    key = (mask | (1 << nxt), nxt)
                    value = (old[0]+float(distance[last, nxt]+scan_costs[nxt]), old[1]+(nxt,))
                    if key not in dp or value < dp[key]:
                        dp[key] = value
        cost, path = min(value for (mask, _), value in dp.items() if possible[mask])
        return cost, list(path)
