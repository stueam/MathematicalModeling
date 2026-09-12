"""Continuous-cell certificates for unknown source position AND emission heading.

A cell is excluded only if its entire convex polygon is in the convex hull of
stations that are all within 1000 m of every point in that cell. No angle bins
or point sampling are used for the completion certificate.
"""
import math
from functools import lru_cache
import numpy as np


def clip_polygon(poly, normal, bound):
    if not len(poly):
        return poly
    following = np.roll(poly, -1, axis=0)
    f = poly@normal-bound
    g = np.roll(f, -1)
    result = []
    for a, b, fa, fb in zip(poly, following, f, g):
        if (fa <= 0) != (fb <= 0):
            result.append(a+(b-a)*fa/(fa-fb))
        if fb <= 0:
            result.append(b)
    return np.asarray(result).reshape(-1, 2)


@lru_cache(maxsize=4)
def domain_cells(step=80.):
    axis = np.r_[np.arange(-1800., 1800., step), 1800.]
    normals = np.column_stack((np.cos(np.arange(96)*math.tau/96), np.sin(np.arange(96)*math.tau/96)))
    polygons, areas = [], []
    for x0, x1 in zip(axis[:-1], axis[1:]):
        for y0, y1 in zip(axis[:-1], axis[1:]):
            # Retain cells intersecting a circumscribed polygon (also all disk cells).
            poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            minimum = math.hypot(max(x0, -x1, 0.), max(y0, -y1, 0.))
            if minimum > 1800/math.cos(math.pi/96)+1e-7:
                continue
            if np.max(np.linalg.norm(poly, axis=1)) > 1800:
                for normal in normals:
                    poly = clip_polygon(poly, normal, 1800.+1e-8)
                    if len(poly) < 3:
                        break
            if len(poly) < 3:
                continue
            area = abs(np.sum(poly[:, 0]*np.roll(poly[:, 1], -1)-poly[:, 1]*np.roll(poly[:, 0], -1)))/2
            if area <= 0:
                continue
            polygons.append(poly)
            areas.append(area)
    size = max(map(len, polygons))
    vertices = np.array([np.vstack([p, np.repeat(p[-1:], size-len(p), axis=0)]) for p in polygons])
    return vertices, np.asarray(areas)


class CoverageCertificate:
    def __init__(self, step=80.):
        self.vertices, self.areas = domain_cells(step)
        self.centers = self.vertices.mean(axis=1)
        self.step = step

    def refine(self, indices):
        indices = set(map(int, indices))
        polygons = []
        for i, padded in enumerate(self.vertices):
            poly = np.asarray([p for j, p in enumerate(padded) if j == 0 or np.linalg.norm(p-padded[j-1]) > 1e-12])
            if i not in indices:
                polygons.append(poly)
                continue
            lo, hi = poly.min(axis=0), poly.max(axis=0)
            middle = (lo+hi)/2
            for sx in (-1, 1):
                for sy in (-1, 1):
                    part = clip_polygon(poly, np.array([sx, 0.]), sx*middle[0])
                    part = clip_polygon(part, np.array([0., sy]), sy*middle[1])
                    if len(part) >= 3:
                        area = abs(np.sum(part[:, 0]*np.roll(part[:, 1], -1)-part[:, 1]*np.roll(part[:, 0], -1)))/2
                        if area > 0:
                            polygons.append(part)
        size = max(map(len, polygons))
        self.vertices = np.array([np.vstack([p, np.repeat(p[-1:], size-len(p), axis=0)]) for p in polygons])
        self.areas = np.array([abs(np.sum(p[:, 0]*np.roll(p[:, 1], -1)-p[:, 1]*np.roll(p[:, 0], -1)))/2 for p in polygons])
        self.centers = self.vertices.mean(axis=1)

    def refine_to_certify(self, stations, rounds=6):
        for _ in range(rounds):
            covered, _ = self.evaluate(stations)
            if covered.all():
                return True
            self.refine(np.flatnonzero(~covered))
        return self.complete(stations)

    def prepare(self, stations):
        stations = np.asarray(stations, float).reshape(-1, 2)
        delta = stations[None, None, :, :]-self.vertices[:, :, None, :]
        distance2 = np.sum(delta*delta, axis=-1)
        # Norm is convex: checking ALL polygon vertices bounds the whole cell.
        eligible = np.max(distance2, axis=1) <= (1000.-1e-6)**2
        angle = np.arctan2(delta[:, :, :, 1], delta[:, :, :, 0]) % math.tau
        return angle, eligible, np.all(delta == 0., axis=-1)

    def evaluate_prepared(self, prepared, keep=None):
        angles, eligible, coincidence = prepared
        if keep is not None:
            angles, eligible, coincidence = angles[:, :, keep], eligible[:, keep], coincidence[:, :, keep]
        if angles.shape[2] == 0:
            return np.zeros(len(self.vertices), dtype=bool), np.ones(len(self.vertices))
        count = eligible.sum(axis=1)
        ordered = np.sort(np.where(eligible[:, None, :], angles, 10.), axis=2)
        last = np.take_along_axis(ordered, np.maximum(count-1, 0)[:, None, None], axis=2)[:, :, 0]
        gap = ordered[:, :, 0]+math.tau-last
        if ordered.shape[2] > 1:
            differences = np.diff(ordered, axis=2)
            differences = np.where(np.arange(ordered.shape[2]-1)[None, None, :] < count[:, None, None]-1, differences, 0.)
            gap = np.maximum(gap, differences.max(axis=2))
        at_station = np.any(coincidence & eligible[:, None, :], axis=2)
        # Largest circular angular gap <= pi iff the point is in the convex hull.
        # A strict margin rejects ambiguous floating-point boundary cases.
        contained = ((gap <= math.pi-1e-10) | at_station) & (count[:, None] > 0)
        covered = contained.all(axis=1)
        residual = np.maximum(0., gap.max(axis=1)-math.pi)/math.tau
        residual[count == 0] = 1.
        residual[covered] = 0.
        residual[~covered] = np.maximum(residual[~covered], 1e-9)
        return covered, residual

    def evaluate(self, stations):
        return self.evaluate_prepared(self.prepare(stations))

    def complete(self, stations):
        return bool(self.evaluate(stations)[0].all())
