"""Fast observable-state tracker and candidate generation (Numba CPU kernels).

Convex polygons are conservative OUTER regions. Negative observations are recorded
but their non-convex disk exclusions are deliberately not applied to these polygons.
This sacrifices information, not containment. They are not Bayesian posteriors.
"""
import numpy as np
from numba import njit

MAX_VERTICES = 96
ANGLE_EPS = np.deg2rad(1.005)  # 1 degree error plus possible 0.005 degree rounding
t = np.arange(6) * np.pi / 3
STATIONS = np.vstack((np.zeros((1, 2)), 1500 * np.column_stack((np.cos(t), np.sin(t)))))
t = (np.arange(32) + .5) * 2 * np.pi / 32
OUTER_DISK = 1800 / np.cos(np.pi / 32) * np.column_stack((np.cos(t), np.sin(t)))
t = np.arange(16) * 2 * np.pi / 16
NORMALS = np.column_stack((np.cos(t), np.sin(t)))


@njit(cache=True)
def clip(poly, size, nx, ny, limit):
    out = np.empty_like(poly)
    k = 0
    for j in range(size):
        prev = (j - 1) % size
        ax, ay = poly[prev, 0], poly[prev, 1]
        bx, by = poly[j, 0], poly[j, 1]
        da, db = nx * ax + ny * ay - limit, nx * bx + ny * by - limit
        ia, ib = da <= 1e-8, db <= 1e-8
        if ia != ib:
            ratio = da / (da - db)
            if k >= len(out):
                raise ValueError("Polygon capacity exceeded")
            out[k, 0], out[k, 1] = ax + ratio * (bx - ax), ay + ratio * (by - ay)
            k += 1
        if ib:
            if k >= len(out):
                raise ValueError("Polygon capacity exceeded")
            out[k] = poly[j]
            k += 1
    poly[:k] = out[:k]
    return k


@njit(cache=True)
def update_geometry(polys, sizes, status, centers, radii, area, last_angle,
                    coverage, counts, last_result, last_pos, fail_pos,
                    positions, channels, outcomes, angles):
    for i in range(len(channels)):
        c, result = channels[i], outcomes[i]
        x, y = positions[i, 0], positions[i, 1]
        last_result[i, c] = result
        counts[i, c] += 1
        last_pos[i, c] = positions[i]
        if result <= 2:
            for s in range(7):
                if (x - STATIONS[s, 0])**2 + (y - STATIONS[s, 1])**2 < 1e-10:
                    coverage[i, c, s] = True
        if result == 3:
            status[i, c] = 2
        elif result == 4:
            fail_pos[i, c] = positions[i]
        elif result == 0:
            if status[i, c] == 0 and np.all(coverage[i, c]):
                status[i, c] = 3
        else:
            status[i, c] = 1
            poly, size = polys[i, c], sizes[i, c]
            if size == 0:
                poly[:32] = OUTER_DISK
                size = 32
            radius = 5. if result == 1 else 1500.
            for j in range(16):
                nx, ny = NORMALS[j, 0], NORMALS[j, 1]
                size = clip(poly, size, nx, ny, nx * x + ny * y + radius)
            if result == 2:
                theta = np.deg2rad(angles[i])
                last_angle[i, c] = theta
                lo, hi = theta - ANGLE_EPS, theta + ANGLE_EPS
                nx, ny = np.sin(lo), -np.cos(lo)
                size = clip(poly, size, nx, ny, nx * x + ny * y)
                nx, ny = -np.sin(hi), np.cos(hi)
                size = clip(poly, size, nx, ny, nx * x + ny * y)
            if size == 0:
                raise ValueError("Inconsistent observations: empty outer polygon")
            sizes[i, c] = size
            # Bbox midpoint is cheap. Max vertex distance certifies an enclosing
            # circle of the entire convex polygon, although not its minimum one.
            xmin, ymin = np.min(poly[:size, 0]), np.min(poly[:size, 1])
            xmax, ymax = np.max(poly[:size, 0]), np.max(poly[:size, 1])
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            centers[i, c, 0], centers[i, c, 1] = cx, cy
            max_r2, a = 0., 0.
            for j in range(size):
                max_r2 = max(max_r2, (poly[j, 0] - cx)**2 + (poly[j, 1] - cy)**2)
                k = (j + 1) % size
                a += poly[j, 0] * poly[k, 1] - poly[k, 0] * poly[j, 1]
            radii[i, c], area[i, c] = np.sqrt(max_r2), abs(a) / 2


@njit(cache=True)
def build_candidates(status, centers, radii, areas, coverage, counts, last_angle,
                     last_result, last_pos, fail_pos, position, current_channel,
                     time, steps, max_steps, out, glob, targets, mask):
    n, cmax = status.shape
    for i in range(n):
        glob[i, 0], glob[i, 1] = position[i, 0] / 1800, position[i, 1] / 1800
        glob[i, 2] = time[i] / 10000
        glob[i, 3] = steps[i] / max_steps
        for s in range(4):
            glob[i, 4+s] = np.sum(status[i] == s) / cmax
        for c in range(cmax):
            state, r = status[i, c], radii[i, c]
            cx, cy = centers[i, c, 0], centers[i, c, 1]
            # Two nearest unvisited coverage anchors for unknown channels.
            first, second, d1, d2 = -1, -1, 1e100, 1e100
            for s in range(7):
                if not coverage[i, c, s]:
                    dd = (STATIONS[s, 0]-position[i, 0])**2 + (STATIONS[s, 1]-position[i, 1])**2
                    if dd < d1:
                        second, d2, first, d1 = first, d1, s, dd
                    elif dd < d2:
                        second, d2 = s, dd
            for k in range(4):
                a = c * 4 + k
                valid = state < 2
                x, y = position[i, 0], position[i, 1]
                if state == 0:
                    if k == 1 or k == 2:
                        anchor = first if k == 1 else second
                        if anchor >= 0:
                            x, y = STATIONS[anchor, 0], STATIONS[anchor, 1]
                        else:
                            valid = False
                    elif k == 3:
                        valid = False
                else:
                    x, y = cx, cy
                    if k == 1 or k == 2:
                        offset = min(500., max(50., r * .8)) * (1 if k == 1 else -1)
                        x -= np.sin(last_angle[i, c]) * offset
                        y += np.cos(last_angle[i, c]) * offset
                # Skip exact repeated uninformative action, without hiding truth.
                if k != 3 and counts[i, c] > 0 and last_result[i, c] <= 2:
                    if (x-last_pos[i, c, 0])**2 + (y-last_pos[i, c, 1])**2 < 1e-10:
                        valid = False
                if k == 3 and (x-fail_pos[i, c, 0])**2 + (y-fail_pos[i, c, 1])**2 < 1e-10:
                    valid = False
                targets[i, a, 0], targets[i, a, 1] = x, y
                distance = np.sqrt((x-position[i, 0])**2 + (y-position[i, 1])**2)
                f = out[i, a]
                for j in range(len(f)):
                    f[j] = 0
                f[k] = 1
                f[4] = state == 1
                f[5] = c == current_channel[i]
                f[6], f[7] = x / 1800, y / 1800
                f[8], f[9] = (x-position[i, 0]) / 1800, (y-position[i, 1]) / 1800
                f[10] = (distance / 5 + (3 if k == 3 else 5 + (c != current_channel[i]))) / 1000
                f[11] = r / 1800
                f[12] = np.log1p(areas[i, c]) / 17
                f[13] = np.sum(coverage[i, c]) / 7
                f[14] = min(counts[i, c], 30) / 30
                f[15] = state == 1 and r <= 20 - 1e-6
                f[16], f[17] = np.cos(last_angle[i, c]), np.sin(last_angle[i, c])
                f[18] = last_result[i, c] == 0
                f[19] = last_result[i, c] == 4
                mask[i, a] = valid
            # Same-channel duplicate candidate positions would bias sampling.
            for k in range(3):
                a = c * 4 + k
                if mask[i, a]:
                    for j in range(k):
                        b = c * 4 + j
                        if mask[i, b] and np.sum((targets[i, a] - targets[i, b])**2) < 1e-10:
                            mask[i, a] = False


class BeliefBatch:
    def __init__(self, n, channels):
        self.status = np.zeros((n, channels), np.int8)
        self.polys = np.zeros((n, channels, MAX_VERTICES, 2))
        self.sizes = np.zeros((n, channels), np.int32)
        self.centers = np.zeros((n, channels, 2))
        self.radii = np.full((n, channels), 1800.)
        self.areas = np.full((n, channels), np.pi * 1800**2)
        self.last_angle = np.zeros((n, channels))
        self.coverage = np.zeros((n, channels, 7), bool)
        self.counts = np.zeros((n, channels), np.int32)
        self.last_result = np.full((n, channels), -1, np.int8)
        self.last_pos = np.full((n, channels, 2), 1e10)
        self.fail_pos = np.full((n, channels, 2), 1e10)
        self.features = np.zeros((n, channels * 4, 20), np.float32)
        self.global_features = np.zeros((n, 8), np.float32)
        self.targets = np.zeros((n, channels * 4, 2))
        self.mask = np.zeros((n, channels * 4), bool)

    def reset(self, ids):
        self.status[ids] = self.sizes[ids] = self.counts[ids] = 0
        self.centers[ids] = self.last_angle[ids] = 0
        self.radii[ids] = 1800
        self.areas[ids] = np.pi * 1800**2
        self.coverage[ids] = False
        self.last_result[ids] = -1
        self.last_pos[ids] = self.fail_pos[ids] = 1e10

    def update(self, positions, channels, outcome, angle):
        update_geometry(self.polys, self.sizes, self.status, self.centers, self.radii,
                        self.areas, self.last_angle, self.coverage, self.counts,
                        self.last_result, self.last_pos, self.fail_pos,
                        positions, channels, outcome, angle)

    def observe(self, position, channel, time, steps, max_steps):
        build_candidates(self.status, self.centers, self.radii, self.areas,
                         self.coverage, self.counts, self.last_angle, self.last_result,
                         self.last_pos, self.fail_pos, position, channel, time, steps,
                         max_steps, self.features, self.global_features, self.targets, self.mask)
        return {"candidates": self.features, "global": self.global_features, "mask": self.mask}
