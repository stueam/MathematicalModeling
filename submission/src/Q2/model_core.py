"""Paper Q2: fixed R0, full-region expected enclosing radius."""

import numpy as np
from scipy.spatial.distance import pdist
import shapely
from shapely.geometry import Point, Polygon


def sector(p, angle, half, low, high, segments=128):
    t = np.linspace(angle - half, angle + half, segments + 1)
    u = np.column_stack((np.cos(t), np.sin(t)))
    return Polygon(np.vstack((p + high * u, p + low * u[::-1])))


def wedge(p, angle, half):
    # This triangle extends beyond every source location in the finite domain.
    t = np.array([angle - half, angle + half])
    return Polygon(np.vstack((p, p + 20000 * np.column_stack((np.cos(t), np.sin(t))))))


def metrics(region):
    if region.is_empty:
        return 0.0, 0.0
    hull = region.convex_hull
    if hull.geom_type == 'Point':
        return 0.0, 0.0
    pts = np.asarray(hull.exterior.coords)[:-1] if hull.geom_type == 'Polygon' else np.asarray(hull.coords)
    return float(shapely.minimum_bounding_radius(hull)), float(pdist(pts).max())


class Design:
    def __init__(self, p1=(-700.0, -250.0), theta=25.0, reception=1200.0, nr=80, na=16, arc=128):
        self.p1 = np.asarray(p1, float)
        self.theta, self.eps = np.deg2rad([theta, 1.0])
        self.reception, self.arc = reception, arc
        self.domain = Point(0, 0).buffer(1800, quad_segs=arc)
        self.p1_region = sector(self.p1, self.theta, self.eps, 5.0, reception).intersection(self.domain)
        if self.p1_region.area <= 0:
            raise ValueError('First observation has zero feasible area')
        radial = np.sqrt(np.linspace(25.0, reception**2, nr + 1))
        angular = np.linspace(self.theta - self.eps, self.theta + self.eps, na + 1)
        points, weights = [], []
        for i in range(nr):
            for j in range(na):
                cell = sector(
                    self.p1,
                    (angular[j] + angular[j + 1]) / 2,
                    (angular[j + 1] - angular[j]) / 2,
                    radial[i],
                    radial[i + 1],
                    max(1, 128 // na),
                ).intersection(self.domain)
                if cell.area <= 1e-12:
                    continue
                c = cell.centroid
                if not cell.covers(c):
                    c = cell.representative_point()
                points.append([c.x, c.y])
                weights.append(cell.area)
        self.points = np.array(points)
        self.weights = np.array(weights) / sum(weights)
        self.area_partition_relative_error = abs(sum(weights) / self.p1_region.area - 1)
        self.initial_radius, self.initial_diameter = metrics(self.p1_region)

    def probabilities(self, q, width):
        nbin = int(round(360 / width))
        h = 2 * np.pi / nbin
        vec = self.points - q
        distance = np.linalg.norm(vec, axis=1)
        near = distance <= 5
        normal = (distance > 5) & (distance <= self.reception)
        none = distance > self.reception
        phi = np.mod(np.arctan2(vec[normal, 1], vec[normal, 0]), 2 * np.pi)
        left, right = phi - self.eps, phi + self.eps
        idx = np.floor(left / h).astype(int)[:, None] + np.arange(int(np.ceil(2 * self.eps / h)) + 2)
        overlap = np.maximum(
            0.0, np.minimum(right[:, None], (idx + 1) * h) - np.maximum(left[:, None], idx * h)
        )
        mass = overlap / (2 * self.eps) * self.weights[normal, None]
        bins = np.bincount((idx % nbin).ravel(), weights=mass.ravel(), minlength=nbin)
        pn, p0 = float(self.weights[near].sum()), float(self.weights[none].sum())
        assert abs(bins.sum() + pn + p0 - 1) < 1e-10
        return bins, pn, p0, h

    def evaluate(self, q, width=0.5):
        q = np.asarray(q, float)
        if np.linalg.norm(q - self.p1) < 1e-9:
            return dict(
                x=float(q[0]),
                y=float(q[1]),
                J=self.initial_radius,
                ED=self.initial_diameter,
                p_near=0.0,
                p_none=0.0,
                probability_sum=1.0,
            )
        probs, pn, p0, h = self.probabilities(q, width)
        receiving = Point(*q).buffer(self.reception, quad_segs=self.arc)
        close = Point(*q).buffer(5, quad_segs=self.arc)
        normal = self.p1_region.intersection(receiving).difference(close)
        rn, dn = metrics(self.p1_region.intersection(close)) if pn else (0.0, 0.0)
        r0, d0 = metrics(self.p1_region.difference(receiving)) if p0 else (0.0, 0.0)
        jr, jd = pn * rn + p0 * r0, pn * dn + p0 * d0
        for b in np.flatnonzero(probs > 0):
            region = normal.intersection(wedge(q, (b + 0.5) * h, self.eps + h / 2))
            if region.is_empty:
                raise ArithmeticError('Positive probability with empty support; refine geometry')
            r, d = metrics(region)
            jr += probs[b] * r
            jd += probs[b] * d
        return dict(
            x=float(q[0]),
            y=float(q[1]),
            J=float(jr),
            ED=float(jd),
            p_near=pn,
            p_none=p0,
            probability_sum=float(probs.sum() + pn + p0),
        )
