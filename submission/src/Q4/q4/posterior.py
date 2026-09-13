"""Position-cell quadrature; exact R/heading interval marginal at each point.

Uniform area, uniform fixed R, uniform heading and independent errors at distinct
sites are WORKING PRIORS only. Probability zero is never an absence certificate.
"""

from dataclasses import dataclass
import math

import numpy as np
import shapely


class QuadratureError(RuntimeError):
    pass


def illuminated_fraction(alpha, left, right):
    """Fraction of each non-wrapping heading interval in a closed semicircle."""
    total = np.zeros_like(left)
    for shift in (-math.tau, 0.0, math.tau):
        lo, hi = alpha[:, None] - math.pi / 2 + shift, alpha[:, None] + math.pi / 2 + shift
        total += np.maximum(0.0, np.minimum(right, hi) - np.maximum(left, lo))
    return np.clip(total / np.maximum(right - left, 1e-300), 0.0, 1.0)


def history_atoms(history, xy, directional_prior=0.5):
    """Unnormalized atoms = (heading interval, uniform conditional radius).

    Column 0 is omnidirectional. Other columns cover the full heading circle.
    Bearings do not depend on emitter heading; rounding likelihood is factored.
    """
    n = len(xy)
    measure = [o for o in history if o.action.kind == 'measure']
    valid = np.linalg.norm(xy, axis=1) <= 1800
    angular = np.ones(n)
    for o in history:
        dxy = xy - o.action.position
        distance = np.linalg.norm(dxy, axis=1)
        if o.result == 'direction':
            valid &= distance > 5
            phi = np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0]))
            diff = (o.bearing - phi + 180) % 360 - 180
            angular *= np.maximum(0.0, np.minimum(1.0, diff + 0.005) - np.maximum(-1.0, diff - 0.005)) / 0.01
        elif o.result == 'near':
            valid &= distance <= 5
        elif o.result == 'success':
            valid &= distance <= 20
        elif o.result == 'no_target_in_range':
            valid &= distance > 20
    if measure:
        delta = np.asarray([o.action.position for o in measure])[None, :, :] - xy[:, None, :]
        distances = np.linalg.norm(delta, axis=2)
        bearings = np.arctan2(delta[:, :, 1], delta[:, :, 0]) % math.tau
        edges = np.sort(
            np.concatenate(
                ((bearings - math.pi / 2) % math.tau, (bearings + math.pi / 2) % math.tau), axis=1
            ),
            axis=1,
        )
        edges = np.column_stack((np.zeros(n), edges, np.full(n, math.tau)))
        left, right = edges[:, :-1], edges[:, 1:]
        mid = (left + right) / 2
        lit = np.cos(mid[:, :, None] - bearings[:, None, :]) >= 0
        lit |= distances[:, None, :] == 0
        positive = np.array([o.result != 'no_signal' for o in measure])
        lower = (
            np.maximum(1000.0, np.max(distances[:, positive], axis=1))
            if positive.any()
            else np.full(n, 1000.0)
        )
        allowed = np.all(lit[:, :, positive], axis=2)
        upper = (
            np.minimum(
                1500.0, np.min(np.where(lit[:, :, ~positive], distances[:, None, ~positive], np.inf), axis=2)
            )
            if (~positive).any()
            else np.full_like(left, 1500.0)
        )
        omni_upper = (
            np.minimum(1500.0, np.min(distances[:, ~positive], axis=1))
            if (~positive).any()
            else np.full(n, 1500.0)
        )
    else:
        left, right = np.zeros((n, 1)), np.full((n, 1), math.tau)
        allowed = np.ones((n, 1), dtype=bool)
        lower, omni_upper = np.full(n, 1000.0), np.full(n, 1500.0)
        upper = np.full_like(left, 1500.0)
    atom_hi = np.column_stack((omni_upper, upper))
    fractions = np.column_stack(
        (np.full(n, 1 - directional_prior), directional_prior * (right - left) / math.tau * allowed)
    )
    mass = np.maximum(atom_hi - lower[:, None], 0.0) / 500 * fractions
    mass *= (valid * angular)[:, None]
    return mass, lower, atom_hi, left, right


@dataclass
class Posterior:
    xy: np.ndarray
    atoms: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    left: np.ndarray
    right: np.ndarray
    evidence: float

    @property
    def weights(self):
        return self.atoms.sum(axis=1)

    @property
    def mean(self):
        return self.weights @ self.xy

    def reception(self, point):
        delta = np.asarray(point) - self.xy
        d = np.linalg.norm(delta, axis=1)
        alpha = np.arctan2(delta[:, 1], delta[:, 0]) % math.tau
        heading = illuminated_fraction(alpha, self.left, self.right)
        heading[d == 0] = 1
        heading = np.column_stack((np.ones(len(d)), heading))
        span = self.upper - self.lower[:, None]
        radial = np.clip(
            (self.upper - np.maximum(self.lower[:, None], d[:, None])) / np.maximum(span, 1e-300), 0.0, 1.0
        )
        received_mass = (self.atoms * heading * radial).sum(axis=1)
        return d, received_mass

    def clear_probability(self, point):
        return float(self.weights @ (np.linalg.norm(self.xy - point, axis=1) <= 20))

    def clear_point(self):
        ids = np.argsort(-self.weights)[:12]
        points = np.vstack((self.mean, self.xy[ids]))
        scores = (np.linalg.norm(points[:, None] - self.xy[None], axis=2) <= 20) @ self.weights
        return tuple(map(float, points[int(np.argmax(scores))]))


class Model:
    def __init__(self, resolution=16, directional_prior=0.5, existence_prior=0.65):
        if not 8 <= resolution <= 64 or not 0 < directional_prior < 1 or not 0 < existence_prior < 1:
            raise ValueError('Invalid quadrature/prior configuration')
        self.resolution, self.directional_prior, self.existence_prior = (
            resolution,
            directional_prior,
            existence_prior,
        )
        self.cache = {}

    def posterior(self, channel):
        key = tuple(channel.history)
        # Channel numbers have no role under the exchangeable working prior.
        key = tuple((o.action.kind, o.action.position, o.result, o.bearing) for o in key)
        if key in self.cache:
            return self.cache[key]
        rectangle = shapely.minimum_rotated_rectangle(channel.region)
        if rectangle.geom_type != 'Polygon' or rectangle.area <= 0:
            raise QuadratureError('Degenerate support needs geometric completion')
        corners = shapely.get_coordinates(rectangle)
        basis = np.asarray([corners[1] - corners[0], corners[3] - corners[0]])
        for resolution in (self.resolution, 2 * self.resolution):
            u, v = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing='ij')
            uv = np.column_stack((u.ravel(), v.ravel())) / resolution
            offsets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) / resolution
            vertices = corners[0] + (uv[:, None] + offsets) @ basis
            cells = shapely.intersection(shapely.polygons(vertices), channel.region)
            area = shapely.area(cells)
            keep = area > 0
            xy = shapely.get_coordinates(shapely.point_on_surface(cells[keep]))
            raw, lower, upper, left, right = history_atoms(channel.history, xy, self.directional_prior)
            raw *= area[keep, None]
            good = raw.sum(axis=1) > 0
            total = float(raw.sum())
            if total > 0:
                post = Posterior(
                    xy[good],
                    raw[good] / total,
                    lower[good],
                    upper[good],
                    left[good],
                    right[good],
                    total / (math.pi * 1800**2),
                )
                if len(self.cache) >= 32:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[key] = post
                return post
        raise QuadratureError('No quadrature mass; NOT an absence certificate')

    def existence(self, belief):
        unknown = [c for c, p in belief.channels.items() if p.status == 'unresolved']
        if len(belief.known) == 16:
            return dict.fromkeys(unknown, 0.0)
        q = []
        for c in unknown:
            e = min(1.0, self.posterior(belief.channels[c]).evidence)
            p = self.existence_prior
            q.append(p * e / (1 - p + p * e))
        return dict(zip(unknown, count_marginals(q, len(belief.known))))


def count_marginals(probabilities, known):
    low, high = max(0, 10 - known), min(len(probabilities), 16 - known)
    if low > high:
        raise QuadratureError('Inconsistent source count')

    def distribution(q):
        result = np.ones(1)
        for p in q:
            result = np.convolve(result, [1 - p, p])
        return result

    q = list(probabilities)
    denominator = float(distribution(q)[low : high + 1].sum())
    if denominator <= 0:
        raise QuadratureError('Zero count-conditioning mass')
    return np.asarray(
        [
            p * distribution(q[:i] + q[i + 1 :])[max(0, low - 1) : high].sum() / denominator
            for i, p in enumerate(q)
        ]
    )


def finish_proxy(point, xy, weights):
    """Explicit heuristic local completion cost, not a bound or full rollout.

    Mean approach + attempted clear + failure recovery (two measurements and
    distance-dependent detour). Separately audited against actual continuations.
    """
    mean = weights @ xy
    d = np.linalg.norm(xy - mean, axis=1)
    hit = d <= 20
    p = float(weights @ hit)
    recovery = float(weights @ np.where(hit, 0.0, d))
    return float(np.linalg.norm(mean - point)) / 5 + 3 + 2 * p + 12 * (1 - p) + 2 * recovery / 5


def expected_measure(post, point, bin_deg=2.0):
    """All received bearing bins, near AND no_signal, using the same R/heading."""
    if bin_deg <= 0 or abs(round(360 / bin_deg) * bin_deg - 360) > 1e-9:
        raise ValueError('Angle-bin width must divide 360')
    d, received = post.reception(point)
    weights = post.weights
    near_mass = received * (d <= 5)
    missed = np.maximum(0.0, weights - received)
    p_near, p_none = float(near_mass.sum()), float(missed.sum())
    future = 5 * p_near
    if p_none > 1e-15:
        future += p_none * finish_proxy(point, post.xy, missed / p_none)
    delta = post.xy - point
    angle = np.degrees(np.arctan2(delta[:, 1], delta[:, 0])) % 360
    first = np.floor((angle - 1) / bin_deg).astype(int)
    bins = first[:, None] + np.arange(math.ceil(2 / bin_deg) + 1)
    width = (
        np.maximum(
            0.0,
            np.minimum(angle[:, None] + 1, (bins + 1) * bin_deg)
            - np.maximum(angle[:, None] - 1, bins * bin_deg),
        )
        / 2
    )
    masses = width * (received * (d > 5))[:, None]
    bins %= round(360 / bin_deg)
    for k in np.unique(bins[masses > 0]):
        w = np.sum(np.where(bins == k, masses, 0.0), axis=1)
        p = float(w.sum())
        if p > 1e-15:
            future += p * finish_proxy(point, post.xy, w / p)
    return future, {
        'near_probability': p_near,
        'no_signal_probability': p_none,
        'direction_probability': float((received * (d > 5)).sum()),
    }
