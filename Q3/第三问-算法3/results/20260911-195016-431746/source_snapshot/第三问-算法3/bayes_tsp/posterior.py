"""Area-cell Bayesian quadrature. No RNG, particles, worlds, or rollouts.

The continuous fixed R is integrated out analytically; position integration is
approximate. These probabilities are NEVER used as a clearance/absence proof.
"""
from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np
import shapely

from .shared import point_key


class QuadratureError(RuntimeError):
    pass


def radius_likelihood(history, xy):
    """Uniform area/R/error working prior, conditioned on rounded observations.

Angular factors are scaled by the constant probability of a central .01-degree
bin. This cancels on normalization; evidence is used only for unknown channels
(which have no positive bearings). Duplicate point observations count once.
"""
    lo, hi = np.full(len(xy), 1000.), np.full(len(xy), 1500.)
    angular = np.ones(len(xy))
    valid = np.linalg.norm(xy, axis=1) <= 1800.
    seen = set()
    for obs in history:
        key = (obs.action.kind, point_key(obs.action.position), obs.result, obs.bearing)
        if key in seen:
            continue
        seen.add(key)
        delta = xy - obs.action.position
        d = np.linalg.norm(delta, axis=1)
        if obs.result == 'direction':
            lo = np.maximum(lo, d)
            valid &= d > 5
            phi = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
            diff = (obs.bearing - phi + 180) % 360 - 180
            overlap = np.maximum(0, np.minimum(1, diff+.005)-np.maximum(-1, diff-.005))
            angular *= overlap/.01
        elif obs.result == 'no_signal':
            hi = np.minimum(hi, d)
        elif obs.result == 'near':
            valid &= d <= 5
        elif obs.result == 'no_target_in_range':
            valid &= d > 20
        elif obs.result == 'success':
            valid &= d <= 20
    return np.maximum(hi-lo, 0)/500 * angular * valid, lo, hi


@dataclass(frozen=True)
class Posterior:
    xy: np.ndarray
    weights: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    mean: np.ndarray
    evidence: float
    cells: int
    resolution: int

    def clear_probability(self, point):
        return float(self.weights @ (np.linalg.norm(self.xy-point, axis=1) <= 20))

    @cached_property
    def best_clear_point(self):
        # A finite local search, not the exact continuous maximum. Include both
        # high-mass cells and spatially spread nodes so a mean between modes is
        # not the sole attempt location.
        indices = np.unique(np.concatenate((np.argsort(-self.weights)[:16],
                                             np.linspace(0, len(self.xy)-1, min(32, len(self.xy))).astype(int))))
        points = np.vstack((self.mean, self.xy[indices]))
        probabilities = (np.linalg.norm(points[:, None, :]-self.xy[None, :, :], axis=2) <= 20) @ self.weights
        return tuple(map(float, points[int(np.argmax(probabilities))]))

    def reception(self, point):
        d = np.linalg.norm(self.xy-point, axis=1)
        probability = np.clip((self.hi-np.maximum(self.lo, d))/(self.hi-self.lo), 0, 1)
        return d, probability

    def scan_stats(self, point):
        d, reception = self.reception(point)
        return float(self.weights @ reception), float(self.weights @ (d <= 1000))


class BayesModel:
    def __init__(self, resolution=24, p0=.65):
        if not 8 <= resolution <= 96 or not 0 < p0 < 1:
            raise ValueError('resolution must be 8..96 and p0 must be in (0,1)')
        self.resolution, self.p0 = resolution, p0
        self.cache = {}

    def posterior(self, channel):
        # Channel labels are exchangeable under the working prior. Identical
        # no-signal histories can share deterministic integration, not truth.
        key = tuple((o.action.kind, o.action.position, o.result, o.bearing)
                    for o in channel.history)
        if key in self.cache:
            return self.cache[key]
        region = channel.region
        rect = shapely.minimum_rotated_rectangle(region)
        corners = shapely.get_coordinates(rect)
        if rect.geom_type != 'Polygon' or rect.area <= 0:
            raise QuadratureError('Zero-area support: retain geometry and use completion policy')
        # Align with a thin bearing intersection. Every boundary cell is clipped
        # and weighted by its retained area, even when its original center misses.
        for resolution in (self.resolution, 2*self.resolution):
            u, v = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing='ij')
            uv = np.column_stack((u.ravel(), v.ravel())) / resolution
            offsets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) / resolution
            basis = np.array([corners[1]-corners[0], corners[3]-corners[0]])
            vertices = corners[0] + (uv[:, None, :] + offsets) @ basis
            cells = shapely.intersection(shapely.polygons(vertices), region)
            areas = shapely.area(cells)
            positive = areas > 0
            cells, areas = cells[positive], areas[positive]
            xy = shapely.get_coordinates(shapely.point_on_surface(cells))
            likelihood, lo, hi = radius_likelihood(channel.history, xy)
            mass = areas * likelihood
            valid = mass > 0
            if valid.any():
                xy, lo, hi, mass = xy[valid], lo[valid], hi[valid], mass[valid]
                evidence = float(mass.sum() / (math.pi*1800**2))
                weights = mass/mass.sum()
                result = Posterior(xy, weights, lo, hi, weights @ xy,
                                   evidence, len(cells), resolution)
                if len(self.cache) >= 128:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[key] = result
                return result
        raise QuadratureError('No positive quadrature mass after refinement; NOT an absence certificate')

    def existence(self, belief):
        unknown = [c for c, p in belief.channels.items() if p.status == 'unresolved']
        known = sum(p.status in ('detected', 'cleared') for p in belief.channels.values())
        if known == 16:
            return dict.fromkeys(unknown, 0.)
        q = []
        for c in unknown:
            evidence = min(1., self.posterior(belief.channels[c]).evidence)
            q.append(self.p0*evidence/(1-self.p0+self.p0*evidence))
        return dict(zip(unknown, count_conditioned_marginals(q, known)))


def count_conditioned_marginals(q, known):
    """Exact finite Bernoulli count conditioning, with deterministic convolution."""
    q = np.asarray(q, dtype=float)
    n = len(q)
    low, high = max(0, 10-known), min(n, 16-known)
    if low > high or np.any((q < 0) | (q > 1)):
        raise QuadratureError('Inconsistent count evidence')
    total = np.array([1.])
    for p in q:
        total = np.convolve(total, [1-p, p])
    normalizer = total[low:high+1].sum()
    if normalizer <= 0:
        raise QuadratureError('Count model has zero mass; retain geometric evidence')
    result = []
    for i, p in enumerate(q):
        others = np.array([1.])
        for j, other in enumerate(q):
            if j != i:
                others = np.convolve(others, [1-other, other])
        result.append(p*others[max(0, low-1):high].sum()/normalizer)
    return np.asarray(result)


def finish_cost(point, xy, weights):
    """Seconds surrogate: approach mean, try clear, one-probe failure recovery.

Recovery is an explicit heuristic, NOT a complete simulated continuation or a
bound. Actual failures are fed back and geometry retains all remaining work.
"""
    mean = weights @ xy
    d = np.linalg.norm(xy-mean, axis=1)
    hit = d <= 20
    chance = float(weights @ hit)
    return (float(np.linalg.norm(mean-point))/5 + 3+2*chance +
            11*(1-chance) + float(weights @ np.where(hit, 0, np.maximum(d-18, 0)))/5)


def clear_finish_cost(posterior, point):
    d = np.linalg.norm(posterior.xy-point, axis=1)
    hit = d <= 20
    chance = float(posterior.weights @ hit)
    return 3+2*chance+11*(1-chance)+float(
        posterior.weights @ np.where(hit, 0, np.maximum(d-18, 0)))/5


def expected_after_measure(posterior, point, bin_width=1.):
    """Integrate near, no_signal and ALL positive coarse bearing bins.

Actual updates use the .01-degree likelihood. Prediction groups the continuous
noisy bearing into bins as a quadrature approximation, just as in Q2. No Monte
Carlo sampling and no tree/rollout. Return remaining cost excluding this measure.
"""
    if bin_width <= 0:
        raise ValueError('bearing bin width must be positive')
    nbin = round(360/bin_width)
    if abs(nbin*bin_width-360) > 1e-9:
        raise ValueError('bearing bin width must divide 360')
    xy, w = posterior.xy, posterior.weights
    d, reception = posterior.reception(point)
    near = d <= 5
    p_near = float(w @ near)
    missed = w * (1-reception) * ~near
    p_none = float(missed.sum())
    cost = 5*p_near
    if p_none > 0:
        cost += p_none*finish_cost(point, xy, missed/p_none)
    delta = xy-point
    phi = np.degrees(np.arctan2(delta[:, 1], delta[:, 0])) % 360
    lower, upper = phi-1, phi+1
    first = np.floor(lower/bin_width).astype(int)
    indices = first[:, None]+np.arange(math.ceil(2/bin_width)+1)
    overlap = np.maximum(0, np.minimum(upper[:, None], (indices+1)*bin_width)-
                         np.maximum(lower[:, None], indices*bin_width))/2
    mass = (w*reception*~near)[:, None]*overlap
    ids = (indices % nbin).ravel()
    mass = mass.ravel()
    nodes = np.repeat(xy, indices.shape[1], axis=0)
    keep = mass > 0
    ids, mass, nodes = ids[keep], mass[keep], nodes[keep]
    probability = np.bincount(ids, mass, minlength=nbin)
    active = probability > 0
    means = np.zeros((nbin, 2))
    for axis in (0, 1):
        means[active, axis] = np.bincount(ids, mass*nodes[:, axis], minlength=nbin)[active]/probability[active]
    spread = np.linalg.norm(nodes-means[ids], axis=1)
    hit_mass = np.bincount(ids, mass*(spread <= 20), minlength=nbin)
    correction = np.bincount(ids, mass*np.where(spread <= 20, 0, np.maximum(spread-18, 0)), minlength=nbin)
    costs = (probability*np.linalg.norm(means-point, axis=1)/5 + 3*probability +
             2*hit_mass + 11*(probability-hit_mass) + correction/5)
    cost += float(costs.sum())
    total = p_near+p_none+float(probability.sum())
    if abs(total-1) > 1e-8:
        raise QuadratureError(f'Predictive feedback mass does not sum to one: {total}')
    return cost, {'near': p_near, 'no_signal': p_none, 'direction': float(probability.sum()),
                  'positive_bearing_bins': int(active.sum())}
