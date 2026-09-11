"""Approximate conditional worlds, distinct from conservative certificates.

Uniform area/R working prior; rounded-angle interval likelihood; existence is
conditioned jointly on total source count. Finite proposal pools are explicitly
an approximation, not an exact posterior or a nonexistence proof.
"""
import math
import time

import numpy as np
import shapely

from .core import point_key
from .simulator import Source, World


class SamplingError(RuntimeError):
    pass


def uniform_domain(rng, size):
    r = 1800*np.sqrt(rng.random(size))
    a = rng.uniform(0, 2*math.pi, size)
    return np.column_stack((r*np.cos(a), r*np.sin(a)))


def likelihood(channel, xy):
    """Marginalize R analytically; include rounding likelihood at error edges."""
    n = len(xy)
    lo, hi = np.full(n, 1000.), np.full(n, 1500.)
    angular = np.ones(n)
    valid = np.linalg.norm(xy, axis=1) <= 1800
    for obs in channel.history:
        delta = xy - obs.action.position
        d = np.linalg.norm(delta, axis=1)
        if obs.result == 'direction':
            lo = np.maximum(lo, d)
            valid &= d > 5
            phi = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
            diff = (obs.bearing - phi + 180) % 360 - 180
            # P(round(phi+epsilon,2)=bearing), epsilon~U[-1,1].
            overlap = np.maximum(0., np.minimum(1., diff+.005) - np.maximum(-1., diff-.005))
            angular *= overlap/.01  # Constant .005 per direction cancels in location weights.
        elif obs.result == 'no_signal':
            hi = np.minimum(hi, d)
        elif obs.result == 'near':
            valid &= d <= 5
        elif obs.result == 'no_target_in_range':
            valid &= d > 20
        elif obs.result == 'success':
            valid &= d <= 20
    weights = np.maximum(hi-lo, 0)/500 * angular * valid
    return weights, lo, hi


def conditional_existence(q, known, rng):
    """Bernoulli probabilities conditioned on 10 <= known+sum(z) <= 16."""
    n = len(q)
    low, high = max(0, 10-known), min(n, 16-known)
    if low > high:
        raise SamplingError('Count evidence contradicts 10..16')
    dp = np.zeros((n+1, n+1))
    dp[n, 0] = 1
    for i in range(n-1, -1, -1):
        dp[i] = (1-q[i])*dp[i+1]
        dp[i, 1:] += q[i]*dp[i+1, :-1]
    mass = dp[0, low:high+1]
    if mass.sum() <= 0:
        raise SamplingError('No admissible count mass')
    k = int(rng.choice(np.arange(low, high+1), p=mass/mass.sum()))
    result = []
    for i in range(n):
        yes = q[i]*dp[i+1, k-1] if k else 0.
        no = (1-q[i])*dp[i+1, k]
        z = bool(rng.random() < yes/(yes+no))
        result.append(z)
        k -= int(z)
    return result


class WorldSampler:
    def __init__(self, seed=2026, pool_size=512, evidence_size=4096, p0=.65):
        self.rng = np.random.default_rng(seed)
        self.pool_size, self.evidence_size, self.p0 = pool_size, evidence_size, p0
        self.cache = {}

    def pool(self, c, channel, deadline=math.inf):
        key = (c, tuple(channel.history))
        if key in self.cache:
            return self.cache[key]
        if time.monotonic() >= deadline:
            raise TimeoutError('Sampling budget exhausted')
        evidence = 1.
        if channel.status == 'unresolved':
            pilot = uniform_domain(self.rng, self.evidence_size)
            w, _, _ = likelihood(channel, pilot)
            evidence = float(w.mean())
        # Uniform rectangle proposals remain area-uniform after conditioning.
        rectangle = shapely.minimum_rotated_rectangle(channel.region)
        coords = shapely.get_coordinates(rectangle)
        if rectangle.geom_type != 'Polygon' or rectangle.area <= 0:
            raise SamplingError('Zero-area posterior cannot be sampled by area proposals')
        origin, edge1, edge2 = coords[0], coords[1]-coords[0], coords[3]-coords[0]
        chunks, ws, lows, highs = [], [], [], []
        count = 0
        for _ in range(24):
            if time.monotonic() >= deadline:
                raise TimeoutError('Sampling budget exhausted')
            uv = self.rng.random((max(self.pool_size*2, 256), 2))
            xy = origin + uv[:, :1]*edge1 + uv[:, 1:]*edge2
            w, lo, hi = likelihood(channel, xy)
            good = w > 0
            if good.any():
                chunks.append(xy[good]); ws.append(w[good]); lows.append(lo[good]); highs.append(hi[good])
                count += int(good.sum())
            if count >= self.pool_size:
                break
        if count == 0:
            raise SamplingError('No compatible samples; geometry is retained, baseline must take over')
        xy, w, lo, hi = map(np.concatenate, (chunks, ws, lows, highs))
        # A zero Monte Carlo evidence estimate is not a certificate of absence.
        evidence = max(evidence, 1e-12)
        result = (xy, w/w.sum(), lo, hi, evidence)
        # Keep only current revision per channel (bounded real-run memory).
        self.cache = {k: v for k, v in self.cache.items() if k[0] != c}
        self.cache[key] = result
        return result

    def sample(self, b, count, deadline=math.inf):
        pools, q, unknown, known = {}, [], [], []
        for c, channel in b.channels.items():
            if channel.status == 'absent_certified':
                continue
            pools[c] = self.pool(c, channel, deadline)
            if channel.status in ('detected', 'cleared'):
                known.append(c)
            else:
                unknown.append(c)
                evidence = pools[c][-1]
                q.append(self.p0*evidence/(1-self.p0+self.p0*evidence))
        bearings = {(c, point_key(o.action.position)): o.bearing
                    for c, channel in b.channels.items() for o in channel.history if o.result == 'direction'}
        worlds = []
        for _ in range(count):
            if time.monotonic() >= deadline:
                raise TimeoutError('Sampling budget exhausted')
            present = known + [c for c, z in zip(unknown, conditional_existence(q, len(known), self.rng)) if z]
            sources = []
            for c in present:
                xy, weights, lo, hi, _ = pools[c]
                j = int(self.rng.choice(len(xy), p=weights))
                radius = float(self.rng.uniform(lo[j], hi[j]))
                sources.append(Source(c, tuple(map(float, xy[j])), radius))
            worlds.append(World(sources, seed=int(self.rng.integers(0, 2**62)),
                                cleared=b.cleared, known_bearings=bearings))
        return worlds
