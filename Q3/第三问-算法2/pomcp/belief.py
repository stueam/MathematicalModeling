"""Particle posterior for decisions; outer geometry remains the only certificate.

Area-uniform locations, uniform fixed R and independent uniform bounded errors
at distinct sites are working priors, not additional problem rules.
"""
from dataclasses import dataclass
import copy
import hashlib
import numpy as np

from q3.core import Belief, point_key
from q3.sampling import WorldSampler, likelihood


@dataclass(frozen=True)
class Particles:
    xyz: np.ndarray
    weights: np.ndarray
    history: tuple
    ess_before: float

    @property
    def ess(self):
        return float(1 / (self.weights @ self.weights))


def systematic(weights, n, rng):
    return np.minimum(np.searchsorted(np.cumsum(weights),
                      (np.arange(n) + rng.random()) / n), len(weights)-1)


class ParticleBelief(Belief):
    def __init__(self, particles=3000, seed=2026, **kwargs):
        super().__init__(**kwargs)
        self.particle_count, self.seed = particles, seed
        self.clouds = {}

    def clone(self):
        b = super().clone()
        b.clouds = self.clouds.copy()  # Clouds are immutable, including arrays.
        return b

    def cloud(self, c):
        channel = self.channels[c]
        history = tuple(channel.history)
        old = self.clouds.get(c)
        if old is not None and old.history == history:
            return old
        token = repr((self.seed, c, history)).encode()
        seed = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), 'big')
        rng = np.random.default_rng(seed)
        ess = 0.
        xyz = weights = None
        if old is not None:
            # Conditional likelihood ratio: R was sampled once and remains fixed.
            marginal, lo, hi = likelihood(channel, old.xyz[:, :2])
            before = copy.copy(channel)
            before.history = list(old.history)
            prior, plo, phi = likelihood(before, old.xyz[:, :2])
            angular = marginal / np.maximum(hi-lo, 1e-300)
            old_angular = prior / np.maximum(phi-plo, 1e-300)
            weights = old.weights * angular / np.maximum(old_angular, 1e-300)
            weights *= (old.xyz[:, 2] >= lo) & (old.xyz[:, 2] < hi)
            if weights.sum() > 0:
                weights /= weights.sum()
                ess = float(1/(weights @ weights))
                xyz = old.xyz
        if xyz is None or ess < .55*self.particle_count:
            # Rejuvenation from the COMPLETE-history posterior, instead of
            # unconstrained Gaussian jitter or declaring a vanished cloud absent.
            sampler = WorldSampler(seed, pool_size=max(512, self.particle_count))
            xy, w, lo, hi, _ = sampler.pool(c, channel)
            ids = systematic(w, self.particle_count, rng)
            radii = rng.uniform(lo[ids], hi[ids])
            xyz = np.column_stack((xy[ids], radii))
            weights = np.full(self.particle_count, 1/self.particle_count)
        elif np.any(weights == 0):
            ids = systematic(weights, self.particle_count, rng)
            xyz = xyz[ids].copy()
            weights = np.full(self.particle_count, 1/self.particle_count)
        xyz.setflags(write=False)
        weights.setflags(write=False)
        result = Particles(xyz, weights, history, ess)
        self.clouds[c] = result
        return result

    def clear_point(self, c):
        p = self.cloud(c)
        cached = getattr(self.channels[c], '_particle_clear', None)
        if cached is not None and cached[0] == p.history:
            return cached[1]
        # Weighted disk coverage maximization over a bounded candidate set.
        # This approximates a continuous optimum, never a clearance certificate.
        ids = systematic(p.weights, min(96, len(p.weights)), np.random.default_rng(0))
        xy = p.xyz[:, :2]
        mean = p.weights @ xy
        candidates = np.vstack((mean, xy[ids]))
        for _ in range(2):
            masks = np.linalg.norm(candidates[:, None, :]-xy[None, :, :], axis=2) <= 20
            prob = masks @ p.weights
            best = int(np.argmax(prob))
            local = p.weights*masks[best]
            if local.sum() > 0:
                candidates = np.vstack((candidates, (local @ xy)/local.sum()))
        masks = np.linalg.norm(candidates[:, None, :]-xy[None, :, :], axis=2) <= 20
        probabilities = masks @ p.weights
        best = int(np.argmax(probabilities))
        result = (tuple(map(float, candidates[best])), float(probabilities[best]))
        self.channels[c]._particle_clear = (p.history, result)
        return result

    def status(self, c):
        p = self.channels[c]
        names = {'unresolved': 'UNKNOWN', 'cleared': 'CLEARED', 'absent_certified': 'ABSENT'}
        if p.status != 'detected':
            return names[p.status]
        return 'LOCATED' if self.clear_point(c)[1] >= .95 else 'DETECTED'

    def apply(self, action, response, request_id):
        changed = super().apply(action, response, request_id)
        if changed:
            known = sum(p.status in ('detected', 'cleared') for p in self.channels.values())
            if known > 16:
                raise ValueError('Observations violate source count upper bound')
            if known == 16:
                for p in self.channels.values():
                    if p.status == 'unresolved':
                        p.status = 'absent_certified'
        return changed

    def diagnostics(self):
        result = {}
        for c, channel in self.channels.items():
            label = {'unresolved': 'UNKNOWN', 'detected': 'DETECTED', 'cleared': 'CLEARED',
                     'absent_certified': 'ABSENT'}[channel.status]
            row = {'status': label, 'outer_area_m2': channel.region.area}
            if channel.status == 'detected':
                try:
                    cloud = self.cloud(c)
                    row.update(status=self.status(c), ess=cloud.ess,
                               ess_before_rejuvenation=cloud.ess_before,
                               p_clear=self.clear_point(c)[1], E_m=channel.summary()[1])
                except RuntimeError as exc:
                    row['particle_error'] = str(exc)
            result[c] = row
        return result
