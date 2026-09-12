"""Fixed-R interval certificates and optional two-sided probe candidates.

Inspired by brl/bilateral.py in shxjm-B-RL at b15bfaae4bc67e9a4bcb0974a89de3d5e4a2610c.
This implementation derives every bound from ordinary public channel history;
there is no forced macro, simulator reference, or probability-based exclusion.
See Q4_GitHub部件改进实验_20260912.md for the distance/half-plane proof.
"""
from dataclasses import dataclass
import math

import numpy as np
import shapely
from shapely.geometry import Polygon

from .core import Channel
from .localization import guaranteed_clear, local_attempts
from .posterior import QuadratureError
from .sector import ProbePolicy
from .shared import GeometryError, distance


K = math.tan(math.radians(1.01))  # Includes the 0.005 degree rounding allowance.
SLACK_M = 1e-4


def frame(channel):
    first = channel.directions[0]
    angle = math.radians(first.bearing)
    basis = np.array([[math.cos(angle), math.sin(angle)],
                      [-math.sin(angle), math.cos(angle)]])
    return np.asarray(first.action.position), basis


def negative_pair_bound(channel):
    """Opposing no-signal sites imply x <= bound in the first-bearing frame.

For q=(a,b), x>=max(a,(a*a+b*b)/(2*(a-K*abs(b)))) implies
|G-q| <= |G-origin| for every |y|<=K*x. Two sites strictly above/below
the whole bearing cone have a front-side witness on their joining segment.
Since the origin was received with this same R, both sites cannot be negative
when x is beyond both thresholds. Outward slack protects all comparisons.
"""
    if not channel.directions:
        return None
    origin, basis = frame(channel)
    best = {}
    for obs in channel.history:
        if obs.result != 'no_signal':
            continue
        a, b = (np.asarray(obs.action.position)-origin) @ basis.T
        denominator = a-K*abs(b)
        if a <= SLACK_M or denominator <= SLACK_M or abs(b) <= K*a+SLACK_M:
            continue
        threshold = max(float(a), float((a*a+b*b)/(2*denominator)))+SLACK_M
        side = 1 if b > 0 else -1
        if side not in best or threshold < best[side][0]:
            best[side] = (threshold, obs.action.position)
    if len(best) != 2:
        return None
    return {'upper_m': max(best[-1][0], best[1][0]),
            'positive_origin': tuple(map(float, origin)),
            'negative_pair': (best[-1][1], best[1][1]),
            'fixed_same_source_radius_required': True, 'probability_used': False}


@dataclass
class BilateralChannel(Channel):
    bilateral_updates: int = 0
    bilateral_certificate: dict | None = None

    def update(self, obs):
        if obs in self.history or self.status in ('cleared', 'absent_certified'):
            return super().update(obs)
        candidate = self.clone()
        Channel.update(candidate, obs)
        if candidate.status == 'detected' and obs.result in ('direction', 'no_signal'):
            proof = negative_pair_bound(candidate)
            if proof:
                origin, basis = frame(candidate)
                coordinates = (shapely.get_coordinates(candidate.region)-origin) @ basis.T
                lo, hi = coordinates.min(axis=0), coordinates.max(axis=0)
                bound = proof['upper_m']
                if bound < hi[0]:
                    if bound <= lo[0]:
                        raise GeometryError('Bilateral certificate conflicts with detected support')
                    # All edges except the certified x bound lie outside the
                    # existing outer support; never discard a small component.
                    local = np.array([[lo[0]-1, lo[1]-1], [bound, lo[1]-1],
                                      [bound, hi[1]+1], [lo[0]-1, hi[1]+1]])
                    clipped = candidate.region.intersection(Polygon(local @ basis+origin))
                    if clipped.is_empty or not clipped.is_valid:
                        raise GeometryError('Invalid bilateral support intersection')
                    candidate.region = clipped
                    candidate._summary = None
                    candidate.bilateral_updates += 1
                    candidate.bilateral_certificate = proof
        self.__dict__.update(candidate.__dict__)


class BilateralProbePolicy(ProbePolicy):
    def __init__(self, config=None, add_candidates=True):
        super().__init__(config)
        self.add_candidates = add_candidates
        self.implementation = ('q4_github_bilateral_probes_v1' if add_candidates
                               else 'q4_github_bilateral_bound_v1')
        self.counters['bilateral_probe_candidates'] = 0

    def record(self, b, action, reason, **extra):
        extra['bilateral_certificate_updates'] = sum(c.bilateral_updates for c in b.channels.values())
        extra['channel_bilateral_certificate'] = b.channels[action.channel].bilateral_certificate
        return super().record(b, action, reason, **extra)

    def local_choices(self, b, c):
        choices, mean = super().local_choices(b, c)
        channel = b.channels[c]
        if (not self.add_candidates or guaranteed_clear(b, c)
                or local_attempts(channel) >= self.config.local_limit):
            return choices, mean
        origin, basis = frame(channel)
        vertices = (shapely.get_coordinates(channel.region)-origin) @ basis.T
        low, high = float(vertices[:, 0].min()), float(vertices[:, 0].max())
        local_points = []
        if high-low > 24:
            mid = (max(0., low)+high)/2
            if mid >= 12:
                h = K*mid+5
                local_points += [(mid, h), (mid, -h)]
        # Reconstruct unfinished pairs from accepted feedback, so replanning
        # cannot restart a macro or depend on a private continuation cache.
        mirrors = []
        for obs in channel.history:
            if obs.result == 'no_signal':
                x, y = (np.asarray(obs.action.position)-origin) @ basis.T
                if 12 <= x <= high and abs(y) > K*x+1:
                    mirrors.append((float(x), -math.copysign(max(abs(y), K*x+5), y)))
        mirrors.sort(key=lambda q: distance(b.position, tuple(np.asarray(q) @ basis+origin)))
        local_points += mirrors[:2]
        existing = {a.position for a, _, _ in choices}
        points = [tuple(map(float, np.asarray(q) @ basis+origin)) for q in local_points]
        points = [p for p in dict.fromkeys(points) if p not in existing and
                  all(distance(p, o.action.position) >= 10 for o in channel.history
                      if o.action.kind == 'measure')]
        self.counters['bilateral_probe_candidates'] += len(points)
        try:
            coarse = sorted((self.probe(b, c, p, 10.)[1], i) for i, p in enumerate(points))
            for _, i in coarse[:2]:
                action, cost, detail = self.probe(b, c, points[i], self.config.bearing_bin)
                choices.append((action, cost, {**detail, 'bilateral_template': True,
                                               'original_choices_retained': True}))
        except QuadratureError:
            pass
        return choices, mean
