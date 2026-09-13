"""Fixed-radius and rounded-bearing likelihood for deterministic support points."""

import numpy as np


def likelihood(channel, xy):
    """Marginalize R analytically; include rounding likelihood at error edges."""
    n = len(xy)
    lo, hi = np.full(n, 1000.0), np.full(n, 1500.0)
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
            overlap = np.maximum(0.0, np.minimum(1.0, diff + 0.005) - np.maximum(-1.0, diff - 0.005))
            angular *= overlap / 0.01  # Constant .005 per direction cancels in location weights.
        elif obs.result == 'no_signal':
            hi = np.minimum(hi, d)
        elif obs.result == 'near':
            valid &= d <= 5
        elif obs.result == 'no_target_in_range':
            valid &= d > 20
        elif obs.result == 'success':
            valid &= d <= 20
    weights = np.maximum(hi - lo, 0) / 500 * angular * valid
    return weights, lo, hi
