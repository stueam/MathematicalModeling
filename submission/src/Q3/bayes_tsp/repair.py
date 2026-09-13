"""Strict feasible-segment repair for tiny numerical constraint violations."""

import numpy as np


def repair_segment(original, proposed, constraint, limit=1e-5):
    original, proposed = np.asarray(original), np.asarray(proposed)
    if not np.isfinite(proposed).all() or np.min(constraint(original)) < 0:
        return None
    margin = float(np.min(constraint(proposed)))
    if margin >= 0:
        return proposed.copy()
    if margin < -limit:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        point = original + mid * (proposed - original)
        if np.min(constraint(point)) >= 0:
            lo = mid
        else:
            hi = mid
    point = original + lo * (proposed - original)
    return point if np.min(constraint(point)) >= 0 else None
