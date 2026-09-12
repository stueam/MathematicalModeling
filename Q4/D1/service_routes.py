"""Fixed-point open routes; Q3 algorithm3/open_routes.py (242b878) DP idea.

Exactness applies only to the supplied, fixed geometric task points, not to
the adaptive search-and-clear problem or the service regions as a whole.
"""
import numpy as np


def exact_open_route(start, points):
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(xy)
    if not n:
        return []
    distance = np.linalg.norm(xy[:, None]-xy[None, :], axis=2)
    # dp[mask,i]: length from i through every point in mask, ending anywhere.
    dp = np.full((1 << n, n), np.inf)
    dp[0] = 0.
    masks = np.arange(1, 1 << n)
    counts = np.array([int(mask).bit_count() for mask in masks])
    for size in range(1, n+1):
        layer = masks[counts == size]
        for j in range(n):
            selected = layer[(layer & (1 << j)) != 0]
            rest = selected ^ (1 << j)
            dp[selected] = np.minimum(dp[selected], dp[rest, j, None]+distance[j][None, :])
    mask, current, order = (1 << n)-1, np.asarray(start), []
    while mask:
        choices = [j for j in range(n) if mask & (1 << j)]
        nxt = min(choices, key=lambda j: (float(np.linalg.norm(current-xy[j]))+dp[mask ^ (1 << j), j], j))
        order.append(nxt)
        mask ^= 1 << nxt
        current = xy[nxt]
    return order
