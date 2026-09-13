"""Exact Held-Karp open TSP up to 16 nodes; heuristic for larger mixed tours."""
from functools import lru_cache

import numpy as np


def length(start, order, points):
    if not order:
        return 0.
    xy = np.asarray([start]+[points[k] for k in order])
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


@lru_cache(maxsize=4)
def _suffix_table(items):
    """dp[mask,i] = cost from i through precisely mask, ending anywhere.

    Only immutable fixed coordinates are cached, never observations or worlds.
    At 16 nodes the float64 table occupies 8 MiB.
    """
    xy = np.asarray([p for _, p in items], dtype=float)
    n = len(xy)
    d = np.linalg.norm(xy[:, None]-xy[None, :], axis=2)
    dp = np.full((1 << n, n), np.inf)
    dp[0] = 0.
    masks = np.arange(1, 1 << n)
    counts = np.array([int(m).bit_count() for m in masks])
    for count in range(1, n+1):
        layer = masks[counts == count]
        for j in range(n):
            selected = layer[(layer & (1 << j)) != 0]
            rest = selected ^ (1 << j)
            dp[selected] = np.minimum(dp[selected], dp[rest, j, None]+d[j][None])
    return xy, dp


def exact_route(start, points, remaining=None):
    if len(points) > 16:
        raise ValueError('Exact table limit is 16 nodes')
    if not points:
        return [], 0.
    items = tuple((k, tuple(map(float, points[k]))) for k in sorted(points))
    xy, dp = _suffix_table(items)
    allowed = set(points if remaining is None else remaining)
    if not allowed.issubset(points):
        raise ValueError('Unknown remaining route node')
    mask = sum(1 << i for i, (k, _) in enumerate(items) if k in allowed)
    current, order, cost = np.asarray(start), [], 0.
    while mask:
        options = [j for j in range(len(items)) if mask & (1 << j)]
        j = min(options, key=lambda j: (float(np.linalg.norm(current-xy[j]))+dp[mask ^ (1 << j), j], j))
        cost += float(np.linalg.norm(current-xy[j]))
        order.append(items[j][0])
        current, mask = xy[j], mask ^ (1 << j)
    return order, cost


def open_route(start, points, starts=3, exact_limit=16, remaining=None):
    if not points:
        return [], 0.
    if len(points) <= exact_limit:
        return exact_route(start, points, remaining)
    if remaining is not None:
        points = {k: points[k] for k in remaining}
        if len(points) <= exact_limit:
            return exact_route(start, points)
        if not points:
            return [], 0.
    keys = sorted(points)
    xy = np.asarray([points[k] for k in keys])
    d = np.linalg.norm(xy[:, None]-xy[None, :], axis=2)
    initial = np.linalg.norm(xy-start, axis=1)
    best = None
    for first in sorted(range(len(keys)), key=lambda i: (initial[i], i))[:starts]:
        route, left = [first], set(range(len(keys)))-{first}
        while left:
            j = min(left, key=lambda i: (d[route[-1], i], i))
            route.append(j)
            left.remove(j)
        for _ in range(12):
            gain, change = 1e-7, None
            for i in range(len(route)-1):
                for j in range(i+1, len(route)):
                    before = initial[route[i]] if i == 0 else d[route[i-1], route[i]]
                    after = initial[route[j]] if i == 0 else d[route[i-1], route[j]]
                    if j+1 < len(route):
                        before += d[route[j], route[j+1]]
                        after += d[route[i], route[j+1]]
                    if before-after > gain:
                        gain, change = before-after, (i, j)
            if change is None:
                break
            i, j = change
            route[i:j+1] = reversed(route[i:j+1])
        cost = float(initial[route[0]]+sum(d[a, b] for a, b in zip(route, route[1:])))
        result = cost, tuple(route)
        if best is None or result < best:
            best = result
    return [keys[i] for i in best[1]], best[0]
