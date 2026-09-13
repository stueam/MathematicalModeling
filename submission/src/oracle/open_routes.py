"""Open TSP and conditional suffix routes, with no return-to-origin edge."""
import numpy as np


class OpenRoutes:
    def __init__(self, points, exact_limit=12):
        self.keys = tuple(sorted(points))
        self.xy = np.asarray([points[k] for k in self.keys], dtype=float).reshape(-1, 2)
        self.index = {k: i for i, k in enumerate(self.keys)}
        n = len(self.keys)
        self.dist = np.linalg.norm(self.xy[:, None, :]-self.xy[None, :, :], axis=2)
        self.exact = n <= exact_limit
        self._plans = {}
        self.dp = None
        if self.exact and n:
            # dp[mask,i]: start at i, visit precisely mask, then end anywhere.
            # Vectorize each subset-cardinality layer; no sampled permutations.
            self.dp = np.full((1 << n, n), np.inf)
            self.dp[0] = 0.
            masks = np.arange(1, 1 << n)
            counts = np.array([int(m).bit_count() for m in masks])
            for count in range(1, n+1):
                layer = masks[counts == count]
                for j in range(n):
                    selected = layer[(layer & (1 << j)) != 0]
                    rest = selected ^ (1 << j)
                    costs = self.dp[rest, j, None] + self.dist[j][None, :]
                    self.dp[selected] = np.minimum(self.dp[selected], costs)

    def length(self, start, order):
        points = np.vstack((np.asarray(start), self.xy[order]))
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    def plan(self, start, remaining=None):
        allowed = set(self.keys if remaining is None else remaining)
        if not allowed:
            return 0., []
        cache_key = (tuple(map(float, start)), tuple(sorted(allowed)))
        if cache_key in self._plans:
            cost, route = self._plans[cache_key]
            return cost, list(route)
        ids = [self.index[k] for k in sorted(allowed)]
        if self.exact:
            mask = sum(1 << i for i in ids)
            current = np.asarray(start)
            order = []
            while mask:
                options = [i for i in ids if mask & (1 << i)]
                nxt = min(options, key=lambda j: (float(np.linalg.norm(current-self.xy[j]))+
                                                   self.dp[mask ^ (1 << j), j], j))
                order.append(nxt)
                mask ^= 1 << nxt
                current = self.xy[nxt]
        else:
            # Multi-start nearest neighbor, open 2-opt AND relocate moves.
            # Relocate can improve routes that are locally optimal under 2-opt.
            best = None
            starts = sorted(ids, key=lambda i: (np.linalg.norm(self.xy[i]-start), i))[:4]
            for first in starts:
                left = set(ids)-{first}
                route = [first]
                while left:
                    nxt = min(left, key=lambda i: (self.dist[route[-1], i], i))
                    route.append(nxt)
                    left.remove(nxt)
                while True:
                    cost = self.length(start, route)
                    candidate, new_cost = route, cost
                    for i in range(len(route)):
                        for j in range(i+2, len(route)+1):
                            proposal = route[:i]+route[i:j][::-1]+route[j:]
                            value = self.length(start, proposal)
                            if value < new_cost-1e-7:
                                candidate, new_cost = proposal, value
                        removed = route[:i]+route[i+1:]
                        for j in range(len(route)):
                            proposal = removed[:j]+[route[i]]+removed[j:]
                            value = self.length(start, proposal)
                            if value < new_cost-1e-7:
                                candidate, new_cost = proposal, value
                    if new_cost >= cost-1e-7:
                        break
                    route = candidate
                result = (self.length(start, route), tuple(route))
                if best is None or result < best:
                    best = result
            order = list(best[1])
        cost, route = self.length(start, order), tuple(self.keys[i] for i in order)
        if len(self._plans) >= 128:
            self._plans.pop(next(iter(self._plans)))
        # Only pure geometric route costs, for this immutable coordinate table.
        # This is not a cache of simulated worlds or policy continuation costs.
        self._plans[cache_key] = (cost, route)
        return cost, list(route)
