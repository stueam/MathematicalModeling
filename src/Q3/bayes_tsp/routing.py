"""Dynamic open TSP: remove stale tasks, cheapest insertion, then open 2-opt."""

import numpy as np


def route_length(start, route, points):
    xy = np.asarray([start] + [points[k] for k in route], dtype=float)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


class DynamicTour:
    def __init__(self):
        self.route = []

    def update(self, start, points):
        route = [k for k in self.route if k in points]
        if not route:
            remaining = set(points)
            position = np.asarray(start)
            while remaining:
                key = min(remaining, key=lambda k: (float(np.linalg.norm(points[k] - position)), k))
                route.append(key)
                remaining.remove(key)
                position = np.asarray(points[key])
        for key in sorted(set(points) - set(route)):
            alternatives = [route[:i] + [key] + route[i:] for i in range(len(route) + 1)]
            route = min(alternatives, key=lambda r: route_length(start, r, points))
        # Geometry and start change after each action, so reconsider the first
        # node too. No return edge is introduced by insertion or 2-opt.
        while True:
            old = route_length(start, route, points)
            best, best_cost = route, old
            for i in range(len(route)):
                for j in range(i + 2, len(route) + 1):
                    candidate = route[:i] + list(reversed(route[i:j])) + route[j:]
                    cost = route_length(start, candidate, points)
                    if cost < best_cost - 1e-7:
                        best, best_cost = candidate, cost
            if best_cost >= old - 1e-7:
                break
            route = best
        self.route = route
        return route.copy()
