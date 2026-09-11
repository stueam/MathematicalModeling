"""Open tours; insertion and 2-opt never add a return leg."""
from q3.core import distance
from q3.mission import open_tour


def length(start, points):
    return sum(distance(a, b) for a, b in zip([start]+list(points), points))


def insert(start, route, point):
    routes = [route[:i]+[point]+route[i:] for i in range(len(route)+1)]
    return min(routes, key=lambda r: length(start, r))


def improve(start, route):
    route = list(route)
    for _ in range(4):
        old = length(start, route)
        for i in range(len(route)):
            for j in range(i+2, len(route)+1):
                candidate = route[:i]+list(reversed(route[i:j]))+route[j:]
                if length(start, candidate) < length(start, route)-1e-8:
                    route = candidate
        if length(start, route) >= old-1e-8:
            break
    return route


class OpenRoute:
    def __init__(self):
        self.order = []

    def update(self, start, centers):
        self.order = [c for c in self.order if c in centers]
        for c in centers:
            if c not in self.order:
                options = [self.order[:i]+[c]+self.order[i:] for i in range(len(self.order)+1)]
                self.order = min(options, key=lambda r: length(start, [centers[k] for k in r]))
        if self.order:
            points = [centers[c] for c in self.order]
            fresh = [self.order[i] for i in open_tour(start, points)]
            if length(start, [centers[c] for c in fresh]) < length(start, points):
                self.order = fresh
        return self.order.copy()
