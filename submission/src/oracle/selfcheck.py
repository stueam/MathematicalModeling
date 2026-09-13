"""Exact-route checks extracted from Q3/全知纯移动基准.py."""
import itertools
import math

SPEED = 5.0

def exact_route(points, router):
    length, order = router(points, exact_limit=16).plan((0., 0.))
    assert len(order) == len(set(order)) == len(points)
    positions = [(0., 0.)] + [points[c] for c in order]
    independent_length = sum(math.dist(a, b) for a, b in zip(positions, positions[1:]))
    assert abs(length - independent_length) < 1e-7
    return length, order

def selfcheck(router):
    fixtures = [[], [(0., 0.)], [(3., 4.)], [(1., 0.), (5., 0.), (2., 0.)],
                [(3., 4.), (-1., 2.), (5., -2.), (0., -3.)],
                [(1., 1.), (1., 1.), (-2., 0.), (0., 0.), (2., -1.)]]
    for coords in fixtures:
        points = dict(enumerate(coords))
        result, _ = exact_route(points, router)
        values = []
        for perm in itertools.permutations(coords):
            path = [(0., 0.)] + list(perm)
            values.append(sum(math.dist(a, b) for a, b in zip(path, path[1:])))
        assert abs(result - min(values)) < 1e-7
    regular_polygons = []
    for n in (10, 16):
        points = {j: (1800 * math.cos(2 * math.pi * j / n),
                      1800 * math.sin(2 * math.pi * j / n)) for j in range(n)}
        length, _ = exact_route(points, router)
        # First edge >=1800; every inter-source edge >= the polygon side.
        analytic = 1800 + (n - 1) * 3600 * math.sin(math.pi / n)
        assert abs(length - analytic) < 1e-7
        regular_polygons.append({'n': n, 'length_m': length,
                                 'pure_move_s_per_source': length / SPEED / n})
    return {'bruteforce_fixtures': len(fixtures),
            'analytic_regular_polygons': regular_polygons}
