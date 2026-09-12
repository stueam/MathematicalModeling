import sys
from pathlib import Path
import unittest
from itertools import permutations
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compact_strategy import compact_template
from service_geometry import station_constraints, optimize_service
from service_routes import exact_open_route


class ServiceChecks(unittest.TestCase):
    def test_exact_route_against_permutations(self):
        rng = np.random.default_rng(2026091181)
        self.assertEqual(exact_open_route([0., 0.], []), [])
        for n in (1, 4, 7):
            points = rng.normal(size=(n, 2))
            start = rng.normal(size=2)
            def length(order):
                return np.linalg.norm(np.diff(np.vstack([start, points[list(order)]]), axis=0), axis=1).sum()
            route = exact_open_route(start, points)
            self.assertEqual(sorted(route), list(range(n)))
            self.assertAlmostEqual(length(route), min(length(p) for p in permutations(range(n))), places=10)

    def test_disk_projection_analytic(self):
        point, info = optimize_service([0., 0.], [3., 0.], None, [[0., 0.]], 1.)
        self.assertTrue(info['accepted'])
        np.testing.assert_allclose(point, [1., 0.], atol=1e-5)
        self.assertLessEqual(np.linalg.norm(point), 1.)

    def test_entire_clear_polygon_and_leg_cost(self):
        vertices = np.array([[-8., -3.], [7., -5.], [10., 3.], [-6., 6.]])
        before, after = np.array([-80., 40.]), np.array([100., 50.])
        point, info = optimize_service([0., 0.], before, after, vertices, 19.8)
        self.assertTrue(info['accepted'])
        self.assertLessEqual(np.linalg.norm(vertices-point, axis=1).max(), 19.8)
        self.assertLess(np.linalg.norm(point-before)+np.linalg.norm(point-after),
                        np.linalg.norm(before)+np.linalg.norm(after))

    def test_halfplane_restricts_projection(self):
        point, _ = optimize_service([0., 0.], [3., 0.], None, [[0., 0.]], 2.,
                                    np.array([[1., 0.]]), np.array([.3]))
        self.assertLessEqual(point[0], .3)
        np.testing.assert_allclose(point, [.3, 0.], atol=1e-5)

    def test_scan_replacement_independent_hull_membership(self):
        sites, certificate = compact_template()
        for k in (1, 10, 21):
            others = np.delete(sites, k, axis=0)
            constraints = station_constraints(certificate, others)
            self.assertIsNotNone(constraints)
            vertices, normals, bounds = constraints
            point, info = optimize_service(sites[k], sites[0], sites[(k+1) % len(sites)],
                                            vertices, 1000.-2e-6, normals, bounds)
            self.assertTrue(info['accepted'])
            replaced = np.vstack([others, point])
            self.assertTrue(certificate.complete(replaced))
            uncovered = np.flatnonzero(~certificate.evaluate(others)[0])
            # Independent convex-combination feasibility, including small cells.
            ids = uncovered[np.linspace(0, len(uncovered)-1, min(12, len(uncovered)), dtype=int)]
            for i in ids:
                polygon = certificate.vertices[i]
                eligible = replaced[np.linalg.norm(polygon[:, None]-replaced, axis=2).max(axis=0) < 1000.]
                for vertex in polygon:
                    fit = linprog(np.zeros(len(eligible)),
                                  A_eq=np.vstack([eligible.T, np.ones(len(eligible))]),
                                  b_eq=np.r_[vertex, 1.], bounds=(0, None), method='highs')
                    self.assertTrue(fit.success)


if __name__ == '__main__': unittest.main()
