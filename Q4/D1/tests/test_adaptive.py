from pathlib import Path
import sys
import math
import unittest
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_coverage import CoverageCertificate
from adaptive_strategy import AdaptiveB2, template
from offline import OfflineEnvironment


class AdaptiveChecks(unittest.TestCase):
    def test_initial_continuous_certificate(self):
        sites, c = template()
        self.assertEqual(len(sites), 25)
        self.assertTrue(c.complete(sites))
        self.assertGreaterEqual(c.areas.sum(), math.pi*1800**2)
        self.assertLess(c.areas.sum(), math.pi*1800**2*1.001)

    def test_cell_corners_not_only_center(self):
        c = CoverageCertificate()
        c.vertices = np.array([[[-100., -100.], [100., -100.], [100., 100.], [-100., 100.]]])
        stations = np.array([[-150., -150.], [150., -150.], [0., 150.]])
        self.assertFalse(c.complete(stations))
        self.assertTrue(c.complete([[-160, -160], [160, -160], [160, 160], [-160, 160]]))

    def test_distance_must_cover_entire_cell(self):
        c = CoverageCertificate()
        c.vertices = np.array([[[-20., -20.], [20., -20.], [20., 20.], [-20., 20.]]])
        # All stations are within 1000 m of the cell center, but not of all corners.
        self.assertFalse(c.complete([[-999, 0], [999, 0], [0, 999], [0, -999]]))

    def test_near_station_is_not_exact_coincidence(self):
        c = CoverageCertificate()
        c.vertices = np.array([[[-1e-9, 0.], [1., -.1], [1., .1]]])
        self.assertFalse(c.complete([[0., 0.], [2., -1.], [2., 1.]]))
        self.assertNotEqual(AdaptiveB2.observation_key([0., 0.], 1),
                            AdaptiveB2.observation_key([1e-9, 0.], 1))

    def test_refinement_preserves_area(self):
        c = CoverageCertificate()
        before = c.areas.sum()
        c.refine(np.arange(0, len(c.areas), 17))
        self.assertAlmostEqual(c.areas.sum(), before, places=5)

    def test_angular_hull_check_against_linear_program(self):
        rng = np.random.default_rng(12)
        c = CoverageCertificate()
        for _ in range(80):
            stations = rng.uniform(-500, 500, (7, 2))
            vertices = rng.uniform(-100, 100, (4, 2))
            c.vertices = vertices[None]
            result = c.complete(stations)
            independent = all(linprog(np.zeros(len(stations)), A_eq=np.vstack([stations.T, np.ones(len(stations))]),
                                      b_eq=np.r_[v, 1.], bounds=(0, None), method='highs').success for v in vertices)
            self.assertEqual(result, independent)

    def test_future_scans_not_completion(self):
        e = OfflineEnvironment({'targets': [], 'error_mode': 'zero'})
        s = AdaptiveB2(lambda *args: e.action(*args))
        self.assertTrue(s.certify(s.sites))
        s.scan(np.zeros(2))
        self.assertFalse(s.update_completion())
        before = len(s.scan_history)
        s.measure([100, 100], 1)
        self.assertEqual(len(s.scan_history), before)

    def test_negative_pair_boundary_localization(self):
        for mode in ('zero', 'positive', 'negative'):
            case = {'targets': [{'channel': 1, 'position': [1800., 0.], 'radius': 1000., 'heading_deg': 0.}], 'error_mode': mode}
            e = OfflineEnvironment(case)
            s = AdaptiveB2(lambda *args: e.action(*args))
            s.measure([1875., 0.], 1)
            s.localize(1)
            self.assertIn(1, e.cleared)
            self.assertGreater(s.negative_pair_cuts+s.history_pair_cuts, 0)
            self.assertEqual(s.fallback_count, 0)

    def test_pair_cut_rejects_unproven_range_or_bracket(self):
        s = AdaptiveB2(lambda *args: {})
        s.polys[1] = np.array([[0., -10.], [1500., -10.], [1500., 10.], [0., 10.]])
        before = s.polys[1].copy()
        self.assertFalse(s.apply_negative_pair(1, np.zeros(2), np.array([1., 0.]), np.array([0., 1.]),
                                               100., 10., [np.array([100., -10.]), np.array([100., 10.])]))
        np.testing.assert_array_equal(s.polys[1], before)

    def test_blended_radius_bound_and_range_counterexample(self):
        s = AdaptiveB2(lambda *args: {})
        rectangle = np.array([[0., -20.], [1500., -20.], [1500., 20.], [0., 20.]])
        s.polys[1] = rectangle.copy()
        # Neither R>=1000 alone nor R>=distance(anchor,G) alone proves both
        # endpoints in range across this polygon; a convex combination does.
        self.assertTrue(s.cut_negative_segment(1, np.array([1500., 0.]),
                                               np.array([700., -900.]), np.array([700., 900.])))
        self.assertGreater(s.polys[1][:, 0].min(), 699.99)
        self.assertGreater(s.polys[1][:, 0].max(), 1200.)
        s.polys[1] = rectangle.copy()
        self.assertFalse(s.cut_negative_segment(1, np.array([1500., 0.]),
                                                np.array([700., -1600.]), np.array([700., 1600.])))
        np.testing.assert_array_equal(s.polys[1], rectangle)

    def test_history_cuts_preserve_random_true_sources(self):
        rng = np.random.default_rng(2026091131)
        for k in range(32):
            angle = rng.uniform(0, math.tau)
            g = rng.uniform(0, 1800)*np.array([math.cos(angle), math.sin(angle)])
            heading = rng.uniform(0, math.tau)
            radius = rng.uniform(1000, 1500)
            case = {'targets': [{'channel': 1, 'position': g.tolist(), 'radius': radius,
                                  'heading_deg': math.degrees(heading)}],
                    'error_mode': ['positive', 'negative', 'hashed'][k % 3], 'error_seed': k}
            e = OfflineEnvironment(case)
            s = AdaptiveB2(lambda *args: e.action(*args))
            bearing = heading+rng.uniform(-1.55, 1.55)
            positive = g+rng.uniform(20, radius)*np.array([math.cos(bearing), math.sin(bearing)])
            # First collect negative history, then receive the source, then add more negatives.
            negatives = []
            for j in range(12):
                a = heading+math.pi+rng.uniform(-1.55, 1.55)
                negatives.append(g+rng.uniform(20, 1800)*np.array([math.cos(a), math.sin(a)]))
            for p in negatives[:6]:
                s.measure(p, 1)
            s.measure(positive, 1)
            for p in negatives[6:]:
                s.measure(p, 1)
                poly = s.polys[1]
                edges = np.roll(poly, -1, axis=0)-poly
                delta = g-poly
                cross = edges[:, 0]*delta[:, 1]-edges[:, 1]*delta[:, 0]
                self.assertTrue(np.all(cross >= -1e-5) or np.all(cross <= 1e-5))
                self.assertTrue(np.all(g >= poly.min(axis=0)-1e-5) and np.all(g <= poly.max(axis=0)+1e-5))


if __name__ == '__main__':
    unittest.main()
