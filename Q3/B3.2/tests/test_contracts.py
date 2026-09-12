import math
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from q3_strategy import initial_envelope, observe
from offline_environment import OfflineEnvironment
from solver import build_strategy

class Contracts(unittest.TestCase):
    def test_virtual_cost_and_clear_boundary(self):
        env = OfflineEnvironment(dict(error_mode='zero', targets=[
            dict(channel=2, position=[20., 0.], radius=1000.)]))
        self.assertEqual(env.action('/measure', [0., 0.], 2)['virtual_time_s'], 6.)
        self.assertEqual(env.action('/clear', [0., 0.], 2)['clear_result'], 'success')
        self.assertEqual(env.clock, 11.)
        self.assertEqual(env.cleared, {2})

    def test_wedge_keeps_truth_at_error_extremes(self):
        for angle in [-179.999, -1., 0., 179.999]:
            point = 999*np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
            for error in [-1., 1.]:
                poly = observe(initial_envelope(), np.zeros(2), round(angle+error, 2)%360)
                edges = np.roll(poly, -1, axis=0)-poly
                delta = point-poly
                cross = edges[:,0]*delta[:,1]-edges[:,1]*delta[:,0]
                self.assertTrue(np.all(cross >= -1e-5) or np.all(cross <= 1e-5))

    def test_initial_scans_cover_whole_annulus(self):
        # Sample the interior as well as endpoints, independent of four-corner checks.
        model = build_strategy(lambda *args: None)
        angles = np.linspace(-math.pi/7, math.pi/7, 201)
        points = np.array([[r*math.cos(a), r*math.sin(a)]
                           for r in np.linspace(1000, 1800, 31) for a in angles])
        site = np.array([1100., 0.])
        self.assertLess(np.linalg.norm(points-site, axis=1).max(), 1000.)
        adjusted = model.adjust(0, np.array([0., 0.]), np.array([0., 1500.]), site)
        self.assertLess(np.linalg.norm(points-adjusted, axis=1).max(), 1000.)

if __name__ == '__main__':
    unittest.main()
