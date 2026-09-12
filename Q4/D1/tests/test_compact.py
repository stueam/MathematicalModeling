import sys
import math
import copy
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compact_strategy import compact_template, CompactB2
from offline import OfflineEnvironment


class CompactChecks(unittest.TestCase):
    def test_compact_continuous_domain_and_rotations(self):
        sites, certificate = compact_template()
        self.assertEqual(len(sites), 22)
        self.assertTrue(certificate.complete(sites))
        self.assertGreaterEqual(certificate.areas.sum(), math.pi*1800**2)
        self.assertLess(certificate.areas.sum(), math.pi*1800**2*1.001)
        before = certificate.vertices.copy()
        for angle in np.arange(12)*math.pi/18:
            rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            rotated = copy.copy(certificate)
            rotated.vertices = certificate.vertices@rotation.T
            self.assertTrue(rotated.complete(sites@rotation.T))
        np.testing.assert_array_equal(certificate.vertices, before)

    def test_compact_cells_independent_convex_combination(self):
        sites, certificate = compact_template()
        rng = np.random.default_rng(2026091151)
        # Include the smallest cells where the near-range geometry needs refinement.
        ids = np.unique(np.r_[np.argsort(certificate.areas)[:30], rng.choice(len(certificate.areas), 30, replace=False)])
        for i in ids:
            vertices = certificate.vertices[i]
            eligible = sites[np.max(np.linalg.norm(sites[None]-vertices[:, None], axis=2), axis=0) < 1000.]
            for vertex in vertices:
                result = linprog(np.zeros(len(eligible)), A_eq=np.vstack([eligible.T, np.ones(len(eligible))]),
                                 b_eq=np.r_[vertex, 1.], bounds=(0, None), method='highs')
                self.assertTrue(result.success)

    def test_compact_no_early_absence_and_hard_boundary_detection(self):
        environment = OfflineEnvironment({'targets': [], 'error_mode': 'zero'})
        strategy = CompactB2(environment.action)
        strategy.scan(np.zeros(2))
        self.assertFalse(strategy.update_completion())
        sites, _ = compact_template()
        # Adversarial reception cross-check; full continuous guarantee is above.
        for angle in np.linspace(0, math.tau, 721):
            direction = np.array([math.cos(angle), math.sin(angle)])
            for radius in (0., .1, 998.9, 999., 999.1, 1000., 1799.9, 1800.):
                target = {'position': radius*direction, 'radius': 1000., 'heading_deg': math.degrees(angle)}
                self.assertTrue(any(environment.visible(station, target) for station in sites))


if __name__ == '__main__': unittest.main()
