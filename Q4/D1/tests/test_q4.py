from pathlib import Path
import sys
import math
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coverage import search_mesh, triangle_distance
from offline import OfflineEnvironment
from strategy import DirectionalB2


def single(heading=0., position=(0., 0.), radius=1000., error_mode='zero'):
    return {'targets': [{'channel': 2, 'position': list(position), 'radius': radius, 'heading_deg': heading}],
            'error_mode': error_mode, 'error_seed': 91}


class Q4Rules(unittest.TestCase):
    def test_halfplane_boundary_and_range(self):
        e = OfflineEnvironment(single())
        self.assertEqual(e.action('/measure', [1000, 0], 2)['measure_result'], 'direction')
        self.assertEqual(e.action('/measure', [0, 1000], 2)['measure_result'], 'direction')
        self.assertEqual(e.action('/measure', [0, -1000], 2)['measure_result'], 'direction')
        self.assertEqual(e.action('/measure', [1000.001, 0], 2)['measure_result'], 'no_signal')
        self.assertEqual(e.action('/measure', [-.001, 10], 2)['measure_result'], 'no_signal')

    def test_backside_near_clear_and_channel_cost(self):
        e = OfflineEnvironment(single())
        self.assertEqual(e.action('/measure', [-1, 0], 2)['measure_result'], 'no_signal')
        self.assertEqual(e.action('/clear', [-1, 0], 2)['clear_result'], 'success')
        self.assertEqual(e.channel, 2)
        self.assertAlmostEqual(e.clock, .2+6+5)
        e.action('/clear', [0, 0], 3)
        self.assertEqual(e.channel, 2)
        self.assertEqual(e.action('/measure', [-1, 0], 2)['measure_result'], 'no_signal')

    def test_near_boundary(self):
        e = OfflineEnvironment(single())
        self.assertEqual(e.action('/measure', [5, 0], 2)['measure_result'], 'near')
        self.assertEqual(e.action('/measure', [5.001, 0], 2)['measure_result'], 'direction')
        self.assertEqual(e.action('/clear', [20.001, 0], 2)['clear_result'], 'no_target_in_range')
        self.assertEqual(e.action('/clear', [20, 0], 2)['clear_result'], 'success')

    def test_fixed_error_and_heading_not_bearing(self):
        e = OfflineEnvironment(single(heading=0., error_mode='hashed'))
        a = e.action('/measure', [100, 50], 2)['svd_deg']
        e.action('/measure', [60, 70], 2)
        b = e.action('/measure', [100, 50], 2)['svd_deg']
        self.assertEqual(a, b)
        self.assertLess(abs(a-(math.degrees(math.atan2(-50, -100)) % 360)), 1.005001)

    def test_no_signal_does_not_remove_truth(self):
        e = OfflineEnvironment(single())
        s = DirectionalB2(lambda *a: e.action(*a))
        s.measure([100, 100], 2)
        before = s.polys[2].copy()
        s.measure([-1, 0], 2)
        np.testing.assert_array_equal(s.polys[2], before)
        self.assertNotIn(2, s.absent)
        self.assertFalse(s.update_completion())

    def test_extreme_bearing_truth_in_envelope(self):
        for mode in ['positive', 'negative', 'hashed']:
            e = OfflineEnvironment(single(heading=None, position=(123.456, -77.89), error_mode=mode))
            s = DirectionalB2(lambda *a: e.action(*a))
            for p in ([0, 0], [200, 0], [50, -200], [300, 400]):
                s.measure(p, 2)
                poly = s.polys[2]
                edge = np.roll(poly, -1, axis=0)-poly
                delta = np.array([123.456, -77.89])-poly
                cross = edge[:, 0]*delta[:, 1]-edge[:, 1]*delta[:, 0]
                self.assertTrue(np.all(cross >= -1e-5) or np.all(cross <= 1e-5))

    def test_mesh_continuous_cells_and_dense_headings(self):
        sites, triangles = search_mesh()
        pts = sites[triangles]
        self.assertLess(np.linalg.norm(pts[:, :, None]-pts[:, None, :], axis=-1).max(), 1000)
        self.assertTrue(any(np.linalg.norm(p) > 1800 for p in sites))
        for tri in pts:
            self.assertLessEqual(triangle_distance(tri), 1800+1e-7)
        rng = np.random.default_rng(44)
        angles = np.linspace(0, 2*math.pi, 720, endpoint=False)
        sources = np.vstack([1800*np.column_stack((np.cos(angles), np.sin(angles))),
                             rng.uniform(-1, 1, (500, 2))*1200, [[0, 0], [1800, 0]]])
        headings = np.column_stack((np.cos(angles[::4]), np.sin(angles[::4])))
        for g in sources:
            delta = sites-g
            eligible = delta[np.linalg.norm(delta, axis=1) <= 1000]
            self.assertGreaterEqual((eligible@headings.T).max(axis=0).min(), -1e-7)

    def test_outward_boundary_requires_external_sites(self):
        e = OfflineEnvironment(single(heading=0., position=(1800, 0)))
        for p in ([0, 0], [1700, 0], [1750, 100]):
            self.assertEqual(e.action('/measure', p, 2)['measure_result'], 'no_signal')
        sites, _ = search_mesh()
        self.assertTrue(any(e.visible(p, e.targets[2]) for p in sites))

    def test_finite_optical_fallback_from_backside(self):
        e = OfflineEnvironment(single(heading=0., position=(10, 3)))
        s = DirectionalB2(lambda *a: e.action(*a))
        s.polys[2] = np.array([[-40., -10.], [60., -10.], [60., 20.], [-40., 20.]])
        s.optical_cover(2)
        self.assertIn(2, e.cleared)
        self.assertLessEqual(e.clear_attempts, 15)

    def test_safe_end_to_end_boundary_case(self):
        # A one-source diagnostic exercises the coverage-based termination branch.
        e = OfflineEnvironment(single(heading=0., position=(1800, 0), error_mode='positive'))
        s = DirectionalB2(lambda *a: e.action(*a))
        result = s.run()
        self.assertEqual(result['cleared_channels'], [2])
        self.assertEqual(len(result['absent_channels']), 19)
        self.assertEqual(result['completion_reason'], 'all_triangle_vertices_scanned')


if __name__ == '__main__':
    unittest.main()
