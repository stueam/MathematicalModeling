"""Torch feature arithmetic preserves canonical particles and file boundaries."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnq4.bridge import Action, Belief, Config
from nnq4.state import EXTENT, Frame, MenuPlanner, _weighted_quantile_indices
from q4.simulator import LocalSimulator, Source, World


class StateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_particle_lexsort_ties_and_left_quantile_boundaries(self):
        xy = np.array([[2., 1.], [1., 5.], [1., 2.], [1., 2.], [2., -1.], [-3., 0.]])
        weights = np.array([.125, .125, .125, .125, .25, .25])
        order = np.lexsort((xy[:, 1], xy[:, 0]))
        for count in (1, 2, 4, 7, 16):
            expected = order[np.minimum(np.searchsorted(np.cumsum(weights[order]),
                             (np.arange(count) + .5) / count), len(order) - 1)]
            actual = _weighted_quantile_indices(torch.from_numpy(xy), torch.from_numpy(weights), count)
            np.testing.assert_array_equal(actual.numpy(), expected)

    def test_particle_quantiles_keep_float64_near_threshold(self):
        xy = torch.tensor([[0., 0.], [1., 0.], [2., 0.]], dtype=torch.float64)
        weights = torch.tensor([.5 - 1e-10, 2e-10, .5 - 1e-10], dtype=torch.float64)
        self.assertEqual(_weighted_quantile_indices(xy, weights, 1).tolist(), [1])

    def public_fixture(self):
        b = Belief()
        env = LocalSimulator(World([Source(1, (900., 0.), 1000., 180.)], seed=1))
        for i, point in enumerate(((0., 0.), (250., 10.))):
            action = Action('measure', point, 1)
            b.apply(action, env.execute(action, str(i)), str(i))
        return b

    def test_public_posterior_moments_match_numpy_reference(self):
        b = self.public_fixture()
        planner = MenuPlanner(Config(resolution=8))
        frame = planner.frame(b)
        for tensor in (frame.nodes, frame.candidates, frame.global_features):
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(tensor.device.type, 'cpu')
            self.assertTrue(torch.isfinite(tensor).all())
        post = planner.model.posterior(b.channels[1])
        delta = post.xy - post.mean
        covariance = (delta.T * post.weights) @ delta / EXTENT**2
        node = frame.nodes[0].numpy()
        np.testing.assert_allclose(node[:2], post.mean / EXTENT, rtol=0, atol=1e-6)
        np.testing.assert_allclose(node[20:23], [covariance[0, 0], covariance[1, 1], covariance[0, 1]],
                                   rtol=0, atol=1e-6)
        self.assertAlmostEqual(float(node[13]), float(np.sqrt(np.trace(covariance))), places=6)
        radius = float(np.sum(post.atoms * (post.upper + post.lower[:, None]) / 2)) / 1500
        self.assertAlmostEqual(float(node[30]), radius, places=6)
        order = np.lexsort((post.xy[:, 1], post.xy[:, 0]))
        ix = order[np.minimum(np.searchsorted(np.cumsum(post.weights[order]),
                   (np.arange(planner.particles) + .5) / planner.particles), len(order) - 1)]
        particles = frame.nodes[frame.nodes[:, 6] == 1]
        np.testing.assert_allclose(particles[:, :2].numpy(), post.xy[ix] / EXTENT, rtol=0, atol=1e-6)
        np.testing.assert_allclose(particles[:, 23].numpy(), post.weights[ix], rtol=0, atol=1e-6)

    def test_arrays_and_current_episode_writer_remain_compatible(self):
        from nnq4.experiment import pack_episode
        frame = Frame(torch.arange(64, dtype=torch.float32).reshape(2, 32),
                      torch.ones(3, 32), torch.zeros(16),
                      [Action('measure', (0., 0.), c) for c in (1, 2, 3)],
                      [True, True, True], 1, 'fixture')
        arrays = frame.arrays()
        for key, value in arrays.items():
            self.assertIsInstance(value, np.ndarray)
            self.assertEqual(value.dtype, np.bool_ if key.endswith('_mask') else np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'episode.npz'
            self.assertEqual(pack_episode([frame], [20.], 120., path), 1)
            with np.load(path) as saved:
                for key in arrays:
                    np.testing.assert_array_equal(saved[key][0], arrays[key])
                self.assertEqual(saved['target'][0], 1)
                self.assertEqual(saved['remaining'][0], 100.)


if __name__ == '__main__':
    unittest.main()
