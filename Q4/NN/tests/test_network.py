"""Invariance and learning-path checks for the Q4 candidate decoder."""

import io
import json
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnq4.network import NetworkConfig, PolicyNetwork


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(21)
        self.network = PolicyNetwork(d_model=32, n_heads=4, ffn_dim=64).eval()
        self.inputs = (torch.randn(2, 5, 32), torch.ones(2, 5, dtype=torch.bool),
                       torch.randn(2, 4, 32), torch.ones(2, 4, dtype=torch.bool),
                       torch.randn(2, 16))

    def test_output_shape_and_finite_costs(self):
        out = self.network(*self.inputs)
        self.assertEqual(out["scores"].shape, (2, 4))
        self.assertEqual(out["q"].shape, (2, 4))
        self.assertEqual(out["value"].shape, (2,))
        for value in out.values():
            self.assertTrue(torch.isfinite(value).all())
        self.assertTrue((out["q"] >= 0).all())
        self.assertTrue((out["value"] >= 0).all())

    def test_padding_does_not_change_valid_outputs(self):
        for candidate_attention in (False, True):
            with self.subTest(candidate_attention=candidate_attention):
                net = PolicyNetwork(d_model=32, ffn_dim=64,
                                    candidate_attention=candidate_attention).eval()
                nodes, node_mask, candidates, mask, global_features = self.inputs
                expected = net(*self.inputs)
                padded = (F.pad(nodes, (0, 0, 0, 3), value=float("nan")),
                          F.pad(node_mask, (0, 3), value=False),
                          F.pad(candidates, (0, 0, 0, 2), value=float("inf")),
                          F.pad(mask, (0, 2), value=False), global_features)
                actual = net(*padded)
                for key in ("scores", "q"):
                    torch.testing.assert_close(expected[key], actual[key][:, :4], atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(expected["value"], actual["value"], atol=2e-6, rtol=1e-5)
                self.assertTrue((actual["scores"][:, 4:] == -10000).all())
                (actual["scores"][:, :4].square().mean() + actual["q"][:, :4].mean()).backward()
                for param in net.parameters():
                    if param.grad is not None:
                        self.assertTrue(torch.isfinite(param.grad).all())

    def test_candidate_permutation_equivariance(self):
        nodes, node_mask, candidates, mask, global_features = self.inputs
        mask[1, 2] = False
        permutation = torch.tensor([3, 0, 2, 1])
        for candidate_attention in (False, True):
            with self.subTest(candidate_attention=candidate_attention):
                net = PolicyNetwork(d_model=32, ffn_dim=64,
                                    candidate_attention=candidate_attention).eval()
                expected = net(*self.inputs)
                actual = net(nodes, node_mask, candidates[:, permutation], mask[:, permutation], global_features)
                for key in ("scores", "q"):
                    torch.testing.assert_close(expected[key][:, permutation], actual[key], atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(expected["value"], actual["value"])

    def test_node_permutation_invariance(self):
        nodes, node_mask, candidates, mask, global_features = self.inputs
        node_mask[0, 2] = False
        permutation = torch.tensor([4, 2, 0, 3, 1])
        expected = self.network(*self.inputs)
        actual = self.network(nodes[:, permutation], node_mask[:, permutation], candidates, mask, global_features)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], atol=2e-6, rtol=1e-5)

    def test_policy_and_q_gradients_reach_candidate_cross_attention(self):
        for objective in ("policy", "q"):
            with self.subTest(objective=objective):
                self.network.zero_grad(set_to_none=True)
                out = self.network(*self.inputs)
                loss = (F.cross_entropy(out["scores"], torch.tensor([1, 3])) if objective == "policy"
                        else F.smooth_l1_loss(out["q"], torch.full((2, 4), 3.0)))
                loss.backward()
                gradient = self.network.candidate_cross_attention.in_proj_weight.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.norm()), 1e-6)
                self.assertGreater(float(self.network.node_embedding.weight.grad.norm()), 1e-6)

    def test_no_valid_candidate_is_rejected(self):
        nodes, node_mask, candidates, mask, global_features = self.inputs
        mask[1] = False
        with self.assertRaisesRegex(ValueError, "at least one valid candidate"):
            self.network(nodes, node_mask, candidates, mask, global_features)
        with self.assertRaisesRegex(ValueError, "at least one valid candidate"):
            self.network(nodes, node_mask, candidates[:, :0], mask[:, :0], global_features)

    def test_empty_nodes_use_global_token(self):
        nodes, node_mask, candidates, mask, global_features = self.inputs
        out = self.network(nodes[:, :0], node_mask[:, :0], candidates, mask, global_features)
        masked_out = self.network(torch.full_like(nodes, float("nan")),
                                  torch.zeros_like(node_mask), candidates, mask, global_features)
        for key in out:
            self.assertTrue(torch.isfinite(out[key]).all())
            torch.testing.assert_close(out[key], masked_out[key], atol=2e-6, rtol=1e-5)

    def test_checkpoint_roundtrip(self):
        config = json.loads(json.dumps(self.network.get_config()))
        self.assertEqual(NetworkConfig(**config), self.network.config)
        stream = io.BytesIO()
        torch.save(self.network.checkpoint(epoch=3), stream)
        stream.seek(0)
        payload = torch.load(stream, weights_only=True)
        restored = PolicyNetwork.from_checkpoint(payload).eval()
        self.assertEqual(payload["metadata"]["epoch"], 3)
        for key, expected in self.network(*self.inputs).items():
            torch.testing.assert_close(expected, restored(*self.inputs)[key])


if __name__ == "__main__":
    unittest.main()
