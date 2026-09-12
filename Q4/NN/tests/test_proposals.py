"""External probes remain optional, distinct and based on public observations."""
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnq4.bridge import Action, Belief, Config, ProbePolicy, QuadratureError, distance
from nnq4.proposals import SHAPES, external_choices
from q4.shaped import existing_proposals, shaped_pair
from q4.simulator import LocalSimulator, Source, World


class ProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def fixture(self, points=((0., 0.), (250., 10.))):
        # Truth belongs to this test environment only. The proposal interface
        # receives the belief built by actual execute/apply feedback.
        world = World([Source(1, (900., 0.), 1000., 180.)], seed=1)
        env, belief = LocalSimulator(world), Belief()
        for i, point in enumerate(points):
            action = Action('measure', point, 1)
            belief.apply(action, env.execute(action, str(i)), str(i))
        planner = ProbePolicy(Config(resolution=8))
        choices, mean = planner.local_choices(belief, 1)
        return planner, belief, choices, mean, env

    def test_certified_shape_family_keeps_wide_templates_and_all_new_points(self):
        planner, b, choices, mean, _ = self.fixture()
        before = (b.position, b.steps, b.virtual_time, tuple(b.channels[1].history),
                  b.channels[1].region.wkb, dict(planner.points), planner.batch)
        original = list(choices)
        pool = existing_proposals(planner, b, 1, mean, choices)
        added = external_choices(planner, b, 1, choices, mean)
        shaped = [row for row in added if row[2]['candidate_family'] == 'shaped_probe']
        self.assertGreater(len(shaped), 3)  # No inherited top-three prefilter.
        self.assertTrue(any(row[2]['shape']['aspect'] == 1. for row in shaped))
        self.assertEqual(len(SHAPES), 9)
        for action, cost, detail in shaped:
            proof = detail['shape']
            points, checked = shaped_pair(b.channels[1], proof['fraction'], proof['aspect'])
            self.assertIn(action.position, points)
            self.assertEqual(proof, checked)
            self.assertLess(proof['max_squared_distance_change'], -1e-4)
            self.assertGreater(proof['minimum_forward_gap_m'], 1e-4)
            self.assertGreater(proof['minimum_cone_slack'], 1e-4)
            self.assertTrue(all(distance(action.position, p) >= 10. for p in pool))
            expected = planner.probe(b, 1, action.position, 10.)
            self.assertEqual(action, expected[0])
            self.assertAlmostEqual(cost, expected[1])
            self.assertTrue(detail['single_point_only'])
            self.assertFalse(detail['probability_used_for_proof'])
            self.assertEqual(len(detail['source_commit']), 40)
            self.assertTrue(detail['source_repo'])
        self.assertEqual(choices, original)
        self.assertEqual(before, (b.position, b.steps, b.virtual_time,
                                 tuple(b.channels[1].history), b.channels[1].region.wkb,
                                 dict(planner.points), planner.batch))

    def test_heading_and_heard_side_are_not_removed_by_documented_suffix(self):
        planner, b, choices, mean, _ = self.fixture(((0., 0.), (300., 0.), (1100., 0.)))
        self.assertEqual(b.channels[1].history[-1].result, 'no_signal')
        added = external_choices(planner, b, 1, choices, mean)
        families = {row[2]['candidate_family'] for row in added}
        self.assertIn('forward_heading', families)
        self.assertIn('heard_side_after_miss', families)
        self.assertTrue(all(row[0].kind == 'measure' and row[0].channel == 1 for row in added))
        # Supplying the expanded real menu again must not duplicate its actions.
        self.assertEqual(external_choices(planner, b, 1, choices + added, mean), [])
        for i, (action, _, _) in enumerate(added):
            self.assertTrue(all(distance(action.position, row[0].position) >= 10.
                                for row in added[:i] + choices))
            self.assertTrue(all(distance(action.position, obs.action.position) >= 10.
                                for obs in b.channels[1].history if obs.action.kind == 'measure'))

    def test_unmeasured_current_stop_can_be_restored_after_coarse_filtering(self):
        planner, b, _, _, env = self.fixture()
        action = Action('measure', (200., 40.), 2)
        b.apply(action, env.execute(action, 'move-on-channel-2'), 'move-on-channel-2')
        choices, mean = planner.local_choices(b, 1)
        choices = [row for row in choices if distance(row[0].position, b.position) >= 10.]
        added = external_choices(planner, b, 1, choices, mean)
        current = [row for row in added if row[2]['candidate_family'] == 'shared_current']
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0][0], Action('measure', b.position, 1))
        self.assertTrue(current[0][2]['zero_extra_movement'])
        # A same-channel real reading at this point suppresses it subsequently.
        b.apply(current[0][0], env.execute(current[0][0], 'shared'), 'shared')
        choices, mean = planner.local_choices(b, 1)
        self.assertTrue(all(distance(row[0].position, b.position) >= 10.
                            for row in external_choices(planner, b, 1, choices, mean)))

    def test_completed_or_unknown_channels_never_generate_local_probes(self):
        planner, b, choices, mean, env = self.fixture()
        self.assertEqual(external_choices(planner, b, 2, choices, mean), [])
        action = Action('clear', (900., 0.), 1)
        b.apply(action, env.execute(action, 'clear'), 'clear')
        self.assertEqual(b.channels[1].status, 'cleared')
        self.assertEqual(external_choices(planner, b, 1, choices, mean), [])

    def test_finite_modes_and_local_budget_disable_all_additions(self):
        planner, b, choices, mean, _ = self.fixture()
        planner.completion_mode = True
        self.assertEqual(external_choices(planner, b, 1, choices, mean), [])
        planner.completion_mode = False
        b.steps = planner.config.completion_after
        self.assertEqual(external_choices(planner, b, 1, choices, mean), [])
        b.steps = 2
        b.deadline = time.monotonic() + planner.config.reserve_s - 1.
        self.assertEqual(external_choices(planner, b, 1, choices, mean), [])
        b.deadline = float('inf')
        planner.config = Config(resolution=8, local_limit=2)
        self.assertEqual(external_choices(planner, b, 1, choices, mean), [])

    def test_quadrature_failure_preserves_original_choices(self):
        planner, b, choices, mean, _ = self.fixture()
        original, history, geometry = list(choices), tuple(b.channels[1].history), b.channels[1].region.wkb
        with patch.object(planner.model, 'posterior', side_effect=QuadratureError('empty quadrature')):
            self.assertEqual(external_choices(planner, b, 1, choices, mean), [])
        self.assertEqual(choices, original)
        self.assertEqual(tuple(b.channels[1].history), history)
        self.assertEqual(b.channels[1].region.wkb, geometry)


if __name__ == '__main__':
    unittest.main()
