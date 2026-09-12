"""Directed feature proxies preserve all other public service obligations."""
from dataclasses import replace
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnq4.bridge import Action, Belief, Config, DOMAIN
from nnq4.features_v2 import (CANDIDATE_EXTRA as C, NODE_EXTRA as N,
                             GLOBAL_EXTRA as G, augment_frame)
from nnq4.state import EXTENT, Frame, MenuPlanner
from q4.simulator import LocalSimulator, Source, World


class FeaturesV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def fixture(self):
        # Public posterior point estimates, not an evaluator World. Source 1
        # and survey A deliberately share a coordinate but are separate tasks.
        means = {1: (100., 0.), 2: (0., 100.)}
        tracks = {c: SimpleNamespace(status='detected', mean=p,
                  region=SimpleNamespace(area=1.), directions=[], negatives=[],
                  history=[], measured=set()) for c, p in means.items()}
        tracks[3] = SimpleNamespace(status='unresolved')
        b = SimpleNamespace(channels=tracks, position=(0., 0.), receiver=1)
        planner = SimpleNamespace(
            points={-1: (100., 0.), -2: (0., 100.)},
            model=SimpleNamespace(posterior=lambda channel: SimpleNamespace(
                xy=torch.tensor([channel.mean], dtype=torch.float64),
                weights=torch.ones(1, dtype=torch.float64))),
            needed=lambda belief, point: [3], records=None)
        nodes = torch.zeros(4, 32)
        for i, c in enumerate((1, 2)):
            nodes[i, :2] = torch.tensor(means[c]) / EXTENT
            nodes[i, 2:4] = nodes[i, :2]
            nodes[i, 4] = nodes[i, 8] = 1.
            nodes[i, 11] = float(c == b.receiver)
            nodes[i, 12] = math.log(2.) / math.log1p(DOMAIN.area)
        for i, p in enumerate(planner.points.values(), 2):
            nodes[i, :2] = torch.tensor(p) / EXTENT
            nodes[i, 5] = 1.
        actions = [Action('measure', (40., 0.), 1), Action('clear', (100., 0.), 1),
                   Action('measure', (0., 40.), 2), Action('measure', (100., 0.), 3),
                   Action('measure', (0., 100.), 3)]
        costs = [35., 25., 50., 26., 26.]
        survey = [False, False, False, True, True]
        candidates = torch.zeros(5, 32)
        for i, (action, cost) in enumerate(zip(actions, costs)):
            candidates[i, :2] = torch.tensor(action.position) / EXTENT
            candidates[i, 4:7] = torch.tensor([action.kind == 'measure' and not survey[i],
                                              action.kind == 'clear', survey[i]])
            move = math.hypot(*action.position) / 5
            candidates[i, 7] = move / 880.
            candidates[i, 16] = 0. if survey[i] else (cost - move) / 1000.
        frame = Frame(nodes, candidates, torch.arange(16, dtype=torch.float32),
                      actions, survey, 0, 'arbitrary reference', ['local'] * 3 + ['survey'] * 2)
        return frame, b, planner

    def test_prefix_and_metadata_are_preserved_without_mutation(self):
        frame, b, planner = self.fixture()
        original = [frame.nodes.clone(), frame.candidates.clone(), frame.global_features.clone()]
        points = dict(planner.points)
        actual = augment_frame(frame, b, planner)
        self.assertEqual(actual.nodes.shape, (4, 40))
        self.assertEqual(actual.candidates.shape, (5, 48))
        self.assertEqual(actual.global_features.shape, (20,))
        for before, after, current in zip(original, (actual.nodes[:, :32], actual.candidates[:, :32],
                      actual.global_features[:16]), (frame.nodes, frame.candidates, frame.global_features)):
            self.assertTrue(torch.equal(before, after))
            self.assertTrue(torch.equal(before, current))
        self.assertIs(actual.actions, frame.actions)
        self.assertIs(actual.families, frame.families)
        self.assertEqual(planner.points, points)
        self.assertEqual(b.channels[1].status, 'detected')
        for value in (actual.nodes, actual.candidates, actual.global_features):
            self.assertTrue(torch.isfinite(value).all())

    def test_service_costs_and_directed_completion_endpoints(self):
        frame, b, planner = self.fixture()
        result = augment_frame(frame, b, planner)
        # Channel 1 service = min(35,25)-100/5=5; channel 2=50-100/5=30.
        torch.testing.assert_close(result.nodes[:, N['service_s']] * 1000,
                                   torch.tensor([5., 30., 6., 6.]), atol=1e-5, rtol=0)
        self.assertAlmostEqual(float(result.global_features[G['all_service_s']]) * 1000, 47., places=4)
        expected_endpoints = torch.tensor([[100., 0.], [100., 0.], [0., 100.], [100., 0.], [0., 100.]])
        torch.testing.assert_close(result.candidates[:, C['completion_x']:C['completion_y']+1] * EXTENT,
                                   expected_endpoints)
        torch.testing.assert_close(result.candidates[:, C['completion_position_is_proxy']],
                                   torch.tensor([1., 0., 1., 0., 0.]))
        self.assertAlmostEqual(float(result.candidates[0, C['completion_minus_action_x']]) * EXTENT, 60., places=4)
        # One co-located obligation costs no extra movement, but still service.
        self.assertAlmostEqual(float(result.candidates[0, C['other_min_distance']]), 0.)
        self.assertAlmostEqual(float(result.candidates[0, C['open_tail_route_s']]) * 1000,
                               math.sqrt(20000.) / 5, places=4)

    def test_survey_and_known_actions_remove_only_their_own_obligation(self):
        frame, b, planner = self.fixture()
        result = augment_frame(frame, b, planner).candidates
        # Known channel 1 is completed only in this cost proxy: both stations
        # remain, including station A at the same (100,0) endpoint.
        for i in (0, 1):
            self.assertAlmostEqual(float(result[i, C['other_known_count_over_16']]) * 16, 1.)
            self.assertAlmostEqual(float(result[i, C['other_station_count_over_49']]) * 49, 2., places=5)
            self.assertAlmostEqual(float(result[i, C['other_service_s']]) * 1000, 42., places=4)
        # Survey A cannot erase co-located known channel 1 or survey B.
        self.assertAlmostEqual(float(result[3, C['other_known_count_over_16']]) * 16, 2.)
        self.assertAlmostEqual(float(result[3, C['other_station_count_over_49']]) * 49, 1., places=5)
        self.assertAlmostEqual(float(result[3, C['other_service_s']]) * 1000, 41., places=4)
        self.assertEqual(b.channels[1].status, 'detected')

    def test_node_and_candidate_permutations_preserve_appended_features(self):
        frame, b, planner = self.fixture()
        expected = augment_frame(frame, b, planner)
        nodes = torch.tensor([3, 1, 0, 2])
        candidates = torch.tensor([4, 2, 0, 3, 1])
        order = candidates.tolist()
        shuffled = replace(frame, nodes=frame.nodes[nodes], candidates=frame.candidates[candidates],
                           actions=[frame.actions[i] for i in order], survey=[frame.survey[i] for i in order],
                           families=[frame.families[i] for i in order], target=order.index(frame.target))
        actual = augment_frame(shuffled, b, planner)
        torch.testing.assert_close(actual.nodes, expected.nodes[nodes], rtol=0, atol=0)
        torch.testing.assert_close(actual.candidates, expected.candidates[candidates], rtol=0, atol=0)
        torch.testing.assert_close(actual.global_features, expected.global_features, rtol=0, atol=0)

    def test_reference_target_reason_and_family_are_never_features(self):
        frame, b, planner = self.fixture()
        expected = augment_frame(frame, b, planner)
        changed = replace(frame, target=4, reference_reason='different teacher decision',
                          families=['different provenance'] * len(frame.actions))
        actual = augment_frame(changed, b, planner)
        for a, e in zip((actual.nodes, actual.candidates, actual.global_features),
                        (expected.nodes, expected.candidates, expected.global_features)):
            self.assertTrue(torch.equal(a, e))

    def test_optional_off_station_survey_keeps_all_existing_obligations(self):
        frame, b, planner = self.fixture()
        action = Action('measure', (50., 50.), 3)
        changed = replace(frame, candidates=torch.cat([frame.candidates, torch.zeros(1, 32)]),
                          actions=frame.actions+[action], survey=frame.survey+[True])
        result = augment_frame(changed, b, planner).candidates[-1]
        self.assertAlmostEqual(float(result[C['other_known_count_over_16']]) * 16, 2.)
        self.assertAlmostEqual(float(result[C['other_station_count_over_49']]) * 49, 2., places=5)
        self.assertAlmostEqual(float(result[C['other_service_s']]) * 1000, 47., places=4)

    def test_real_public_frame_integrates_without_feature_or_action_changes(self):
        env = LocalSimulator(World([Source(1, (900., 0.), 1000., 180.)], seed=1))
        b = Belief()
        for i, point in enumerate(((0., 0.), (250., 10.))):
            action = Action('measure', point, 1)
            b.apply(action, env.execute(action, str(i)), str(i))
        planner = MenuPlanner(Config(resolution=8), menu='external')
        frame = planner.frame(b)
        expected_state = (b.steps, b.position, b.channels[1].region.wkb, tuple(b.channels[1].history))
        result = augment_frame(frame, b, planner)
        self.assertTrue(torch.equal(result.nodes[:, :32], frame.nodes))
        self.assertEqual(result.actions, frame.actions)
        self.assertEqual(result.target, frame.target)
        self.assertEqual(expected_state, (b.steps, b.position, b.channels[1].region.wkb,
                                         tuple(b.channels[1].history)))
        self.assertEqual(float(result.nodes[0, N['missing_service_or_ambiguous_identity']]), 0.)
        self.assertTrue(torch.isfinite(result.candidates).all())
        with self.assertRaisesRegex(ValueError, 'v1'):
            augment_frame(result, b, planner)


if __name__ == '__main__':
    unittest.main()
