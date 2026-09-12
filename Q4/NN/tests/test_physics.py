"""Q4 physical invariants and an independent evaluator-log audit."""

import copy
from dataclasses import asdict
import math
from pathlib import Path
import sys
import unittest

from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nnq4.audit import audit_episode, _exact_surround_certificate
from nnq4.bridge import Action, Belief, DOMAIN, distance, ring_stations
from q4.core import Channel
from q4.shared import Observation, core as q3_core
from q4.simulator import LocalSimulator, Source, World


def _trajectory(world, actions):
    manifest = world.manifest()
    simulator = LocalSimulator(world)
    events = []
    for index, action in enumerate(actions):
        move = distance(simulator.position, action.position)
        response = simulator.execute(action, f"request-{index}")
        events.append({"request_id": f"request-{index}", "action": asdict(action),
                       "response": response, "move_m": move, "executor": "test"})
    return manifest, events


class PhysicsTests(unittest.TestCase):
    def test_directional_closed_half_plane_and_fixed_radius(self):
        source = Source(1, (0.0, 0.0), 1000.0, 0.0)
        world = World([source], seed=3)
        for position, expected in (((-2.0, 0.0), "no_signal"), ((2.0, 0.0), "near"),
                                   ((0.0, 0.0), "near"), ((0.0, 1000.0), "direction"),
                                   ((0.0, -1000.0), "direction"), ((1000.0001, 0.0), "no_signal"),
                                   ((-0.0001, 500.0), "no_signal")):
            with self.subTest(position=position):
                action = Action("measure", position, 1)
                self.assertEqual(world.feedback(action)[0]["measure_result"], expected)
                manifest, events = _trajectory(World([source], seed=3), [action])
                result = audit_episode(manifest, events)
                self.assertTrue(result["ok"], result["errors"])

    def test_clear_uses_20m_disk_even_behind_emitter(self):
        source = Source(2, (0.0, 0.0), 1500.0, 0.0)
        actions = [Action("clear", (-20.0001, 0.0), 2), Action("clear", (-20.0, 0.0), 2)]
        manifest, events = _trajectory(World([source]), actions)
        result = audit_episode(manifest, events)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(events[0]["response"]["clear_result"], "no_target_in_range")
        self.assertEqual(events[1]["response"]["clear_result"], "success")
        self.assertEqual(result["receiver"], 1)
        self.assertEqual(result["costs"]["clear_success_s"], 5)
        self.assertEqual(result["costs"]["clear_failure_s"], 3)
        self.assertEqual(result["costs"]["switch_s"], 0)

    def test_no_signal_cannot_apply_q3_disk_exclusion(self):
        action = Action("measure", (0.0, 0.0), 1)
        truth = Point(2.0, 0.0)
        q4_channel, wrong_q3_channel = Channel(), q3_core.Channel()
        observation = Observation(action, "no_signal")
        q4_channel.update(observation)
        wrong_q3_channel.update(observation)
        self.assertTrue(q4_channel.region.equals(DOMAIN))
        self.assertTrue(q4_channel.region.covers(truth))
        self.assertFalse(wrong_q3_channel.region.covers(truth))
        # A real source two metres away can legitimately be invisible from
        # behind: this makes the Q3 disk subtraction a concrete counterexample.
        manifest, events = _trajectory(World([Source(1, (2.0, 0.0), 1000.0, 0.0)]), [action])
        result = audit_episode(manifest, events)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["truth_support_checks"], 1)

    def test_tiny_nonempty_region_is_not_absence(self):
        channel = Channel()
        channel.region = Point(0.0, 0.0).buffer(1e-5)
        channel.update(Observation(Action("measure", (1500.0, 0.0), 1), "no_signal"))
        self.assertLess(channel.region.area, 1.0)
        self.assertFalse(channel.region.is_empty)
        self.assertEqual(channel.status, "unresolved")
        self.assertTrue(channel.region.covers(Point(0.0, 0.0)))

    def test_exact_surround_certificate_is_geometric_not_an_area_cutoff(self):
        # A sizeable support is certified when four actual negative stations
        # surround all its vertices within R_min. No GEOS union is involved.
        support = Point(0.0, 0.0).buffer(100)
        stations = ((-400.0, 0.0), (0.0, -400.0), (400.0, 0.0), (0.0, 400.0))
        witness = _exact_surround_certificate(support, stations)
        self.assertIsNotNone(witness)
        self.assertGreater(witness[0]['minimum_strict_hull_cross_m2'], 0)
        # Even arbitrarily tiny support has no proof from one-sided stations,
        # or surrounding stations outside the guaranteed reception radius.
        tiny = Point(0.0, 0.0).buffer(1e-12)
        self.assertIsNone(_exact_surround_certificate(tiny, ((10.0, 0.0), (20.0, 5.0), (20.0, -5.0))))
        far = tuple((4*x, 4*y) for x, y in stations)
        self.assertIsNone(_exact_surround_certificate(tiny, far))

    def test_direction_error_includes_rounding_and_is_fixed_by_point(self):
        for mode in ("iid", "extreme", "correlated"):
            with self.subTest(mode=mode):
                source = Source(1, (900.0, 10.0), 1500.0, None)
                first = Action("measure", (0.0, 0.0), 1)
                other = Action("measure", (20.0, 30.0), 1)
                manifest, events = _trajectory(World([source], seed=22, error_mode=mode), [first, other, first])
                result = audit_episode(manifest, events)
                self.assertTrue(result["ok"], result["errors"])
                self.assertLessEqual(result["max_bearing_error_deg"], 1.0050001)
                self.assertEqual(events[0]["response"]["svd_deg"], events[2]["response"]["svd_deg"])
                self.assertEqual(result["fixed_point_checks"], 1)
                tampered = copy.deepcopy(events)
                tampered[0]["response"]["svd_deg"] = (math.degrees(math.atan2(10, 900)) + 1.006) % 360
                invalid = audit_episode(manifest, tampered)
                self.assertFalse(invalid["ok"])
                self.assertIn("Direction error", invalid["errors"][0]["message"])

    def test_rejection_zero_clock_and_duplicate_are_not_recharged(self):
        world = World([Source(2, (300.0, 400.0), 1000.0, 0.0)])
        clear = Action("clear", (300.0, 400.0), 2)
        measure = Action("measure", (301.1234567, 402.2345678), 3)
        manifest, events = _trajectory(world, [clear, measure])
        rejected = {"request_id": "temporary-reject", "action": asdict(measure),
                    "response": {"accepted": False, "virtual_time_s": 0}, "move_m": 0.0}
        duplicate = copy.deepcopy(events[0])
        duplicate["response"]["real_timestamp_ms"] += 12345
        # The duplicate refers to an earlier clock and receiver, after a later
        # accepted request. It must leave the current state exactly unchanged.
        result = audit_episode(manifest, [events[0], rejected, events[1], duplicate])
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["accepted_steps"], 2)
        self.assertEqual(result["duplicate_requests"], 1)
        self.assertEqual(result["rejected_requests"], 1)
        self.assertEqual(result["receiver"], 3)
        self.assertAlmostEqual(result["spent_virtual_s"], events[1]["response"]["virtual_time_s"], places=6)
        self.assertEqual(result["costs"]["switch_s"], 1)
        self.assertEqual(result["costs"]["clear_success_s"], 5)
        changed = copy.deepcopy(duplicate)
        changed["action"]["position"] = (302.0, 400.0)
        invalid = audit_episode(manifest, [events[0], changed])
        self.assertFalse(invalid["ok"])
        self.assertIn("changed action", invalid["errors"][0]["message"])

    def test_accepted_retry_after_rejection_executes_once(self):
        action = Action("measure", (100.0, 20.0), 2)
        manifest, events = _trajectory(World([Source(1, (0.0, 0.0), 1000.0)]), [action])
        rejected = {"request_id": events[0]["request_id"], "action": asdict(action),
                    "response": {"accepted": False, "virtual_time_s": 0}, "move_m": 0.0}
        result = audit_episode(manifest, [rejected, events[0]])
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["accepted_steps"], 1)
        self.assertEqual(result["rejected_requests"], 1)

    def test_rejected_id_does_not_prevent_a_corrected_action(self):
        corrected = Action("measure", (100.0, 20.0), 2)
        manifest, events = _trajectory(World([Source(1, (0.0, 0.0), 1000.0)]), [corrected])
        rejected = {"request_id": events[0]["request_id"],
                    "action": asdict(Action("measure", (0.0, 0.0), 1)),
                    "response": {"accepted": False, "virtual_time_s": 0}, "move_m": 0.0}
        result = audit_episode(manifest, [rejected, events[0]])
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["accepted_steps"], 1)
        self.assertEqual(result["receiver"], 2)

    def test_audit_identifies_local_scope_and_timestamp_only_difference(self):
        action = Action("measure", (0.0, 0.0), 1)
        manifest, events = _trajectory(World([Source(1, (100.0, 0.0), 1000.0)]), [action])
        events[0]["response"]["real_timestamp"] = "intentionally ignored"
        result = audit_episode(manifest, events)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["audit_scope"], "frozen_local_q4_simulator")
        self.assertFalse(result["official_validation"])
        self.assertTrue(result["world_distribution_is_local_assumption"])

    def test_cost_tampering_is_not_a_valid_fast_result(self):
        manifest, events = _trajectory(World([Source(1, (0.0, 0.0), 1000.0)]),
                                       [Action("measure", (1000.0, 0.0), 1)])
        self.assertEqual(events[0]["response"]["virtual_time_s"], 205)
        events[0]["response"]["virtual_time_s"] = 5
        result = audit_episode(manifest, events)
        self.assertFalse(result["ok"])
        self.assertFalse(result["complete"])
        self.assertIsNone(result["seconds_per_source"])
        self.assertIsNone(result["completed_virtual_s"])

    def test_completed_sixteen_clears_and_incomplete_prefix(self):
        sources = [Source(channel, (0.0, 0.0), 1000.0, 0.0) for channel in range(1, 17)]
        manifest, events = _trajectory(World(sources), [Action("clear", (0.0, 0.0), c) for c in range(1, 17)])
        result = audit_episode(manifest, events)
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(result["complete"])
        self.assertTrue(result["all_cleared"] and result["certified_complete"])
        self.assertEqual(result["seconds_per_source"], 5)
        partial = audit_episode(manifest, events[:10])
        self.assertTrue(partial["ok"], partial["errors"])
        self.assertFalse(partial["complete"])
        self.assertIsNone(partial["seconds_per_source"])
        self.assertEqual(partial["spent_virtual_s"], 50)

    def test_ten_clears_need_remaining_channel_certificates(self):
        sources = [Source(channel, (0.0, 0.0), 1000.0) for channel in range(1, 11)]
        actions = [Action("clear", (0.0, 0.0), c) for c in range(1, 11)]
        actions += [Action("measure", position, c) for position in ring_stations() for c in range(11, 21)]
        manifest, events = _trajectory(World(sources), actions)
        partial = audit_episode(manifest, events[:10])
        self.assertTrue(partial["ok"], partial["errors"])
        self.assertTrue(partial["all_cleared"])
        self.assertFalse(partial["certified_complete"])
        self.assertIsNone(partial["seconds_per_source"])
        complete = audit_episode(manifest, events)
        self.assertTrue(complete["ok"], complete["errors"])
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["absence_certificates_checked"], 10)


if __name__ == "__main__":
    unittest.main()
