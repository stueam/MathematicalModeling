import itertools
import math
import unittest

from audit_movement import route_metrics


def clear(channel, point):
    return {'action': {'kind': 'clear', 'channel': channel, 'position': point},
            'response': {'accepted': True, 'clear_result': 'success'}}


class MovementAuditTests(unittest.TestCase):
    def test_exact_reference_matches_brute_force_and_accounts_for_tail(self):
        points = [(100., 0.), (0., 100.), (110., 0.)]
        actions = [clear(i+1, p) for i, p in enumerate(points)]
        actions.append({'action': {'kind': 'measure', 'channel': 4, 'position': (200., 200.)},
                        'response': {'accepted': True, 'measure_result': 'no_signal'}})
        result = route_metrics(actions)
        exact = min(sum(math.dist(a, b) for a, b in zip([(0., 0.)]+list(order), order))
                    for order in itertools.permutations(points))
        self.assertAlmostEqual(result['hindsight_point_tsp_m'], exact)
        self.assertAlmostEqual(result['intermediate_and_tail_excess_m'], math.dist(points[-1], (200., 200.)))
        self.assertAlmostEqual(result['actual_m']-exact,
                               result['intermediate_and_tail_excess_m']+result['hindsight_order_gap_m'])

    def test_bound_allows_opposite_sides_of_clear_disks(self):
        # Source at 100: historical success at 120, alternative success at 80.
        result = route_metrics([clear(1, (120., 0.))])
        self.assertLessEqual(result['source_travel_lower_bound_m'], 80.)
        self.assertAlmostEqual(result['source_travel_lower_bound_m'], 79.99)

    def test_invalid_success_or_rejected_action_is_not_silently_counted(self):
        with self.assertRaises(ValueError):
            route_metrics([clear(1, (100., 0.)), clear(1, (100., 0.))])
        row = clear(1, (100., 0.))
        row['response']['accepted'] = False
        with self.assertRaises(ValueError):
            route_metrics([row])
