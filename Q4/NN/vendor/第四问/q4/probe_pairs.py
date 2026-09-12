"""Conditional two-stop probes using the existing fixed-R/heading prior.

Only a real first-stop miss commits the mirrored second stop. Prediction is
used for costs, never to update hard geometry or certify absence/completion.
"""
import math
import time

import numpy as np
import shapely

from .bilateral import BilateralProbePolicy, K, frame
from .localization import guaranteed_clear, local_attempts
from .posterior import Posterior, QuadratureError, expected_measure, finish_proxy
from .shared import Action, distance, point_key


def condition_miss(post, point):
    """Split heading intervals and truncate the SAME R after no_signal.

    In an illuminated heading interval a miss means R < distance; in a dark
    interval it says nothing about R. Position quadrature points stay fixed.
    Returns (normalized conditional posterior, probability of the miss).
    """
    delta = np.asarray(point)-post.xy
    d = np.linalg.norm(delta, axis=1)
    alpha = np.arctan2(delta[:, 1], delta[:, 0]) % math.tau
    edges = np.sort(np.column_stack((post.left, post.right[:, -1],
                    (alpha-math.pi/2) % math.tau, (alpha+math.pi/2) % math.tau)), axis=1)
    left, right = edges[:, :-1], edges[:, 1:]
    mid = (left+right)/2
    index = np.sum(mid[:, :, None] >= post.right[:, None, :], axis=2)
    index = np.minimum(index, post.left.shape[1]-1)
    old_left = np.take_along_axis(post.left, index, axis=1)
    old_right = np.take_along_axis(post.right, index, axis=1)
    old_hi = np.column_stack((post.upper[:, 0], np.take_along_axis(post.upper[:, 1:], index, axis=1)))
    old_mass = np.take_along_axis(post.atoms[:, 1:], index, axis=1)
    fraction = np.divide(right-left, old_right-old_left, out=np.zeros_like(left),
                         where=old_right > old_left)
    atoms = np.column_stack((post.atoms[:, 0], old_mass*fraction))
    lit = (np.cos(mid-alpha[:, None]) >= 0) | (d[:, None] == 0)
    lit = np.column_stack((np.ones(len(d), dtype=bool), lit))
    hi = np.where(lit, np.minimum(old_hi, d[:, None]), old_hi)
    span = old_hi-post.lower[:, None]
    radial = np.divide(np.maximum(0., hi-post.lower[:, None]), span,
                       out=np.zeros_like(span), where=span > 0)
    atoms *= radial
    mass = float(atoms.sum())
    if mass <= 1e-15:
        return None, mass
    return Posterior(post.xy, atoms/mass, post.lower, hi, left, right,
                     post.evidence*mass), mass


def expected_pair(post, first, second, bins=2.):
    """Positive first feedback replans; a miss pays the second stop in full."""
    future, detail = expected_measure(post, first, bins)
    missed, probability = condition_miss(post, first)
    if missed is not None:
        second_future, second_detail = expected_measure(missed, second, bins)
        # Replace exactly the first miss branch, not the received branches.
        recovery = finish_proxy(first, post.xy, missed.weights)
        future += probability*(distance(first, second)/5+5+second_future-recovery)
    else:
        second_detail = None
    return future, {**detail, 'conditional_second': second_detail,
                    'pair_miss_probability': probability,
                    'mirror_move_m': distance(first, second),
                    'fixed_R_and_heading_conditioned': True,
                    'cost_is_proxy': True, 'full_rollout': False}


class PairProbePolicy(BilateralProbePolicy):
    def __init__(self, config=None, direct=False):
        super().__init__(config, add_candidates=False)
        self.direct = direct
        self.implementation = 'q4_github_bilateral_direct_v1' if direct else 'q4_github_bilateral_pair_v1'
        self.pending_pair = None
        self.pair_cache = {}
        self.counters.update(pair_candidates=0, pair_started=0, pair_second_stops=0,
                             pair_positive_replans=0, pair_cancelled=0)

    def local_choices(self, b, c):
        choices, mean = super().local_choices(b, c)
        channel = b.channels[c]
        if (self.completion_mode or not channel.directions or guaranteed_clear(b, c)
                or local_attempts(channel)+1 >= self.config.local_limit):
            return choices, mean
        origin, basis = frame(channel)
        xy = (shapely.get_coordinates(channel.region)-origin) @ basis.T
        lo, hi = max(0., float(xy[:, 0].min())), float(xy[:, 0].max())
        if hi-lo <= 24:
            return choices, mean
        mid = (lo+hi)/2
        points = [tuple(map(float, np.array([mid, sign*(K*mid+5)]) @ basis+origin))
                  for sign in (-1, 1)]
        if any(distance(q, o.action.position) < 10 for q in points
               for o in channel.history if o.action.kind == 'measure'):
            return choices, mean
        try:
            post = self.model.posterior(channel)
            pairs = []
            for first, second in (points, points[::-1]):
                key = (c, channel.revision, first, second, self.config.bearing_bin)
                if key not in self.pair_cache:
                    if len(self.pair_cache) >= 256:
                        self.pair_cache.clear()
                    self.pair_cache[key] = expected_pair(post, first, second, self.config.bearing_bin)
                future, detail = self.pair_cache[key]
                cost = distance(b.position, first)/5+5+int(c != b.receiver)+future
                pairs.append((Action('measure', first, c), cost,
                    {**detail, 'probe_pair': {'first': first, 'second': second},
                     'original_choices_retained': not self.direct,
                     'direct_bisection_localizer': self.direct}))
                self.counters['pair_candidates'] += 1
            choices = pairs if self.direct else choices+pairs
        except QuadratureError:
            pass
        return choices, mean

    def choose(self, b):
        fallback = (self.completion_mode or b.steps >= self.config.completion_after
                    or b.deadline-time.monotonic() < self.config.reserve_s)
        if self.pending_pair is not None:
            pending = self.pending_pair
            c = pending['channel']
            channel = b.channels[c]
            if fallback or b.done() or channel.status != 'detected' or guaranteed_clear(b, c):
                self.pending_pair = None
                self.counters['pair_cancelled'] += 1
            else:
                first = Action('measure', pending['first'], c)
                second = Action('measure', pending['second'], c)
                feedback = channel.history[pending['history_start']:]
                first_obs = next((o for o in feedback if o.action == first), None)
                second_obs = next((o for o in feedback if o.action == second), None)
                if first_obs is None:
                    return self.record(b, first, 'pair_wait_first_feedback', pair_state=dict(pending))
                if first_obs.result == 'no_signal' and second_obs is None:
                    if not pending['second_issued']:
                        self.counters['pair_second_stops'] += 1
                        pending['second_issued'] = True
                    self.batch = None
                    return self.record(b, second, 'pair_mirror_after_actual_miss', pair_state=dict(pending))
                self.counters['pair_positive_replans'] += int(first_obs.result != 'no_signal')
                self.pending_pair = None
        action = super().choose(b)
        candidates = self.records[-1].get('candidates', [])
        selected = min(candidates, key=lambda row: row['score_s']) if candidates else {}
        pair = selected.get('probe_pair')
        if pair and point_key(pair['first']) == point_key(action.position):
            self.pending_pair = {**pair, 'channel': action.channel,
                'history_start': len(b.channels[action.channel].history), 'second_issued': False}
            self.counters['pair_started'] += 1
            self.records[-1]['pair_state'] = dict(self.pending_pair)
        return action
