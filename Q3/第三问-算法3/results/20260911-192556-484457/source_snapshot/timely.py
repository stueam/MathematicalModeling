"""Compare useful measurements now with discounted, uncommitted future stops.

The discount is a scheduling heuristic, not a probability or completion proof.
Every executed measurement still updates the original conservative geometry.
"""
from dataclasses import dataclass

from .posterior import expected_after_measure, finish_cost
from .refinement import RefinementConfig, RefinementPolicy
from .shared import Action


@dataclass(frozen=True)
class TimelyConfig(RefinementConfig):
    future_discount: float = 0.


class TimelyPolicy(RefinementPolicy):
    implementation = 'bayes_timely_measurements'

    def __init__(self, config=None):
        super().__init__(config or TimelyConfig())
        if not 0 <= self.config.future_discount <= 1:
            raise ValueError('future_discount must be between zero and one')

    def _known_sensing(self, b, posts, future, target):
        choices, reviews = [], []
        for c, post in posts.items():
            p = b.channels[c]
            if (self.guaranteed_clear(b, c) is not None or len(p.directions) >= self.max_bearings
                    or not self._fresh(p, b.position)):
                continue
            before = finish_cost(b.position, post.xy, post.weights)
            after, probabilities = expected_after_measure(post, b.position, self.config.bearing_bin)
            gain = before-after-5-int(c != b.receiver)
            future_gain = 0.
            if c != target and self.config.future_discount:
                for pos in future:
                    if self._fresh(p, pos):
                        later, _ = expected_after_measure(post, pos, self.config.bearing_bin)
                        future_gain = max(future_gain, finish_cost(pos, post.xy, post.weights)-later-6)
            discounted = self.config.future_discount*future_gain
            execute = gain > 2 and gain > discounted+2
            reviews.append({'channel': c, 'now_gain_s': gain, 'future_gain_s': future_gain,
                            'discounted_future_gain_s': discounted, 'execute_candidate': execute})
            if execute:
                choices.append((gain-discounted, Action('measure', b.position, c), probabilities))
        self._decision_context['known_sensing_review'] = reviews
        return max(choices, key=lambda item: item[0]) if choices else None
