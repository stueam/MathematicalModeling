"""Experimental one-stop initial triangulation, scored against the V6 route."""
from dataclasses import dataclass
import math

import numpy as np

from .posterior import expected_after_measure, finish_cost
from .refinement import RefinementConfig, RefinementPolicy
from .shared import Action, distance


@dataclass(frozen=True)
class ScoutConfig(RefinementConfig):
    scout_radius: float = 250.


class ScoutPolicy(RefinementPolicy):
    implementation = 'bayes_initial_triangulation'

    def _select_task(self, b, posts, stable, pending, points):
        selected, candidates, route, future = super()._select_task(b, posts, stable, pending, points)
        if distance(b.position, (0., 0.)) > 1e-6 or len(posts) < 3:
            return selected, candidates, route, future
        destination = selected['action'].position
        length = distance(b.position, destination)
        if length < self.config.scout_radius:
            return selected, candidates, route, future
        theta = math.atan2(destination[1], destination[0])
        proposals = []
        for offset in (-math.pi/6, 0., math.pi/6):
            pos = (self.config.scout_radius*math.cos(theta+offset),
                   self.config.scout_radius*math.sin(theta+offset))
            gains = []
            for c, post in posts.items():
                if not self._fresh(b.channels[c], pos) or self.guaranteed_clear(b, c):
                    continue
                after, _ = expected_after_measure(post, pos, self.config.bearing_bin)
                gain = finish_cost(pos, post.xy, post.weights)-after-6
                if gain > 2:
                    gains.append((gain, c))
            detour = (distance(b.position, pos)+distance(pos, destination)-length)/5
            net = sum(g for g, _ in gains)-detour
            if gains and net > 5:
                channels = [c for _, c in sorted(gains, reverse=True)]
                proposal = {**selected, 'action': Action('measure', pos, channels[0]),
                    'score_s': selected['score_s']-net, 'initial_scout': True,
                    'scout_channels': channels, 'scout_gain_s': net,
                    'scout_detour_s': detour, 'score_kind': 'sum_local_information_proxy'}
                proposals.append(proposal)
        if not proposals:
            return selected, candidates, route, future
        selected = min(proposals, key=lambda item: item['score_s'])
        self._joint_details['service_channels'] = selected['scout_channels']
        return selected, candidates+proposals, route, [selected['action'].position]+future

    def __init__(self, config=None):
        super().__init__(config or ScoutConfig())
