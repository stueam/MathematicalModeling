"""A small fixed family of certified V shapes, scored by existing probes.

The coordinate family follows luxury221's shaped-probe idea; its hypothesis
weights and geometry updates are not imported. The existing front/miss/heading
branches already implement the separate forward-heading suggestion.
"""
import math

import numpy as np
import shapely
from shapely.geometry import Point

from .localization import guaranteed_clear, local_attempts, measure_points
from .posterior import QuadratureError
from .sector import ProbePolicy
from .shared import distance


SHAPES = ((.7, .2), (.7, .5), (.85, .2), (.85, .5), (.95, .2), (.95, .5))


def shaped_pair(channel, fraction, aspect):
    """Both points are nearer than a real positive anchor for the WHOLE region.

    The source-to-anchor segment crosses the interior of the probe segment.
    Thus at least one endpoint is on the illuminated side for any admissible
    heading. Squared-distance differences are affine: checking all polygon
    vertices proves the conditions continuously, including disconnected parts.
    """
    if not 0 < fraction < 1 or not 0 < aspect <= 1:
        raise ValueError('Invalid shape')
    if not channel.directions:
        return None
    obs = channel.directions[-1]
    origin = np.asarray(obs.action.position)
    lower = max(0., channel.region.distance(Point(origin))-1e-4)
    if lower < 1:
        return None
    angle = math.radians(obs.bearing)
    basis = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
    vertices = (shapely.get_coordinates(channel.region)-origin) @ basis.T
    forward = fraction*lower
    lateral = aspect*forward
    pair = np.array([[forward, lateral], [forward, -lateral]])
    gap = float(np.min(vertices[:, 0]-forward))
    cone_slack = float(np.min(lateral*vertices[:, 0]-forward*np.abs(vertices[:, 1])))
    # |G-q|^2-|G-anchor|^2 = |q-anchor|^2-2(G-anchor).(q-anchor).
    max_change = float(np.max(np.sum(pair*pair, axis=1)[:, None]-2*pair @ vertices.T))
    if gap <= 1e-4 or cone_slack <= 1e-4 or max_change >= -1e-4:
        return None
    return [tuple(map(float, p)) for p in pair @ basis+origin], {
        'fraction': fraction, 'aspect': aspect, 'forward_m': forward, 'lateral_m': lateral,
        'max_squared_distance_change': max_change, 'minimum_forward_gap_m': gap,
        'minimum_cone_slack': cone_slack, 'pair_has_received_witness': True,
        'proof_used_for_exclusion': False}


def existing_proposals(policy, b, c, mean, choices):
    """Coordinates from the whole old pool, not only the refined finalists."""
    channel = b.channels[c]
    points = [a.position for a, _, _ in choices]+measure_points(b, c, mean)
    first = np.asarray(channel.directions[-1].action.position)
    center = np.asarray(mean)
    vector = center-first
    normal = np.array([-vector[1], vector[0]])/max(float(np.linalg.norm(vector)), 1.)
    points += [tuple(center+s*o*normal) for s in (-1, 1) for o in (20., 100., 200.)]
    points += [tuple(first+f*vector+s*100*normal) for f in (.35, .65) for s in (-1, 1)]
    goals = {j: p.summary()[0] for j, p in b.channels.items() if p.status == 'detected'}
    nearest = sorted(policy.points, key=lambda k: distance(policy.points[k], goals[c]))[:2]
    goals.update({k: policy.points[k] for k in nearest if policy.needed(b, policy.points[k])})
    for j in sorted((j for j in goals if j != c), key=lambda j: distance(center, goals[j]))[:3]:
        goal = np.asarray(goals[j])
        delta = goal-b.position
        t = np.clip(float((center-b.position) @ delta)/max(float(delta @ delta), 1e-12), 0, 1)
        points += [tuple(goal), tuple((goal+center)/2), tuple(np.asarray(b.position)+t*delta)]
    # Compare the previously tested front/miss/heading coordinates as well.
    from .external_candidates import heading_moment
    u, kappa, probability = heading_moment(policy.model.posterior(channel))
    if kappa >= .4 and probability >= .5:
        points.append(tuple(center+350*u))
    last = max(i for i, o in enumerate(channel.history) if o.result in ('direction', 'near'))
    misses = sum(o.result == 'no_signal' for o in channel.history[last+1:])
    if misses:
        toward = first-center
        toward /= max(float(np.linalg.norm(toward)), 1.)
        sideways = np.array([-toward[1], toward[0]])
        offset = 100.*min(misses, 6)
        points += [tuple(center+offset*toward+s*.5*offset*sideways) for s in (-1, 1)]
    return points


class ShapedPolicy(ProbePolicy):
    def __init__(self, config=None):
        super().__init__(config)
        self.implementation = 'q4_github_shaped_candidates_v1'
        self.counters.update(shaped_candidates=0, shaped_duplicates=0, shaped_selected=0)

    def local_choices(self, b, c):
        choices, mean = super().local_choices(b, c)
        channel = b.channels[c]
        if (not channel.directions or guaranteed_clear(b, c)
                or local_attempts(channel) >= self.config.local_limit):
            return choices, mean
        try:
            existing = existing_proposals(self, b, c, mean, choices)
            additions = []
            for fraction, aspect in SHAPES:
                proposal = shaped_pair(channel, fraction, aspect)
                if proposal is None:
                    continue
                points, proof = proposal
                for point in points:
                    separation = min(distance(point, x) for x in existing)
                    if separation < 10 or any(distance(point, o.action.position) < 10
                            for o in channel.history if o.action.kind == 'measure'):
                        self.counters['shaped_duplicates'] += 1
                        continue
                    if any(distance(point, x[0]) < 10 for x in additions):
                        continue
                    additions.append((point, {**proof, 'distance_to_old_pool_m': separation}))
            self.counters['shaped_candidates'] += len(additions)
            coarse = sorted((self.probe(b, c, point, 10.)[1], i) for i, (point, _) in enumerate(additions))
            for _, i in coarse[:3]:
                point, proof = additions[i]
                action, cost, detail = self.probe(b, c, point, self.config.bearing_bin)
                choices.append((action, cost, {**detail, 'shape': proof, 'original_choices_retained': True}))
        except QuadratureError:
            pass
        return choices, mean

    def choose(self, b):
        action = super().choose(b)
        candidates = self.records[-1].get('candidates', [])
        if candidates and 'shape' in min(candidates, key=lambda r: r['score_s']):
            self.counters['shaped_selected'] += 1
        return action
