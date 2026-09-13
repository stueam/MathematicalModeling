"""Hypothetical worlds sampled exclusively from public Q4 observations.

Uniform area/R/heading and independent point errors are working priors. The
position quadrature is approximate. Sample exhaustion never proves absence.
"""
import math

import numpy as np

from .posterior import Model, QuadratureError
from .shared import point_key
from .simulator import Source, World


def conditional_subset(probabilities, known, rng):
    """Joint Bernoulli draw conditioned on 10 <= known + count <= 16."""
    q = np.asarray(probabilities, dtype=float)
    n = len(q)
    low, high = max(0, 10-known), min(n, 16-known)
    if low > high or np.any(q < 0) or np.any(q > 1):
        raise QuadratureError('Inconsistent source-count support')
    suffix = np.zeros((n+1, n+1))
    suffix[n, 0] = 1.
    for i in range(n-1, -1, -1):
        suffix[i] = (1-q[i])*suffix[i+1]
        suffix[i, 1:] += q[i]*suffix[i+1, :-1]
    counts = np.arange(low, high+1)
    weights = suffix[0, counts]
    if weights.sum() <= 0:
        raise QuadratureError('Zero conditional source-count probability')
    left = int(rng.choice(counts, p=weights/weights.sum()))
    chosen = []
    for i in range(n):
        if left == 0:
            break
        denominator = suffix[i, left]
        chance = q[i]*suffix[i+1, left-1]/denominator if denominator > 0 else 0.
        if rng.random() < np.clip(chance, 0., 1.):
            chosen.append(i)
            left -= 1
    if left:
        raise QuadratureError('Conditional subset failed to allocate count')
    return chosen


def verify_history(world, belief):
    """Independent feedback replay on the sampled world, never the real one."""
    replay = world.clone()
    replay._cleared.clear()
    for c, channel in belief.channels.items():
        for obs in channel.history:
            data, _ = replay.feedback(obs.action)
            result = data.get('measure_result', data.get('clear_result'))
            if result != obs.result:
                raise QuadratureError(f'Sampled world contradicts channel {c} history')
            if obs.result == 'direction':
                if data['svd_deg'] != obs.bearing:
                    raise QuadratureError('Sampled bearing does not replay the public value')
                source = world._sources[c]
                phi = math.degrees(math.atan2(source.position[1]-obs.action.position[1],
                                             source.position[0]-obs.action.position[0]))
                delta = (obs.bearing-phi+180) % 360-180
                if abs(delta) > 1.00500001:
                    raise QuadratureError('Replayed bearing requires an impossible error')
    if replay._cleared != set(belief.cleared):
        raise QuadratureError('Sampled clear history disagrees with public state')


class WorldSampler:
    def __init__(self, model=None):
        self.model = model or Model()

    def sample(self, belief, rng):
        known = sorted(belief.known)
        unknown = [c for c, p in belief.channels.items() if p.status == 'unresolved'] if len(known)<16 else []
        posts = {c: self.model.posterior(belief.channels[c]) for c in known+unknown}
        prior = self.model.existence_prior
        probability = []
        for c in unknown:
            evidence = min(1., posts[c].evidence)
            probability.append(prior*evidence/(1-prior+prior*evidence))
        selected = known+[unknown[i] for i in conditional_subset(probability, len(known), rng)]
        sources, bearings = [], {}
        for c in selected:
            post = posts[c]
            index = int(rng.choice(post.atoms.size, p=post.atoms.ravel()/post.atoms.sum()))
            row, atom = divmod(index, post.atoms.shape[1])
            xy = tuple(map(float, post.xy[row]))
            radius = float(rng.uniform(post.lower[row], post.upper[row, atom]))
            heading = (math.degrees(float(rng.uniform(post.left[row, atom-1], post.right[row, atom-1]))) % 360
                       if atom else None)
            sources.append(Source(c, xy, radius, heading))
            for obs in belief.channels[c].directions:
                bearings[c, point_key(obs.action.position)] = obs.bearing
        world = World(sources, seed=int(rng.integers(0, 2**63)), error_mode='iid',
                      cleared=belief.cleared, known_bearings=bearings)
        verify_history(world, belief)
        return world
