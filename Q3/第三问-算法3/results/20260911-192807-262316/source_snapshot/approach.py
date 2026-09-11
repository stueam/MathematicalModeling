"""Nearest fully certified clearance point from the complete feasible lens.

The intersection of radius-20 disks centered on every convex-hull vertex is
exactly the set of clearing locations that contain the complete hard support.
This module only improves an already certified endpoint; no posterior mass or
unobserved source coordinate enters the geometric decision.
"""
from dataclasses import dataclass

import numpy as np
import shapely

from .refinement import RefinementConfig, RefinementPolicy
from .repair import repair_segment
from .shared import Action, distance, point_key


def nearest_safe_point(vertices, position, witness, radius=19.999):
    """Project onto an equal-radius disk intersection, with a feasible fallback.

For a point outside a convex disk intersection the nearest boundary point lies
either on one circle's radial projection or at a pairwise circle intersection.
All candidates are checked against every vertex. Small floating-point errors
are repaired toward the known feasible witness without relaxing the radius.
    """
    vertices = np.unique(np.asarray(vertices, dtype=float), axis=0)
    origin = np.asarray(position, dtype=float)
    witness = np.asarray(witness, dtype=float)
    if (not len(vertices) or not np.isfinite(vertices).all() or
            not np.isfinite(origin).all() or not np.isfinite(witness).all() or
            not np.isfinite(radius) or radius <= 0):
        raise ValueError('Nonempty finite vertices and positive radius required')
    def constraint(point):
        return radius-np.linalg.norm(vertices-point, axis=1)
    if np.min(constraint(witness)) < 0:
        raise ValueError('The fallback witness must satisfy every safety disk')
    if np.min(constraint(origin)) >= 0:
        return tuple(map(float, origin))
    delta = origin-vertices
    lengths = np.linalg.norm(delta, axis=1)
    candidates = [witness[None, :], vertices+radius*delta/np.maximum(lengths[:, None], 1e-12)]
    i, j = np.triu_indices(len(vertices), 1)
    difference = vertices[j]-vertices[i]
    separation = np.linalg.norm(difference, axis=1)
    usable = (separation > 1e-12) & (separation <= 2*radius)
    if usable.any():
        i, j = i[usable], j[usable]
        difference, separation = difference[usable], separation[usable]
        middle = (vertices[i]+vertices[j])/2
        perpendicular = np.column_stack((-difference[:, 1], difference[:, 0]))/separation[:, None]
        height = np.sqrt(np.maximum(0., radius**2-(separation/2)**2))
        candidates.extend([middle+perpendicular*height[:, None], middle-perpendicular*height[:, None]])
    candidates = np.vstack(candidates)
    margins = radius-np.linalg.norm(candidates[:, None, :]-vertices[None, :, :], axis=2).max(axis=1)
    candidates = candidates[margins >= -1e-7]
    if not len(candidates):
        return tuple(map(float, witness))
    chosen = candidates[np.argmin(np.linalg.norm(candidates-origin, axis=1))]
    repaired = repair_segment(witness, chosen, constraint)
    if repaired is None or distance(origin, repaired) > distance(origin, witness)+1e-9:
        repaired = witness
    return tuple(map(float, repaired))


@dataclass(frozen=True)
class ApproachConfig(RefinementConfig):
    safe_lens_clear: bool = True


class ApproachPolicy(RefinementPolicy):
    implementation = 'bayes_tsp_experimental_safe_lens_clear'

    def __init__(self, config=None):
        super().__init__(config or ApproachConfig())
        self._safe_endpoint_cache = {}

    def guaranteed_clear(self, b, c):
        reference = super().guaranteed_clear(b, c)
        if (reference is None or not self.config.safe_lens_clear or
                distance(b.position, reference.position) < 1e-9):
            return reference
        channel = b.channels[c]
        key = (c, channel.revision, point_key(b.position))
        if key not in self._safe_endpoint_cache:
            vertices = shapely.get_coordinates(channel.region.convex_hull)
            point = nearest_safe_point(vertices, b.position, reference.position)
            if len(self._safe_endpoint_cache) >= 64:
                self._safe_endpoint_cache.pop(next(iter(self._safe_endpoint_cache)))
            self._safe_endpoint_cache[key] = Action('clear', point, c)
        return self._safe_endpoint_cache[key]
