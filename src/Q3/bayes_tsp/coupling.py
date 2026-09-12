"""Algorithm-three-only conservative geometry for a fixed unknown radius.

A reception at a and a no-signal at b imply |x-a| < |x-b|. No source truth,
probability threshold, or alteration of the shared q3 core is involved.
"""
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import Polygon

from .shared import Belief, GeometryError, core


def receive_halfplane(region, positive, negative):
    a, b = np.asarray(positive, dtype=float), np.asarray(negative, dtype=float)
    delta = b-a
    norm = float(np.linalg.norm(delta))
    if norm == 0:
        raise GeometryError('Same location cannot both receive and miss a fixed active source')
    if norm < 1e-9:
        return region  # Weak/ill-conditioned cut: retaining an outer set is safe.
    n = delta/norm
    midpoint = a+delta/2
    vertices = shapely.get_coordinates(region.convex_hull)
    # Evaluate around the midpoint instead of subtracting large squared norms.
    scale = max(float(np.abs(a).max()), float(np.abs(b).max()), float(np.abs(vertices).max()), 1.)
    slack = 1e-6+32*np.finfo(float).eps*scale
    signed = (vertices-midpoint) @ n
    if signed.max() <= slack:
        return region
    if signed.min() > slack:
        raise GeometryError('Fixed-radius reception/no-signal constraints have no feasible position')
    tangent = np.array([-n[1], n[0]])
    anchor = midpoint+slack*n
    along = (vertices-anchor) @ tangent
    lo, hi = float(along.min())-1., float(along.max())+1.
    depth = max(1., float(-signed.min())+2.)
    allowed = Polygon([anchor+lo*tangent, anchor+hi*tangent,
                       anchor+hi*tangent-depth*n, anchor+lo*tangent-depth*n])
    clipped = region.intersection(allowed)
    if clipped.is_empty:
        raise GeometryError('Empty coupled support for a detected source; not an absence proof')
    return clipped


@dataclass
class CoupledChannel(core.Channel):
    coupling_checks: int = 0
    coupling_cuts: int = 0
    coupling_area_removed_m2: float = 0.

    def update(self, obs):
        # Transactional: even an accepted but inconsistent observation must not
        # leave half-applied local geometry before the controller reports failure.
        candidate = self.clone()
        core.Channel.update(candidate, obs)
        if candidate.status == 'detected' and obs not in self.history:
            if obs.result == 'direction':
                pairs = [(obs.action.position, old.action.position) for old in candidate.history if old.result == 'no_signal']
            elif obs.result == 'no_signal':
                pairs = [(old.action.position, obs.action.position) for old in candidate.history if old.result == 'direction']
            else:
                pairs = []
            for positive, negative in pairs:
                before = candidate.region
                candidate.region = receive_halfplane(before, positive, negative)
                candidate.coupling_checks += 1
                if not candidate.region.equals(before):
                    candidate.coupling_cuts += 1
                    candidate.coupling_area_removed_m2 += max(0., before.area-candidate.region.area)
                candidate._summary = None
        self.__dict__.update(candidate.__dict__)


@dataclass
class CoupledBelief(Belief):
    channels: dict = field(default_factory=lambda: {c: CoupledChannel() for c in range(1, 21)})
