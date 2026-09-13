"""Optional whole-convex-piece certificate with an explicit 0.5 m margin."""
import numpy as np
import shapely
from shapely.errors import GEOSException
from shapely.geometry import MultiPoint, Polygon

from .core import Channel
from .coverage import full_exclusion
from .sector import ProbePolicy


def convex_piece_certifies(region,positions):
    if region.is_empty:
        return True
    if not positions:
        return False
    points=np.asarray(tuple(sorted(set(positions))),dtype=float)
    parts=list(region.geoms) if hasattr(region,'geoms') else [region]
    try:
        for part in parts:
            if part.is_empty:
                continue
            q=part.convex_hull  # Outer enclosure; holes never justify exclusion.
            vertices=shapely.get_coordinates(q)
            farthest=np.linalg.norm(points[:,None]-vertices[None],axis=2).max(axis=1)
            witnesses=points[farthest<=999.5]
            if not len(witnesses) or not MultiPoint(witnesses).convex_hull.covers(q):
                return False
        return True
    except GEOSException:
        return False


def hull_certifies(region,positions):
    points=tuple(sorted(set(tuple(map(float,p)) for p in positions)))
    try:
        residual=region.difference(full_exclusion(points))
        return convex_piece_certifies(residual,points)
    except GEOSException:
        return False


class ConvexChannel(Channel):
    def update(self,obs):
        super().update(obs)
        if obs.result=='no_signal' and self.status=='unresolved' and convex_piece_certifies(self.region,self.negatives):
            self.region=Polygon()
            self.status='absent_certified'
            self._summary=None


class HullPolicy(ProbePolicy):
    def __init__(self,config=None):
        super().__init__(config)
        self.implementation='q4_external_convex_piece_certificate_v1'

    def valid_future(self,b,points):
        self.counters['certificate_reviews']+=1
        seen=set()
        for channel in b.channels.values():
            if channel.status!='unresolved':
                continue
            key=(channel.region.wkb,tuple(sorted(channel.negatives)))
            if key in seen:
                continue
            seen.add(key)
            if not hull_certifies(channel.region,channel.negatives+tuple(points.values())):
                return False
        return True
