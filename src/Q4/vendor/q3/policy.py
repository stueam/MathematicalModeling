"""Deterministic, observation-only completion policy and bounded proposals."""
import math

import numpy as np
import shapely
from shapely.geometry import box

from .core import Action, distance, point_key, GeometryError


class Baseline:
    def __init__(self, ring_radius=1500., max_bearings=4):
        worst = max(math.sqrt(r*r + ring_radius**2 - 2*r*ring_radius*math.cos(math.pi/6))
                    for r in (1000, 1800))
        if worst >= 995:  # Margin also covers our circumscribed domain polygon.
            raise ValueError('Patrol ring lacks sufficient coverage margin')
        self.stations = [(0., 0.)] + [(ring_radius*math.cos(k*math.pi/3),
                                      ring_radius*math.sin(k*math.pi/3)) for k in range(6)]
        self.max_bearings = max_bearings
        self.fallback_calls = 0

    def unknown_at(self, b, position):
        key = point_key(position)
        return [c for c, p in b.channels.items()
                if p.status == 'unresolved' and key not in p.measured]

    def scan_action(self, b, position):
        channels = self.unknown_at(b, position)
        if not channels:
            return None
        c = b.receiver if b.receiver in channels else min(channels)
        return Action('measure', position, c)

    def guaranteed_clear(self, b, c):
        p = b.channels[c]
        if p.status != 'detected':
            return None
        center, radius = p.summary()
        vertices = shapely.get_coordinates(p.region.convex_hull)
        if np.linalg.norm(vertices - b.position, axis=1).max() <= 19.999:
            return Action('clear', b.position, c)
        if radius > 19.999:
            return None
        # A disk of radius (20-r) around this center consists of safe clearing
        # positions. Choose its nearest point, not always the center itself.
        d = distance(center, b.position)
        step = min(d, 19.999-radius)
        pos = center if d == 0 else tuple(center[i] + step*(b.position[i]-center[i])/d for i in range(2))
        return Action('clear', pos, c)

    def measurement_candidates(self, b, c):
        p = b.channels[c]
        history = p.directions
        if not history:
            return []
        if len(history) == 1:
            first = history[0]
            theta = math.radians(first.bearing)
            u, v = np.array([math.cos(theta), math.sin(theta)]), np.array([-math.sin(theta), math.cos(theta)])
            points = [np.asarray(first.action.position) + 750*u + side*500*v for side in (-1, 1)]
        else:
            center, radius = p.summary()
            last = history[-1]
            theta = math.radians(last.bearing)
            v = np.array([-math.sin(theta), math.cos(theta)])
            offset = min(200., max(35., 1.3*radius))
            points = [np.asarray(center) + sign*offset*v for sign in (-1, 1)]
        actions = [Action('measure', tuple(map(float, q)), c) for q in points
                   if point_key(q) not in p.measured]
        return sorted(actions, key=lambda a: distance(b.position, a.position))

    def grid_clear(self, b, c):
        """Finite fallback: include every intersecting 20 m square, even slivers."""
        self.fallback_calls += 1
        p = b.channels[c]
        x0, y0, x1, y1 = p.region.bounds
        failed = {point_key(o.action.position) for o in p.history if o.result == 'no_target_in_range'}
        candidates = []
        for ix in range(math.floor(x0/20), math.floor(x1/20)+1):
            for iy in range(math.floor(y0/20), math.floor(y1/20)+1):
                pos = (ix*20+10., iy*20+10.)
                if point_key(pos) not in failed:
                    candidates.append(pos)
        for pos in sorted(candidates, key=lambda q: distance(b.position, q)):
            if p.region.intersects(box(pos[0]-10, pos[1]-10, pos[0]+10, pos[1]+10)):
                return Action('clear', pos, c)
        raise GeometryError('No untried fallback cell covers the nonempty region')

    def for_channel(self, b, c):
        p = b.channels[c]
        if p.status != 'detected':
            raise ValueError('Localizer requires a detected source')
        for o in reversed(p.history):
            if o.result == 'near':
                return Action('clear', o.action.position, c)
        clear = self.guaranteed_clear(b, c)
        if clear is not None:
            return clear
        # Count all informative/nonrepeated local observations too: unsuccessful
        # measurements must not allow an infinite localizer loop.
        measurements = sum(o.action.kind == 'measure' for o in p.history)
        if len(p.directions) >= self.max_bearings or measurements >= 10:
            return self.grid_clear(b, c)
        candidates = self.measurement_candidates(b, c)
        return candidates[0] if candidates else self.grid_clear(b, c)

    def choose(self, b):
        if b.done():
            raise ValueError('No next action after completion')
        detected = [c for c, p in b.channels.items() if p.status == 'detected']
        for c in detected:
            clear = self.guaranteed_clear(b, c)
            if clear is not None and distance(clear.position, b.position) < 1e-5:
                return clear
            if any(o.result == 'near' for o in b.channels[c].history):
                return self.for_channel(b, c)
        # Complete the necessary scan at a patrol stop before departing.
        for station in self.stations:
            if distance(station, b.position) < 1e-7:
                scan = self.scan_action(b, station)
                if scan is not None:
                    return scan
        if detected:
            candidates = [self.for_channel(b, c) for c in detected]
            return min(candidates, key=lambda a: distance(b.position, a.position)/5 +
                       (0 if a.kind == 'clear' else 10 + b.channels[a.channel].summary()[1]/5))
        for station in sorted(self.stations, key=lambda s: distance(b.position, s)):
            scan = self.scan_action(b, station)
            if scan is not None:
                return scan
        raise GeometryError('Patrol exhausted but completion not certified')

    def proposals(self, b, max_candidates=8, speculative_clear=False):
        base = self.choose(b)
        actions = [base]
        if base.kind == 'clear' and distance(base.position, b.position) < 1e-6:
            return actions
        unknown = self.unknown_at(b, b.position)
        for c in sorted(unknown, key=lambda c: (c != b.receiver, c))[:2]:
            actions.append(Action('measure', b.position, c))
        detected = sorted((c for c, p in b.channels.items() if p.status == 'detected'),
                          key=lambda c: distance(b.position, b.channels[c].summary()[0]))[:2]
        for c in detected:
            clear = self.guaranteed_clear(b, c)
            if clear:
                actions.append(clear)
            else:
                actions.extend(self.measurement_candidates(b, c)[:2])
                center, radius = b.channels[c].summary()
                if speculative_clear and radius < 100:
                    actions.append(Action('clear', center, c))
        stops = sorted(self.stations, key=lambda s: distance(b.position, s))
        for station in stops:
            scan = self.scan_action(b, station)
            if scan is not None and distance(station, b.position) > 1e-6:
                actions.append(scan)
                break
        return list(dict.fromkeys(actions))[:max_candidates]
