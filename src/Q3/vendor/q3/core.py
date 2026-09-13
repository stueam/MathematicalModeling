"""Public types and conservative geometry. No simulator truth is imported here."""

from dataclasses import dataclass, field
import math
import copy

import numpy as np
import shapely
from shapely.geometry import Point, Polygon

EPS_DEG = 1.0050001  # ±1° physical error plus two-decimal rounding
SAFETY = 1e-6


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_key(p):
    # Exact submitted coordinates; normalize signed zero only.
    return tuple(float(x or 0).hex() for x in p)


@dataclass(frozen=True)
class Action:
    kind: str
    position: tuple[float, float]
    channel: int

    def __post_init__(self):
        if self.kind not in ('measure', 'clear'):
            raise ValueError('Only measure and clear are physical actions')
        if type(self.channel) is not int or not 1 <= self.channel <= 20:
            raise ValueError('Invalid channel')
        if len(self.position) != 2 or any(not math.isfinite(x) or abs(x) > 2e6 for x in self.position):
            raise ValueError('Invalid position')

    def payload(self):
        return {
            'position': {'x': float(self.position[0]), 'y': float(self.position[1])},
            'channel': self.channel,
        }


@dataclass(frozen=True)
class Observation:
    action: Action
    result: str
    bearing: float | None = None


def disk(p, r, outer=False):
    """Allowed disks circumscribed; excluded disks inscribed, with safe margin."""
    n = 64
    radius = (r + SAFETY) / math.cos(math.pi / n) if outer else max(0, r - SAFETY)
    return Point(p).buffer(radius, quad_segs=n // 4)


DOMAIN = disk((0, 0), 1800, outer=True)


def bearing_wedge(p, angle):
    th, eps = math.radians(angle), math.radians(EPS_DEG)
    reach = 4000  # Beyond the allowed 1500 m receiving disk, including outer error.
    return Polygon(
        [
            p,
            (p[0] + reach * math.cos(th - eps), p[1] + reach * math.sin(th - eps)),
            (p[0] + reach * math.cos(th + eps), p[1] + reach * math.sin(th + eps)),
        ]
    )


def region_summary(p):
    if p.is_empty:
        raise ValueError('Cannot summarize empty region')
    circle = shapely.minimum_bounding_circle(p)
    x0, y0, x1, y1 = circle.bounds
    center = ((x0 + x1) / 2, (y0 + y1) / 2)
    vertices = shapely.get_coordinates(p.convex_hull)
    radius = float(np.linalg.norm(vertices - center, axis=1).max()) + SAFETY
    return center, radius


class GeometryError(RuntimeError):
    pass


@dataclass
class Channel:
    status: str = 'unresolved'
    region: object = field(default_factory=lambda: DOMAIN)
    history: list[Observation] = field(default_factory=list)
    measured: set = field(default_factory=set)
    revision: int = 0
    _summary: tuple | None = None

    def clone(self):
        result = copy.copy(self)
        result.history = self.history.copy()
        result.measured = self.measured.copy()
        return result  # Shapely 2 geometries are immutable.

    def summary(self):
        if self._summary is None:
            self._summary = region_summary(self.region)
        return self._summary

    @property
    def directions(self):
        return [o for o in self.history if o.result == 'direction']

    def update(self, obs):
        a, result = obs.action, obs.result
        if self.status in ('cleared', 'absent_certified'):
            if result in ('direction', 'near', 'success'):
                raise GeometryError('Positive feedback conflicts with completed channel')
            return
        p = self.region
        new_status = self.status
        if result == 'direction':
            p = p.intersection(bearing_wedge(a.position, obs.bearing))
            p = p.intersection(disk(a.position, 1500, True)).difference(disk(a.position, 5))
            new_status = 'detected'
        elif result == 'near':
            p = p.intersection(disk(a.position, 5, True))
            new_status = 'detected'
        elif result == 'no_signal':
            p = p.difference(disk(a.position, 1000))
        elif result == 'no_target_in_range':
            p = p.difference(disk(a.position, 20))
        elif result == 'success':
            p = p.intersection(disk(a.position, 20, True))
            new_status = 'cleared'
        else:
            raise ValueError(f'Unknown feedback: {result}')
        if p.is_empty:
            if new_status != 'unresolved':
                raise GeometryError('Detected source has empty conservative region; do not certify absence')
            new_status = 'absent_certified'
        self.region, self.status = p, new_status
        if a.kind == 'measure':
            self.measured.add(point_key(a.position))
        # Deterministic repeats must not multiply observation likelihoods.
        if obs not in self.history:
            self.history.append(obs)
        self.revision += 1
        self._summary = None


@dataclass
class Belief:
    position: tuple = (0.0, 0.0)
    receiver: int = 1
    virtual_time: float = 0.0
    channels: dict = field(default_factory=lambda: {c: Channel() for c in range(1, 21)})
    applied: dict = field(default_factory=dict)
    steps: int = 0
    deadline: float = math.inf
    virtual_limit: float = 360000.0

    @property
    def cleared(self):
        return {c for c, p in self.channels.items() if p.status == 'cleared'}

    def done(self):
        return len(self.cleared) == 16 or all(
            p.status in ('cleared', 'absent_certified') for p in self.channels.values()
        )

    def apply(self, action, response, request_id):
        if response.get('accepted') is not True:
            return False
        if request_id in self.applied:
            if self.applied[request_id] != (action, response):
                raise ValueError('Reused request ID has different action/response')
            return False
        t = float(response['virtual_time_s'])
        if not math.isfinite(t) or t + 1e-6 < self.virtual_time:
            raise ValueError('Invalid or backwards virtual clock')
        result = response['measure_result' if action.kind == 'measure' else 'clear_result']
        allowed = (
            ('direction', 'near', 'no_signal')
            if action.kind == 'measure'
            else ('success', 'no_target_in_range')
        )
        if result not in allowed:
            raise ValueError('Feedback does not match action')
        angle = float(response['svd_deg']) if result == 'direction' else None
        if angle is not None and (not math.isfinite(angle) or not 0 <= angle < 360):
            raise ValueError('Invalid bearing')
        # Channel.update commits only after checking the resulting geometry.
        self.channels[action.channel].update(Observation(action, result, angle))
        self.position = action.position
        if action.kind == 'measure':
            self.receiver = action.channel
        self.virtual_time = t
        self.steps += 1
        self.applied[request_id] = (action, dict(response))
        return True
