"""Hidden Q4 worlds. Controllers receive only observations, never a World."""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .shared import load

base = load('simulator')
LocalSimulator = base.LocalSimulator


@dataclass(frozen=True)
class Source:
    channel: int
    position: tuple
    radius: float
    heading_deg: float | None = None


class World(base.World):
    def __init__(self, sources, seed=0, error_mode='iid', **kwargs):
        super().__init__(sources, seed, error_mode, **kwargs)
        for s in sources:
            if s.heading_deg is not None and (not math.isfinite(s.heading_deg) or not 0 <= s.heading_deg < 360):
                raise ValueError('Invalid fixed emitter heading')

    def feedback(self, action):
        source = self._sources.get(action.channel)
        active = source is not None and action.channel not in self._cleared
        if action.kind == 'clear' or not active:
            return super().feedback(action)
        if source.heading_deg is not None:
            dx, dy = (action.position[i]-source.position[i] for i in (0, 1))
            theta = math.radians(source.heading_deg)
            # Exactly co-located receiver is treated as illuminated. Nonzero
            # vectors use a closed half-plane; only roundoff at ±90° is padded.
            projection = math.cos(theta)*dx+math.sin(theta)*dy
            tolerance = 8*np.finfo(float).eps*max(1., math.hypot(dx, dy))
            if projection < -tolerance:
                return {'measure_result': 'no_signal'}, 5
        return super().feedback(action)

    def manifest(self):
        """Evaluation-only; written separately and never passed to the policy."""
        return {'seed': self.seed, 'error_mode': self.error_mode,
                'sources': [asdict(s) for s in self._sources.values()]}

    def score(self):
        result = super().score()
        result['directional_count'] = sum(s.heading_deg is not None for s in self._sources.values())
        return result


SCENARIOS = ('uniform', 'boundary', 'outward', 'tangent', 'cluster', 'near', 'backside')


def generate_world(seed, n=None, scenario='uniform', radius=None, error_mode='iid', directional_count=None):
    if scenario not in SCENARIOS or radius is not None and not 1000 <= radius <= 1500:
        raise ValueError('Invalid scenario/radius')
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 17)) if n is None else n
    if not 10 <= n <= 16:
        raise ValueError('Q4 requires 10..16 sources')
    # Preserve the original mixed-type default distribution and RNG sequence.
    # Explicit endpoints cover pure types observed in official practice.
    nd = int(rng.integers(1, n)) if directional_count is None else directional_count
    if not isinstance(nd, (int, np.integer)) or not 0 <= nd <= n:
        raise ValueError('Directional count must be an integer in 0..n')
    channels = rng.choice(np.arange(1, 21), n, replace=False)
    angles = rng.uniform(0, math.tau, n)
    radial = 1800*np.sqrt(rng.random(n))
    if scenario in ('boundary', 'outward', 'tangent'):
        radial = rng.uniform(1750, 1800, n)
    if scenario == 'outward':
        radial[:nd] = 1800  # Hard case: at the boundary, emit away from the arena.
    if scenario == 'cluster':
        angles = rng.normal(.8, .035, n)
        radial = rng.uniform(1400, 1550, n)
    if scenario in ('near', 'backside'):
        radial[0] = 2.
    radii = rng.uniform(1000, 1500, n) if radius is None else np.full(n, radius)
    headings = rng.uniform(0, 360, n)
    if scenario in ('outward', 'backside'):
        headings = np.degrees(angles) % 360
    if scenario == 'tangent':
        headings = (np.degrees(angles)+90) % 360
    if scenario == 'near':
        headings[0] = (math.degrees(angles[0])+180) % 360
    sources = [Source(int(c), (float(r*math.cos(a)), float(r*math.sin(a))), float(rad),
                      float(h) if i < nd else None)
               for i, (c, r, a, rad, h) in enumerate(zip(channels, radial, angles, radii, headings))]
    return World(sources, seed, error_mode)
