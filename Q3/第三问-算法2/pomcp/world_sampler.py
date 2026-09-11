"""Joint count-constrained posterior sampling, using active (x,y,R) particles."""
import math
import time
import numpy as np
from q3.core import point_key
from q3.sampling import WorldSampler, conditional_existence
from q3.simulator import Source, World


class ParticleWorldSampler(WorldSampler):
    def sample(self, b, count, deadline=math.inf):
        known, unknown, pools, q = [], [], {}, []
        for c, p in b.channels.items():
            if time.monotonic() >= deadline:
                raise TimeoutError('World sampling deadline')
            if p.status == 'absent_certified':
                continue
            if p.status in ('detected', 'cleared'):
                known.append(c)
                pools[c] = b.cloud(c)
            else:
                unknown.append(c)
                pools[c] = self.pool(c, p, deadline)
                evidence = pools[c][-1]
                q.append(self.p0*evidence/(1-self.p0+self.p0*evidence))
        bearings = {(c, point_key(o.action.position)): o.bearing
                    for c, p in b.channels.items() for o in p.history if o.result == 'direction'}
        worlds = []
        for _ in range(count):
            present = known + [c for c, z in zip(unknown, conditional_existence(q, len(known), self.rng)) if z]
            sources = []
            for c in present:
                if c in known:
                    cloud = pools[c]
                    j = int(self.rng.choice(len(cloud.weights), p=cloud.weights))
                    x, y, radius = cloud.xyz[j]
                else:
                    xy, w, lo, hi, _ = pools[c]
                    j = int(self.rng.choice(len(w), p=w))
                    x, y = xy[j]
                    radius = self.rng.uniform(lo[j], hi[j])
                sources.append(Source(c, (float(x), float(y)), float(radius)))
            worlds.append(World(sources, seed=int(self.rng.integers(0, 2**62)),
                                cleared=b.cleared, known_bearings=bearings))
        return worlds
