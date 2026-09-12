"""Continuous position/heading coverage by closed equilateral triangles."""
import math
import numpy as np


def triangle_distance(triangle):
    """Exact geometric minimum distance to origin, not a sampled boundary test."""
    a, b, c = np.asarray(triangle, float)
    uv = np.linalg.solve(np.column_stack((b-a, c-a)), -a)
    if min(uv) >= -1e-12 and sum(uv) <= 1+1e-12:
        return 0.
    distances = []
    for p, q in ((a, b), (b, c), (c, a)):
        t = np.clip(-p@(q-p)/((q-p)@(q-p)), 0., 1.)
        distances.append(np.linalg.norm(p+t*(q-p)))
    return float(min(distances))


def search_mesh(step=950., radius=1800.):
    if not 0 < step < 1000 or radius <= 0:
        raise ValueError('Require 0 < triangle side < 1000 and positive domain radius')
    vertical = step*math.sqrt(3)/2
    jlim = math.ceil(radius/vertical)+2
    ilim = math.ceil(radius/step)+math.ceil(jlim/2)+2

    def point(key):
        i, j = key
        return np.array([step*(i+j/2), vertical*j])

    keys, triangles = set(), []
    for j in range(-jlim, jlim):
        for i in range(-ilim, ilim):
            a, b, c, d = (i, j), (i+1, j), (i, j+1), (i+1, j+1)
            for tri in ((a, b, c), (b, d, c)):
                if triangle_distance([point(k) for k in tri]) <= radius+1e-8:
                    triangles.append(tri)
                    keys.update(tri)
    keys = sorted(keys)
    index = {k: i for i, k in enumerate(keys)}
    return np.array([point(k) for k in keys]), np.array([[index[k] for k in tri] for tri in triangles])
