"""Smallest enclosing circle and area quadrature for convex position envelopes.

All routines operate on public feasible geometry, never simulator ground truth.
"""
from functools import lru_cache
import math
import numpy as np
from scipy.stats import qmc


def minimum_circle(points):
    """Randomized incremental MEC, with deterministic order and final coverage check."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all():
        raise ValueError('Circle requires nonempty finite 2D points')
    origin = p.mean(axis=0)
    p = (p-origin)[np.random.default_rng(17).permutation(len(p))]
    center, r2 = p[0].copy(), 0.

    def outside(x):
        return float((x-center)@(x-center)) > r2+1e-9

    for i, a in enumerate(p):
        if not outside(a):
            continue
        center, r2 = a.copy(), 0.
        for j in range(i):
            b = p[j]
            if not outside(b):
                continue
            center = (a+b)/2
            r2 = float((a-center)@(a-center))
            for k in range(j):
                c = p[k]
                if not outside(c):
                    continue
                u, v = b-a, c-a
                det = 2*(u[0]*v[1]-u[1]*v[0])
                if abs(det) <= 1e-12*max(1., np.linalg.norm(u)*np.linalg.norm(v)):
                    triple = np.array([a,b,c])
                    ii,jj = np.unravel_index(np.argmax(np.sum(
                        (triple[:,None]-triple[None,:])**2,axis=2)), (3,3))
                    center = (triple[ii]+triple[jj])/2
                else:
                    uu, vv = u@u, v@v
                    center = a+np.array([v[1]*uu-u[1]*vv, u[0]*vv-v[0]*uu])/det
                r2 = float(np.max(np.sum((np.array([a,b,c])-center)**2,axis=1)))
    center = center+origin
    # Recompute against every original vertex; never understate a clearing radius.
    radius = float(np.linalg.norm(np.asarray(points)-center,axis=1).max())
    return center, radius


@lru_cache(maxsize=8)
def unit_samples(power):
    return qmc.Sobol(d=3, scramble=True, seed=20260911).random_base2(power)


def area_samples(poly, power=4):
    """Nested quasi-Monte Carlo under area-uniform planning prior.

    Exact triangle areas select a fan triangle; sqrt barycentric mapping is
    uniform inside it. This integrates an arbitrary convex current polygon;
    it is not a square grid and does not modify Q2's annular-sector code.
    Equal sample weights are a quadrature approximation, not equal cell masses.
    """
    poly = np.asarray(poly, dtype=float)
    if len(poly)<3 or not np.isfinite(poly).all():
        raise ValueError('Positive-area finite polygon required')
    anchor = poly.mean(axis=0)
    u, v = poly-anchor, np.roll(poly,-1,axis=0)-anchor
    areas = np.abs(u[:,0]*v[:,1]-u[:,1]*v[:,0])/2
    if areas.sum() <= 1e-10:
        raise ValueError('Cannot assign area prior to zero-area region')
    uvw = unit_samples(power)
    cdf = np.cumsum(areas)/areas.sum()
    indices = np.minimum(np.searchsorted(cdf,uvw[:,0],side='right'),len(poly)-1)
    r = np.sqrt(uvw[:,1])
    points = anchor+r[:,None]*((1-uvw[:,2,None])*u[indices]+uvw[:,2,None]*v[indices])
    return points, np.full(len(points),1/len(points))


def direction_region(poly, station, bearing):
    """Two bearing halfplanes; radius constraints already guaranteed by candidate filter."""
    from q3_strategy import EPS, clip
    a = math.radians(bearing)
    for n in (np.array([math.sin(a-EPS),-math.cos(a-EPS)]),
              np.array([-math.sin(a+EPS),math.cos(a+EPS)])):
        poly = clip(poly,n,float(n@station)+1e-7)
    return poly
