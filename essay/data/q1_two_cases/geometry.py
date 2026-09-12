# Numerical routines extracted unchanged from ../q1_n10_geometry/generate.py.
from collections import deque
import numpy as np
TOL = 1e-8
PARALLEL = 1e-12

def cross(u, v):
    return float(u[0]*v[1]-u[1]*v[0])

def normalize(A, b):
    A, b = np.asarray(A, float), np.asarray(b, float)
    if A.ndim != 2 or A.shape[1] != 2 or b.shape != (len(A),):
        raise ValueError('Halfplanes must have shapes (k,2) and (k,).')
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
        raise ValueError('Non-finite halfplane input.')
    lengths = np.linalg.norm(A, axis=1)
    if np.any(lengths <= PARALLEL):
        raise ValueError('Zero halfplane normal.')
    return A/lengths[:, None], b/lengths

def wedge_constraints(detectors, bearings_deg, error_deg=1.):
    """Return unit-normal inequalities A*x <= b for the closed wedges."""
    p, theta = np.asarray(detectors, float), np.deg2rad(bearings_deg)
    if p.shape != (len(theta), 2) or not 0 < error_deg < 90:
        raise ValueError('Invalid detector coordinates or angular error.')
    eps = np.deg2rad(error_deg)
    lo = np.column_stack((np.cos(theta-eps), np.sin(theta-eps)))
    hi = np.column_stack((np.cos(theta+eps), np.sin(theta+eps)))
    A = np.empty((2*len(p), 2))
    A[0::2] = np.column_stack((lo[:, 1], -lo[:, 0]))
    A[1::2] = np.column_stack((-hi[:, 1], hi[:, 0]))
    return normalize(A, np.einsum('ij,ij->i', A, np.repeat(p, 2, axis=0)))

def intersect_lines(n1, b1, n2, b2):
    det = cross(n1, n2)
    if abs(det) < PARALLEL:
        raise ValueError('Parallel supporting lines.')
    return np.array([(b1*n2[1]-n1[1]*b2)/det,
                     (n1[0]*b2-b1*n2[0])/det])

def halfplane_deque(A, b, tol=TOL):
    """O(k log k) sorted halfplane intersection for full-dimensional polygons.

    Feasibility/boundedness classification is separate. No artificial bounding
    box and no fallback to the enumeration algorithm are used here.
    """
    A, b = np.asarray(A, float), np.asarray(b, float)
    directions = np.column_stack((-A[:, 1], A[:, 0]))
    angles = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2*np.pi)
    order = np.argsort(angles, kind='stable')
    lines = []
    for i in order:
        if lines and abs(cross(A[lines[-1]], A[i])) < PARALLEL and A[lines[-1]]@A[i] > 0:
            if b[i] < b[lines[-1]]:
                lines[-1] = i
        else:
            lines.append(i)
    if len(lines) > 1 and abs(cross(A[lines[0]], A[lines[-1]])) < PARALLEL and A[lines[0]]@A[lines[-1]] > 0:
        first, last = lines[0], lines.pop()
        lines[0] = first if b[first] <= b[last] else last
    q = deque()
    def point(i, j):
        return intersect_lines(A[i], b[i], A[j], b[j])
    def outside(k, x):
        return A[k]@x > b[k]+tol
    for k in lines:
        while len(q) > 1 and outside(k, point(q[-2], q[-1])):
            q.pop()
        while len(q) > 1 and outside(k, point(q[0], q[1])):
            q.popleft()
        if q and abs(cross(A[q[-1]], A[k])) < PARALLEL:
            raise RuntimeError('Unexpected parallel active lines; check region status/tolerance.')
        q.append(k)
    while len(q) > 2 and outside(q[0], point(q[-2], q[-1])):
        q.pop()
    while len(q) > 2 and outside(q[-1], point(q[0], q[1])):
        q.popleft()
    if len(q) < 3:
        raise RuntimeError('No full-dimensional bounded polygon after deque closure.')
    indices = list(q)
    vertices = np.array([point(i, j) for i, j in zip(indices, indices[1:]+indices[:1])])
    # Remove consecutive duplicate intersections (multiple boundaries at a vertex).
    kept = []
    for v in vertices:
        if not kept or np.linalg.norm(v-kept[-1]) > tol:
            kept.append(v)
    if len(kept) > 1 and np.linalg.norm(kept[0]-kept[-1]) <= tol:
        kept.pop()
    return np.asarray(kept)

def diameter(v):
    distances = np.sum((v[:, None]-v[None, :])**2, axis=2)
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    return float(np.sqrt(distances[i, j])), int(i), int(j)
