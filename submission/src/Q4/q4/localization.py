"""Public-feedback local candidates and a finite rotated optical grid."""

import math

import numpy as np
import shapely

from .shared import Action, GeometryError, distance, point_key


def guaranteed_clear(b, c):
    near = next((o for o in reversed(b.channels[c].history) if o.result == 'near'), None)
    if near:
        return Action('clear', near.action.position, c)
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
    step = min(d, 19.999 - radius)
    pos = center if d == 0 else tuple(center[i] + step * (b.position[i] - center[i]) / d for i in range(2))
    return Action('clear', pos, c)


def local_attempts(channel):
    first = next(
        (i for i, o in enumerate(channel.history) if o.result in ('direction', 'near')), len(channel.history)
    )
    return len(channel.history) - first


def finite_clear(b, c):
    channel = b.channels[c]
    first = next(o for o in channel.history if o.result == 'direction')
    angle = math.radians(first.bearing)
    basis = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
    origin = np.asarray(first.action.position)
    vertices = (shapely.get_coordinates(channel.region) - origin) @ basis.T
    lo, hi = np.floor(vertices.min(axis=0) / 20).astype(int), np.floor(vertices.max(axis=0) / 20).astype(int)
    ii, jj = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing='ij')
    centers = (np.column_stack((ii.ravel(), jj.ravel())) * 20 + 10) @ basis + origin
    failed = {point_key(o.action.position) for o in channel.history if o.result == 'no_target_in_range'}
    order = np.argsort(np.linalg.norm(centers - b.position, axis=1), kind='stable')
    corners = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]]) @ basis
    for i in order:
        p = tuple(map(float, centers[i]))
        if point_key(p) not in failed and channel.region.intersects(shapely.Polygon(centers[i] + corners)):
            return Action('clear', p, c)
    raise GeometryError('Nonempty support has no untried finite optical cell')


def measure_points(b, c, mean):
    channel = b.channels[c]
    origin, mean = np.asarray(b.position), np.asarray(mean)
    direction = mean - origin
    norm = float(np.linalg.norm(direction))
    if norm < 1:
        theta = math.radians(channel.directions[-1].bearing)
        direction = np.array([math.cos(theta), math.sin(theta)])
    u = direction / max(float(np.linalg.norm(direction)), 1e-12)
    v = np.array([-u[1], u[0]])
    points = [origin, mean]
    points += [mean - r * u for r in (100.0, 200.0) if norm > r + 20]
    for sign in (-1, 1):
        points += [
            origin + sign * 50 * v,
            mean + sign * 50 * v,
            origin + 0.65 * (mean - origin) + sign * 50 * v,
        ]
    first = channel.directions[0]
    theta = math.radians(first.bearing)
    fu = np.array([math.cos(theta), math.sin(theta)])
    fv = np.array([-fu[1], fu[0]])
    points += [np.asarray(first.action.position) + 750 * fu + sign * 350 * fv for sign in (-1, 1)]
    result = []
    for p in points:
        p = tuple(map(float, p))
        if p not in result and all(
            distance(p, o.action.position) >= 10 for o in channel.history if o.action.kind == 'measure'
        ):
            result.append(p)
    return result
