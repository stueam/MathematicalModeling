"""Inner exclusion certificates for half-disk emitters, independently by channel."""

from functools import lru_cache
from itertools import combinations

import shapely
from shapely.errors import GEOSException
from shapely.geometry import MultiPoint, Polygon

from .shared import disk, distance

DIAGNOSTICS = {
    'invalid_pieces_omitted': 0,
    'union_retries': 0,
    'union_pieces_omitted': 0,
    'candidate_certificate_errors': 0,
}


def safe_union(pieces):
    """Union a trustworthy subset; never repair by expanding an exclusion."""
    valid = []
    for p in pieces:
        if not p.is_valid:
            DIAGNOSTICS['invalid_pieces_omitted'] += 1
        elif not p.is_empty:
            valid.append(p)
    if not valid:
        return Polygon()
    try:
        result = shapely.union_all(valid)
        if result.is_valid:
            return result
    except GEOSException:
        pass
    DIAGNOSTICS['union_retries'] += 1
    result = Polygon()
    for p in valid:
        try:
            candidate = result.union(p)
            if not candidate.is_valid:
                raise GEOSException('Invalid union result')
            result = candidate
        except GEOSException:
            DIAGNOSTICS['union_pieces_omitted'] += 1
    return result


def square_stations():
    """The outer domain fits inside this grid; each cell diagonal < 1000 m."""
    return tuple((float(x), float(y)) for x in range(-2100, 2101, 700) for y in range(-2100, 2101, 700))


@lru_cache(maxsize=4096)
def receiving_disk(p):
    return disk(p, 1000)


@lru_cache(maxsize=8192)
def triangle_exclusion(points):
    _a, _b, _c = points
    if any(distance(p, q) >= 2000 for p, q in combinations(points, 2)):
        return Polygon()
    triangle = MultiPoint(points).convex_hull
    if triangle.geom_type != 'Polygon':
        return Polygon()  # Degenerate support is never an area-based absence proof.
    # Full circle radius is never increased; allowed boundary uncertainty stays.
    for p in points:
        try:
            triangle = triangle.intersection(receiving_disk(p))
        except GEOSException:
            DIAGNOSTICS['invalid_pieces_omitted'] += 1
            return Polygon()
        if not triangle.is_valid:
            DIAGNOSTICS['invalid_pieces_omitted'] += 1
            return Polygon()
        if triangle.is_empty:
            break
    return triangle


@lru_cache(maxsize=256)
def added_exclusion(new, old):
    close = [p for p in old if distance(p, new) < 2000]
    pieces = [
        triangle_exclusion(tuple(sorted((new, a, b))))
        for a, b in combinations(close, 2)
        if distance(a, b) < 2000
    ]
    return safe_union(pieces)


@lru_cache(maxsize=64)
def full_exclusion(points):
    """Hypothetical all-negative coverage: planning only until actually measured."""
    pieces = []
    for i, p in enumerate(points):
        pieces.append(added_exclusion(p, tuple(points[:i])))
    return safe_union(pieces)


def certifies(region, positions):
    points = tuple(sorted(set(tuple(map(float, p)) for p in positions)))
    try:
        return region.difference(full_exclusion(points)).is_empty
    except GEOSException:
        DIAGNOSTICS['candidate_certificate_errors'] += 1
        return False


def relevant(region, position):
    # Outer query disk: never omit a station that could contribute at R_min.
    return region.intersects(disk(position, 1000, outer=True))
