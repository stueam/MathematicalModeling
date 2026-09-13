"""Deterministic now-versus-route sensing and geometrically complete patrols."""

import numpy as np


def insertion_costs(start, future, xy):
    """Extra metres to insert each quadrature location along every open edge."""
    vertices = np.asarray([start] + list(future), dtype=float)
    d = np.linalg.norm(vertices[:, None, :] - xy[None, :, :], axis=2)
    if not future:
        return d
    edges = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    return np.vstack((d[:-1] + d[1:] - edges[:, None], d[-1:]))


def scan_route_value(posterior, existence, position, future, recovery_s):
    """Expected discovery-insertion regret avoided by measuring NOW.

    R is fixed across all future stops. Integrate the interval for 'now receives,
    first future reception is j', rather than multiplying independent hit chances.
    Coverage is a planning estimate; actual no-signal disks alone certify absence.
    """
    xy, w, lo, hi = (posterior.xy, posterior.weights, posterior.lo, posterior.hi)
    d0 = np.linalg.norm(xy - position, axis=1)
    width = hi - lo
    receive_now = np.clip((hi - np.maximum(lo, d0)) / width, 0, 1)
    best_seen = np.full(len(xy), np.inf)
    extra = insertion_costs(position, future, xy)
    best_now = extra.min(axis=0)
    gain = 0.0
    for j, stop in enumerate(future):
        dj = np.linalg.norm(xy - stop, axis=1)
        probability = np.maximum(0, np.minimum(hi, best_seen) - np.maximum.reduce((lo, d0, dj))) / width
        late_extra = extra[j + 1 :].min(axis=0)
        gain += existence * float(w @ (probability * np.maximum(late_extra - best_now, 0))) / 5
        best_seen = np.minimum(best_seen, dj)
    exclusive = np.maximum(0, np.minimum(hi, best_seen) - np.maximum(lo, d0)) / width
    unique_cover = (d0 <= 1000) & (best_seen > 1000)
    exclusive_probability = float(w @ exclusive)
    coverage = float(w @ unique_cover)
    discovery = existence * exclusive_probability * recovery_s
    certification = 0.5 * coverage * recovery_s
    return {
        'gross_saving_s': gain + discovery + certification,
        'early_insertion_s': gain,
        'exclusive_discovery_s': discovery,
        'coverage_value_s': certification,
        'reception_now': float(w @ receive_now),
        'exclusive_reception': exclusive_probability,
        'unique_coverage': coverage,
    }
