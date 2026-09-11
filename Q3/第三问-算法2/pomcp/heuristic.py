"""Potential Phi=-H, gamma=1. All terms are estimated virtual seconds.

No entropy/discovery reward. H is an approximation, not an admissible bound.
"""
import math
import numpy as np
import shapely
from q3.core import distance, disk
from q3.movement import support, measurement_costs, clear_costs
from q3.policy import Baseline
from .routing import open_tour, length


def components(b):
    if b.done():
        return dict(route=0., clear=0., localize=0., search=0.)
    points, localize, known = [], 0., 0
    unresolved = []
    for c, p in b.channels.items():
        if p.status in ('detected', 'cleared'):
            known += 1
        if p.status == 'detected':
            _, _, _, _, mean, cov = support(p)
            points.append(tuple(mean))
            radius = 2*math.sqrt(float(np.linalg.eigvalsh(cov)[-1]))
            if p.summary()[1] > 19.999:
                localize += max(0., radius-18)/5 + 6*math.log2(max(1., radius/18))
        elif p.status == 'unresolved':
            unresolved.append(p)
    # Expected UNKNOWN count, conditioned at least to the hard total bounds.
    # Area fraction is a cheap conservative proxy for no-signal evidence.
    expected = sum(.65*(p.region.area/(math.pi*1800**2)) /
                   (.35+.65*(p.region.area/(math.pi*1800**2))) for p in unresolved)
    expected = min(max(expected, 10-known, 0.), 16-known, len(unresolved))
    search = 0.
    if unresolved:
        regions = [p.region for p in unresolved]
        stations = Baseline().stations
        search_points = []
        # Greedy cover with INSCRIBED exclusion disks. Any unrepresented tiny
        # residue is charged explicitly; it is never a stopping certificate.
        for _ in range(len(stations)):
            if all(r.is_empty for r in regions):
                break
            scores = [sum(r.intersection(disk(q, 1000)).area for r in regions) for q in stations]
            j = int(np.argmax(scores))
            q = stations.pop(j)
            cover = disk(q, 1000)
            count = sum(not r.is_empty and r.intersection(cover).area > 1e-6 for r in regions)
            search += 6*count
            regions = [r.difference(cover) for r in regions]
            search_points.append(q)
        search += 6*sum(not r.is_empty for r in regions)
        # Route and search stops share one tour. Unknown discoveries also
        # require travel/localization, estimated from their feasible regions.
        points.extend(search_points)
        search += expected*(2*math.sqrt(sum(p.region.area for p in unresolved)/
                                      max(1, len(unresolved))/math.pi)/5 + 12)
    route = length(b.position, [points[i] for i in open_tour(b.position, points)])/5 if points else 0.
    return dict(route=route, clear=5*(sum(p.status == 'detected' for p in b.channels.values())+expected),
                localize=localize, search=search)


def remaining_time(b):
    return sum(components(b).values())


def quick_score(b, macro):
    """Cheap ranking only; actual action time is solely charged by simulator."""
    travel = distance(b.position, macro.position)/5
    if macro.actions[0].kind == 'clear':
        c = macro.actions[0].channel
        return travel+float(clear_costs(b.channels[c], [macro.position])[0][0])
    known = [a.channel for a in macro.actions if b.channels[a.channel].status == 'detected']
    if known:
        # Amortize movement over targets served, in seconds per completed task.
        return travel/len(known)+sum(float(measurement_costs(b.channels[c], [macro.position])[0]) for c in known)/len(known)
    gain = sum(b.channels[a.channel].region.intersection(disk(macro.position, 1000)).area /
               max(b.channels[a.channel].region.area, 1e-20) for a in macro.actions)
    return (travel+6*len(macro.actions))/max(.01, gain)
