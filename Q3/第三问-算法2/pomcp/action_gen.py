"""Unified search/localization/clear proposals, followed by cheap pruning."""
import math
import numpy as np
import shapely
from q3.core import Action, distance, disk, point_key
from q3.movement import MovementPolicy, support, measurement_costs
from q3.mission import MissionPolicy
from .generative_model import MacroAction
from .routing import OpenRoute, length


def scan(b, pos, channels, kind):
    channels = sorted(set(channels), key=lambda c: (c != b.receiver, c))
    actions = tuple(Action('measure', tuple(map(float, pos)), c) for c in channels
                    if point_key(pos) not in b.channels[c].measured)
    return MacroAction(kind, actions) if actions else None


class MacroPolicy:
    def __init__(self):
        self.base = MovementPolicy()
        self.mission = MissionPolicy()
        self.route = OpenRoute()

    def unknown(self, b, pos, threshold=.12):
        cover = disk(pos, 1000)
        result = []
        for c, p in b.channels.items():
            if p.status != 'unresolved' or point_key(pos) in p.measured:
                continue
            fraction = p.region.intersection(cover).area / max(p.region.area, 1e-20)
            if fraction >= threshold:
                result.append(c)
        return result

    def batch(self, b, a, kind=None):
        if a.kind == 'clear':
            return MacroAction.atomic(a, kind)
        if b.channels[a.channel].status == 'unresolved':
            cs = self.unknown(b, a.position)
            return scan(b, a.position, cs+[a.channel], 'SEARCH_BLIND')
        return MacroAction.atomic(a, kind)

    def choose(self, b):
        detected = [c for c, p in b.channels.items() if p.status == 'detected']
        for c in detected:
            a = self.base.guaranteed_clear(b, c)
            if a is not None and distance(a.position, b.position) < 1e-6:
                return MacroAction.atomic(a)
        # Same-site batches share motion; fixed-error repeats are excluded.
        informative = []
        for c in detected:
            p = b.channels[c]
            if len(p.directions) == 1 and all(distance(b.position, o.action.position) >= 75 for o in p.history):
                if distance(b.position, support(p)[4]) < 950:
                    informative.append(c)
        if informative:
            return scan(b, b.position, informative, 'SHARED_PROBE')
        cs = self.unknown(b, b.position, .32)
        if cs:
            return scan(b, b.position, cs, 'SEARCH_BLIND')
        if detected:
            centers = {c: tuple(support(b.channels[c])[4]) for c in detected}
            order = self.route.update(b.position, centers)
            a = self.base.for_channel(b, order[0])
            return self.batch(b, a, 'ROUTE_CONTINUE')
        options = self.mission.coverage_options(b)
        return self.batch(b, options[0] if options else self.base.choose(b), 'SEARCH_BLIND')

    def generate(self, b):
        base = MacroAction.atomic(self.base.choose(b), 'BASELINE')
        actions = [base, self.choose(b)]
        detected = [c for c, p in b.channels.items() if p.status == 'detected']
        centers = {c: tuple(support(b.channels[c])[4]) for c in detected}
        order = self.route.update(b.position, centers)
        points = [b.position]
        for c in order[:5]:
            p = b.channels[c]
            safe = self.base.guaranteed_clear(b, c)
            if safe:
                actions.append(MacroAction.atomic(safe))
            # Every target can propose a speculative clear, regardless of .95.
            q, probability = b.clear_point(c)
            if point_key(q) not in {point_key(o.action.position) for o in p.history if o.result == 'no_target_in_range'}:
                actions.append(MacroAction.atomic(Action('clear', q, c)))
            if p.directions and safe is None:
                menu = self.base.ranked_local(b, c, limit=4)
                actions.extend(MacroAction.atomic(a) for _, a in menu)
                center = np.asarray(centers[c])
                direction = center-b.position
                d = np.linalg.norm(direction)
                u = direction/max(d, 1e-8)
                for offset in (100., 175., 250.):
                    q = tuple(center-min(d, offset)*u)
                    if all(distance(q, o.action.position) >= 10 for o in p.history if o.action.kind == 'measure'):
                        actions.append(MacroAction.atomic(Action('measure', q, c)))
                        points.append(q)
        for i, c in enumerate(order[:4]):
            for other in order[i+1:4]:
                points.append(tuple((np.asarray(centers[c])+centers[other])/2))
        for radius in (100., 300.):
            for angle in np.arange(6)*math.pi/3:
                points.append((b.position[0]+radius*math.cos(angle), b.position[1]+radius*math.sin(angle)))
        shared = []
        for pos in points:
            cs = []
            for c in detected:
                p = b.channels[c]
                if p.summary()[1] <= 19.999 or not p.directions:
                    continue
                if any(distance(pos, o.action.position) < 25 for o in p.history if o.action.kind == 'measure'):
                    continue
                xy, weights, _, _, mean, _ = support(p)
                if float(weights @ (np.linalg.norm(xy-pos, axis=1) <= 1000)) < .6:
                    continue
                cs.append(c)
            if len(cs) >= 2:
                shared.append(scan(b, pos, cs, 'SHARED_PROBE'))
        # Bound raw geometry work while retaining all action families.
        shared.sort(key=lambda a: distance(b.position, a.position)/5 +
                    sum(float(measurement_costs(b.channels[x.channel], [a.position])[0]) for x in a.actions)/len(a.actions))
        actions.extend(shared[:12])
        search_points = [b.position] + [a.position for a in self.mission.coverage_options(b)[:6]]
        search_points += self.base.stations
        for pos in search_points:
            cs = self.unknown(b, pos)
            a = scan(b, pos, cs, 'SEARCH_BLIND')
            if a:
                actions.append(a)
        # Deduplicate physical sequences, but keep the baseline label/slot.
        seen, result = set(), []
        for a in actions:
            if a and a.actions not in seen:
                result.append(a)
                seen.add(a.actions)
        return result

    def prune(self, b, actions, keep):
        from .heuristic import quick_score
        ranked = sorted(actions, key=lambda a: quick_score(b, a))
        # Both the original atomic baseline and the macro rollout policy must
        # survive screening; otherwise policy improvement has no comparator.
        result = list(dict.fromkeys(actions[:2]))[:keep]
        for kind in ('CLEAR_TARGET', 'PROBE_TARGET', 'SHARED_PROBE', 'SEARCH_BLIND', 'ROUTE_CONTINUE'):
            a = next((a for a in ranked if a.kind == kind), None)
            if a and a not in result and len(result) < keep:
                result.append(a)
        for a in ranked:
            if a not in result and len(result) < keep:
                result.append(a)
        return result
