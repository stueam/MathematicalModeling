"""Movement-aware completion policy and diverse, cost-ranked proposals.

Geometry is authoritative for stopping. Covariance/area quadrature here only
orders actions; its approximations never remove feasible source positions.
All inputs are public observation state, never simulator truth.
"""
import math

import numpy as np
import shapely

from .core import Action, distance, point_key
from .policy import Baseline
from .sampling import likelihood


def _radical_inverse(n, base):
    out, scale = 0., 1./base
    while n:
        n, digit = divmod(n, base)
        out += digit*scale
        scale /= base
    return out


UV = np.array([(_radical_inverse(i, 2), _radical_inverse(i, 3)) for i in range(1, 513)])


def support(channel):
    """Cached deterministic area quadrature with the same R/angle likelihood.

    Finite nodes guide candidate scores only. No source existence/clearance
    decision is made from this discretization.
    """
    cached = getattr(channel, '_movement_support', None)
    if cached is not None and cached[0] == channel.revision:
        return cached[1]
    region = channel.region
    rect = shapely.minimum_rotated_rectangle(region)
    coords = shapely.get_coordinates(rect)
    if rect.geom_type == 'Polygon' and rect.area > 0:
        xy = coords[0]+UV[:, :1]*(coords[1]-coords[0])+UV[:, 1:]*(coords[3]-coords[0])
        weights, lo, hi = likelihood(channel, xy)
        good = (weights > 0) & shapely.contains_xy(region, xy[:, 0], xy[:, 1])
        xy, weights, lo, hi = xy[good], weights[good], lo[good], hi[good]
    else:
        xy, weights = np.empty((0, 2)), np.empty(0)
    if not len(xy):
        p = region.representative_point()
        xy, weights = np.array([[p.x, p.y]]), np.ones(1)
        lo, hi = np.array([1000.]), np.array([1500.])
    # Systematic resampling bounds per-action scoring cost; retain all original
    # region geometry separately. The extra covariance floor avoids believing
    # that one surviving quadrature point is an exact source coordinate.
    weights /= weights.sum()
    if len(xy) > 32:
        indices = np.searchsorted(np.cumsum(weights), (np.arange(32)+.5)/32)
        xy, lo, hi = xy[indices], lo[indices], hi[indices]
        weights = np.full(32, 1./32)
    mean = weights @ xy
    residual = xy-mean
    cov = (residual*weights[:, None]).T @ residual + np.eye(2)*4.
    data = (xy, weights, lo, hi, mean, cov)
    channel._movement_support = (channel.revision, data)
    return data


def measurement_costs(channel, positions):
    """Cheap predicted local finishing cost *after arriving* at each point.

    Linearized bearing update estimates remaining uncertainty, not a safety
    certificate. Expected movement toward the source is charged explicitly.
    Covariance contraction, no-signal risk and measurement count are charged
    in seconds; information/entropy is not used as an arbitrary reward.
    """
    xy, weights, lo, hi, mean, cov = support(channel)
    positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    delta = xy[None, :, :]-positions[:, None, :]
    dd = np.maximum(np.linalg.norm(delta, axis=2), 1e-6)
    nx, ny = -delta[:, :, 1]/dd, delta[:, :, 0]/dd
    pn_x = cov[0, 0]*nx+cov[0, 1]*ny
    pn_y = cov[1, 0]*nx+cov[1, 1]*ny
    noise_variance = (dd*math.radians(1.)/math.sqrt(3))**2 + 1.
    denominator = np.maximum(nx*pn_x+ny*pn_y+noise_variance, 1e-8)
    xx = np.maximum(0., cov[0, 0]-pn_x**2/denominator)
    yy = np.maximum(0., cov[1, 1]-pn_y**2/denominator)
    cross = cov[0, 1]-pn_x*pn_y/denominator
    eigen = .5*(xx+yy+np.sqrt((xx-yy)**2+4*cross**2))
    posterior_radius = 2*np.sqrt(np.maximum(eigen, 0.))
    reception = np.clip((hi[None, :]-np.maximum(lo[None, :], dd))/
                        np.maximum(hi-lo, 1e-9)[None, :], 0, 1)
    near = dd <= 5
    # A missed reception leaves existing uncertainty: do not score it as a
    # perfect measurement. These are heuristic future travel/time estimates.
    prior_radius = 2*math.sqrt(float(np.linalg.eigvalsh(cov)[-1]))
    residual_time = np.maximum(posterior_radius-12, 0)/5 + 8*np.log2(np.maximum(1, posterior_radius/18))
    missed_time = prior_radius/5+20
    finish = np.maximum(dd-18, 0)/5 + np.where(near, 0,
             reception*residual_time+(1-reception)*missed_time)
    return 5 + finish @ weights


def clear_costs(channel, positions):
    xy, weights, _, _, _, _ = support(channel)
    dd = np.linalg.norm(xy[None, :, :]-np.asarray(positions)[:, None, :], axis=2)
    chance = (dd <= 20) @ weights
    # Used for proposal ranking only. Rollout actually executes failure,
    # excludes its disk and pays its subsequent recovery cost.
    recover = np.maximum(dd-18, 0)/5 + 12
    return 5*chance + 3*(1-chance) + (np.where(dd > 20, recover, 0) @ weights), chance


class MovementPolicy(Baseline):
    """Improved pi0; cheap enough to be the completion policy inside rollout."""
    def __init__(self, ring_radius=1500., max_bearings=8, speculative_clear=True):
        super().__init__(ring_radius, max_bearings)
        self.speculative_clear = speculative_clear

    def local_menu(self, channel):
        cached = getattr(channel, '_movement_menu', None)
        if cached is not None and cached[0] == channel.revision:
            return cached[1]
        xy, w, _, _, mean, _ = support(channel)
        obs = channel.directions[-1]
        origin = np.asarray(obs.action.position)
        toward = mean-origin
        d = float(np.linalg.norm(toward))
        u = toward/max(d, 1e-8)
        if d < 1:
            th = math.radians(obs.bearing)
            u = np.array([math.cos(th), math.sin(th)])
        v = np.array([-u[1], u[0]])
        points = []
        # Cheap parallax, partial progress along the route, and near-source
        # looks coexist with the original guaranteed-reception backups.
        for side in (-1, 1):
            for step in (30., 75., 150.):
                points.append(origin+side*step*v)
            for fraction in (.25, .55, .85):
                for lateral in (35., 90.):
                    points.append(origin+fraction*d*u+side*lateral*v)
            for lateral in (25., 60.):
                points.append(mean+side*lateral*v)
        points += [mean, origin+.5*d*u]
        first = channel.directions[0]
        th = math.radians(first.bearing)
        fu, fv = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
        points += [np.asarray(first.action.position)+750*fu+sign*500*fv for sign in (-1, 1)]
        points = np.unique(np.round(np.asarray(points), 8), axis=0)
        measured = np.asarray([o.action.position for o in channel.history if o.action.kind == 'measure'])
        if len(measured):
            points = points[np.linalg.norm(points[:, None, :]-measured[None, :, :], axis=2).min(axis=1) >= 10]
        costs = measurement_costs(channel, points)
        data = (points, costs)
        channel._movement_menu = (channel.revision, data)
        return data

    def ranked_local(self, b, c, extra_points=(), limit=4):
        p = b.channels[c]
        if any(o.result == 'near' for o in p.history):
            obs = next(o for o in reversed(p.history) if o.result == 'near')
            a = Action('clear', obs.action.position, c)
            return [(distance(b.position, a.position)/5+5, a)]
        safe = self.guaranteed_clear(b, c)
        if safe is not None:
            return [(distance(b.position, safe.position)/5+5, safe)]
        first_index = next(i for i, o in enumerate(p.history) if o.result == 'direction')
        attempts = sum(o.action.kind == 'measure' for o in p.history[first_index:])
        failures = sum(o.result == 'no_target_in_range' for o in p.history)
        if len(p.directions) >= self.max_bearings or attempts >= 12 or failures >= 6:
            a = self.grid_clear(b, c)
            return [(distance(b.position, a.position)/5+10, a)]
        points, costs = self.local_menu(p)
        extra = [b.position] + list(extra_points)
        if extra:
            points = np.vstack((points, extra))
            costs = np.concatenate((costs, measurement_costs(p, extra)))
        travel = np.linalg.norm(points-b.position, axis=1)/5
        total = costs+travel+int(b.receiver != c)
        ranked = []
        for i in np.argsort(total):
            pos = tuple(map(float, points[i]))
            if point_key(pos) in p.measured:
                continue
            # Avoid proposals separated by insignificant jitter; no assumption
            # that microscopic movement resets electromagnetic conditions.
            if any(distance(pos, o.action.position) < 10 for o in p.history if o.action.kind == 'measure'):
                continue
            ranked.append((float(total[i]), Action('measure', pos, c)))
            if len(ranked) >= 3*limit+2:
                break
        if self.speculative_clear:
            xy, w, _, _, mean, _ = support(p)
            positions = np.asarray([b.position, mean])
            finishing, chance = clear_costs(p, positions)
            failed = {point_key(o.action.position) for o in p.history if o.result == 'no_target_in_range'}
            for pos, value, probability in zip(positions, finishing, chance):
                pos = tuple(map(float, pos))
                if probability >= .2 and point_key(pos) not in failed:
                    ranked.append((distance(b.position, pos)/5+float(value), Action('clear', pos, c)))
        ranked.sort(key=lambda item: item[0])
        # Keep geometrically distinct alternatives, not four virtually
        # identical offsets on the same side of the same source.
        selected = []
        for cost, action in ranked:
            if any(action.kind == old.kind and distance(action.position, old.position) < 25 for _, old in selected):
                continue
            selected.append((cost, action))
            if len(selected) >= limit:
                break
        if not selected:
            a = self.grid_clear(b, c)
            return [(distance(b.position, a.position)/5+10, a)]
        return selected

    def for_channel(self, b, c):
        return self.ranked_local(b, c, limit=1)[0][1]

    def scan_merit(self, b, c, pos):
        p = b.channels[c]
        cache = getattr(p, '_scan_merits', None)
        if cache is None or cache[0] != p.revision:
            cache = (p.revision, {})
            p._scan_merits = cache
        key = point_key(pos)
        if key not in cache[1]:
            xy, weights, lo, hi, _, _ = support(p)
            d = np.linalg.norm(xy-pos, axis=1)
            reception = np.clip((hi-np.maximum(lo, d))/np.maximum(hi-lo, 1e-9), 0, 1)
            detectable = float(weights @ reception)
            # Coverage evidence matters even if reception is unlikely. This
            # is a proposal priority, not a task reward or certificate.
            coverage = float(weights @ (d <= 1000))
            cache[1][key] = (detectable, coverage)
        detectable, coverage = cache[1][key]
        return (detectable+.5*coverage)/(5+int(c != b.receiver))

    def scan_action(self, b, position):
        channels = self.unknown_at(b, position)
        if not channels:
            return None
        c = max(channels, key=lambda c: (self.scan_merit(b, c, position), c == b.receiver, -c))
        return Action('measure', position, c)

    def choose(self, b):
        if b.done():
            raise ValueError('No next action after completion')
        detected = [c for c, p in b.channels.items() if p.status == 'detected']
        for c in detected:
            safe = self.guaranteed_clear(b, c)
            if safe is not None and distance(safe.position, b.position) < 1e-6:
                return safe
        # Use each stop to measure other known bearings if this cheaply
        # replaces a dedicated trip. Current-position looks compete by cost.
        local = sorted((self.ranked_local(b, c, limit=1)[0] for c in detected), key=lambda z: z[0])
        inplace = [(cost, a) for cost, a in local if distance(a.position, b.position) < 1e-6]
        if inplace:
            return min(inplace, key=lambda z: z[0])[1]
        for station in self.stations:
            if distance(station, b.position) < 1e-6:
                scan = self.scan_action(b, station)
                if scan is not None:
                    return scan
        # At an incidental localization stop, check an unknown channel only
        # if its predicted coverage is useful. Otherwise proceed with a source.
        if detected:
            unknown = self.unknown_at(b, b.position)
            if unknown:
                c = max(unknown, key=lambda c: self.scan_merit(b, c, b.position))
                if self.scan_merit(b, c, b.position) > .11:
                    return Action('measure', b.position, c)
            return local[0][1]
        # Evaluate an open patrol route (six outer stops at most) rather than
        # committing to nearest-neighbor without considering the remaining leg.
        remaining = [s for s in self.stations if self.unknown_at(b, s)]
        if not remaining:
            return super().choose(b)  # Raises an explicit geometry error.
        order = shortest_open_route(b.position, remaining)
        return self.scan_action(b, remaining[order[0]])

    def proposals(self, b, max_candidates=12, speculative_clear=None):
        base = self.choose(b)
        if base.kind == 'clear' and distance(base.position, b.position) < 1e-6:
            return [base]
        detected = sorted((c for c, p in b.channels.items() if p.status == 'detected'),
                          key=lambda c: distance(b.position, support(b.channels[c])[4]))[:3]
        centers = [tuple(support(b.channels[c])[4]) for c in detected]
        stops = [s for s in self.stations if self.unknown_at(b, s)]
        stops.sort(key=lambda s: distance(b.position, s))
        # Shared/route points can serve different channels. Each is still an
        # atomic legal action; future scans react to intermediate feedback.
        shared = centers + [tuple((np.asarray(b.position)+s)/2) for s in stops[:2]]
        shared += [tuple((np.asarray(x)+y)/2) for i, x in enumerate(centers) for y in centers[i+1:]]
        groups = []
        for c in detected:
            groups.append([a for _, a in self.ranked_local(b, c, extra_points=shared, limit=4)])
        unknown = self.unknown_at(b, b.position)
        unknown.sort(key=lambda c: self.scan_merit(b, c, b.position), reverse=True)
        scan_group = [Action('measure', b.position, c) for c in unknown[:2]]
        patrol_group = [self.scan_action(b, s) for s in stops[:2]]
        shared_group = []
        shared_ranked = []
        for c in detected:
            if not shared or self.guaranteed_clear(b, c) is not None:
                continue
            p = b.channels[c]
            values = measurement_costs(p, shared)
            other_centers = [support(b.channels[k])[4] for k in detected if k != c]
            for pos, value in zip(shared, values):
                if point_key(pos) in p.measured or distance(b.position, pos) < 10:
                    continue
                connection = min((distance(pos, q) for q in other_centers), default=0.)/5
                score = distance(b.position, pos)/5+float(value)+.25*connection
                shared_ranked.append((score, Action('measure', tuple(map(float, pos)), c)))
        for _, action in sorted(shared_ranked, key=lambda z: z[0]):
            if not any(distance(action.position, old.position) < 25 for old in shared_group):
                shared_group.append(action)
            if len(shared_group) == 2:
                break
        # Reserve slots for all action classes before filling by round-robin.
        result = [base]
        for group in [scan_group, patrol_group, shared_group] + groups:
            if group and group[0] not in result:
                result.append(group[0])
        for i in range(1, 4):
            for group in groups + [scan_group, patrol_group, shared_group]:
                if len(group) > i and group[i] not in result:
                    result.append(group[i])
        return result[:max_candidates]


def shortest_open_route(start, points):
    """Held-Karp DP, no return-to-origin requirement. n <= 7."""
    n = len(points)
    if n == 0:
        return []
    if n > 8:
        raise ValueError('Patrol DP is only for a small fixed station set')
    dp = {(1 << i, i): (distance(start, p), (i,)) for i, p in enumerate(points)}
    for mask in range(1, 1 << n):
        for last in range(n):
            value = dp.get((mask, last))
            if value is None:
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                key = (mask | (1 << j), j)
                proposed = (value[0]+distance(points[last], points[j]), value[1]+(j,))
                if key not in dp or proposed < dp[key]:
                    dp[key] = proposed
    return list(min(dp[((1 << n)-1, i)] for i in range(n))[1])
