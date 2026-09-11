"""Q3 B5: sector backbone with interruptible probes and clearing tasks.

Only the action callback supplies observations. Uniform polygon area and bounded
uniform bearing errors are planning approximations, never ground-truth priors.
The inherited conservative geometry and complete search stations certify absence;
scores cannot certify absence. Existing methods are unchanged.
"""
import math
import time
from collections import defaultdict
import numpy as np
from q2_lookahead import Q2Proposal
from q2_geometry import area_samples, direction_region, minimum_circle
from route_search_probe import improved_tour
from b4_strategy import channel_order, service_seconds


B5_CONFIG = dict(sectors=7, ring_radius=1000., proposals_per_source=3,
                 proposal_slack_s=25., active_sources=5, max_candidates=22,
                 max_batch=5, coarse_power=2, fine_power=4, error_order=3,
                 finalists=3, cross_detour_m=250., sector_action_limit=6,
                 max_wait=18, max_probes=8, min_gain_s=2.,
                 min_measure_gain_s=7., trial_radius_m=80., max_macros=2000,
                 adaptive_sectors=True, batching=True)


def path_length(start, points):
    return sum(float(np.linalg.norm(b-a)) for a, b in
               zip([np.asarray(start), *points[:-1]], points)) if points else 0.


class B5Strategy(Q2Proposal):
    """One stop/one clear per macro action; a near response clears immediately."""
    def __init__(self, action, b5_config=None, **kwargs):
        super().__init__(action, **kwargs)
        self.b5_config = dict(B5_CONFIG, **(b5_config or {}))
        cfg = self.b5_config
        if cfg['sectors'] != 7 or cfg['ring_radius'] != 1000.:
            raise ValueError('This B5 version certifies the fixed seven-station backbone')
        for key in ('proposals_per_source', 'active_sources', 'max_candidates',
                    'max_batch', 'finalists', 'sector_action_limit', 'max_wait',
                    'max_probes', 'max_macros'):
            if not isinstance(cfg[key], int) or cfg[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('proposal_slack_s', 'cross_detour_m', 'min_gain_s',
                    'min_measure_gain_s', 'trial_radius_m'):
            if not np.isfinite(cfg[key]) or cfg[key] < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        for key in ('coarse_power', 'fine_power'):
            if not isinstance(cfg[key], int) or not 1 <= cfg[key] <= 8:
                raise ValueError(f'{key} outside supported quadrature range')
        if not isinstance(cfg['error_order'], int) or not 1 <= cfg['error_order'] <= 5:
            raise ValueError('error_order outside supported range')
        self.wait_age = defaultdict(int)
        self.probe_count = defaultdict(int)
        self.failed_points = defaultdict(list)
        self.optical = {}
        self.b5_log = []
        self.b5_stats = defaultdict(int)
        self._metrics_cache = {}
        self._macro = 0
        self._sector_actions = 0
        self.backbone = []

    def sector(self, p):
        angle = math.atan2(p[1], p[0])
        return int(math.floor((angle+math.pi/7)/(math.tau/7))) % 7

    def new_measurement(self, c, p):
        if c in self.cleared or c in self.absent:
            return False
        history = [q for q, _ in self.measurements.get(c, [])]
        history += self.negative_stations[c]
        return not any(np.linalg.norm(p-q) < 1. for q in history)

    def admissible(self, c, p):
        return (c in self.polys and self.new_measurement(c, p)
                and np.max(np.linalg.norm(self.polys[c]-p, axis=1)) <= 999.)

    def unknown_channels(self):
        return sorted(set(range(1, 21))-set(self.polys)-self.cleared-self.absent)

    def refresh_absence(self):
        found = set(self.polys) | self.cleared
        if len(found) >= 16 or not self.backbone:
            self.absent.update(set(range(1, 21))-found)
            if len(found) >= 16:
                self.backbone.clear()

    def state_info(self):
        return {c: self.region_circle(self.polys[c])
                for c in sorted(set(self.polys)-self.cleared)}

    def estimate_route(self, start, centers):
        """Insert known centers into a fixed-order search backbone; open endpoint.

        This is an approximate remaining route, not an optimal value function.
        Every travel leg is counted once even when many channels share a stop.
        """
        route = [p.copy() for _, p in self.backbone]
        left = {c: np.asarray(p) for c, p in centers.items()}
        while left:
            best = None
            for c, p in left.items():
                for i in range(len(route)+1):
                    a = np.asarray(start) if i == 0 else route[i-1]
                    extra = np.linalg.norm(a-p)
                    if i < len(route):
                        extra += np.linalg.norm(p-route[i])-np.linalg.norm(a-route[i])
                    candidate = (float(extra), c, i)
                    if best is None or candidate < best:
                        best = candidate
            _, c, i = best
            route.insert(i, left.pop(c))
        return path_length(start, route)/5

    @staticmethod
    def residual(radius):
        return 0. if radius <= 19.8 else 5.+2*radius/5

    def metrics(self, c, p, power):
        """Expected next envelope, approach and unresolved-localization cost.

        Cache excludes robot position; movement is added once by the scheduler.
        """
        poly = self.polys[c]
        key = (c, poly.tobytes(), np.asarray(p).tobytes(), power)
        if key in self._metrics_cache:
            return self._metrics_cache[key]
        if not self.admissible(c, p):
            return None
        try:
            points, weights = area_samples(poly, power)
        except ValueError:
            return None  # Degenerate regions use geometric recovery, not fake mass.
        nodes, ew = np.polynomial.legendre.leggauss(self.b5_config['error_order'])
        ew /= 2
        result = dict(residual=0., approach=0., ready=0., center=np.zeros(2))
        angles = np.degrees(np.arctan2(points[:, 1]-p[1], points[:, 0]-p[0]))
        observed = {}
        for x, weight, bearing in zip(points, weights, angles):
            if np.linalg.norm(x-p) <= 5:
                # near is immediately cleared; its 5 s is included in execution score.
                result['ready'] += weight
                result['center'] += weight*p
                continue
            for e, eweight in zip(nodes, ew):
                angle = round(float(bearing+e), 2) % 360
                if angle not in observed:
                    observed[angle] = minimum_circle(direction_region(poly, p, angle))
                center, radius = observed[angle]
                mass = float(weight*eweight)
                result['residual'] += mass*self.residual(radius)
                result['approach'] += mass*np.linalg.norm(center-p)/5
                result['ready'] += mass*(radius <= 19.8)
                result['center'] += mass*center
        self._metrics_cache[key] = result
        return result

    def route_eligible(self, p, urgent=False):
        if urgent or not self.backbone or not self.b5_config['adaptive_sectors']:
            return True
        k, anchor = self.backbone[0]
        if self.sector(p) == k:
            return True
        detour = (np.linalg.norm(self.position-p)+np.linalg.norm(p-anchor)
                  -np.linalg.norm(self.position-anchor))
        return detour <= self.b5_config['cross_detour_m']

    def proposed_points(self, c, info):
        center, radius = info[c]
        _, _, vh = np.linalg.svd(self.polys[c]-center, full_matrices=False)
        normal = np.array([-vh[0, 1], vh[0, 0]])
        defaults = [center+30*normal, center-30*normal]
        default = min(defaults, key=lambda p: np.linalg.norm(p-self.position))
        options = self.candidates(c, self.polys[c], center, radius, default)
        # Nearby backbone sites may be slightly worse for one source but shared.
        for _, anchor in self.backbone[:2]:
            if self.admissible(c, anchor):
                options.append(anchor.copy())
        ranked = []
        for p in options:
            m = self.metrics(c, p, self.b5_config['coarse_power'])
            if m is not None:
                value = np.linalg.norm(self.position-p)/5+5+m['approach']+5+m['residual']
                ranked.append((float(value), p))
        ranked.sort(key=lambda item: item[0])
        if not ranked:
            return []
        ceiling = ranked[0][0]+self.b5_config['proposal_slack_s']
        selected = [p for value, p in ranked if value <= ceiling]
        return selected[:self.b5_config['proposals_per_source']]

    def evaluate_probe(self, p, info, centers, baseline, required=None, power=None):
        power = self.b5_config['coarse_power'] if power is None else power
        credits = []
        for c, (_, radius) in info.items():
            if radius <= 19.8 or self.probe_count[c] >= self.b5_config['max_probes']:
                continue
            m = self.metrics(c, p, power)
            if m is None:
                continue
            credit = self.residual(radius)-m['residual']
            if c == required or credit > self.b5_config['min_measure_gain_s']:
                credits.append((c, credit, m))
        credits.sort(key=lambda row: (row[0] != required, -row[1], row[0]))
        limit = self.b5_config['max_batch'] if self.b5_config['batching'] else 1
        credits = credits[:limit]
        if not credits or (required is not None and required not in [r[0] for r in credits]):
            return None
        channels = channel_order([r[0] for r in credits], self.receiver_channel)
        # Preserve joint route geometry here. Sum only localization residuals;
        # do not add each channel's independent approach distance to shared travel.
        after_centers = dict(centers)
        for c, _, m in credits:
            after_centers[c] = m['center']
        route = self.estimate_route(p, after_centers)
        cost = np.linalg.norm(self.position-p)/5+service_seconds(channels, self.receiver_channel)
        gain = baseline-(cost+route)+sum(r[1] for r in credits)
        return dict(kind='probe', point=np.asarray(p), channels=channels, gain_s=float(gain),
                    credits={str(c):float(g) for c, g, _ in credits})

    def evaluate_clear(self, c, info, centers, baseline):
        center, radius = info[c]
        if radius > self.b5_config['trial_radius_m'] and radius > 19.8:
            return None
        if any(np.linalg.norm(center-p) < 1 for p in self.failed_points[c]):
            return None
        guaranteed = radius <= 19.8
        probability = 1.
        if not guaranteed:
            try:
                points, weights = area_samples(self.polys[c], self.b5_config['fine_power'])
            except ValueError:
                return None
            probability = float(weights[np.linalg.norm(points-center, axis=1) <= 20].sum())
            if probability <= 0:
                return None
        other = {k:p for k,p in centers.items() if k != c}
        success_route = self.estimate_route(center, other)
        failed_route = self.estimate_route(center, centers)
        future = probability*success_route+(1-probability)*failed_route
        cost = np.linalg.norm(self.position-center)/5+3+2*probability
        gain = baseline-cost-future+probability*(5+self.residual(radius))
        return dict(kind='clear', point=center, channels=[c], gain_s=float(gain),
                    guaranteed=guaranteed, predicted_success=probability)

    def choose_local(self, info, urgent=None):
        cfg = self.b5_config
        centers = {c:m for c,(m,_) in info.items()}
        baseline = self.estimate_route(self.position, centers)
        plans = []
        for c, (center, _) in info.items():
            if urgent is not None and c != urgent:
                continue
            if self.route_eligible(center, c == urgent):
                plan = self.evaluate_clear(c, info, centers, baseline)
                if plan is not None:
                    plans.append(plan)
        active = [c for c, (_,r) in info.items()
                  if r > 19.8 and self.probe_count[c] < cfg['max_probes']]
        active.sort(key=lambda c:(c != urgent, not self.route_eligible(info[c][0]),
                                  np.linalg.norm(info[c][0]-self.position), c))
        points = [self.position.copy()]
        points.extend(p.copy() for _,p in self.backbone[:2])
        if urgent is not None:
            active = [urgent] if urgent in active else []
        for c in active[:cfg['active_sources']]:
            points.extend(self.proposed_points(c, info))
        # Means are candidates only, never mandatory moves or evidence.
        original = list(points)
        if cfg['batching']:
            for i,p in enumerate(original):
                for q in original[i+1:]:
                    if 1 < np.linalg.norm(p-q) <= 180:
                        points.append((p+q)/2)
        unique = []
        for p in points:
            if self.route_eligible(p, urgent is not None) and not any(np.linalg.norm(p-q)<1 for q in unique):
                unique.append(p)
        unique.sort(key=lambda p:np.linalg.norm(p-self.position))
        ranked = []
        for p in unique[:cfg['max_candidates']]:
            plan = self.evaluate_probe(p, info, centers, baseline, required=urgent)
            if plan is not None:
                ranked.append(plan)
        ranked.sort(key=lambda row:row['gain_s'], reverse=True)
        for candidate in ranked[:cfg['finalists']]:
            plan = self.evaluate_probe(candidate['point'], info, centers, baseline,
                                       required=urgent, power=cfg['fine_power'])
            if plan is not None:
                plans.append(plan)
        if not plans:
            return None
        return max(plans, key=lambda p:p['gain_s'])

    def recover_one(self, c):
        """One optical grid attempt; finite conservative cover, no source monopoly."""
        if c not in self.optical:
            poly = self.polys[c]
            lo, hi = poly.min(axis=0), poly.max(axis=0)
            axes = [np.linspace(lo[k], hi[k], max(1, math.ceil((hi[k]-lo[k])/25))+1)
                    for k in range(2)]
            if len(axes[0])*len(axes[1]) > 10000:
                raise RuntimeError('B5 optical cover exceeds budget; no completeness claim')
            grid = [np.array([x,y]) for i,x in enumerate(axes[0])
                    for y in (axes[1] if i%2 == 0 else axes[1][::-1])]
            # Start at the closer endpoint; keep the cover even as region shrinks.
            if np.linalg.norm(grid[-1]-self.position) < np.linalg.norm(grid[0]-self.position):
                grid.reverse()
            self.optical[c] = grid
            self.fallback_count += 1
        if not self.optical[c]:
            raise ArithmeticError('B5 optical cover exhausted without clearing')
        point = self.optical[c].pop(0)
        self.b5_stats['recovery_actions'] += 1
        return dict(kind='recovery', point=point, channels=[c], gain_s=None)

    def execute_local(self, plan):
        p = plan['point']
        served = []
        if plan['kind'] == 'probe':
            # Another channel's near clear does not change this channel's geometry.
            # Re-check all eligibility after each real response.
            for c in plan['channels']:
                if not self.admissible(c, p):
                    continue
                state = self.measure(p, c)
                self.probe_count[c] += 1
                self.b5_stats['probe_measurements'] += 1
                served.append(c)
            if len(served) > 1:
                self.b5_stats['shared_stops'] += 1
        else:
            c = plan['channels'][0]
            if not self.clear(p, c, must_succeed=plan.get('guaranteed', False)):
                self.failed_points[c].append(np.asarray(p).copy())
            served.append(c)
        return served

    def scan_backbone(self):
        k, point = self.backbone[0]
        self.scan(point)
        self.backbone.pop(0)  # Partial known-channel probes never remove anchors.
        self._sector_actions = 0
        self.b5_stats['backbone_scans'] += 1
        return dict(kind='scan', point=point, channels=[], sector=k, gain_s=None)

    def run(self):
        entered = self.send('/enter')
        self.deadline = time.monotonic()+max(0, entered.get('remaining_real_duration_s',1200)-5)
        self.scan(np.zeros(2))
        sites = [1000*np.array([math.cos(k*math.tau/7), math.sin(k*math.tau/7)]) for k in range(7)]
        # The initial route is influenced by detected sources; subsequently its
        # sector order stays fixed to prevent oscillation between convenient zones.
        info = self.state_info()
        route = improved_tour(self.position, sites+[m for m,_ in info.values()])
        self.backbone = [(i, sites[i]) for i in route if i < 7]
        for macro in range(self.b5_config['max_macros']):
            self._macro = macro
            self.refresh_absence()
            if len(self.cleared | self.absent) == 20:
                break
            info = self.state_info()
            for c in info:
                self.wait_age[c] += 1
            urgent = max(info, key=lambda c:(self.wait_age[c],-c)) if info else None
            if urgent is not None and self.wait_age[urgent] < self.b5_config['max_wait']:
                urgent = None
            before = len(self.trace)
            start = self.position.copy()
            active_sector = self.backbone[0][0] if self.backbone else None
            tick = time.perf_counter()
            # Budget of local stops gives the backbone finite progress independent
            # of gain optimism. Urgency resumes after the mandatory scan.
            if self.backbone and self._sector_actions >= self.b5_config['sector_action_limit']:
                plan = self.scan_backbone()
                served = []
            else:
                plan = self.choose_local(info, urgent) if info else None
                if plan is None or (self.backbone and urgent is None and plan['gain_s'] < self.b5_config['min_gain_s']):
                    if self.backbone and urgent is None:
                        plan = self.scan_backbone()
                        served = []
                    else:
                        c = urgent if urgent is not None else min(info, key=lambda c:np.linalg.norm(info[c][0]-self.position))
                        plan = self.recover_one(c)
                        served = self.execute_local(plan)
                        self._sector_actions += 1
                else:
                    served = self.execute_local(plan)
                    self._sector_actions += 1
            self.planning_time_s += time.perf_counter()-tick
            self.planning_calls += 1
            # Real scans can serve known channels too, including a newly discovered
            # one. A failed trial clear does not reset its wait-age indefinitely.
            touched = set(served)
            for event in self.trace[before:]:
                if event['path'] == '/measure' and event['response']['measure_result'] in ('direction','near'):
                    touched.add(event['channel'])
            for c in touched:
                self.wait_age[c] = 0
            self.b5_stats['max_wait_seen'] = max(self.b5_stats['max_wait_seen'], max(self.wait_age.values(), default=0))
            self.b5_log.append(dict(macro=macro, kind=plan['kind'], active_sector=active_sector,
                position_before=start.tolist(), point=plan['point'].tolist(),
                planned_channels=plan['channels'], served_channels=sorted(touched),
                gain_s=plan['gain_s'], urgent_channel=urgent,
                remaining_sectors=[k for k,_ in self.backbone],
                wait_age={str(c):self.wait_age[c] for c in info if c not in self.cleared},
                cleared_channels=sorted(self.cleared), action_count=len(self.trace)-before))
            if len(self._metrics_cache) > 12000:
                self._metrics_cache.clear()
        else:
            raise RuntimeError('B5 macro budget exhausted; cannot claim full clearance')
        self.refresh_absence()
        if len(self.cleared | self.absent) != 20:
            raise RuntimeError('B5 stopped without complete observable evidence')
        result = self.send('/exit')
        return dict(method='b5', virtual_time_s=result['virtual_time_s'],
                    cleared_channels=sorted(self.cleared), absent_channels=sorted(self.absent),
                    **dict(self.b5_stats))
