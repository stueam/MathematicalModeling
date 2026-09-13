"""Compact, certified search obligations on top of the unchanged V2 policy."""
from functools import lru_cache
from itertools import combinations
import math
import time

import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

from .core import DOMAIN
from .coverage import certifies
from .policy import Policy
from .shared import distance


@lru_cache(maxsize=1)
def triangular_stations():
    """950 m triangles covering the entire outer source domain, including edges.

    Every point of a retained triangle is within 950 m of all its vertices.
    The same full inner-exclusion certificate used at runtime is checked here.
    """
    edge = 950.
    xy = np.asarray([(edge*(i+.5*(j % 2)), edge*math.sqrt(3)/2*j)
                     for j in range(-4, 5) for i in range(-4, 5)])
    indices = set()
    for triangle in Delaunay(xy).simplices:
        if Polygon(xy[triangle]).intersects(DOMAIN):
            indices.update(map(int, triangle))
    points = tuple(tuple(map(float, xy[i])) for i in sorted(indices))
    if not certifies(DOMAIN, points):
        raise ValueError('Compact initial search skeleton lacks a full certificate')
    return points


class CoverPolicy(Policy):
    def __init__(self, mode='compact', config=None):
        if mode not in ('compact', 'sparse', 'trim', 'adaptive'):
            raise ValueError('Unknown compact-cover ablation')
        super().__init__('mobile', config)
        self.variant = mode
        self.implementation = f'q4_v3_{mode}_cover'
        # Keep the original square grid in self.fixed for the finite fallback.
        if mode in ('compact', 'sparse'):
            self.points = {-i-1: p for i, p in enumerate(triangular_stations())}
        self.prune_signature = None
        self.prune_failed = set()
        self.pending_cover_audit = []
        self.counters.update(cover_prune_reviews=0, cover_pruned_stations=0,
                             cover_prune_cache_hits=0, cover_exhaustion_fallbacks=0,
                             shared_merged_stations=0, initial_search_stations=len(self.points))

    def record(self, b, action, reason, **extra):
        if self.pending_cover_audit:
            extra['coverage_plan_audit'] = self.pending_cover_audit
            self.pending_cover_audit = []
        extra['planned_search_stations'] = len(self.points)
        return super().record(b, action, reason, **extra)

    def prune_stations(self, b):
        active = {k: p for k, p in self.points.items() if self.needed(b, p)}
        # Actual history accounts for already measured sites. The remaining
        # required sites alone must still certify every unresolved channel.
        if len(active) != len(self.points) and self.valid_future(b, active):
            self.points = active
        if len(active) < 2:
            return
        signature = (tuple(sorted(self.points.items())),
                     tuple(sorted({(c.region.wkb, tuple(sorted(c.negatives)))
                                   for c in b.channels.values() if c.status == 'unresolved'})))
        if signature != self.prune_signature:
            self.prune_signature, self.prune_failed = signature, set()
        route, _ = self.route(b.position, active)
        trials = []
        for i, k in enumerate(route):
            if distance(b.position, active[k]) < 1e-6:
                continue  # Keep the current free stop available for discovery.
            previous = b.position if i == 0 else active[route[i-1]]
            saving_m = distance(previous, active[k])
            if i+1 < len(route):
                following = active[route[i+1]]
                saving_m += distance(active[k], following)-distance(previous, following)
            predicted_saving = saving_m/5+6*len(self.needed(b, active[k]))
            trials.append((predicted_saving, k, saving_m))
        reviews = 0
        for saving, key, saving_m in sorted(trials, reverse=True):
            if key in self.prune_failed:
                self.counters['cover_prune_cache_hits'] += 1
                continue
            if reviews >= 4:
                break
            reviews += 1
            proposed = {k: p for k, p in self.points.items() if k != key}
            valid = self.valid_future(b, proposed)
            self.counters['cover_prune_reviews'] += 1
            self.pending_cover_audit.append({'operation': 'remove_station', 'station': key,
                'position': self.points[key], 'route_shortcut_m': saving_m,
                'predicted_saved_s': saving, 'full_certificate_valid': valid,
                'sites_before': len(self.points), 'sites_after': len(proposed),
                'probability_used_for_proof': False})
            if valid:
                self.points = proposed
                self.counters['cover_pruned_stations'] += 1
                self.prune_signature = None
                break  # Rebuild the complete route after the accepted change.
            self.prune_failed.add(key)

    def shared_replacement(self, b):
        if self.variant != 'adaptive':
            return super().shared_replacement(b)
        if not b.applied or self.shared_checked == b.steps:
            return None
        self.shared_checked = b.steps
        _, response = b.applied[next(reversed(b.applied))]
        # Reuse an actual local observation/clear stop, including a bearing
        # measurement. Search batches are still completed before this hook.
        if not (response.get('clear_result') == 'success'
                or response.get('measure_result') in ('direction', 'near')):
            return None
        current_channels = self.needed(b, b.position)
        if not current_channels:
            return None
        remaining = {k: p for k, p in self.points.items() if self.needed(b, p)}
        _, before = self.route(b.position, remaining)
        nearest = sorted(remaining, key=lambda k: distance(b.position, remaining[k]))[:5]
        groups = [(k,) for k in nearest] + list(combinations(nearest[:4], 2))
        trials = []
        for group in groups:
            proposed = {k: p for k, p in self.points.items() if k not in group}
            proposed[group[0]] = b.position
            adjusted = {k: p for k, p in remaining.items() if k not in group}
            adjusted[group[0]] = b.position
            _, after = self.route(b.position, adjusted)
            scans_saved = sum(len(self.needed(b, remaining[k])) for k in group)-len(current_channels)
            saving = (before-after)/5+6*scans_saved
            if saving > 1:
                trials.append((saving, group, proposed, before-after))
        audit = []
        for saving, group, proposed, distance_saved in sorted(trials, key=lambda t: -t[0]):
            valid = self.valid_future(b, proposed)
            audit.append({'stations': group, 'predicted_saving_s': saving,
                          'route_saved_m': distance_saved, 'full_certificate_valid': valid,
                          'probability_used_for_proof': False})
            if valid:
                self.points = proposed
                self.counters['shared_replacements'] += 1
                self.counters['shared_merged_stations'] += len(group)-1
                self.pending_cover_audit.extend(audit)
                return self.scan(b, b.position, 'adaptive_shared_station', certificate_audit=audit)
        self.pending_cover_audit.extend(audit)
        return None

    def choose(self, b):
        fallback = (self.completion_mode or b.steps >= self.config.completion_after
                    or b.deadline-time.monotonic() < self.config.reserve_s)
        batch_pending = (self.batch is not None and distance(self.batch, b.position) < 1e-6
                         and bool(self.needed(b, self.batch)))
        if self.variant != 'sparse' and not fallback and not b.done() and not batch_pending:
            self.prune_stations(b)
        # Incremental polygon differences can leave conservative fragments even
        # when an all-at-once future certificate was valid. Never exit from the
        # plan or round away fragments: execute the original finite fallback.
        if (not fallback and not b.done()
                and not any(c.status == 'detected' for c in b.channels.values())
                and not any(self.needed(b, p) for p in self.points.values())):
            self.completion_mode = True
            self.counters['cover_exhaustion_fallbacks'] += 1
            self.pending_cover_audit.append({'operation': 'restore_finite_grid',
                                            'reason': 'plan_exhausted_without_actual_certificate'})
        return super().choose(b)


def make_belief(mode):
    from .core import Belief
    if mode=='hull':
        from .hull_certificate import ConvexChannel
        return Belief(channels={c:ConvexChannel() for c in range(1,21)})
    return Belief()


def make_policy(mode, config=None, **options):
    if mode in ('collect','collect-value'):
        from .collected import CollectedPolicy
        return CollectedPolicy(config,value_gate=mode=='collect-value')
    if mode=='mc-stable':
        from .monte_carlo import StableMonteCarloPolicy
        return StableMonteCarloPolicy(config,**options)
    if mode=='pooled':
        from .pooled import PooledPolicy
        return PooledPolicy(config)
    if mode=='phased':
        from .phased import PhasedPolicy
        return PhasedPolicy(config)
    if mode=='deferred':
        from .deferred import DeferredPolicy
        return DeferredPolicy(config)
    if mode=='boundary':
        from .external_candidates import BoundaryPolicy
        return BoundaryPolicy(config)
    if mode=='hull':
        from .hull_certificate import HullPolicy
        return HullPolicy(config)
    if mode in ('heading','front','miss','witness','polar22'):
        from .external_candidates import HeadingPolicy, WitnessPolicy, PolarPolicy
        if mode=='witness':
            return WitnessPolicy(config)
        if mode=='polar22':
            return PolarPolicy(config)
        return HeadingPolicy(config,mode)
    if mode == 'share':
        from .shared_probes import SharedProbePolicy
        return SharedProbePolicy(config)
    if mode == 'bundle':
        from .bundles import BundlePolicy
        return BundlePolicy(config)
    if mode == 'mc-probes':
        from .monte_carlo import MonteCarloPolicy
        return MonteCarloPolicy(config, **options)
    if mode == 'corridor':
        from .corridor import CorridorPolicy
        return CorridorPolicy(config)
    if mode in ('ring', 'sector', 'probes', 'reuse', 'aligned'):
        from .sector import RingPolicy, SectorPolicy, ProbePolicy, ReusePolicy, AlignedPolicy
        return {'ring': RingPolicy, 'sector': SectorPolicy, 'probes': ProbePolicy,
                'reuse': ReusePolicy, 'aligned': AlignedPolicy}[mode](config)
    if mode in ('compact', 'sparse', 'trim', 'adaptive'):
        return CoverPolicy(mode, config)
    return Policy(mode, config)
