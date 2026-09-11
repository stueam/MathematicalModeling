"""Exact reuse of V6 computations; no change to its action menu or objective.

Route tables retain their existing eight-table bound. Candidate reuse exists
only during one choose() call, when the observed belief and posterior objects
are fixed; no simulated-world cost or observation is reused across decisions.
"""
import numpy as np

from .open_routes import OpenRoutes
from .refinement import RefinementPolicy
from .sectors import SectorPolicy


class CachedOpenRoutes(OpenRoutes):
    """Use the immutable distance matrix when evaluating route permutations.

    Keep the original heuristic, tie breaking and NumPy summation order. This
    avoids allocating coordinates and recomputing every edge norm for each
    2-opt/relocation proposal, including proposals previously considered.
    """

    def __init__(self, points, exact_limit=12):
        super().__init__(points, exact_limit)
        self._distance_start = None
        self._start_distances = None

    def length(self, start, order):
        if not len(order):
            return 0.
        start_key = tuple(map(float, start))
        if start_key != self._distance_start:
            self._distance_start = start_key
            self._start_distances = np.linalg.norm(self.xy-start, axis=1)
        ids = np.asarray(order, dtype=int)
        edges = np.empty(len(ids), dtype=float)
        edges[0] = self._start_distances[ids[0]]
        edges[1:] = self.dist[ids[:-1], ids[1:]]
        return float(edges.sum())


class CachedComputationMixin:
    """Reusable speed layer for V6-compatible policy subclasses.

    Local menus are evaluated twice by the V3 reference and joint route menu.
    Joint scoring also repeats exactly the local cost already in that menu.
    Reuse those values without reconstructing a residual by subtraction, which
    would introduce floating-point roundoff into the original comparison.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._decision_menus = None
        self._decision_local_costs = None

    def choose(self, b):
        self._decision_menus, self._decision_local_costs = {}, {}
        try:
            return super().choose(b)
        finally:
            # Invalidate on success and on errors. A subsequent choose() can
            # receive new feedback, a new receiver or a different belief.
            self._decision_menus = self._decision_local_costs = None

    def _table(self, points):
        key = tuple((k, tuple(map(float, points[k]))) for k in sorted(points))
        if key not in self._tables:
            if len(self._tables) >= 8:
                self._tables.pop(next(iter(self._tables)))
            self._tables[key] = CachedOpenRoutes(points, self.config.exact_limit)
        return self._tables[key]

    @staticmethod
    def _copy_menu(menu):
        return [(action, cost, dict(detail)) for action, cost, detail in menu]

    def _local_candidates(self, b, c, post):
        if self._decision_menus is None:
            return super()._local_candidates(b, c, post)
        key = (c, id(post))
        if key in self._decision_menus:
            menu, counters = self._decision_menus[key]
            # Preserve existing diagnostic counters as if the duplicate
            # candidate construction had run, including finite-grid calls.
            for name, increment in counters.items():
                setattr(self, name, getattr(self, name)+increment)
            return self._copy_menu(menu)
        before = {name: getattr(self, name) for name in ('fallback_calls', 'recovery_filtered')
                  if hasattr(self, name)}
        menu = super()._local_candidates(b, c, post)
        counters = {name: getattr(self, name, 0)-value for name, value in before.items()}
        self._decision_menus[key] = (self._copy_menu(menu), counters)
        for action, local, detail in menu:
            cost_key = (c, id(post), action.kind, tuple(action.position), bool(detail.get('guaranteed')))
            probabilities = ({name: detail[name] for name in
                ('near', 'no_signal', 'direction', 'positive_bearing_bins')}
                if action.kind == 'measure' else {})
            self._decision_local_costs[cost_key] = (local, probabilities)
        return menu

    def _score(self, b, action, c, plan, posts, kind='finish', detail=None):
        detail = dict(detail or {})
        key = (c, id(posts[c]), action.kind, tuple(action.position), bool(detail.get('guaranteed'))) if c >= 0 else None
        cached = self._decision_local_costs.get(key) if self._decision_local_costs is not None else None
        if cached is None or kind != 'finish':
            # Fine posterior review and route probes need their own numerical
            # integration. They never inherit coarse-menu metadata or costs.
            return super()._score(b, action, c, plan, posts, kind=kind, detail=detail)
        local_cost, probabilities = cached
        detail.update(probabilities)
        local_cost += plan['scans'][c]
        points = plan['points']
        endpoint = action.position if action.kind == 'clear' else points[c]
        rest = [k for k in points if k != c]
        length, route = plan['table'].plan(endpoint, rest)
        other = sum(plan['local'][k]+plan['scans'][k] for k in rest)
        return {'action': action, 'task': f'source:{c}', 'kind': kind,
                'local_s': local_cost, 'tail_route_s': length/5,
                'other_work_s': other, 'score_s': local_cost+length/5+other,
                'continuation': route, 'mask': plan['mask'], **detail}


class CachedRefinementPolicy(CachedComputationMixin, RefinementPolicy):
    implementation = 'bayes_tsp_v6_cached_computation'


class CachedSectorPolicy(CachedComputationMixin, SectorPolicy):
    implementation = 'bayes_tsp_v5_cached_computation'
