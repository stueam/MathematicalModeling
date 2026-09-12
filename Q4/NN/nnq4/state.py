"""Public Q4 belief nodes and exact, variable candidate actions.

The frozen planner supplies certified coverage geometry and local proposals.
Its selected action is retained as a reference for supervision, never a feature.
Candidate order is canonical; there is no reference-action flag or route score.
"""
from dataclasses import dataclass
import math
import torch

from .bridge import (Action, DOMAIN, ProbePolicy, Config, distance, point_key,
                     guaranteed_clear, local_attempts, QuadratureError)

NODE_DIM = CANDIDATE_DIM = 32
GLOBAL_DIM = 16
EXTENT = 2200.
FEATURE_VERSION = 'q4-public-nodes-v1'


@dataclass
class Frame:
    nodes: torch.Tensor
    candidates: torch.Tensor
    global_features: torch.Tensor
    actions: list
    survey: list
    target: int
    reference_reason: str
    families: list | None = None

    def tensors(self):
        return {'nodes': self.nodes, 'node_mask': torch.ones(len(self.nodes), dtype=torch.bool),
                'candidates': self.candidates, 'candidate_mask': torch.ones(len(self.actions), dtype=torch.bool),
                'global_features': self.global_features}

    def arrays(self):
        """NumPy conversion is confined to the NPZ/current inference boundary."""
        return {k: v.detach().numpy() for k, v in self.tensors().items()}


def _double(value):
    """Frozen quadrature arrays enter feature arithmetic at full precision."""
    return torch.as_tensor(value, dtype=torch.float64, device='cpu')


def _weighted_quantile_indices(xy, weights, count):
    """Match lexsort((y, x)) and left-search cumulative weighted quantiles."""
    # Stable least-significant-key first, retaining original order on x/y ties.
    by_y = torch.argsort(xy[:, 1], stable=True)
    order = by_y[torch.argsort(xy[by_y, 0], stable=True)]
    cumulative = torch.cumsum(weights[order], dim=0, dtype=torch.float64)
    quantiles = (torch.arange(count, dtype=torch.float64) + .5) / count
    selected = torch.searchsorted(cumulative, quantiles, right=False).clamp_max(len(order) - 1)
    return order[selected]


class MenuPlanner(ProbePolicy):
    def __init__(self, config=None, menu='full', particles=4, layout='ring22', features='v1'):
        super().__init__(config or Config())
        if menu not in ('full', 'external'):
            raise ValueError('Unknown candidate menu')
        if layout != 'ring22':
            from q4.repo_layouts import certified_layout
            self.points = {-i-1: p for i, p in enumerate(certified_layout(layout))}
            self.counters['initial_search_stations'] = len(self.points)
        self.layout = layout
        if features not in ('v1', 'v2'):
            raise ValueError('Unknown feature schema')
        self.features = features
        self.menu = menu
        self.particles = particles
        self.stats = {'frames': 0, 'reference_missing': 0, 'posterior_failures': 0,
                      'candidate_families': {}}

    def pending_action(self, b):
        """A selected survey macro performs separate paid channel requests."""
        if self.batch is not None and distance(self.batch, b.position) < 1e-6:
            channels = self.needed(b, self.batch)
            if channels:
                c = b.receiver if b.receiver in channels else channels[0]
                return Action('measure', self.batch, c)
        self.batch = None
        return None

    def commit(self, action, survey):
        self.batch = action.position if survey else None

    def frame(self, b):
        # Deterministic coverage pruning/adjustment is shared by training and
        # deployment. Reference selection is not exposed to the neural model.
        reference = super().choose(b)
        reason = self.records[-1]['reason']
        self.records.clear()
        self.batch = None  # Only the ultimately executed action can start a batch.
        self.stats['frames'] += 1
        posts, post_tensors, channel_nodes = {}, {}, {}
        position = _double(b.position)
        for c, p in b.channels.items():
            row = torch.zeros(NODE_DIM, dtype=torch.float32)
            row[4] = 1
            row[7 + ('unresolved', 'detected', 'cleared', 'absent_certified').index(p.status)] = 1
            row[11] = c == b.receiver
            row[12] = math.log1p(p.region.area) / math.log1p(DOMAIN.area)
            row[14] = min(len(p.directions), 8) / 8
            row[15] = min(len(p.negatives), 32) / 32
            row[16] = min(len(p.history), 64) / 64
            row[24] = point_key(b.position) in p.measured
            row[29] = min(local_attempts(p), 16) / 16
            if p.status in ('detected', 'unresolved'):
                try:
                    posts[c] = post = self.model.posterior(p)
                    xy, weights = _double(post.xy), _double(post.weights)
                    post_tensors[c] = (xy, weights)
                    mean = weights @ xy
                    delta = xy - mean
                    covariance = (delta.T * weights) @ delta / EXTENT**2
                    row[13] = torch.sqrt(torch.trace(covariance).clamp_min(0.))
                    row[17] = _double(post.atoms)[:, 1:].sum()
                    row[18] = math.log1p(max(0., post.evidence) * 1000) / math.log(1001)
                    row[20:23] = torch.stack((covariance[0, 0], covariance[1, 1], covariance[0, 1]))
                    row[25] = post.clear_probability(b.position)
                    row[30] = (_double(post.atoms) *
                               (_double(post.upper) + _double(post.lower)[:, None]) / 2).sum() / 1500
                    row[31] = _double(post.reception(b.position)[1]).sum()
                except QuadratureError:
                    self.stats['posterior_failures'] += 1
                    mean, radius = p.summary()
                    row[13] = radius / EXTENT
            else:
                mean = p.summary()[0] if not p.region.is_empty else (0., 0.)
            row[:2] = _double(mean) / EXTENT
            row[2:4] = (_double(mean) - position) / EXTENT
            if p.directions:
                a = math.radians(p.directions[-1].bearing)
                row[26:28] = torch.tensor((math.cos(a), math.sin(a)), dtype=torch.float32)
            row[28] = any(o.result == 'near' for o in p.history)
            channel_nodes[c] = row
        unknown = [c for c, p in b.channels.items() if p.status == 'unresolved']
        try:
            existence = self.model.existence(b)
        except QuadratureError:
            existence = {c: self.config.existence_prior for c in unknown}
        for c, row in channel_nodes.items():
            row[19] = existence.get(c, float(b.channels[c].status == 'detected'))

        nodes = list(channel_nodes.values())
        sites = [(k, p, self.needed(b, p)) for k, p in sorted(self.points.items())]
        for _, p, channels in sites:
            if not channels:
                continue
            row = torch.zeros(NODE_DIM, dtype=torch.float32)
            row[:2] = _double(p) / EXTENT
            row[2:4] = (_double(p) - position) / EXTENT
            row[5] = 1
            row[15] = len(channels) / 20
            row[19] = sum(existence.get(c, 0.) for c in channels) / 16
            row[24] = distance(p, b.position) < 1e-6
            nodes.append(row)
        for c, post in posts.items():
            if b.channels[c].status != 'detected' or self.particles <= 0:
                continue
            # Weighted quantiles, canonical order and no sampled hidden truth.
            xy, weights = post_tensors[c]
            ix = _weighted_quantile_indices(xy, weights, self.particles)
            for j in ix:
                row = channel_nodes[c].clone()
                row[4], row[6] = 0, 1
                row[:2] = xy[j] / EXTENT
                row[2:4] = (xy[j] - position) / EXTENT
                row[23] = weights[j]
                nodes.append(row)

        entries = {}
        def add(action, survey=False, detail=None, local_cost=None):
            key = (action.kind, action.channel, point_key(action.position))
            if key in entries:
                return
            p = b.channels[action.channel]
            if p.status not in ('unresolved', 'detected'):
                return
            if action.kind == 'measure' and point_key(action.position) in p.measured:
                return
            if action.kind == 'clear' and any(o.result == 'no_target_in_range' and
                    point_key(o.action.position) == point_key(action.position) for o in p.history):
                return
            row = torch.zeros(CANDIDATE_DIM, dtype=torch.float32)
            row[:2] = _double(action.position) / EXTENT
            row[2:4] = (_double(action.position) - position) / EXTENT
            row[4:7] = torch.tensor((action.kind == 'measure' and not survey,
                                    action.kind == 'clear', survey), dtype=torch.float32)
            move = distance(b.position, action.position) / 5
            row[7] = move / 880
            channels = self.needed(b, action.position) if survey else []
            row[8] = (6 * len(channels) if survey else (5 if action.kind == 'measure' else 3)) / 120
            row[9] = action.kind == 'measure' and action.channel != b.receiver
            post = posts.get(action.channel)
            if post is not None:
                row[10] = post.clear_probability(action.position)
                row[11] = _double(post.reception(action.position)[1]).sum()
            detail = detail or {}
            row[12] = bool(detail.get('guaranteed'))
            source = channel_nodes[action.channel]
            row[13:15] = source[:2] - row[:2]
            row[15] = source[13]
            row[16] = max(0., local_cost-move) / 1000 if local_cost is not None else 0
            row[17] = len(channels) / 20
            # Public posterior predicted reception, not actual future feedback.
            if survey:
                row[18] = sum(existence.get(c, 0.) * float(_double(posts[c].reception(action.position)[1]).sum())
                              for c in channels if c in posts) / 16
            row[19:24] = source[torch.tensor((8, 17, 15, 14, 28))]
            row[24] = distance(action.position, b.position) < 1e-6
            row[25] = source[29]
            row[26:29] = source[20:23]
            row[29:31] = source[26:28]
            row[31] = source[19]
            family = detail.get('candidate_family', 'survey' if survey else 'local')
            entries[key] = (action, survey, row, family)

        for _, p, channels in sites:
            if channels:
                c = b.receiver if b.receiver in channels else channels[0]
                add(Action('measure', p, c), True)
        if self.needed(b, b.position):
            channels = self.needed(b, b.position)
            c = b.receiver if b.receiver in channels else channels[0]
            add(Action('measure', b.position, c), True)
        for c, p in b.channels.items():
            if p.status == 'detected':
                choices, mean = self.local_choices(b, c)
                if self.menu == 'external':
                    from .proposals import external_choices
                    choices = choices + external_choices(self, b, c, choices, mean)
                for action, cost, detail in choices:
                    add(action, False, detail, cost)
        # Reference action may be a certified moved/shared survey station.
        # The identical deterministic generator runs at deployment as well.
        refkey = (reference.kind, reference.channel, point_key(reference.position))
        if refkey not in entries:
            self.stats['reference_missing'] += 1
            add(reference, reference.kind == 'measure' and b.channels[reference.channel].status == 'unresolved')
        if refkey not in entries:
            raise ValueError('Reference action not representable as an admissible candidate')
        ordered = sorted(entries)
        actions, surveys, features, families = zip(*(entries[k] for k in ordered))
        for family in families:
            self.stats['candidate_families'][family] = self.stats['candidate_families'].get(family, 0)+1
        extra = torch.zeros(GLOBAL_DIM, dtype=torch.float32)
        extra[:2] = position/EXTENT
        extra[2:7] = torch.tensor((b.virtual_time/10000, b.steps/1000, len(b.cleared)/16,
                                   len(b.known)/16, len(unknown)/20), dtype=torch.float32)
        extra[7] = sum(p.status == 'detected' for p in b.channels.values())/16
        extra[8] = len(sites)/49
        extra[9] = sum(existence.values())/16
        extra[10] = torch.stack([_double(p.atoms)[:, 1:].sum() for p in posts.values()]).mean() if posts else 0
        extra[11] = b.position[0]**2+b.position[1]**2 < 1800**2
        if b.applied:
            action, response = b.applied[next(reversed(b.applied))]
            extra[12:16] = torch.tensor((response.get('measure_result') == 'no_signal',
                                         response.get('measure_result') == 'direction',
                                         response.get('clear_result') == 'success',
                                         response.get('clear_result') == 'no_target_in_range'), dtype=torch.float32)
        frame = Frame(torch.stack(nodes), torch.stack(features), extra,
                      list(actions), list(surveys), ordered.index(refkey), reason, list(families))
        if self.features == 'v2':
            from .features_v2 import augment_frame
            frame = augment_frame(frame, b, self)
        return frame
