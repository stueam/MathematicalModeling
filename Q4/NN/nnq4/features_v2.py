"""Public service and directed continuation features appended to a v1 Frame.

All costs are explicit planning proxies, never bounds or teacher rankings.
A local measure represents eventual completion of that channel at its public
posterior mean; a survey/clear ends at its requested coordinate. In particular,
neither a measure nor a speculative clear is assumed to succeed in the belief.
Complete-rollout supervision must learn the omitted failure/replanning costs.

Known service = max(0, min(frame local cost) - robot-to-mean travel / 5).
The v1 local cost is recovered from candidate[16] * 1000 + actual move / 5.
Missing local proxies use an explicitly flagged 5-second service placeholder.
Required survey service is 6 * needed-channel count, as in the frozen planner.

The open tail uses deterministic torch nearest-neighbor routing over ALL other
current obligations. It does not predict discoveries or future certified pruning.
No method reads frame.target, reference_reason, planner.records, or a World.
"""
from dataclasses import dataclass, replace
import math

import torch

from .bridge import DOMAIN, QuadratureError, local_attempts, point_key
from .state import EXTENT, Frame


FEATURE_VERSION = 'q4-public-service-directed-v2'
NODE_DIM, CANDIDATE_DIM, GLOBAL_DIM = 40, 48, 20
TIME_SCALE = 1000.
TASK_SCALE = 64.

# Positions/displacements/distances use EXTENT=2200 m; times use TIME_SCALE.
# Counts use TASK_SCALE except the separately named /16 and /49 fields.
NODE_EXTRA = {
    'service_s': 32, 'paid_channels_or_local_candidates': 33,
    'completion_x': 34, 'completion_y': 35,
    'other_min_distance': 36, 'other_mean_distance': 37,
    'other_task_count': 38, 'missing_service_or_ambiguous_identity': 39,
}
CANDIDATE_EXTRA = {
    'completion_x': 32, 'completion_y': 33,
    'completion_minus_action_x': 34, 'completion_minus_action_y': 35,
    'selected_task_service_s': 36, 'other_service_s': 37,
    'other_min_distance': 38, 'other_mean_distance': 39,
    'other_known_min_distance': 40, 'other_station_min_distance': 41,
    'open_tail_route_s': 42, 'full_remaining_proxy_s': 43,
    'other_known_count_over_16': 44, 'other_station_count_over_49': 45,
    'completion_position_is_proxy': 46, 'selected_service_missing': 47,
}
GLOBAL_EXTRA = {
    'all_service_s': 16, 'all_open_route_s': 17,
    'all_route_and_service_s': 18, 'task_count': 19,
}

# v1 does not carry node-to-channel IDs. These public values identify a target
# without relying on row order. Indistinguishable nodes receive the mean of the
# matching task features and an ambiguity flag; candidate identities are exact.
_CHANNEL_SIGNATURE = (0, 1, 11, 12, 14, 15, 16, 24, 26, 27, 28, 29)


def _double(value):
    return torch.as_tensor(value, dtype=torch.float64, device='cpu')


def _coord_key(point):
    return tuple((_double(point) / EXTENT).to(torch.float32).tolist())


def _known_signature(channel, c, b, mean):
    angle = math.radians(channel.directions[-1].bearing) if channel.directions else 0.
    values = [*(_double(mean) / EXTENT).tolist(), float(c == b.receiver),
              math.log1p(channel.region.area) / math.log1p(DOMAIN.area),
              min(len(channel.directions), 8) / 8, min(len(channel.negatives), 32) / 32,
              min(len(channel.history), 64) / 64, float(point_key(b.position) in channel.measured),
              math.cos(angle) if channel.directions else 0.,
              math.sin(angle) if channel.directions else 0.,
              float(any(o.result == 'near' for o in channel.history)), min(local_attempts(channel), 16) / 16]
    return tuple(torch.tensor(values, dtype=torch.float32).tolist())


@dataclass
class _Task:
    kind: str
    identity: object
    point: torch.Tensor
    service: float
    count: int
    missing: bool = False
    signature: tuple | None = None


def _tasks(frame, b, planner):
    """Construct obligations from public state, independently of frame ordering."""
    position = _double(b.position)
    tasks = []
    for c, channel in sorted(b.channels.items()):
        if channel.status != 'detected':
            continue
        try:
            post = planner.model.posterior(channel)
            mean = _double(post.weights) @ _double(post.xy)
        except QuadratureError:
            mean = _double(channel.summary()[0])
        costs, count = [], 0
        for row, action, survey in zip(frame.candidates, frame.actions, frame.survey):
            if survey or action.channel != c:
                continue
            count += 1
            overhead = float(row[16]) * TIME_SCALE
            if overhead > 0:
                move = float(torch.linalg.vector_norm(_double(action.position) - position)) / 5
                costs.append(overhead + move)
        missing = not costs
        service = max(0., min(costs) - float(torch.linalg.vector_norm(mean - position)) / 5) if costs else 5.
        tasks.append(_Task('known', c, mean, service, count, missing,
                           _known_signature(channel, c, b, mean)))
    # Multiple keys at one exact stop describe the same paid survey batch.
    # This never merges a known-source task with a co-located survey task.
    seen = set()
    for _, point in sorted(planner.points.items()):
        key = point_key(point)
        if key in seen:
            continue
        seen.add(key)
        channels = planner.needed(b, point)
        if channels:
            tasks.append(_Task('station', key, _double(point), 6. * len(channels), len(channels)))
    # Canonical geometry-first ordering makes routing ties independent of all
    # input node/candidate permutations, including repeated coordinates.
    return sorted(tasks, key=lambda t: (*t.point.tolist(), t.kind, str(t.identity)))


def _open_tail(starts, goals, remaining):
    """Batched deterministic open greedy routes, in meters, with no return leg."""
    costs = torch.zeros(len(starts), dtype=torch.float64)
    if not len(goals):
        return costs
    pending = remaining.clone()
    current = starts.clone()
    rows = torch.arange(len(starts))
    for _ in range(len(goals)):
        active = pending.any(dim=1)
        if not bool(active.any()):
            break
        distances = torch.linalg.vector_norm(current[:, None] - goals[None], dim=2)
        distances = distances.masked_fill(~pending, float('inf'))
        selected = distances.argmin(dim=1)
        costs += torch.where(active, distances[rows, selected], 0.)
        current = torch.where(active[:, None], goals[selected], current)
        pending[rows, selected] = False
    return costs


def _distance_stats(distances, mask):
    count = mask.sum(dim=1)
    nearest = distances.masked_fill(~mask, float('inf')).min(dim=1).values
    nearest = torch.where(count > 0, nearest, 0.)
    average = (distances * mask).sum(dim=1) / count.clamp_min(1)
    return nearest, average


def augment_frame(frame, b, planner):
    """Append v2 features without modifying the frame, belief, actions or plan."""
    if (frame.nodes.shape[1] != 32 or frame.candidates.shape[1] != 32
            or frame.global_features.shape != (16,)):
        raise ValueError('augment_frame expects exactly the v1 32/32/16 prefix')
    if len(frame.actions) != len(frame.candidates) or len(frame.actions) != len(frame.survey):
        raise ValueError('Action, candidate and survey counts must agree')
    tasks = _tasks(frame, b, planner)
    size, count = len(tasks), len(frame.actions)
    goals = torch.stack([t.point for t in tasks]) if tasks else torch.empty(0, 2, dtype=torch.float64)
    service = _double([t.service for t in tasks])
    known = torch.tensor([t.kind == 'known' for t in tasks], dtype=torch.bool)
    positions = _double([a.position for a in frame.actions]).reshape(count, 2)
    endpoints = positions.clone()
    remaining = torch.ones(count, size, dtype=torch.bool)
    extra = torch.zeros(count, 16, dtype=torch.float64)
    local_costs = torch.zeros(count, dtype=torch.float64)
    by_known = {t.identity: i for i, t in enumerate(tasks) if t.kind == 'known'}
    by_station = {t.identity: i for i, t in enumerate(tasks) if t.kind == 'station'}
    robot = _double(b.position)
    for i, (action, survey) in enumerate(zip(frame.actions, frame.survey)):
        task_index = (by_station.get(point_key(action.position)) if survey
                      else by_known.get(action.channel))
        if task_index is not None:
            remaining[i, task_index] = False
            extra[i, 4] = service[task_index] / TIME_SCALE
            extra[i, 15] = tasks[task_index].missing
            if action.kind == 'measure' and not survey:
                endpoints[i] = goals[task_index]
                extra[i, 14] = 1.
        move = float(torch.linalg.vector_norm(positions[i] - robot)) / 5
        if survey:
            survey_cost = 6. * len(planner.needed(b, action.position))
            extra[i, 4] = survey_cost / TIME_SCALE
            local_costs[i] = move + survey_cost
        elif float(frame.candidates[i, 16]) > 0:
            local_costs[i] = move + float(frame.candidates[i, 16]) * TIME_SCALE
        else:
            # Explicitly flagged fallback for an injected reference lacking
            # a local proxy; no target/reference metadata is consulted.
            extra[i, 15] = 1.
            operation = 3. + 2. * float(frame.candidates[i, 10]) if action.kind == 'clear' else 5. + int(action.channel != b.receiver)
            local_costs[i] = move + operation
            if action.kind == 'measure' and task_index is not None:
                local_costs[i] += torch.linalg.vector_norm(endpoints[i] - positions[i]) / 5 + service[task_index]
    extra[:, :2] = endpoints / EXTENT
    extra[:, 2:4] = (endpoints - positions) / EXTENT
    if size:
        distances = torch.linalg.vector_norm(endpoints[:, None] - goals[None], dim=2)
        extra[:, 5] = (remaining * service).sum(dim=1) / TIME_SCALE
        nearest, mean = _distance_stats(distances, remaining)
        extra[:, 6], extra[:, 7] = nearest / EXTENT, mean / EXTENT
        extra[:, 8] = _distance_stats(distances, remaining & known)[0] / EXTENT
        extra[:, 9] = _distance_stats(distances, remaining & ~known)[0] / EXTENT
        extra[:, 10] = _open_tail(endpoints, goals, remaining) / 5 / TIME_SCALE
        extra[:, 12] = (remaining & known).sum(dim=1) / 16
        extra[:, 13] = (remaining & ~known).sum(dim=1) / 49
    extra[:, 11] = local_costs / TIME_SCALE + extra[:, 5] + extra[:, 10]

    task_features = torch.zeros(size, 8, dtype=torch.float64)
    if size:
        task_features[:, 0] = service / TIME_SCALE
        task_features[:, 1] = _double([t.count for t in tasks]) / TASK_SCALE
        task_features[:, 2:4] = goals / EXTENT
        other = ~torch.eye(size, dtype=torch.bool)
        distances = torch.linalg.vector_norm(goals[:, None] - goals[None], dim=2)
        nearest, mean = _distance_stats(distances, other)
        task_features[:, 4], task_features[:, 5] = nearest / EXTENT, mean / EXTENT
        task_features[:, 6] = (size - 1) / TASK_SCALE
        task_features[:, 7] = _double([t.missing for t in tasks])
    node_extra = torch.zeros(len(frame.nodes), 8, dtype=torch.float64)
    channel_signatures, station_positions = {}, {}
    for i, task in enumerate(tasks):
        lookup, key = ((channel_signatures, task.signature) if task.kind == 'known'
                       else (station_positions, _coord_key(task.point)))
        lookup.setdefault(key, []).append(i)
    for i, node in enumerate(frame.nodes):
        matches = []
        if bool(node[4]) and bool(node[8]):
            matches = channel_signatures.get(tuple(node[list(_CHANNEL_SIGNATURE)].tolist()), [])
        elif bool(node[5]):
            matches = station_positions.get(tuple(node[:2].tolist()), [])
        if matches:
            node_extra[i] = task_features[matches].mean(dim=0)
            if len(matches) > 1:
                node_extra[i, 7] = 1.
        elif bool(node[4]) and bool(node[8]) or bool(node[5]):
            node_extra[i, 7] = 1.
    global_extra = torch.zeros(4, dtype=torch.float64)
    global_extra[0] = service.sum() / TIME_SCALE
    global_extra[1] = _open_tail(robot[None], goals, torch.ones(1, size, dtype=torch.bool))[0] / 5 / TIME_SCALE
    global_extra[2] = global_extra[0] + global_extra[1]
    global_extra[3] = size / TASK_SCALE
    # replace preserves all current/future Frame metadata verbatim, including
    # source families, without using any supervision field as an input feature.
    return replace(frame,
                   nodes=torch.cat((frame.nodes, node_extra.to(frame.nodes.dtype)), dim=1),
                   candidates=torch.cat((frame.candidates, extra.to(frame.candidates.dtype)), dim=1),
                   global_features=torch.cat((frame.global_features, global_extra.to(frame.global_features.dtype))))
