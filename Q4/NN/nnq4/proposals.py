"""Optional external single-point probes over frozen Q4 public geometry.

These proposals never update a channel, start a pair macro, or replace the
certified survey obligations. New vector arithmetic uses CPU float64 torch;
the audited geometry and local-cost routines keep their original interfaces.
"""
import time

import torch

from .bridge import QuadratureError, guaranteed_clear, local_attempts
from q4.external_candidates import heading_moment
from q4.shaped import existing_proposals, shaped_pair


SHAPES = tuple((fraction, aspect) for fraction in (.7, .85, .95)
               for aspect in (.2, .5, 1.0))
SOURCES = {
    'shaped_probe': ('luxury221/mathmodel-b',
                     '2a4acf1ebeb44f5d51fde08f4ab8484323cecfc6',
                     'research/shaped_probe_v11/shaped_geometry.py:certified_shaped_probe'),
    'forward_heading': ('li2396803/cumcm2026-b-radio-interference-localization',
                        '1337c54aae777924ece984d2c5109cf3954841d9',
                        'q4.external_candidates.HeadingPolicy.local_choices'),
    'heard_side_after_miss': ('xmuyux/cumcm-2026-b-jammer-strategy',
                              '3e7b5a17dda9135c3ec0c45931dfab248efafea6',
                              'q4.external_candidates.HeadingPolicy.local_choices'),
    'shared_current': ('tong2324919503jl/26-',
                       '8aefbd78df015bf8d2034257a07353176c35798d',
                       'shared-stop idea; q4.shared_probes.SharedProbePolicy.choose'),
}


def _vector(value):
    return torch.as_tensor(value, dtype=torch.float64, device='cpu')


def _point(value):
    return tuple(float(x) for x in value)


def _separated(point, others):
    if not others:
        return True
    distances = torch.linalg.vector_norm(_vector(others) - _vector(point), dim=1)
    return bool(torch.all(distances >= 10.0))


def _heading_choices(post, channel, mean):
    """Same optional suffix as frozen existing_proposals, kept identifiable."""
    center = _vector(mean)
    heading, concentration, directional = heading_moment(post)
    result = []
    if concentration >= .4 and directional >= .5:
        result.append((_point(center + 350 * _vector(heading)), 'forward_heading', {
            'heading_kappa': concentration, 'directional_probability': directional,
            'forward_offset_m': 350.}))
    last = max(i for i, obs in enumerate(channel.history)
               if obs.result in ('direction', 'near'))
    misses = sum(obs.result == 'no_signal' for obs in channel.history[last + 1:])
    if misses:
        toward = _vector(channel.directions[-1].action.position) - center
        toward = toward / torch.linalg.vector_norm(toward).clamp_min(1.)
        sideways = torch.stack((-toward[1], toward[0]))
        offset = 100. * min(misses, 6)
        for side in (-1, 1):
            result.append((_point(center + offset * toward + side * .5 * offset * sideways),
                           'heard_side_after_miss', {'misses_since_signal': misses,
                                                    'toward_heard_side_m': offset}))
    return result


def external_choices(planner, b, c, choices, mean):
    """Return only new ``(Action, local_cost_s, detail)`` single-point options.

    The complete original geometric pool is used for shape/heading deduplication.
    A free current stop can be restored if coarse filtering omitted it from the
    actual choices. Every option still pays the frozen single-channel cost.
    Quadrature failure leaves the caller's original choices entirely intact.
    """
    channel = b.channels[c]
    if (channel.status != 'detected' or not channel.directions
            or getattr(planner, 'completion_mode', False)
            or b.steps >= planner.config.completion_after
            or b.deadline - time.monotonic() < planner.config.reserve_s
            or local_attempts(channel) >= planner.config.local_limit
            or guaranteed_clear(b, c)):
        return []
    try:
        post = planner.model.posterior(channel)
        heading = _heading_choices(post, channel, mean)
        full_pool = existing_proposals(planner, b, c, mean, choices)
        # The frozen helper appends these heading/miss ideas AFTER the actual
        # ProbePolicy pool. They are not existing actions merely by appearing
        # in that audit helper. Preserve earlier duplicates belonging to the
        # actual pool; remove only the known optional suffix.
        original_pool = full_pool[:-len(heading)] if heading else full_pool
        measured = [obs.action.position for obs in channel.history
                    if obs.action.kind == 'measure']
        additions, selected_points = [], []

        def add(point, family, extra, *, restore_current=False):
            point = _point(point)
            if not bool(torch.isfinite(_vector(point)).all()):
                return
            pool = [action.position for action, _, _ in choices] if restore_current else original_pool
            if not _separated(point, pool + measured + selected_points):
                return
            # Keep every new proposal. A coarse local proxy is metadata for
            # learning, not a top-three filter over the new action family.
            action, cost, branch = planner.probe(b, c, point, 10.)
            repo, commit, source_function = SOURCES[family]
            detail = {**branch, **extra, 'candidate_family': family,
                      'source_repo': repo, 'source_commit': commit,
                      'source_function': source_function,
                      'single_point_only': True, 'probability_used_for_proof': False,
                      'local_proxy_bearing_bin_deg': 10.}
            additions.append((action, cost, detail))
            selected_points.append(point)

        for fraction, aspect in SHAPES:
            proposal = shaped_pair(channel, fraction, aspect)
            if proposal is not None:
                points, proof = proposal
                for point in points:
                    add(point, 'shaped_probe', {'shape': proof})
        for point, family, detail in heading:
            add(point, family, detail)
        add(b.position, 'shared_current', {'zero_extra_movement': True}, restore_current=True)
        return additions
    except QuadratureError:
        return []
