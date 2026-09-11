"""Complete paired-review work units using only self-sampled hidden worlds.

The caller must use the same sampled worlds for every candidate, retain its
baseline candidate, and discard an entire comparison if any work unit fails.
This module does not select actions for the live environment or open HTTP.
"""
from dataclasses import dataclass
import math
import time

from .cached import CachedRefinementPolicy
from .refinement import RefinementConfig
from .shared import load


sampling = load('sampling')
simulator = load('simulator')


class ReviewWorldSampler(sampling.WorldSampler):
    """Approximate conditional worlds from an isolated public-state snapshot.

    The shared sampler integrates one fixed radius per source, preserves known
    bearings at previously measured points, conditions existence on 10--16
    total sources, and retains cleared sources in that count. CoupledBelief's
    clone retains its channel subclasses and conservative coupled geometry.
    Empty proposal pools raise; they never certify source absence.
    """

    def sample(self, belief, count, deadline=math.inf):
        if count < 1:
            raise ValueError('Review needs at least one complete sampled world')
        observer = belief.clone()
        return super().sample(observer, count, min(deadline, time.monotonic()+120.))


def capture_continuation_state(policy):
    """Capture V6's decision-bearing memory after the reference choose().

    DynamicTour's route is a warm start for local optimization, not a pure
    cache. Active target, pending service and permanent completion mode also
    affect later actions. Immutable geometry/quadrature/TSP caches can instead
    be recomputed; counters and prior decision records do not select actions.
    This snapshot does not represent a recursive MC review policy's memory.
    """
    selected = policy.records[-1]['selected'] if policy.records else None
    return {'version': 1, 'tour_route': tuple(policy.tour.route),
            'tour_revisions': policy.tour.revisions, 'active_target': policy.active_target,
            'completion_mode': policy.completion_mode, 'after_clear': policy.after_clear,
            'batch_position': policy.batch_position, 'batch_channels': tuple(policy.batch_channels),
            'batch_kind': policy.batch_kind,
            'selected_action': ((selected['kind'], tuple(selected['position']), selected['channel'])
                                if selected else None)}


def restore_continuation_state(policy, state):
    """Restore an isolated lightweight V6 snapshot onto a new continuation."""
    if state.get('version') != 1:
        raise ValueError('Unsupported V6 continuation-state version')
    policy.tour.route = list(state['tour_route'])
    policy.tour.revisions = state['tour_revisions']
    policy.active_target = state['active_target']
    policy.completion_mode = state['completion_mode']
    after_clear = state['after_clear']
    policy.after_clear = ((tuple(after_clear[0]), after_clear[1]) if after_clear is not None else None)
    position = state['batch_position']
    policy.batch_position = tuple(position) if position is not None else None
    policy.batch_channels = list(state['batch_channels'])
    policy.batch_kind = state['batch_kind']


@dataclass(frozen=True)
class RolloutJob:
    belief: object
    world: object
    action: object
    batch_channels: tuple = ()
    config: object = None
    max_steps: int = 4000
    timeout_s: float = 120.
    continuation_state: object = None


def evaluate_task(job):
    """Impose one candidate, then finish with the same V6 continuation.

    Return a residual virtual cost only after both public certification and
    evaluator-only clearance verification. A timeout, error or budget-limited
    prefix has ``cost=None``; its spent work is never a candidate score.
    The job's belief, world, virtual clock and real deadline remain untouched.
    """
    started = time.monotonic()
    steps = 0
    try:
        if isinstance(job, dict):
            job = RolloutJob(**job)
        if job.max_steps < 1 or not math.isfinite(job.timeout_s) or job.timeout_s <= 0:
            raise ValueError('Rollout max_steps and timeout_s must be positive and finite')
        config = job.config or RefinementConfig()
        if isinstance(config, dict):
            config = RefinementConfig(**config)
        observer = job.belief.clone()
        initial_time = observer.virtual_time
        observer.deadline = started+min(120., job.timeout_s)
        world = job.world.clone()
        env = simulator.LocalSimulator(world, observer.position, observer.receiver, initial_time)
        policy = CachedRefinementPolicy(config)
        if job.continuation_state is not None:
            restore_continuation_state(policy, job.continuation_state)
            selected = job.continuation_state.get('selected_action')
            identity = (job.action.kind, tuple(job.action.position), job.action.channel)
            if selected is not None:
                selected = (selected[0], tuple(selected[1]), selected[2])
            if identity != selected:
                # Match the experiment's accepted-challenger state transition.
                # The reference has already warmed the route in the live policy.
                policy.batch_channels = list(dict.fromkeys(job.batch_channels))
                policy.batch_position = job.action.position if policy.batch_channels else None
                policy.active_target = job.action.channel if not policy.batch_channels else None
                policy.after_clear = ((job.action.position, job.action.channel)
                                      if job.action.kind == 'clear' and config.service_scans else None)
            # For the identical reference, retain its actual queue and hooks.
            # Reconstructing a survey batch from coverage can change that queue.
        else:
            policy.batch_position = job.action.position
            policy.batch_channels = list(dict.fromkeys(job.batch_channels))
            policy.batch_kind = 'rollout_review_batch'
            # An imposed clear bypasses _record(), so restore its ordinary
            # observation-only post-clear service hook for the next decision.
            if job.action.kind == 'clear' and config.service_scans:
                policy.after_clear = (job.action.position, job.action.channel)
        while not observer.done():
            if time.monotonic() >= observer.deadline:
                raise TimeoutError('Rollout real deadline before certified completion')
            if steps >= job.max_steps:
                raise sampling.SamplingError('Rollout action limit before certified completion')
            if observer.virtual_time >= observer.virtual_limit:
                raise sampling.SamplingError('Rollout virtual limit before certified completion')
            action = job.action if steps == 0 else policy.choose(observer)
            if time.monotonic() >= observer.deadline:
                raise TimeoutError('Rollout decision exceeded real deadline')
            request_id = f'rollout-review-{observer.steps}-{steps}'
            while request_id in observer.applied:
                request_id += '-new'
            response = env.execute(action, request_id)
            if not observer.apply(action, response, request_id):
                raise sampling.SamplingError('Rollout feedback was not applied')
            steps += 1
            if observer.virtual_time > observer.virtual_limit:
                raise sampling.SamplingError('Rollout exceeded virtual budget')
        # This is an evaluator check AFTER the observation-only controller has
        # stopped. Neither this score nor hidden coordinates select any action.
        if not world.score()['all_cleared']:
            raise sampling.SamplingError('Public completion disagrees with sampled-world clearance')
        return {'cost': observer.virtual_time-initial_time, 'completed': True,
                'steps': steps, 'error': None, 'elapsed_s': time.monotonic()-started}
    except Exception as exc:
        return {'cost': None, 'completed': False, 'steps': steps,
                'error': f'{type(exc).__name__}: {exc}', 'elapsed_s': time.monotonic()-started}
