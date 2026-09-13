"""Complete paired-world continuations. No real HTTP or real world is accepted."""
import math
import time

from .sector import ProbePolicy
from .simulator import LocalSimulator


def continue_world(job):
    """Return no cost unless the entire sampled mission completes legally."""
    started = time.monotonic()
    belief = job['belief'].clone()
    belief.deadline = math.inf  # Wall timeout rejects the job, never changes its score.
    environment = LocalSimulator(job['world'].clone(), position=belief.position,
                                 receiver=belief.receiver, virtual_time=belief.virtual_time)
    policy = ProbePolicy(job['config'])
    policy.points = dict(job['points'])
    policy.batch = job['batch']
    policy.shared_checked = job['shared_checked']
    policy.completion_mode = job['completion_mode']
    initial_clock, steps, movement = belief.virtual_time, 0, 0.
    initial_steps = belief.steps
    reason, error = 'action_limit', None
    try:
        for i in range(job.get('max_steps', 6000)):
            if belief.done():
                reason = 'complete'
                break
            if time.monotonic()-started >= job.get('wall_limit_s', 45.):
                reason = 'wall_limit'
                break
            if belief.virtual_time >= belief.virtual_limit:
                reason = 'virtual_limit'
                break
            action = job['action'] if i == 0 else policy.choose(belief)
            response = environment.execute(action, f'rollout-{initial_steps}-{i}')
            belief.apply(action, response, f'rollout-{initial_steps}-{i}')
            steps += 1
            policy.records.clear()  # These are hypothetical action traces, not real requests.
        complete = belief.done() and environment._world.score()['all_cleared']
        if complete:
            reason = 'complete'
    except Exception as exc:
        complete = False
        reason, error = 'error', f'{type(exc).__name__}: {exc}'
    return {'world_index':job['world_index'], 'candidate_index':job['candidate_index'],
            'complete':complete, 'cost_s':belief.virtual_time-initial_clock if complete else None,
            'steps':steps, 'movement_m':environment.costs['move_s']*5,
            'real_time_s':time.monotonic()-started, 'stop_reason':reason, 'error':error}
