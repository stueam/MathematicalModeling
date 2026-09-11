import time
from q3.core import GeometryError
from q3.sampling import SamplingError
from .action_gen import MacroPolicy
from .generative_model import environment, simulate
from .heuristic import remaining_time


def evaluate_world(job):
    """One worker owns one hypothetical world and ALL paired candidates."""
    belief, world, actions, horizon, tail_mode, deadline = job
    rows = []
    for first in actions:
        if time.monotonic() >= deadline:
            raise TimeoutError('Paired world deadline')
        b = belief.clone()
        # Future policies use quadrature of their own observations, not truth.
        env, policy = environment(world, b), MacroPolicy()
        start, steps = b.virtual_time, 0
        for depth in range(horizon):
            if b.done():
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Rollout deadline')
            a = first if depth == 0 else policy.choose(b)
            obs, _ = simulate(env, b, a)
            steps += len(obs)
        prefix = b.virtual_time-start
        if tail_mode == 'completion':
            # Optional independent tail audit / high-quality estimator: finish
            # the same sampled world under an observation-only rollout policy.
            while not b.done():
                if time.monotonic() >= deadline:
                    raise TimeoutError('Completion tail deadline')
                if steps >= 4000 or b.virtual_time >= b.virtual_limit:
                    raise SamplingError('Unfinished continuation cannot be scored')
                obs, _ = simulate(env, b, policy.choose(b))
                steps += len(obs)
            tail = b.virtual_time-start-prefix
        else:
            tail = remaining_time(b)
        rows.append({'cost': prefix+tail, 'prefix': prefix, 'tail': tail,
                     'physical_steps': steps, 'terminal': b.done()})
    return rows
