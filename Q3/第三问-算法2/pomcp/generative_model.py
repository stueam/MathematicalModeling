"""Macros are serial compositions of the single shared exact physical kernel."""
from dataclasses import dataclass
from q3.core import Action
from q3.simulator import LocalSimulator


@dataclass(frozen=True)
class MacroAction:
    kind: str
    actions: tuple[Action, ...]

    def __post_init__(self):
        if not self.actions or len({a.position for a in self.actions}) != 1:
            raise ValueError('A macro requires at least one action at one stop')

    @property
    def position(self):
        return self.actions[0].position

    @classmethod
    def atomic(cls, a, kind=None):
        return cls(kind or ('CLEAR_TARGET' if a.kind == 'clear' else 'PROBE_TARGET'), (a,))


def simulate(env, belief, macro, prefix='simulation'):
    start = env.virtual_time
    observations = []
    for action in macro.actions:
        if belief.done():
            break
        request_id = f'{prefix}-{belief.steps}'
        response = env.execute(action, request_id)
        belief.apply(action, response, request_id)
        observations.append((action, response))
    return observations, env.virtual_time-start


def environment(world, belief):
    return LocalSimulator(world.clone(), belief.position, belief.receiver, belief.virtual_time)
