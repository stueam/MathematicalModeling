"""Local hidden-state environment, never passed to the real action policy.

Fixed error at a point is keyed by (world seed, channel, exact coordinates),
not by query order, so repeated observations share the same error field.
"""
import copy
import hashlib
import math
import time


from .core import distance, point_key




class World:
    def __init__(self, sources, seed=0, error_mode='iid', cleared=()):
        self._sources = {s.channel: s for s in sources}
        if len(self._sources) != len(sources):
            raise ValueError('Source channels must be unique')
        for s in sources:
            if not 1 <= s.channel <= 20 or not 1000 <= s.radius <= 1500 or distance(s.position, (0, 0)) > 1800 + 1e-6:
                raise ValueError('Source outside problem bounds')
        self._cleared = set(cleared)
        self.seed, self.error_mode = int(seed), error_mode

    def clone(self):
        w = copy.copy(self)
        w._cleared = self._cleared.copy()
        return w

    def error(self, c, p):
        token = repr((self.seed, c, point_key(p))).encode()
        value = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), 'big') / 2**64
        if self.error_mode == 'iid':
            return 2*value-1
        if self.error_mode == 'extreme':
            return 1. if value >= .5 else -1.
        if self.error_mode == 'correlated':
            phase = (self.seed * .137 + c * 1.713) % (2*math.pi)
            return .7*math.sin(p[0]/180 + phase) + .3*math.sin(p[1]/270-phase)
        raise ValueError('Unknown error mode')

    def feedback(self, action):
        s = self._sources.get(action.channel)
        active = s is not None and action.channel not in self._cleared
        d = distance(action.position, s.position) if active else math.inf
        if action.kind == 'clear':
            success = d <= 20
            if success:
                self._cleared.add(action.channel)
            return {'clear_result': 'success' if success else 'no_target_in_range'}, 5 if success else 3
        if not active or d > s.radius:
            return {'measure_result': 'no_signal'}, 5
        if d <= 5:
            return {'measure_result': 'near'}, 5
        phi = math.degrees(math.atan2(s.position[1]-action.position[1], s.position[0]-action.position[0]))
        bearing = round((phi + self.error(action.channel, action.position)) % 360, 2) % 360
        return {'measure_result': 'direction', 'svd_deg': bearing}, 5

    def score(self):
        # Evaluation-only API. Controllers never receive the actual World.
        return {'source_count': len(self._sources), 'cleared_count': len(self._cleared),
                'all_cleared': self._cleared == set(self._sources),
                'clearance_fraction': len(self._cleared)/len(self._sources) if self._sources else 1.}


class LocalSimulator:
    def __init__(self, world, position=(0., 0.), receiver=1, virtual_time=0.):
        self._world = world
        self.position, self.receiver, self.virtual_time = position, receiver, virtual_time
        self._responses = {}
        self.costs = {'move_s': 0., 'switch_s': 0., 'measure_s': 0., 'clear_success_s': 0., 'clear_failure_s': 0.}

    def execute(self, action, request_id):
        if request_id in self._responses:
            old_action, result = self._responses[request_id]
            if action != old_action:
                raise ValueError('409: ID reused with changed request')
            return dict(result)
        move = round(distance(self.position, action.position)/5, 6)
        switch = int(action.kind == 'measure' and action.channel != self.receiver)
        feedback, operation = self._world.feedback(action)
        self.virtual_time = round(self.virtual_time + move + switch + operation, 6)
        self.position = action.position
        self.costs['move_s'] += move
        self.costs['switch_s'] += switch
        if action.kind == 'measure':
            self.receiver = action.channel
            self.costs['measure_s'] += operation
        else:
            self.costs['clear_success_s' if operation == 5 else 'clear_failure_s'] += operation
        response = {'accepted': True, 'virtual_time_s': self.virtual_time,
                    'real_timestamp_ms': int(time.time()*1000), **feedback}
        self._responses[request_id] = (action, response)
        return dict(response)
