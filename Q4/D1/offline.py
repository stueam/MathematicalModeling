"""Local Q4 rules. Only this evaluator holds hidden source locations/types."""
import hashlib
import math
import numpy as np


class OfflineEnvironment:
    def __init__(self, scenario):
        self.targets = {t['channel']: dict(t) for t in scenario['targets']}
        if len(self.targets) != len(scenario['targets']):
            raise ValueError('Channels must be unique')
        for c, t in self.targets.items():
            if c not in range(1, 21) or not 1000 <= t['radius'] <= 1500 or np.linalg.norm(t['position']) > 1800+1e-7:
                raise ValueError('Invalid source specification')
            if t.get('heading_deg') is not None and not math.isfinite(t['heading_deg']):
                raise ValueError('Heading must be finite or null for an omni source')
        self.seed = scenario.get('error_seed', 0)
        self.error_mode = scenario.get('error_mode', 'hashed')
        self.position = np.zeros(2)
        self.channel = 1
        self.clock = 0.
        self.cleared = set()
        self.movement = 0.
        self.measures = 0
        self.clear_attempts = 0
        self.clear_failures = 0
        self.switches = 0
        self.clear_times = []
        self.states = {'direction': 0, 'near': 0, 'no_signal': 0}

    @staticmethod
    def visible(point, target):
        delta = np.asarray(point)-target['position']
        if np.linalg.norm(delta) > target['radius']:
            return False
        heading = target.get('heading_deg')
        if heading is None:
            return True
        a = math.radians(heading)
        # Tiny absolute arithmetic tolerance handles closed ±90° boundary.
        return delta@np.array([math.cos(a), math.sin(a)]) >= -1e-10

    def error(self, point, channel):
        if self.error_mode == 'positive':
            return 1.
        if self.error_mode == 'negative':
            return -1.
        if self.error_mode == 'zero':
            return 0.
        if self.error_mode == 'fixed_extreme':
            return 1. if channel % 2 else -1.
        if self.error_mode == 'spatial_smooth':
            return math.sin(point[0]/170+point[1]/230+channel*.71+self.seed*.00001)
        if self.error_mode != 'hashed':
            raise ValueError('Unknown error mode')
        key = f'{self.seed}:{channel}:{float(point[0]).hex()}:{float(point[1]).hex()}'.encode()
        return 2*int.from_bytes(hashlib.sha256(key).digest()[:8], 'big')/(2**64-1)-1

    def action(self, path, point=None, channel=None):
        if path in ('/enter', '/exit'):
            return {'accepted': True, 'virtual_time_s': self.clock, 'remaining_real_duration_s': 1200}
        if path not in ('/measure', '/clear') or channel not in range(1, 21):
            raise ValueError('Invalid action')
        point = np.asarray(point, float)
        if point.shape != (2,) or not np.isfinite(point).all() or np.max(np.abs(point)) > 2000000:
            raise ValueError('Invalid point')
        move = float(np.linalg.norm(point-self.position))
        self.movement += move
        self.clock += move/5
        self.position = point.copy()
        target = self.targets.get(channel)
        present = target is not None and channel not in self.cleared
        distance = float(np.linalg.norm(point-target['position'])) if present else float('inf')
        result = {'accepted': True}
        if path == '/measure':
            self.measures += 1
            self.switches += int(channel != self.channel)
            self.clock += 5+int(channel != self.channel)
            self.channel = channel
            state = 'no_signal'
            if present and self.visible(point, target):
                state = 'near' if distance <= 5 else 'direction'
            result['measure_result'] = state
            self.states[state] += 1
            if state == 'direction':
                delta = np.asarray(target['position'])-point
                true = math.degrees(math.atan2(delta[1], delta[0]))
                result['svd_deg'] = round(true+self.error(point, channel), 2) % 360
        else:
            self.clear_attempts += 1
            success = present and distance <= 20
            self.clock += 5 if success else 3
            result['clear_result'] = 'success' if success else 'no_target_in_range'
            if success:
                self.cleared.add(channel)
                self.clear_times.append(self.clock)
            else:
                self.clear_failures += 1
        if self.clock > 360000:
            raise TimeoutError('100-hour virtual-time limit exceeded')
        result['virtual_time_s'] = self.clock
        return result


def make_cases(seed=2026091104, count=28):
    rng = np.random.default_rng(seed)
    families = ['uniform_mixed', 'boundary_outward', 'two_clusters', 'narrow_sector',
                'tangent_heading', 'all_directional_stress', 'all_omni_regression']
    cases = []
    for k in range(count):
        family = families[k % len(families)]
        n = 10+(k//len(families)+k % len(families)) % 7
        angle = rng.uniform(0, math.tau, n)
        radius = 1800*np.sqrt(rng.random(n))
        if family in ('boundary_outward', 'tangent_heading'):
            radius = np.full(n, 1800.) if k % 2 == 0 else rng.uniform(1795, 1800, n)
        elif family == 'narrow_sector':
            angle = rng.uniform(0, math.tau)+rng.uniform(-.05, .05, n)
        points = np.column_stack((radius*np.cos(angle), radius*np.sin(angle)))
        if family == 'two_clusters':
            points = np.array([np.array([1100*(-1 if j % 2 else 1), 0])+rng.normal(0, 160, 2) for j in range(n)])
            points *= np.minimum(1, 1800/np.maximum(np.linalg.norm(points, axis=1), 1))[:, None]
        channels = rng.choice(np.arange(1, 21), n, replace=False)
        targets = []
        # Mixed official-format families always contain both types. Extremes are diagnostics.
        directional_n = n if family == 'all_directional_stress' else 0 if family == 'all_omni_regression' else [1, n//2, n-1][k % 3]
        for j, (c, p) in enumerate(zip(channels, points)):
            heading = None
            if j < directional_n:
                heading = float(rng.uniform(0, 360))
                if family == 'boundary_outward':
                    heading = math.degrees(math.atan2(p[1], p[0])) % 360
                elif family == 'tangent_heading':
                    heading = (math.degrees(math.atan2(p[1], p[0]))+90+(1e-8 if j % 2 else -1e-8)) % 360
            reception = 1000. if k % 3 == 0 else 1500. if k % 3 == 1 else float(rng.uniform(1000, 1500))
            targets.append({'channel': int(c), 'position': p.tolist(), 'radius': reception, 'heading_deg': heading})
        cases.append({'id': f'{seed}_{k:03d}_{family}', 'family': family, 'targets': targets,
                      'error_seed': int(rng.integers(0, 2**31)),
                      'error_mode': ['hashed', 'fixed_extreme', 'spatial_smooth'][k % 3]})
    return cases
