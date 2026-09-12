"""Omnidirectional strategy; consumes action feedback only, never target truth."""
import math
import time
import numpy as np

EPS = math.radians(1.005)
NORMALS = np.column_stack((np.cos(np.arange(32)*math.tau/32), np.sin(np.arange(32)*math.tau/32)))


def clip(poly, n, b):
    if not len(poly):
        raise ArithmeticError('Empty position envelope')
    out = []
    for a, c in zip(poly, np.roll(poly, -1, axis=0)):
        fa, fc = float(a@n-b), float(c@n-b)
        if (fa <= 0) != (fc <= 0):
            out.append(a+(c-a)*fa/(fa-fc))
        if fc <= 0:
            out.append(c)
    result = np.asarray(out).reshape(-1, 2)
    if not len(result):
        raise ArithmeticError('Inconsistent observations; no invented source position')
    return result


def initial_envelope():
    poly = np.array([[-1801.,-1801.],[1801.,-1801.],[1801.,1801.],[-1801.,1801.]])
    for n in NORMALS:
        poly = clip(poly, n, 1800+1e-7)
    return poly


def observe(poly, position, bearing):
    a = math.radians(bearing)
    for n in [np.array([math.sin(a-EPS),-math.cos(a-EPS)]),
              np.array([-math.sin(a+EPS), math.cos(a+EPS)])]:
        poly = clip(poly, n, n@position+1e-7)
    for n in NORMALS:
        poly = clip(poly, n, n@position+1500+1e-7)
    return poly


def envelope_circle(poly):
    center = (poly.min(axis=0)+poly.max(axis=0))/2
    radius = float(np.linalg.norm(poly-center, axis=1).max())
    return center, radius


class Strategy:
    region_circle = staticmethod(envelope_circle)

    def select_local_measurement(self, channel, poly, center, radius, default):
        return default

    def __init__(self, action, method='main'):
        if method not in ['main','baseline']:
            raise ValueError('Unknown method')
        self.action, self.method = action, method
        self.position = np.zeros(2)
        self.receiver_channel = 1
        self.polys, self.measurements = {}, {}
        self.cleared, self.absent = set(), set()
        self.trace = []
        self.fallback_count = 0
        self.deadline = float('inf')

    def send(self, path, point=None, channel=None):
        if time.monotonic() > self.deadline:
            raise TimeoutError('Stop before available real-time budget expires')
        if len(self.trace) >= 20000:
            raise RuntimeError('Action limit reached')
        result = self.action(path, None if point is None else list(map(float,point)), channel)
        if result.get('accepted') is not True:
            raise RuntimeError('Simulator rejected action')
        if path == '/measure':
            self.receiver_channel = channel
        if point is not None:
            self.position = np.array(point, dtype=float)
        self.trace.append({'path':path,'position':None if point is None else self.position.tolist(),
                           'channel':channel,'response':result})
        return result

    def clear(self, point, channel, must_succeed=False):
        result = self.send('/clear', point, channel)
        if result['clear_result'] == 'success':
            self.cleared.add(channel)
            return True
        if must_succeed:
            raise ArithmeticError('Guaranteed clearing attempt failed; preserve evidence')
        return False

    def measure(self, point, channel):
        result = self.send('/measure', point, channel)
        state = result['measure_result']
        if state == 'near':
            self.clear(point, channel, must_succeed=True)
        elif state == 'direction':
            if channel not in self.polys:
                self.polys[channel] = initial_envelope()
                self.measurements[channel] = []
            self.polys[channel] = observe(self.polys[channel], np.asarray(point), result['svd_deg'])
            self.measurements[channel].append((np.array(point), result['svd_deg']))
        elif state != 'no_signal':
            raise ValueError('Unknown measurement state')
        return state

    def localize(self, channel):
        for step in range(6):
            poly = self.polys[channel]
            center, radius = self.region_circle(poly)
            if radius <= 19.8:
                self.clear(center, channel, must_succeed=True)
                return
            if len(self.measurements[channel]) == 1:
                point, bearing = self.measurements[channel][0]
                angle = math.radians(bearing)
                u = np.array([math.cos(angle),math.sin(angle)])
                n = np.array([-u[1],u[0]])
                choices = [point+798*u+sign*602*n for sign in [-1,1]]
                target = min(choices,key=lambda p:np.linalg.norm(p-self.position))
            else:
                # A lateral viewpoint avoids repeatedly measuring on the long axis.
                _, _, vh = np.linalg.svd(poly-center, full_matrices=False)
                normal = np.array([-vh[0,1],vh[0,0]])
                offset = min(150., max(25., radius*0.8))
                choices = [center+sign*offset*normal for sign in [-1,1]]
                target = min(choices,key=lambda p:np.linalg.norm(p-self.position))
                if np.linalg.norm(target-self.position) < 1:
                    target = choices[1] if np.allclose(target,choices[0]) else choices[0]
            state = self.measure(target,channel)
            if channel in self.cleared:
                return
            if state == 'no_signal':
                # Region remains valid. Finite covering search below still applies.
                break
        self.fallback_count += 1
        poly = self.polys[channel]
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        # Endpoints included; every bounding-box point within sqrt(2)*10 <20m.
        axes = [np.linspace(lo[k],hi[k],max(1,math.ceil((hi[k]-lo[k])/20))+1) for k in range(2)]
        candidates = [np.array([x,y]) for i,x in enumerate(axes[0])
                      for y in (axes[1] if i%2==0 else axes[1][::-1])]
        if len(candidates)>10000:
            raise RuntimeError('Cover search exceeds budget; not claiming completeness')
        for point in candidates:
            if self.clear(point,channel):
                return
        raise ArithmeticError('Cover search exhausted without clearing; inconsistent model')

    def run(self):
        entered = self.send('/enter')
        self.deadline = time.monotonic()+max(0,entered.get('remaining_real_duration_s',1200)-5)
        sites = [np.zeros(2)]+[1500*np.array([math.cos(k*math.pi/3),math.sin(k*math.pi/3)]) for k in range(6)]
        for point in sites:
            for channel in range(1,21):
                if channel not in self.cleared:
                    self.measure(point,channel)
        self.absent = set(range(1,21))-set(self.polys)-self.cleared
        while set(self.polys)-self.cleared:
            pending = set(self.polys)-self.cleared
            if self.method == 'baseline':
                channel = min(pending)
            else:
                channel = min(pending,key=lambda c:(np.linalg.norm(self.region_circle(self.polys[c])[0]-self.position),c))
            self.localize(channel)
        assert len(self.cleared|self.absent) == 20
        exited = self.send('/exit')
        return {'method':self.method,'cleared_channels':sorted(self.cleared),'absent_channels':sorted(self.absent),
                'virtual_time_s':exited['virtual_time_s'],'actions':len(self.trace),'cover_searches':self.fallback_count}
