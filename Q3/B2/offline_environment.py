import hashlib, math
import numpy as np
SEED = 20261001
FAMILIES = ['uniform_disk', 'outer_ring', 'two_clusters', 'narrow_sector', 'coverage_seams']

class OfflineEnvironment:
    def __init__(self, scenario):
        self.targets = {t['channel']:dict(t) for t in scenario['targets']}
        self.mode = scenario['error_mode']
        self.position = np.zeros(2)
        self.channel = 1
        self.clock = 0.
        self.cleared = set()
        self.clear_times = []
        self.movement = 0.
        self.measures = 0
        self.clear_attempts = 0

    def action(self, path, point=None, channel=None):
        response = {'accepted':True,'virtual_time_s':self.clock}
        if path == '/enter':
            return dict(response,remaining_real_duration_s=1200)
        if path == '/exit':
            return dict(response,exit_reason='user_exit')
        point = np.array(point)
        move = float(np.linalg.norm(point-self.position))
        self.movement += move
        self.clock += move/5
        self.position = point
        target = self.targets.get(channel)
        distance = float(np.linalg.norm(point-np.array(target['position']))) if target else float('inf')
        present = target is not None and channel not in self.cleared
        if path == '/measure':
            self.measures += 1
            self.clock += 5+int(channel != self.channel)
            self.channel = channel
            if not present or distance > target['radius']:
                response['measure_result'] = 'no_signal'
            elif distance <= 5:
                response['measure_result'] = 'near'
            else:
                response['measure_result'] = 'direction'
                delta = np.array(target['position'])-point
                true = math.degrees(math.atan2(delta[1],delta[0]))%360
                if self.mode == 'positive':
                    error = 1.
                elif self.mode == 'negative':
                    error = -1.
                elif self.mode == 'zero':
                    error = 0.
                else:
                    key = f"{channel}:{point[0]:.9f}:{point[1]:.9f}".encode()
                    value = int.from_bytes(hashlib.sha256(key).digest()[:8],'little')/(2**64-1)
                    error = 2*value-1
                response['svd_deg'] = round(true+error,2)%360
        elif path == '/clear':
            self.clear_attempts += 1
            success = present and distance <= 20
            self.clock += 5 if success else 3
            response['clear_result'] = 'success' if success else 'no_target_in_range'
            if success:
                self.cleared.add(channel)
                self.clear_times.append(self.clock)
        else:
            raise ValueError(path)
        response['virtual_time_s'] = self.clock
        return response

class StressEnvironment(OfflineEnvironment):
    def __init__(self,case):
        super().__init__(case);self.seed=case['error_seed'];self.error_mode=case['error_mode']

    def action(self,path,point=None,channel=None):
        response=super().action(path,point,channel)
        if path=='/measure' and response['measure_result']=='direction':
            delta=np.array(self.targets[channel]['position'])-np.array(point)
            angle=math.degrees(math.atan2(delta[1],delta[0]))
            if self.error_mode=='spatial_smooth':
                error=math.sin(point[0]/170+point[1]/230+channel*.71+self.seed*.00001)
            elif self.error_mode=='fixed_extreme':
                error=1. if channel%2 else -1.
            else:
                key=f'{self.seed}:{channel}:{float(point[0]).hex()}:{float(point[1]).hex()}'.encode()
                error=2*int.from_bytes(hashlib.sha256(key).digest()[:8],'big')/(2**64-1)-1
            response['svd_deg']=round(angle+error,2)%360
        return response

def make_cases():
    rng=np.random.default_rng(SEED);cases=[]
    for n in range(10,17):
        for family in FAMILIES:
            for rep in range(4):
                theta=rng.uniform(0,math.tau,n)
                if family=='uniform_disk':
                    radius=1800*np.sqrt(rng.random(n))
                    positions=np.column_stack((radius*np.cos(theta),radius*np.sin(theta)))
                elif family=='outer_ring':
                    radius=rng.uniform(1650,1800,n)
                    positions=np.column_stack((radius*np.cos(theta),radius*np.sin(theta)))
                elif family=='two_clusters':
                    phi=rng.uniform(0,math.tau)
                    centre=1100*np.array([math.cos(phi),math.sin(phi)])
                    positions=np.array([(-1 if i%2 else 1)*centre+rng.normal(0,180,2) for i in range(n)])
                elif family=='narrow_sector':
                    theta=rng.uniform(0,math.tau)+rng.uniform(-.13,.13,n)
                    radius=rng.uniform(100,1800,n)
                    positions=np.column_stack((radius*np.cos(theta),radius*np.sin(theta)))
                else:
                    theta=(rng.integers(0,7,n)+.5)*math.tau/7+rng.uniform(-.003,.003,n)
                    radius=rng.uniform(1795,1800,n)
                    positions=np.column_stack((radius*np.cos(theta),radius*np.sin(theta)))
                for point in positions:
                    point*=min(1.,1800/max(1.,np.linalg.norm(point)))
                channels=rng.choice(np.arange(1,21),n,replace=False)
                targets=[{'channel':int(c),'position':p.tolist(),
                          'radius':float(1000 if rep==0 else 1500 if rep==1 else rng.uniform(1000,1500))}
                         for c,p in zip(channels,positions)]
                cases.append({'id':f'n{n}_{family}_{rep}','family':family,'replicate':rep,
                              'error_seed':int(rng.integers(0,2**31)),
                              'error_mode':['hashed','fixed_extreme','spatial_smooth','hashed'][rep],
                              'targets':targets})
    return cases
