"""Interleave source clearing and certified discovery coverage. No truth access."""
import math
import time
import numpy as np
from q3_strategy import Strategy, envelope_circle


class FastStrategy(Strategy):
    def __init__(self, action):
        super().__init__(action,'main')
        # Closed tiles intersecting target disk; a tile is removed only if its
        # entire square is within a scanned station's guaranteed reception disk.
        h=150.
        axis=np.arange(-1800+h/2,1800,h)
        self.cells=np.array([[x,y] for x in axis for y in axis
                            if max(abs(x)-h/2,0)**2+max(abs(y)-h/2,0)**2 <= 1800**2])
        self.cell_radius=h/math.sqrt(2)
        self.covered=np.zeros(len(self.cells),dtype=bool)
        self.scan_points=[]

    def scan_here(self):
        point=self.position.copy()
        if any(np.linalg.norm(point-old)<0.1 for old in self.scan_points):
            return
        # Every still-unknown channel shares this coverage history.
        for channel in range(1,21):
            if channel in self.cleared or channel in self.absent:
                continue
            if channel not in self.polys:
                self.measure(point,channel)
            elif envelope_circle(self.polys[channel])[1]>19.8:
                if not any(np.linalg.norm(point-p)<1 for p,_ in self.measurements[channel]):
                    self.measure(point,channel)
        self.scan_points.append(point)
        self.covered |= np.linalg.norm(self.cells-point,axis=1)+self.cell_radius <= 1000-1e-7
        if self.covered.all():
            self.absent=set(range(1,21))-set(self.polys)-self.cleared

    def next_scan(self):
        missing=self.cells[~self.covered]
        # Finite set includes every cell centre, hence always permits progress.
        distances=np.linalg.norm(missing[:,None,:]-missing[None,:,:],axis=2)
        counts=np.sum(distances+self.cell_radius<=1000-1e-7,axis=1)
        travel=np.linalg.norm(missing-self.position,axis=1)
        scores=counts/(travel+300)
        return missing[int(np.argmax(scores))]

    def localize(self,channel):
        # A single source is anywhere along a <=1500m ray segment. Moving
        # near its midpoint with a modest lateral offset avoids a 1000m dogleg.
        if len(self.measurements[channel])==1:
            center,radius=envelope_circle(self.polys[channel])
            point,bearing=self.measurements[channel][0]
            angle=math.radians(bearing)
            normal=np.array([-math.sin(angle),math.cos(angle)])
            choices=[center+sign*100*normal for sign in [-1,1]]
            target=min(choices,key=lambda p:np.linalg.norm(p-self.position))
            if np.linalg.norm(self.polys[channel]-target,axis=1).max()<=999.:
                self.measure(target,channel)
                if channel in self.cleared:
                    return
        super().localize(channel)

    def run(self):
        entered=self.send('/enter')
        self.deadline=time.monotonic()+max(0,entered.get('remaining_real_duration_s',1200)-5)
        self.scan_here()
        while len(self.cleared|self.absent)<20:
            pending=set(self.polys)-self.cleared
            if pending:
                channel=min(pending,key=lambda c:(np.linalg.norm(envelope_circle(self.polys[c])[0]-self.position),c))
                self.localize(channel)
                self.scan_here()
            else:
                point=self.next_scan()
                # Use first unresolved channel to move, then scan other channels.
                channel=min(set(range(1,21))-self.cleared-self.absent)
                self.measure(point,channel)
                self.scan_here()
        exited=self.send('/exit')
        return {'method':'fast_interleaved','cleared_channels':sorted(self.cleared),'absent_channels':sorted(self.absent),
                'virtual_time_s':exited['virtual_time_s'],'actions':len(self.trace),'cover_searches':self.fallback_count,
                'coverage_cells':len(self.cells),'covered_cells':int(self.covered.sum()),'scanning_stations':len(self.scan_points)}
