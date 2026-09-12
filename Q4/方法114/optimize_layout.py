"""Offline public-domain station-set search with exact final coverage checks.

No worlds, source coordinates or HTTP are read. Sampled coverage constraints
propose layouts; ONLY the full conservative polygon certificate accepts them.
"""
from datetime import datetime
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack
import shapely
from shapely.geometry import Point

from q4.core import DOMAIN
from q4.coverage import certifies, full_exclusion
from q4.routing import open_route
from q4.sector import ring_stations
from run import ROOT, dump, manifest


def constraints(points, witnesses):
    rows = []
    for x, heading in witnesses:
        delta = points-x
        normal = np.array([math.cos(heading), math.sin(heading)])
        # 2 m range margin stays inside the current inscribed 1000 m disks.
        rows.append((np.linalg.norm(delta, axis=1) < 998.) & (delta @ normal > .001))
    return csr_matrix(np.asarray(rows, dtype=float))


def main():
    out = ROOT/'results'/datetime.now().strftime('layout-%Y%m%d-%H%M%S-%f')
    out.mkdir()
    initial = ring_stations()
    pool = list(initial)
    for radius in (600., 800., 975., 1200., 1450., 1855., 2000., 2200.):
        pool += [(radius*math.cos(i*math.tau/48), radius*math.sin(i*math.tau/48)) for i in range(48)]
    pool = list(dict.fromkeys(tuple(map(float, p)) for p in pool))
    points = np.asarray(pool)
    samples = [(float(x), float(y)) for x in range(-1750,1751,250)
               for y in range(-1750,1751,250) if DOMAIN.covers(Point(x,y))]
    samples += list(map(tuple, shapely.get_coordinates(DOMAIN)))
    witnesses = [(np.asarray(x), k*math.tau/24) for x in samples for k in range(24)]
    matrix = constraints(points, witnesses)
    bounds_low = np.zeros(len(pool)); bounds_low[0] = 1.
    cost = 1+np.linalg.norm(points,axis=1)*.05/2200
    dump(out/'config.json', {'code_sha256': manifest(), 'candidate_points': pool,
                            'initial_samples': samples, 'angles':24, 'distance_margin_m':2,
                            'max_iterations':12, 'time_limit_per_solve_s':20})
    print(out,flush=True)
    attempts=[]
    for iteration in range(12):
        started=time.monotonic()
        result=milp(cost,integrality=np.ones(len(pool)),bounds=Bounds(bounds_low,1.),
                    constraints=LinearConstraint(matrix,1.,np.inf),
                    options={'time_limit':20.,'mip_rel_gap':.005})
        if result.x is None:
            attempts.append({'iteration':iteration,'error':str(result.message)})
            break
        selected=np.flatnonzero(result.x>.5)
        chosen=tuple(sorted(pool[i] for i in selected))
        missing=DOMAIN.difference(full_exclusion(chosen))
        valid=certifies(DOMAIN,chosen)
        _,route_m=open_route((0.,0.),dict(enumerate(chosen)),exact_limit=0)
        record={'iteration':iteration,'selected_points':chosen,'stations':len(chosen),
                'route_m':route_m,'full_certificate_valid':valid,'uncovered_area':missing.area,
                'solve_status':int(result.status),'solve_message':str(result.message),
                'mip_gap':float(result.mip_gap),'solve_s':time.monotonic()-started,
                'constraints':matrix.shape[0]}
        attempts.append(record); dump(out/'attempts.json',attempts)
        print({k:v for k,v in record.items() if k!='selected_points'},flush=True)
        if valid:
            dump(out/'layout.json',record)
            break
        geometry=list(missing.geoms) if hasattr(missing,'geoms') else [missing]
        geometry=sorted(geometry,key=lambda g:-g.area)[:60]
        extra=[]
        for piece in geometry:
            samples2=[piece.representative_point().coords[0]]
            vertices=shapely.get_coordinates(piece)
            samples2 += list(map(tuple,vertices[::max(1,len(vertices)//4)]))
            for x in samples2:
                x=np.asarray(x)
                delta=points[selected]-x
                delta=delta[np.linalg.norm(delta,axis=1)<998.]
                if len(delta):
                    angles=np.sort(np.arctan2(delta[:,1],delta[:,0]) % math.tau)
                    gaps=np.diff(np.r_[angles,angles[0]+math.tau])
                    i=int(np.argmax(gaps)); heading=angles[i]+gaps[i]/2
                    extra.append((x,heading))
                extra += [(x,k*math.tau/24) for k in range(24)]
        matrix=vstack((matrix,constraints(points,extra)),format='csr')
    dump(out/'attempts.json',attempts)
    dump(out/'summary.json',{'full_certificate_found':any(a.get('full_certificate_valid') for a in attempts),
                            'attempts':len(attempts),'policy_uses_layout':False})


if __name__=='__main__':
    main()
