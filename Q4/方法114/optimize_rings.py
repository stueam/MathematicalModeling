"""Search public-domain ring layouts; full exclusion is the acceptance gate."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import math
import multiprocessing

import numpy as np
import shapely

from q4.core import DOMAIN
from q4.coverage import certifies
from q4.routing import open_route
from run import ROOT, dump, manifest


def evaluate(parameters):
    inner_n,outer_n,inner_r,outer_extra,phase=parameters
    outer_r=1805/math.cos(math.pi/outer_n)+outer_extra
    points=[(0.,0.)]
    points += [(inner_r*math.cos((k+phase)*math.tau/inner_n),
                inner_r*math.sin((k+phase)*math.tau/inner_n)) for k in range(inner_n)]
    points += [(outer_r*math.cos(k*math.tau/outer_n),outer_r*math.sin(k*math.tau/outer_n))
               for k in range(outer_n)]
    samples=np.asarray([(x,y) for x in range(-1700,1701,200) for y in range(-1700,1701,200)
                        if x*x+y*y<=1800**2]+list(map(tuple,shapely.get_coordinates(DOMAIN))))
    delta=np.asarray(points)[None]-samples[:,None]
    angles=np.where(np.linalg.norm(delta,axis=2)<998.7,np.arctan2(delta[:,:,1],delta[:,:,0])%math.tau,np.inf)
    angles=np.sort(angles,axis=1)
    counts=np.sum(np.isfinite(angles),axis=1)
    if np.any(counts<3):
        return None
    # Reject obvious angular holes only; this screening never accepts a plan.
    for row,n in zip(angles,counts):
        if np.max(np.diff(np.r_[row[:n],row[0]+math.tau]))>math.pi+1e-10:
            return None
    if not certifies(DOMAIN,tuple(points)):
        return None
    route,meters=open_route((0.,0.),dict(enumerate(points[1:])),exact_limit=16)
    return {'parameters':parameters,'stations':len(points),'points':points,'route_m':meters,
            'route':route,'full_certificate_valid':True,'geometry_only_proxy_s':meters/5+42*(len(points)-1)}


def main():
    out=ROOT/'results'/datetime.now().strftime('rings-%Y%m%d-%H%M%S-%f');out.mkdir()
    parameters=[(ni,no,ri,extra,phase) for ni in range(4,10) for no in (12,14,16,18,20,24,28)
                for ri in (800.,875.,925.,975.,997.) for extra in (0.,30.,100.) for phase in (0.,.5)]
    dump(out/'config.json',{'parameters':parameters,'code_sha256':manifest(),'workers':16,
                          'uses_true_worlds':False,'proxy_scan_cost_per_station_s':42})
    print(out,flush=True)
    accepted=[]
    with ProcessPoolExecutor(16,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(evaluate,p) for p in parameters]
        for i,future in enumerate(as_completed(futures),1):
            row=future.result()
            if row is not None:
                accepted.append(row)
            if i%250==0:
                print('reviewed',i,'accepted',len(accepted),flush=True)
    accepted.sort(key=lambda x:x['geometry_only_proxy_s'])
    dump(out/'accepted.json',accepted)
    dump(out/'summary.json',{'reviewed':len(parameters),'accepted':len(accepted),'best':accepted[:10],
                            'not_a_global_optimum':True,'policy_uses_layout':False})
    print([{k:v for k,v in row.items() if k not in ('points','route')} for row in accepted[:10]],flush=True)


if __name__=='__main__':
    main()
