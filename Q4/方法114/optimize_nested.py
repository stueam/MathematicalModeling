"""Three-ring public-domain search; sampled screening, full certificate gate."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import math
import multiprocessing

import numpy as np
import shapely

from q4.core import DOMAIN
from q4.coverage import certifies
from q4.routing import open_route
from run import ROOT,dump,manifest


def evaluate(params):
    ni,nm,no,ri,rm,ro,pi,pm=params
    points=[(0.,0.)]
    for n,r,phase in ((ni,ri,pi),(nm,rm,pm),(no,ro,0.)):
        points.extend((r*math.cos((k+phase)*math.tau/n),r*math.sin((k+phase)*math.tau/n)) for k in range(n))
    samples=np.asarray([(x,y) for x in range(-1700,1701,200) for y in range(-1700,1701,200)
                        if x*x+y*y<=1800**2]+list(map(tuple,shapely.get_coordinates(DOMAIN))))
    delta=np.asarray(points)[None]-samples[:,None]
    angles=np.where(np.linalg.norm(delta,axis=2)<998.7,np.arctan2(delta[:,:,1],delta[:,:,0])%math.tau,np.inf)
    angles.sort(axis=1);counts=np.isfinite(angles).sum(axis=1)
    if np.any(counts<3):
        return None
    for row,n in zip(angles,counts):
        if np.max(np.diff(np.r_[row[:n],row[0]+math.tau]))>math.pi+1e-10:
            return None
    if not certifies(DOMAIN,points):
        return None
    order,meters=open_route((0.,0.),dict(enumerate(points[1:])),exact_limit=16)
    return {'parameters':params,'points':points,'stations':len(points),'route':order,'route_m':meters,
            'full_certificate_valid':True,'geometry_only_proxy_s':meters/5+42*(len(points)-1)}


def main():
    out=ROOT/'results'/datetime.now().strftime('nested-%Y%m%d-%H%M%S-%f');out.mkdir()
    params=[(ni,nm,no,ri,rm,ro,pi,pm) for ni in (3,4,5) for nm in (7,8,9) for no in (7,8,9,10)
            for ri in (400.,550.,700.) for rm in (1150.,1300.,1450.) for ro in (2000.,2100.,2200.)
            for pi in (0.,.5) for pm in (0.,.5)]
    dump(out/'config.json',{'code_sha256':manifest(),'parameters':params,'workers':16,
                          'reads_source_worlds':False,'policy_uses_new_layout':False})
    print(out,flush=True);accepted=[]
    with ProcessPoolExecutor(16,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(evaluate,p) for p in params]
        for i,future in enumerate(as_completed(futures),1):
            row=future.result()
            if row:accepted.append(row)
            if i%500==0:print(i,'reviewed',len(accepted),'valid',flush=True)
    accepted.sort(key=lambda r:r['geometry_only_proxy_s'])
    dump(out/'accepted.json',accepted)
    dump(out/'summary.json',{'reviewed':len(params),'accepted':len(accepted),'best':accepted[:10],
                            'global_optimum_proved':False})
    print([{k:v for k,v in x.items() if k not in ('points','route')} for x in accepted[:10]],flush=True)


if __name__=='__main__':main()
