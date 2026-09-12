"""Paper Q2 numerical design; output CSV/JSON, with no plotting dependency."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from model_core import Design

def save(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=['symmetric','asymmetric'],default='symmetric')
    parser.add_argument('--R0',type=float,default=1200.)
    parser.add_argument('--evaluate',nargs=2,type=float,metavar=('X','Y'))
    parser.add_argument('--step',type=float,default=60.)
    parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    if not 1000<=a.R0<=1500 or a.step<=0:parser.error('Require 1000 <= R0 <= 1500 and positive step')
    options=dict(p1=(0.,0.),theta=0.) if a.scenario=='symmetric' else dict(p1=(1200.,400.),theta=65.)
    m=Design(**options,reception=a.R0,nr=80,na=16,arc=128)
    if a.evaluate:
        if np.linalg.norm(a.evaluate)>1800:parser.error('Candidate outside target disk')
        print(json.dumps(m.evaluate(a.evaluate,.125),indent=2));return
    out=a.output or Path(__file__).parent/'results'/f'{a.scenario}_R{a.R0:g}'
    out.mkdir(parents=True,exist_ok=True)
    axis=np.arange(-1800,1800+a.step/2,a.step)
    candidates=[(float(x),float(y)) for y in axis for x in axis if x*x+y*y<=1800**2]
    coarse=[]
    for i,q in enumerate(candidates,1):
        coarse.append(m.evaluate(q,.125))
        if i%100==0:print(f'{i}/{len(candidates)}',flush=True)
    best=min(coarse,key=lambda r:r['J'])
    centers=[(best['x'],best['y'])]
    if a.scenario=='symmetric':centers.append((best['x'],-best['y']))
    points=sorted({(float(x),float(y)) for cx,cy in centers
                  for x in np.arange(cx-90,cx+91,15) for y in np.arange(cy-90,cy+91,15)
                  if x*x+y*y<=1800**2})
    fine=[m.evaluate(q,.125) for q in points]
    save(out/'grid.csv',coarse);save(out/'refined.csv',fine)
    result=dict(scenario=a.scenario,R0=a.R0,**options,nr=80,na=16,arc=128,
                angle_bin_deg=.125,step_m=a.step,local_step_m=15,
                objective='expected full-region minimum enclosing radius in metres',
                initial_radius_m=m.initial_radius,best=min(coarse+fine,key=lambda r:r['J']))
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
