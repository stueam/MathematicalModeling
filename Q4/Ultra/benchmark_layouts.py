"""Paired local experiment; only initial station coordinates change."""
from pathlib import Path
import sys, json, hashlib
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'results'/'paired'
import run as baseline
from q4.policy import Config
from q4 import sector
from q4.core import DOMAIN
from q4.coverage import full_exclusion, certifies
from check_s21_certificate import verify
ORIGINAL_RING = sector.ring22_stations

def s21_stations():
    cert = ROOT/'s21_certificate.json'
    verify(cert)
    points = tuple(tuple(map(float,p)) for p in json.loads(cert.read_text())['info']['points'])
    # Same conservative polygons, without lexicographic reordering. No area tolerance.
    if not DOMAIN.difference(full_exclusion(points)).is_empty:
        raise ValueError('S21 fails model-domain initialization coverage')
    return points

def execute(job):
    sector.ring_stations = s21_stations if job['policy']=='s21' else ORIGINAL_RING
    original_factory = baseline.make_policy
    def factory(mode, config, **options):
        p = original_factory('probes', config, **options)
        p.implementation += '_'+mode
        return p
    baseline.make_policy = factory
    try:
        return baseline.run_case(job, OUTPUT/'runs')
    finally:
        baseline.make_policy = original_factory
        sector.ring_stations = ORIGINAL_RING

def main():
    (OUTPUT/'runs').mkdir(parents=True, exist_ok=True)
    configs = [(800+i,'uniform','iid',None,None,None,'random') for i in range(16)]
    for i,scene in enumerate(('uniform','boundary','outward','tangent','cluster','near','backside')):
        configs += [(9100+2*i,scene,'extreme',10,1000.,10,'stress'),
                    (9101+2*i,scene,'correlated',16,1500.,16,'stress')]
    jobs=[dict(seed=s,scenario=c,error_mode=e,n=n,radius=r,directional_count=d,
               group=g,policy=p,config=asdict(Config()),real_limit=1200.,max_steps=6000)
          for s,c,e,n,r,d,g in configs for p in ('s22','s21')]
    certificate=verify(ROOT/'s21_certificate.json')
    points=s21_stations()
    baseline.dump(OUTPUT/'manifest.json',dict(jobs=jobs,code_sha256=baseline.manifest(),
        s21_certificate=certificate,s21_model_domain_passed=True,
        sorted_checker_passed=certifies(DOMAIN,points),
        initialization_note='Exact empty residual with original station ordering; runtime unchanged',
        external_commit='b4af97c4cc6fbec14fcb4dfb76acd56d4753f87c'))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(execute,j) for j in jobs]
        for f in as_completed(futures):
            row=f.result(); rows.append(row)
            baseline.dump(OUTPUT/'summary.json',rows)
            print(f"{len(rows)}/{len(jobs)} {row['case_id']} {row['cleared_count']}/{row['source_count']} {row['virtual_time_s']:.2f}s {row['stop_reason']}",flush=True)
    baseline.dump(OUTPUT/'aggregate.json',baseline.aggregate(rows))

if __name__=='__main__':
    main()
