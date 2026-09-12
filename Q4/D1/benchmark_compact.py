"""Offline paired v3 / v2 evaluation. This file has no live simulator access."""
import argparse
import csv
import json
import hashlib
import platform
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from benchmark import run_case, source_hashes, summarize
from adaptive_strategy import AdaptiveB2
from compact_strategy import CompactB2
from offline import make_cases

def factory(action, method):
    return AdaptiveB2(action) if method == 'q4_adaptive' else CompactB2(action)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=2026091104)
    p.add_argument('--count',type=int,default=14)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenarios',type=Path)
    args=p.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for folder in ('metrics','tables'): (out/folder).mkdir()
    cases=json.loads(args.scenarios.read_text(encoding='utf-8')) if args.scenarios else make_cases(args.seed,args.count)
    source_start=source_hashes()
    (out/'metrics/scenarios.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    rows=[]
    for case in cases:
        for method in ('q4_adaptive','q4_compact'):
            row,failure=run_case(case,method,factory)
            rows.append(row)
            if failure:
                (out/'logs').mkdir(exist_ok=True)
                (out/'logs'/f"{case['id']}_{method}.json").write_text(json.dumps(failure,indent=2),encoding='utf-8')
            with (out/'tables/cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(rows)
            print(f"{case['id']} {method} {row['status']} {row['seconds_per_cleared']:.2f}s/source reused={row['reused_scans']} runtime={row['runtime_s']:.2f}s",flush=True)
    def aggregate(selected):
        result,_=summarize(selected,['q4_adaptive','q4_compact'])
        a={r['case_id']:r for r in selected if r['method']=='q4_adaptive'}
        b={r['case_id']:r for r in selected if r['method']=='q4_compact'}
        valid=all(r['status']=='PASS' for r in selected)
        paired={'faster':sum(b[k]['virtual_time_s']<a[k]['virtual_time_s']-1e-6 for k in a),
                'slower':sum(b[k]['virtual_time_s']>a[k]['virtual_time_s']+1e-6 for k in a),
                'improvement_percent':100*(1-result['q4_compact']['mean_seconds_per_cleared']/result['q4_adaptive']['mean_seconds_per_cleared']) if valid else None}
        return {'methods':result,'paired':paired}
    summary=dict(question='Q4',round=out.name,decision_id='q4_v3_travel_optimization',
                 approved_methods=['q4_compact','q4_adaptive'],roles={'q4_compact':'main_candidate','q4_adaptive':'usable_baseline'},
                 status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
                 seed=args.seed if not args.scenarios else None,source_start_sha256=source_start,
                 source_end_sha256=source_hashes(),inputs=['metrics/scenarios.json'],outputs=['tables/cases.csv'],
                 scenario_sha256=hashlib.sha256((out/'metrics/scenarios.json').read_bytes()).hexdigest(),
                 metrics={'all':aggregate(rows),'mixed':aggregate([r for r in rows if 0<r['directional_sources']<r['sources']])},
                 environment={'python':platform.python_version(),'numpy':np.__version__},
                 network_requests=0,formal_tests_used=0,warnings=['Synthetic cases, not official simulator distribution'],
                 completed_at=datetime.now(timezone.utc).isoformat())
    (out/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary['metrics']['mixed'],ensure_ascii=False),flush=True)

if __name__=='__main__':main()
