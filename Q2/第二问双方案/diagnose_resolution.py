"""固定已选 S2 和全部实际反馈，仅加密贝叶斯后验，诊断覆盖率损失。"""
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
from benchmark import ROOT, infer, prepare_observer, score, stats
from models import Bayesian
from runner import write_csv


def run(output):
    plan = json.loads((output/'plans/boundary_bayes_0.json').read_text())
    s = np.asarray(plan['point'])
    with (output/'trials.csv').open(newline='') as f:
        samples = [r for r in csv.DictReader(f) if r['view']=='boundary' and r['method']=='bayes']
    report = []
    for nr,na in [(256,16),(512,16),(1024,32)]:
        start = time.perf_counter()
        c = {**plan['config'],'radial_bins':nr,'angular_bins':na}
        model = Bayesian(c)
        observer = prepare_observer(model,s)
        scored = []
        for trial in samples:
            truth = np.array([float(trial['gx']),float(trial['gy'])])
            measured = math.radians(float(trial['measured_deg'])) if trial['measured_deg'] else 0
            region = infer(model,s,trial['observation'],measured,observer)
            scored.append(score(region,truth))
        result = dict(radial_bins=nr,angular_bins=na,nonempty_cells=len(model.xy),
                      x2=float(s[0]),y2=float(s[1]),**stats(scored,''),
                      update_seconds=time.perf_counter()-start)
        report.append(result)
        print(result,flush=True)
    write_csv(output/'boundary_resolution.csv',report)


if __name__ == '__main__':
    run(ROOT/'results/benchmark')
