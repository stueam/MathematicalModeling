"""Paper Q2 numerical design; output CSV/JSON, with no plotting dependency."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from model_core import Design


def save(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['symmetric', 'asymmetric'], default='symmetric')
    parser.add_argument('--R0', type=float, default=1200.0)
    parser.add_argument('--evaluate', nargs=2, type=float, metavar=('X', 'Y'))
    parser.add_argument('--step', type=float, default=60.0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not 1000 <= args.R0 <= 1500 or not 0 < args.step <= 1800:
        parser.error('Require 1000 <= R0 <= 1500 and 0 < step <= 1800')
    if args.evaluate is not None and (
        not all(math.isfinite(v) for v in args.evaluate) or np.linalg.norm(args.evaluate) > 1800
    ):
        parser.error('Candidate must be a finite point inside the target disk')

    options = (
        dict(p1=(0.0, 0.0), theta=0.0)
        if args.scenario == 'symmetric'
        else dict(p1=(1200.0, 400.0), theta=65.0)
    )
    model = Design(**options, reception=args.R0, nr=80, na=16, arc=128)
    if args.evaluate is not None:
        print(json.dumps(model.evaluate(args.evaluate, 0.125), indent=2, allow_nan=False))
        return

    output = args.output or Path(__file__).resolve().parent / 'results' / f'{args.scenario}_R{args.R0:g}'
    output.mkdir(parents=True, exist_ok=True)
    axis = np.arange(-1800, 1800 + args.step / 2, args.step)
    candidates = [(float(x), float(y)) for y in axis for x in axis if x * x + y * y <= 1800**2]
    coarse = []
    for index, point in enumerate(candidates, 1):
        coarse.append(model.evaluate(point, 0.125))
        if index % 100 == 0:
            print(f'{index}/{len(candidates)}', flush=True)
    best = min(coarse, key=lambda row: row['J'])
    centers = [(best['x'], best['y'])]
    if args.scenario == 'symmetric':
        centers.append((best['x'], -best['y']))
    points = sorted(
        {
            (float(x), float(y))
            for cx, cy in centers
            for x in np.arange(cx - 90, cx + 91, 15)
            for y in np.arange(cy - 90, cy + 91, 15)
            if x * x + y * y <= 1800**2
        }
    )
    fine = [model.evaluate(point, 0.125) for point in points]
    save(output / 'grid.csv', coarse)
    save(output / 'refined.csv', fine)
    result = dict(
        scenario=args.scenario,
        R0=args.R0,
        **options,
        nr=80,
        na=16,
        arc=128,
        angle_bin_deg=0.125,
        step_m=args.step,
        local_step_m=15,
        objective='expected full-region minimum enclosing radius in metres',
        initial_radius_m=model.initial_radius,
        area_partition_relative_error=model.area_partition_relative_error,
        best=min(coarse + fine, key=lambda row: row['J']),
    )
    text = json.dumps(result, indent=2, allow_nan=False)
    (output / 'summary.json').write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
