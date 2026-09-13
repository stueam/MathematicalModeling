"""Check packaged evidence and byte integrity without accessing a simulator."""

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, record in manifest['files'].items():
        relative = Path(name)
        check(not relative.is_absolute() and '..' not in relative.parts, f'Unsafe path: {name}')
        path = ROOT / relative
        check(path.is_file(), f'Missing file: {name}')
        raw = path.read_bytes()
        check(len(raw) == record['bytes'], f'Length changed: {name}')
        check(hashlib.sha256(raw).hexdigest() == record['sha256'], f'Hash changed: {name}')
    expected_logs = {name for name in manifest['files'] if name.endswith('.jlog')}
    actual_logs = {p.relative_to(ROOT).as_posix() for p in (ROOT / '正式实验日志').rglob('*.jlog')}
    check(expected_logs == actual_logs and len(actual_logs) == 6, 'Expected exactly six formal logs')
    for question in ('3', '4'):
        check(len(list((ROOT / '正式实验日志' / question).glob('*.jlog'))) == 3, 'Missing formal log')
    data = ROOT / 'essay/data/q2_strategy_heatmaps'
    summary = json.loads((data / 'summary.json').read_text(encoding='utf-8'))
    grid_report = {}
    for scenario in ('symmetric', 'asymmetric'):
        with (data / f'{scenario}_grid.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        check(len(rows) == 2821, f'{scenario}: wrong grid size')
        check(all(math.isfinite(float(r['J'])) and float(r['J']) >= 0 for r in rows), 'Invalid objective')
        best = summary['scenarios'][scenario]['best']
        selected = [r for r in rows if float(r['x']) == best['x'] and float(r['y']) == best['y']]
        # Refined optima need not belong to the frozen coarse grid.
        if selected:
            check(math.isclose(float(selected[0]['J']), best['J'], abs_tol=1e-6), 'Selected point mismatch')
        error = max(abs(float(r['probability_sum']) - 1) for r in rows)
        check(error < 1e-10, 'Feedback probabilities do not sum to one')
        grid_report[scenario] = {'coarse_nodes': len(rows), 'max_probability_error': error}
    data = ROOT / 'essay/data/q3_execution_case'
    summary = json.loads((data / 'summary.json').read_text(encoding='utf-8'))
    with (data / 'actions.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    check(len(rows) == summary['steps'] == 121, 'Frozen Q3 action count differs')
    check(sum(r['result'] == 'success' for r in rows) == summary['cleared_count'] == 15, 'Q3 clear count differs')
    check(math.isclose(sum(summary['costs'].values()), summary['virtual_time_s'], abs_tol=1e-6), 'Q3 costs differ')
    check(math.isclose(float(rows[-1]['virtual_time_s']), summary['virtual_time_s'], abs_tol=1e-6), 'Q3 final time differs')
    check(all(v in ('cleared', 'absent_certified') for v in summary['channel_status'].values()), 'Q3 incomplete')
    subprocess.run([sys.executable, str(ROOT / 'evidence/q3_history/verify.py')], check=True)
    print(json.dumps({'status': 'passed', 'files': len(manifest['files']), 'formal_logs': 6,
                      'q2_grids': grid_report, 'q3_frozen_actions': len(rows),
                      'official_connection': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
