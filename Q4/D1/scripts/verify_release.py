"""Verify portable Q4/D1 sources and replay local cases; no network access."""
import csv
import hashlib
import io
import json
import platform
import sys
import unittest
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import source_hashes, run_case
from benchmark_service import factory


def main():
    directory = ROOT/'results/Q4/experiments/round19_service_holdout'
    summary = json.loads((directory/'run_summary.json').read_text(encoding='utf-8'))
    original = summary['source_end_sha256']
    current = source_hashes()
    assert summary['source_start_sha256'] == original
    for name, digest in current.items():
        assert original[name] == digest, f'Historical source differs: {name}'
    omitted = sorted(set(original)-set(current))
    assert omitted == ['practice_adaptive.py'], omitted
    for name in ('round18_service_regression', 'round19_service_holdout'):
        folder = ROOT/'results/Q4/experiments'/name
        data = json.loads((folder/'run_summary.json').read_text(encoding='utf-8'))
        assert data['status'] == 'PASS'
        assert hashlib.sha256((folder/'metrics/scenarios.json').read_bytes()).hexdigest() == data['scenario_sha256']
        rows = list(csv.DictReader((folder/'tables/cases.csv').open(encoding='utf-8-sig')))
        assert len(rows) == 56
        assert all(r['status'] == 'PASS' and r['cleared'] == r['sources'] and r['false_absent'] == '0' and r['envelope_violations'] == '0' for r in rows)
    stream = io.StringIO()
    tested = unittest.TextTestRunner(stream=stream, verbosity=1).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    assert tested.wasSuccessful(), stream.getvalue()
    print(f'{tested.testsRun} tests passed; historical source identity passed', flush=True)
    cases = json.loads((directory/'metrics/scenarios.json').read_text(encoding='utf-8'))
    expected = {(r['case_id'], r['method']): r for r in csv.DictReader((directory/'tables/cases.csv').open(encoding='utf-8-sig'))}
    replays = []
    for i in (1, 5):
        for method in ('q4_compact', 'q4_service'):
            instances = []
            def capture(action, label):
                strategy = factory(action, label)
                instances.append(strategy)
                return strategy
            row, failure = run_case(cases[i], method, capture)
            assert failure is None, failure
            for key in ('service_attempts', 'service_accepts', 'clear_adjustments', 'local_leg_saving_m', 'exact_route_calls'):
                row[key] = getattr(instances[0], key, 0)
            for key, value in row.items():
                if key != 'runtime_s':
                    assert str(value) == expected[(row['case_id'], method)][key], (row['case_id'], method, key)
            replays.append({'case_id': row['case_id'], 'method': method, 'status': 'PASS', 'seconds_per_cleared': row['seconds_per_cleared']})
            print(f"Replay {row['case_id']} {method}: PASS", flush=True)
    output = dict(status='PASS', test_count=tested.testsRun, replays=replays,
                  preserved_source_sha256=current, omitted_historical_files=omitted,
                  official_tests_used=0, practice_tests_used=0, network_requests=0,
                  python=platform.python_version(), verified_at=datetime.now(timezone.utc).isoformat())
    target = ROOT/'local_runs/publication_check.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS: publication verification saved to local_runs/publication_check.json', flush=True)


if __name__ == '__main__':
    main()
