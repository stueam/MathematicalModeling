"""Read-only census of source counts from completed official Q3 practice UI logs.

No simulator is launched. Counts come from the post-practice total, NEVER the
controller's cleared_count or a local world generator. Cases are deduplicated
by case_code across copied batches. Incomplete cases with known totals remain
eligible; missing totals are separately disclosed.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parent
CASE = re.compile(r'[A-Z0-9]{4}(?:-[A-Z0-9]{4}){3}')
MODES = {'problem3_practice', 'problem3_practice_only'}
SKIP_DIRS = {'.git', '__pycache__', 'source_snapshot', 'worktrees',
             'final-source', '.venv', 'node_modules'}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def wilson(k, n):
    z = float(stats.norm.ppf(.975))
    p = k / n
    den = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0., middle - half), min(1., middle + half)]


def ended_ui_count(items, expected_code):
    names = [r.get('name', '') for r in items]
    titles = [re.sub(r'\s+', '', r.get('name', '')) for r in items
              if r.get('id') == 'test-run-title']
    assert titles and all(t == '问题3演练测试' for t in titles), 'Wrong UI mode'
    assert '测试已结束' in names, 'Not an ended case'
    assert {s for s in names if CASE.fullmatch(s)} == {expected_code}, 'UI case mismatch'
    begin = names.index('本次演练测试干扰源数量')
    count = next(int(s) for s in names[begin + 1:] if re.fullmatch(r'\d+', s))
    assert 10 <= count <= 16, 'Out-of-range source count'
    return count


def census(output):
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'batch.json', {'status': 'running', 'started_at': datetime.now().isoformat()})
    roots = [ROOT / f'第三问-算法{i}/results' for i in (1, 2, 3)]
    files = []
    for root in roots:
        if not root.exists():
            continue
        for parent, dirs, names in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith('.'))
            if 'summary.json' in names:
                files.append(Path(parent) / 'summary.json')
    # Freeze file list. Prefer shorter original paths over later copied batches.
    files.sort(key=lambda p: (len(p.parts), str(p)))
    cases = {}
    unknown = []
    conflicts = []
    unreadable = []
    hashes = {}
    matched = 0
    aggregate_only_codes = set()
    for path in files:
        try:
            raw = path.read_bytes()
            data = json.loads(raw)
        except (OSError, ValueError) as exc:
            unreadable.append({'path': str(path.relative_to(ROOT)), 'error': str(exc)})
            continue
        if isinstance(data, list):
            aggregate_only_codes.update(r['case_code'] for r in data if isinstance(r, dict)
                                        and r.get('mode') in MODES and CASE.fullmatch(str(r.get('case_code', ''))))
            continue
        if not isinstance(data, dict) or data.get('mode') not in MODES:
            continue
        code = str(data.get('case_code', ''))
        if not CASE.fullmatch(code):
            continue
        matched += 1
        rel = str(path.relative_to(ROOT))
        hashes[rel] = hashlib.sha256(raw).hexdigest()
        ui_path = path.parent / 'practice-after.json'
        if not ui_path.is_file():
            # A stopped batch may have completed /exit before saving its UI.
            # Preserve the original summary and accept separately archived UI
            # only after the same title/case/ended/total checks below.
            ui_path = path.parent / 'practice-after-stop.json'
        if not ui_path.is_file():
            unknown.append({'case_code': code, 'path': rel, 'reason': 'No ended UI evidence',
                            'summary_source_count': data.get('source_count')})
            continue
        try:
            ui_bytes = ui_path.read_bytes()
            count = ended_ui_count(json.loads(ui_bytes), code)
        except (ValueError, AssertionError, StopIteration, KeyError, TypeError) as exc:
            unknown.append({'case_code': code, 'path': rel, 'reason': str(exc),
                            'summary_source_count': data.get('source_count')})
            continue
        hashes[str(ui_path.relative_to(ROOT))] = hashlib.sha256(ui_bytes).hexdigest()
        if data.get('source_count') is not None and data['source_count'] != count:
            conflicts.append({'case_code': code, 'path': rel, 'summary': data['source_count'], 'UI': count})
            continue
        if code in cases:
            if cases[code]['source_count'] != count:
                conflicts.append({'case_code': code, 'path': rel, 'duplicate_count': count})
            cases[code]['duplicate_summary_paths'].append(rel)
            continue
        cases[code] = {'case_code': code, 'source_count': count, 'summary_path': rel,
                       'UI_evidence_path': str(ui_path.relative_to(ROOT)),
                       'batch': str(path.parent.parent.relative_to(ROOT)),
                       'policy': data.get('policy'), 'error': data.get('error'),
                       'cleared_count_for_completeness_check_only': data.get('cleared_count'),
                       'duplicate_summary_paths': []}
    if conflicts:
        dump(output / 'conflicts.json', conflicts)
        dump(output / 'batch.json', {'status': 'failed', 'reason': 'Conflicting source totals'})
        raise ValueError('Conflicting totals; see conflicts.json')
    rows = list(cases.values())
    assert rows, 'No verified official Q3 cases'
    values = np.array([r['source_count'] for r in rows])
    frequencies = Counter(map(int, values))
    n = len(rows)
    freq_rows = [{'source_count': k, 'cases': frequencies[k], 'frequency': frequencies[k] / n,
                  'individual_wilson_95': wilson(frequencies[k], n)} for k in range(10, 17)]
    goodness = stats.chisquare([frequencies[k] for k in range(10, 17)])
    mean = float(values.mean())
    mean_half = float(stats.t.ppf(.975, n - 1) * values.std(ddof=1) / math.sqrt(n))
    by_batch = defaultdict(list)
    for r in rows:
        by_batch[r['batch']].append(r['source_count'])
    batches = [{'batch': k, 'cases': len(v), 'mean_sources': float(np.mean(v)),
                'frequencies_10_to_16': [v.count(j) for j in range(10, 17)]}
               for k, v in sorted(by_batch.items())]
    unresolved = [r for r in unknown if r['case_code'] not in cases]
    summary = {
        'snapshot_at': datetime.now().isoformat(), 'official_Q3_practice_cases': n,
        'total_sources': int(values.sum()), 'mean_sources': mean,
        'mean_approx_t95': [mean - mean_half, mean + mean_half],
        'median_sources': float(np.median(values)),
        'sample_standard_deviation': float(values.std(ddof=1)),
        'frequencies': freq_rows,
        'uniform_10_to_16_test': {'statistic': float(goodness.statistic),
                                'degrees_of_freedom': 6, 'p_value': float(goodness.pvalue),
                                'expected_cases_each': n / 7},
        'data_checks': {'scanned_summary_files': len(files), 'matching_individual_records': matched,
                        'verified_duplicate_copies': sum(len(r['duplicate_summary_paths']) for r in rows),
                        'unresolved_unique_case_count': len({r['case_code'] for r in unresolved}),
                        'aggregate_codes_without_verified_individual_record': sorted(aggregate_only_codes - cases.keys()),
                        'count_conflicts': 0, 'unreadable_summary_files': len(unreadable),
                        'known_total_cases_with_errors_or_not_all_cleared': sum(
                            bool(r['error']) or r['source_count'] != r['cleared_count_for_completeness_check_only'] for r in rows)},
        'per_batch': batches,
        'interpretation_limits': [
            'Only archived official Q3 practice cases with ended-UI total evidence; no local maps or Q4.',
            'Case codes deduplicated; source_count never substituted by cleared_count.',
            'Intervals and chi-square reference rely on approximately independent cases from a stable distribution.',
            'A large p value does not prove the official RNG is uniform; a small p value does not identify its algorithm.',
            'Archived practice data may have missingness/selection bias; count frequencies do not determine spatial distribution.',
        ]}
    dump(output / 'summary.json', summary)
    dump(output / 'cases.json', rows)
    dump(output / 'missing_or_unverified.json', unresolved)
    dump(output / 'unreadable_files.json', unreadable)
    hashes[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump(output / 'input_and_code_sha256.json', hashes)
    dump(output / 'batch.json', {'status': 'completed', 'unique_cases': n})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' /
                        ('official-q3-counts-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
    args = parser.parse_args()
    print(json.dumps(census(args.output), ensure_ascii=False, indent=2))
    print('Results:', args.output)
