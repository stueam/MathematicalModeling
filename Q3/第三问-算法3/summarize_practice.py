"""Merge completed practice cases across preserved batches; never contacts UI/HTTP."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from run import dump, new_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', type=Path, nargs='+')
    args = parser.parse_args()
    rows, batches, configs = [], [], []
    for folder in args.folders:
        batch = json.loads((folder/'batch.json').read_text())
        config = json.loads((folder/'config.json').read_text())
        cases = json.loads((folder/'summary.json').read_text())
        included = {r['case_code'] for r in cases}
        for path in folder.glob('practice-*/summary.json'):
            recorded = json.loads(path.read_text())
            if recorded.get('case_code') not in included:
                raise ValueError(f'An unmerged round record exists: {path}; review failures before aggregation')
        if config.get('mode') != 'problem3_practice_only':
            raise ValueError('Only explicitly verified practice batches can be merged')
        configs.append(config)
        batches.append({'folder': str(folder), **batch})
        for row in cases:
            if not (row.get('mode') == 'problem3_practice_only' and row.get('exited') and
                    row.get('certified_complete') and not row.get('error')):
                raise ValueError('Unsuccessful case cannot be silently included as completed')
            if row.get('source_count') != row.get('cleared_count') or not row.get('source_count'):
                raise ValueError('Practice source total/clearance is not verified')
            rows.append({**row, 'source_batch': str(folder)})
    if len({r['case_code'] for r in rows}) != len(rows):
        raise ValueError('Duplicate case code: refusing to double count a resumed case')
    if any(c['policy_config'] != configs[0]['policy_config'] or c['code_sha256'] != configs[0]['code_sha256'] for c in configs):
        raise ValueError('Algorithm configuration or recorded source version differs')
    values = np.array([r['average_per_cleared_s'] for r in rows])
    runtimes = np.array([r['real_time_s'] for r in rows])
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.
    half = float(t.ppf(.975, len(values)-1)*std/np.sqrt(len(values))) if len(values) > 1 else 0.
    total_s = sum(r['virtual_time_s'] for r in rows)
    total_n = sum(r['source_count'] for r in rows)
    summary = {'mode': 'problem3_practice_only', 'rounds': len(rows), 'all_rounds_complete': True,
        'source_count': total_n, 'cleared_count': sum(r['cleared_count'] for r in rows),
        'clearance_fraction': 1., 'total_virtual_time_s': total_s,
        'pooled_time_per_source_s': total_s/total_n, 'mean_case_time_per_source_s': float(values.mean()),
        'median_case_time_per_source_s': float(np.median(values)), 'sample_std_s': std,
        'min_case_time_per_source_s': float(values.min()), 'max_case_time_per_source_s': float(values.max()),
        'p90_case_time_per_source_s': float(np.quantile(values, .9)),
        'rounds_under_200_s_per_source': int((values < 200).sum()),
        'approximate_t_interval_for_case_mean_s': [float(values.mean()-half), float(values.mean()+half)],
        'interval_assumption': 'Descriptive t interval assuming independent comparable cases; not an official-distribution guarantee.',
        'mean_client_real_time_s': float(runtimes.mean()), 'max_client_real_time_s': float(runtimes.max()),
        'mean_movement_m': float(np.mean([r['movement_m'] for r in rows])),
        'measure_count': sum(r['measure_count'] for r in rows),
        'clear_failures': sum(r['clear_failures'] for r in rows),
        'actions': sum(r['actions'] for r in rows),
        'max_cost_difference_s': max(r['max_cost_difference_s'] for r in rows)}
    out = new_output()
    dump(out/'config.json', {'inputs': list(map(str, args.folders)), 'policy_config': configs[0]['policy_config'],
                            'code_sha256': configs[0]['code_sha256']})
    dump(out/'source-batches.json', batches)
    dump(out/'cases.json', rows)
    dump(out/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
