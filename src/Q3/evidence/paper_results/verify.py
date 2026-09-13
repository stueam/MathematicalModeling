"""Recheck archived numerical evidence; optionally compare a supplied paper."""

import argparse
from pathlib import Path
import json
import math


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, help='Optional essay.tex to compare against')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    summary = json.loads((here / 'summary.json').read_text(encoding='utf-8'))
    paper = args.paper.read_text(encoding='utf-8') if args.paper else None

    def close(a, b):
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-5), (a, b)

    for name, count in [('random', 40), ('stress', 24)]:
        rows = json.loads((here / (name + '-records.json')).read_text(encoding='utf-8'))
        assert len(rows) == count
        assert (
            len({(r['seed'], r['scenario'], r['error_mode'], r['n_input'], r['radius_input']) for r in rows})
            == count
        )
        assert all(
            r['policy'] == 'bayes-fast'
            and r['all_cleared']
            and r['certified_complete']
            and r['error'] is None
            and r['source_count'] == r['cleared_count']
            for r in rows
        )
        for r in rows:
            close(sum(r['costs'].values()), r['virtual_time_s'])
            close(r['movement_m'] / 5, r['costs']['move_s'])
            close(r['costs']['clear_success_s'], 5 * r['cleared_count'])
        n = sum(r['source_count'] for r in rows)
        assert n == summary[name]['sources']
        for metric, key in [('move_s_per_source', 'move_s'), ('total_s_per_source', None)]:
            value = sum(r['costs'][key] if key else r['virtual_time_s'] for r in rows) / n
            close(value, summary[name][metric])
            if paper is not None:
                assert f'{value:.2f}' in paper
    oracle = json.loads((here / 'oracle-summary.json').read_text(encoding='utf-8'))
    value = oracle['pooled_T_div_total_N_s_per_source']['estimate']
    close(value, oracle['mean_pure_move_s_per_case']['mean'] / oracle['sample_mean_N'])
    close(value, summary['oracle']['move_s_per_source'])
    close(value + 5, summary['oracle']['move_and_clear_s_per_source'])
    delta = summary['random']['total_s_per_source'] - value - 5
    close(delta, summary['random_minus_oracle_s_per_source'])
    if paper is not None:
        assert all(f'{v:.2f}' in paper for v in [value, value + 5])
    assert sum(summary[n]['sources'] for n in ['random', 'stress']) == 857
    print(
        'Verified 64 archived cases, cost accounting and oracle mean.'
        + (' Paper values verified.' if paper is not None else '')
    )


if __name__ == '__main__':
    main()
