"""Reconcile the 50 official rounds and reweight the fixed 100k MC results."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MC = ROOT / 'results/oracle-tsp-mc-20260912-021030-302657/summary.json'
COUNTS = ROOT / 'results/official-q3-counts-20260912-022136-803657/summary.json'
OFFICIAL = ROOT / '第三问-算法3/results/20260911-231356-412606'


def run():
    out = ROOT / 'results' / ('oracle-chain-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    out.mkdir()
    hashes = {}

    def read(p):
        raw = p.read_bytes()
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    mc, counts = read(MC), read(COUNTS)
    official_summary = read(OFFICIAL / 'first50-summary.json')
    rows = [read(OFFICIAL / f'practice-{i}/summary.json') for i in range(1, 51)]
    assert len({r['case_code'] for r in rows}) == 50
    assert all(r['certified_complete'] and r['exited'] and not r['error'] and
               r['cleared_count'] == r['source_count'] for r in rows)
    sources = sum(r['source_count'] for r in rows)
    total = sum(r['virtual_time_s'] for r in rows)
    assert sources == 669 and abs(total - official_summary['total_virtual_time_s']) < 1e-6
    costs = {k: sum(r['costs'][k] for r in rows) for k in rows[0]['costs']}
    assert abs(sum(costs.values()) - total) < 1e-6
    p = {r['source_count']: r['frequency'] for r in counts['frequencies']}
    expected_n = sum(n * w for n, w in p.items())
    conditional = {n: mc['by_N'][str(n)]['s_per_source']['mean'] for n in p}
    oracle = sum(p[n] * n * conditional[n] for n in p) / expected_n
    ratio = total / sources
    answer = {'official_N_cases': counts['official_Q3_practice_cases'],
              'official_N_counts_10_to_16': [r['cases'] for r in counts['frequencies']],
              'uniform_N_test': counts['uniform_10_to_16_test'],
              'MC_samples': 100000,
              'oracle_official152_N_reweighted_pooled_s_per_source': oracle,
              'oracle_official152_N_reweighted_case_equal_s_per_source': sum(p[n] * conditional[n] for n in p),
              'official_algorithm3': {'rounds': 50, 'sources': sources,
                  'all_cleared': True, 'pooled_s_per_source': ratio,
                  'total_virtual_s': total, 'cost_totals_s': costs,
                  'cost_s_per_source': {k: v / sources for k, v in costs.items()},
                  'measurements_per_source': sum(r['measure_count'] for r in rows) / sources,
                  'mean_real_time_s': official_summary['mean_real_time_s']},
              'scale_comparison_NOT_paired': {'pure_movement_oracle_s_per_source': oracle,
                  'oracle_plus_one_successful_clear_s_per_source': oracle + 5,
                  'observed_minus_oracle_and_clear_s_per_source': ratio - oracle - 5,
                  'observed_movement_minus_oracle_s_per_source': costs['move_s'] / sources - oracle,
                  'actual_full_cost_div_oracle_plus_clear': ratio / (oracle + 5)},
              'limits': 'Uniform spatial model; 20m-disk optimum not solved; mixed-map comparison is not an optimality ratio.'}
    for name, data in [('summary.json', answer), ('input_sha256.json', hashes),
                       ('first50_cost_cases.json', [{k: r[k] for k in ('case_code', 'source_count', 'cleared_count',
                           'certified_complete', 'virtual_time_s', 'costs', 'movement_m', 'measure_count', 'clear_failures')} for r in rows]),
                       ('batch.json', {'status': 'completed'})]:
        (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(answer, ensure_ascii=False, indent=2))
    print('Results:', out)


if __name__ == '__main__':
    run()
