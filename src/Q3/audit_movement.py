"""Retrospective route references from completed public practice logs only.

Never call this from a policy: successful clear positions are future information.
Their exact open TSP omits search and sensing, so is not an executable strategy.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

from bayes_tsp.open_routes import OpenRoutes


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def route_metrics(actions):
    previous = (0., 0.)
    actual = 0.
    points = {}
    for row in actions:
        if row['response'].get('accepted') is not True:
            raise ValueError('Expected accepted actions only')
        action = row['action']
        position = tuple(action['position'])
        if len(position) != 2 or not all(math.isfinite(x) for x in position):
            raise ValueError('Invalid position')
        actual += math.dist(previous, position)
        previous = position
        if row['response'].get('clear_result') == 'success':
            channel = action['channel']
            if action['kind'] != 'clear' or channel in points:
                raise ValueError('Successful clears must have distinct channels')
            points[channel] = position
    n = len(points)
    if not 1 <= n <= 16:
        raise ValueError('Need 1..16 successful clears for exact reference')
    ordered = [(0., 0.)]+list(points.values())
    chronological = sum(math.dist(a, b) for a, b in zip(ordered, ordered[1:]))
    exact, order = OpenRoutes(points, exact_limit=16).plan((0., 0.))
    if actual+1e-6 < chronological or chronological+1e-6 < exact:
        raise ValueError('Route reference inequalities failed')
    # Each logged success q and any other successful clear z of the same source
    # are <=40 m apart. Replacing z by q adds <=40 m on the initial edge and
    # <=80 m on each of the other n-1 edges. TSP(q) <= L(z)+40*(2*n-1).
    # A small additional numerical margin weakens, never strengthens, the bound.
    lower = max(0., exact-40*(2*n-1)-.01)
    return {'source_count': n, 'actual_m': actual, 'clear_order_m': chronological,
            'hindsight_point_tsp_m': exact, 'source_travel_lower_bound_m': lower,
            'intermediate_and_tail_excess_m': actual-chronological,
            'hindsight_order_gap_m': chronological-exact,
            'successful_clear_positions': points, 'hindsight_order': order}


def audit(aggregate, output, target):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    dump(output/'batch.json', {'status': 'running'})
    inputs = {}

    def record(path):
        inputs[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        config = {'aggregate': str(aggregate.resolve()), 'target_s_per_source': target,
                  'numeric_margin_m': .01, 'reference_exact_limit': 16,
                  'source_sha256': {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [Path(__file__), Path(__file__).parent/'bayes_tsp/open_routes.py']},
                  'limitations': 'Hindsight diagnosis, no policy calls, no simulator connection; '
                                 'point TSP is not mission cost; fixed-operation bound holds only '
                                 'if total nonmovement cost is unchanged.'}
        dump(output/'config.json', config)
        cases_path = aggregate/'cases.json'
        record(cases_path)
        cases = read(cases_path)
        if not cases or len({c['case_code'] for c in cases}) != len(cases):
            raise ValueError('Empty or duplicate case list')
        rows = []
        for case in cases:
            if (not case['certified_complete'] or case['error'] or not case['exited']
                    or case['cleared_count'] != case['source_count']):
                raise ValueError('Expected completed, fully cleared cases')
            folders = [p.parent for p in Path(case['source_batch']).glob('practice-*/summary.json')
                       if read(p).get('case_code') == case['case_code']]
            if len(folders) != 1:
                raise ValueError('Nonunique case folder')
            path = folders[0]/'actions.jsonl'
            record(path)
            record(folders[0]/'summary.json')
            actions = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
            row = route_metrics(actions)
            if row['source_count'] != case['source_count'] or abs(row['actual_m']-case['movement_m']) > .01:
                raise ValueError('Action history differs from summary')
            operation_s = sum(v for k, v in case['costs'].items() if k != 'move_s')
            if abs(row['actual_m']/5+operation_s-case['virtual_time_s']) > .01:
                raise ValueError('Cost reconciliation failed')
            row.update(case_code=case['case_code'], actions_path=str(path.resolve()),
                       nonmovement_s=operation_s, virtual_time_s=case['virtual_time_s'],
                       target_distance_same_operations_m=5*(target*row['source_count']-operation_s))
            rows.append(row)
        fields = ('source_count', 'actual_m', 'clear_order_m', 'hindsight_point_tsp_m',
                  'source_travel_lower_bound_m', 'intermediate_and_tail_excess_m',
                  'hindsight_order_gap_m', 'nonmovement_s', 'virtual_time_s',
                  'target_distance_same_operations_m')
        totals = {k: sum(r[k] for r in rows) for k in fields}
        n, count = totals['source_count'], len(rows)
        summary = {'rounds': count, 'totals': totals,
                   'mean_case': {k: totals[k]/count for k in fields},
                   'pooled_actual_s_per_source': totals['virtual_time_s']/n,
                   'pooled_move_s_per_source': totals['actual_m']/5/n,
                   'pooled_nonmovement_s_per_source': totals['nonmovement_s']/n,
                   'fixed_operations_lower_bound_s_per_source':
                       (totals['source_travel_lower_bound_m']/5+totals['nonmovement_s'])/n,
                   'target_nonmovement_budget_at_travel_lower_bound_s_per_source':
                       target-totals['source_travel_lower_bound_m']/5/n,
                   'analysis_real_time_s': time.monotonic()-started,
                   'new_policy_runs': 0}
        dump(output/'cases.json', rows)
        dump(output/'input_sha256.json', inputs)
        dump(output/'summary.json', summary)
        dump(output/'batch.json', {'status': 'complete', 'rounds': count})
        return summary
    except BaseException as exc:
        dump(output/'batch.json', {'status': 'stopped', 'error': repr(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('aggregate', type=Path)
    parser.add_argument('--target', type=float, default=150.)
    args = parser.parse_args()
    if not math.isfinite(args.target) or args.target <= 0:
        parser.error('--target must be finite and positive')
    out = Path(__file__).parent/'results'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    print(json.dumps(audit(args.aggregate, out, args.target), ensure_ascii=False, indent=2))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
