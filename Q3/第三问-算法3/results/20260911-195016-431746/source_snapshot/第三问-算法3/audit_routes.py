"""Diagnose existing public action logs; no simulator truth or random sampling."""
import argparse
from collections import Counter
import json
from pathlib import Path

from bayes_tsp.open_routes import OpenRoutes
from bayes_tsp.shared import distance
from run import dump, new_output, code_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--policy', default='bayes-route-scan')
    args = parser.parse_args()
    out = new_output()
    dump(out/'config.json', {'input': str(args.folder), 'policy': args.policy, 'code_sha256': code_manifest()})
    rows = []
    for path in sorted(args.folder.glob(f'*-{args.policy}-actions.json')):
        decisions = json.loads(path.with_name(path.name.replace('-actions.json', '-decisions.json')).read_text())
        decisions = {d['step']: d for d in decisions}
        actions = json.loads(path.read_text())
        last_clear = max((i for i, a in enumerate(actions) if a['response'].get('clear_result') == 'success'), default=-1)
        movement = Counter(total_m=0., cover_m=0., joint_survey_m=0., localize_clear_m=0., after_last_clear_m=0.)
        previous = (0., 0.)
        gaps = []
        for step, row in enumerate(actions):
            action = row['action']
            moved = distance(previous, action['position'])
            decision = decisions.get(step, {})
            movement['total_m'] += moved
            phase = ('joint_survey_m' if (decision.get('selected_task') or '').startswith('survey:') else
                     'cover_m' if decision.get('coverage_phase') else 'localize_clear_m')
            movement[phase] += moved
            if step > last_clear:
                movement['after_last_clear_m'] += moved
            points = decision.get('route_points')
            route = decision.get('route')
            if points and route and len(points) <= 12:
                points = {int(k): v for k, v in points.items()}
                # V3 source task routes use int channel keys.
                if all(k in points for k in route) and set(route) == set(points):
                    solver = OpenRoutes(points)
                    exact, _ = solver.plan(previous)
                    old = sum(distance(a, b) for a, b in zip([previous]+[points[k] for k in route[:-1]],
                                                             [points[k] for k in route]))
                    gaps.append(max(0., old-exact))
            previous = action['position']
        result = {'actions': path.name, **movement,
            'static_route_snapshots': len(gaps), 'mean_static_tsp_gap_m': sum(gaps)/len(gaps) if gaps else None,
            'max_static_tsp_gap_m': max(gaps) if gaps else None}
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    dump(out/'summary.json', rows)
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
