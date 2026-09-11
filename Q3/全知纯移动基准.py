"""Offline, fully informed open-TSP analysis of existing completed cases.

This is an evaluator, never an online policy. Local hidden coordinates are
reconstructed from the original generator and checked against EVERY recorded
feedback before use. Practice clear positions are not labelled source truth.
No new worlds, policy episodes, or HTTP requests are run.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / '第三问-算法3/results/20260911-192929-110364'
PRACTICE = ROOT / '第三问-算法3/results/20260911-195016-431746'
SPEED = 5.0
NUMERIC_MARGIN_M = 0.01


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, hashes):
    hashes[str(path.relative_to(ROOT))] = digest(path)
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def exact_route(points, router):
    length, order = router(points, exact_limit=16).plan((0., 0.))
    assert len(order) == len(set(order)) == len(points)
    positions = [(0., 0.)] + [points[c] for c in order]
    independent_length = sum(math.dist(a, b) for a, b in zip(positions, positions[1:]))
    assert abs(length - independent_length) < 1e-7
    return length, order


def selfcheck(router):
    fixtures = [[], [(0., 0.)], [(3., 4.)], [(1., 0.), (5., 0.), (2., 0.)],
                [(3., 4.), (-1., 2.), (5., -2.), (0., -3.)],
                [(1., 1.), (1., 1.), (-2., 0.), (0., 0.), (2., -1.)]]
    for coords in fixtures:
        points = dict(enumerate(coords))
        result, _ = exact_route(points, router)
        values = []
        for perm in itertools.permutations(coords):
            path = [(0., 0.)] + list(perm)
            values.append(sum(math.dist(a, b) for a, b in zip(path, path[1:])))
        assert abs(result - min(values)) < 1e-7
    regular_polygons = []
    for n in (10, 16):
        points = {j: (1800 * math.cos(2 * math.pi * j / n),
                      1800 * math.sin(2 * math.pi * j / n)) for j in range(n)}
        length, _ = exact_route(points, router)
        # First edge >=1800; every inter-source edge >= the polygon side.
        analytic = 1800 + (n - 1) * 3600 * math.sin(math.pi / n)
        assert abs(length - analytic) < 1e-7
        regular_polygons.append({'n': n, 'length_m': length,
                                 'pure_move_s_per_source': length / SPEED / n})
    return {'bruteforce_fixtures': len(fixtures),
            'analytic_regular_polygons': regular_polygons}


def summarize(rows):
    count = len(rows)
    sources = sum(r['n'] for r in rows)
    return {
        'cases': count, 'total_sources': sources,
        'mean_open_point_tsp_m': statistics.mean(r['point_tsp_m'] for r in rows),
        'mean_pure_move_s_per_case': statistics.mean(r['point_tsp_m'] / SPEED for r in rows),
        'pooled_pure_move_s_per_source': sum(r['point_tsp_m'] for r in rows) / SPEED / sources,
        'case_equal_pure_move_s_per_source': statistics.mean(r['point_tsp_m'] / SPEED / r['n'] for r in rows),
        'min_case_pure_move_s_per_source': min(r['point_tsp_m'] / SPEED / r['n'] for r in rows),
        'max_case_pure_move_s_per_source': max(r['point_tsp_m'] / SPEED / r['n'] for r in rows),
        'pooled_20m_disk_move_lower_s_per_source': sum(r['disk_lower_m'] for r in rows) / SPEED / sources,
        'pooled_20m_disk_move_upper_s_per_source': sum(r['disk_upper_m'] for r in rows) / SPEED / sources,
        'pooled_recorded_full_task_s_per_source': sum(r['recorded_virtual_s'] for r in rows) / sources,
        'pooled_recorded_move_s_per_source': sum(r['recorded_movement_m'] for r in rows) / SPEED / sources,
    }


def run(output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'batch.json', {'status': 'running'})
    hashes = {}
    try:
        config = load(LOCAL / 'config.json', hashes)
        assert config['seed'] == 900 and config['rounds'] == 40
        assert config['scenario'] == 'uniform' and config['n'] is None and config['radius'] is None
        assert config['environment']['numpy'] == np.__version__
        # Fail rather than silently reconstruct maps with a changed generator.
        for name in ('第三问-算法1/q3/core.py', '第三问-算法1/q3/simulator.py'):
            hashes[name] = digest(ROOT / name)
            assert hashes[name] == config['code_sha256'][name], name
        route_path = ROOT / '第三问-算法3/bayes_tsp/open_routes.py'
        hashes[str(route_path.relative_to(ROOT))] = digest(route_path)
        spec = importlib.util.spec_from_file_location('_offline_open_routes', route_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        router = module.OpenRoutes
        checks = selfcheck(router)
        # This import loads environment types only, not a policy or HTTP client.
        sys.path.insert(0, str(ROOT / '第三问-算法1'))
        from q3.core import Action
        from q3.simulator import generate_world, LocalSimulator

        records = load(LOCAL / 'summary.json', hashes)
        records = sorted((r for r in records if r['policy'] == 'bayes-fast'), key=lambda r: r['seed'])
        assert [r['seed'] for r in records] == list(range(900, 940))
        rows = []
        replay_count = 0
        for rec in records:
            assert rec['all_cleared'] and rec['certified_complete'] and rec['error'] is None
            seed = rec['seed']
            world = generate_world(seed, n=rec['n_input'], scenario=rec['scenario'],
                                   radius=rec['radius_input'], error_mode=rec['error_mode'])
            points = {c: s.position for c, s in world._sources.items()}
            assert len(points) == rec['source_count']
            actions = load(LOCAL / f'uniform-iid-nNone-rNone-{seed}-bayes-fast-actions.json', hashes)
            sim = LocalSimulator(world)
            ids = set()
            for item in actions:
                assert item['request_id'] not in ids
                ids.add(item['request_id'])
                a = item['action']
                response = sim.execute(Action(a['kind'], tuple(a['position']), a['channel']), item['request_id'])
                expected = item['response']
                assert set(response) == set(expected), (seed, 'response keys')
                for key in expected:
                    if key != 'real_timestamp_ms':
                        assert response[key] == expected[key], (seed, key, response[key], expected[key])
                replay_count += 1
            assert world.score()['all_cleared']
            assert sim.virtual_time == rec['virtual_time_s']
            length, order = exact_route(points, router)
            n = len(points)
            rows.append({'seed': seed, 'n': n, 'points': points, 'optimal_order': order,
                         'point_tsp_m': length,
                         'disk_lower_m': max(0., length - 20 * (2 * n - 1) - NUMERIC_MARGIN_M),
                         'disk_upper_m': length,
                         'recorded_virtual_s': rec['virtual_time_s'],
                         'recorded_movement_m': rec['movement_m']})
        dump(output / 'local_cases.json', rows)

        practice = load(PRACTICE / 'summary.json', hashes)
        assert len(practice) == 10
        prows = []
        for i, rec in enumerate(practice, 1):
            assert rec['certified_complete'] and rec['exited'] and rec['error'] is None
            case_summary = load(PRACTICE / f'practice-{i}/summary.json', hashes)
            assert case_summary['case_code'] == rec['case_code']
            p = PRACTICE / f'practice-{i}/actions.jsonl'
            hashes[str(p.relative_to(ROOT))] = digest(p)
            actions = [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
            points = {}
            previous = (0., 0.)
            movement = 0.
            for item in actions:
                assert item['response']['accepted'] is True
                a = item['action']
                movement += math.dist(previous, a['position'])
                previous = a['position']
                if item['response'].get('clear_result') == 'success':
                    assert a['kind'] == 'clear' and a['channel'] not in points
                    points[a['channel']] = tuple(a['position'])
            n = len(points)
            assert n == rec['source_count'] == rec['cleared_count']
            assert abs(movement - rec['movement_m']) < .01
            length, order = exact_route(points, router)
            prows.append({'case_code': rec['case_code'], 'n': n,
                          'successful_clear_positions_NOT_source_truth': points,
                          'optimal_clear_point_order': order, 'point_tsp_m': length,
                          'true_center_tsp_lower_m': max(0., length - 20 * (2 * n - 1) - NUMERIC_MARGIN_M),
                          'true_center_tsp_upper_m': length + 20 * (2 * n - 1) + NUMERIC_MARGIN_M,
                          'disk_lower_m': max(0., length - 40 * (2 * n - 1) - NUMERIC_MARGIN_M),
                          'disk_upper_m': length,
                          'recorded_virtual_s': rec['virtual_time_s'],
                          'recorded_movement_m': movement})
        dump(output / 'practice_cases.json', prows)
        summary = {'local_true_source_coordinates': summarize(rows),
                   'practice_clear_point_proxy_NOT_true_coordinates': summarize(prows),
                   'local_by_source_count': {str(n): summarize([r for r in rows if r['n'] == n])
                                            for n in sorted({r['n'] for r in rows})},
                   'verification': {**checks, 'replayed_local_actions': replay_count,
                                    'all_local_feedback_and_costs_match': True,
                                    'core_and_simulator_hashes_match_original': True},
                   'analysis_real_time_s': time.monotonic() - started,
                   'limitations': ['Local 40-map empirical mean, not a distribution-free universal mean.',
                                   'Point TSP exact via Held-Karp in floating-point arithmetic.',
                                   '20m disk optimum bounded, not solved exactly.',
                                   'Practice positions are successful clear submissions, not true source coordinates.',
                                   'Pure movement only: no detection, channel switching, or clear operation costs.',
                                   'If retaining successful clear costs, add exactly 5 seconds per source.']}
        dump(output / 'summary.json', summary)
        hashes[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
        dump(output / 'input_and_code_sha256.json', hashes)
        dump(output / 'batch.json', {'status': 'completed', 'local_cases': 40, 'practice_cases': 10})
        return summary
    except BaseException as exc:
        dump(output / 'batch.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' /
                        ('omniscient-movement-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))
    print('Results:', args.output)
