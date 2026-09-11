"""Fresh random-map Monte Carlo: uniform N=10..16, iid area-uniform disk points.

Every world is solved exactly as a fixed-origin, free-endpoint Euclidean TSP.
Official observations are used ONLY for optional N-frequency reweighting.
Pure movement cost: length / 5; no sensing or clearing operation fees.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy import stats

from 全知纯移动基准 import selfcheck


ROOT = Path(__file__).resolve().parent
ROUTER = None
SEED = None


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def initialize(router_path, seed):
    global ROUTER, SEED
    spec = importlib.util.spec_from_file_location('_mc_exact_open_routes', router_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ROUTER, SEED = module.OpenRoutes, seed


def solve_one(index):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, index]))
    n = int(rng.integers(10, 17))
    theta = 2 * np.pi * rng.random(n)
    radius = 1800 * np.sqrt(rng.random(n))
    xy = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    length, order = ROUTER(dict(enumerate(xy)), exact_limit=16).plan((0., 0.))
    assert sorted(order) == list(range(n))
    path = np.vstack((np.zeros((1, 2)), xy[order]))
    assert abs(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) - length) < 1e-7
    return index, n, xy, order, length


def mean_stats(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(len(values))
    half = float(stats.t.ppf(.975, len(values) - 1)) * se
    return {'samples': len(values), 'mean': mean, 'sample_sd': sd, 'mc_standard_error': se,
            'mean_95_interval': [mean - half, mean + half],
            'map_percentiles_5_50_95': list(map(float, np.quantile(values, [.05, .5, .95])))}


def reweighted(ns, times, probabilities):
    means, variances, lower_means, sizes = [], [], [], []
    for n in range(10, 17):
        selected = times[ns == n] / n
        assert len(selected) >= 2
        means.append(float(selected.mean()))
        variances.append(float(selected.var(ddof=1)))
        sizes.append(len(selected))
        lower_means.append(float(np.maximum(0., selected - 20 * (2 * n - 1) / (5 * n) - .01 / (5 * n)).mean()))
    p, means, variances, sizes = map(np.asarray, (probabilities, means, variances, sizes))
    expected_n = float(p @ np.arange(10, 17))
    answer = {}
    for label, weights in [('expected_case_T_div_N', p),
                           ('expected_T_div_expected_N', p * np.arange(10, 17) / expected_n)]:
        value = float(weights @ means)
        se = float(np.sqrt((weights ** 2 * variances / sizes).sum()))
        answer[label] = {'mean_s_per_source': value, 'conditional_MC_standard_error': se,
                         'conditional_MC_mean_95': [value - 1.959963984540054 * se, value + 1.959963984540054 * se],
                         'mean_20m_visit_lower_bound_s_per_source': float(weights @ np.asarray(lower_means)),
                         'mean_20m_visit_upper_bound_s_per_source': value}
    return {'P_N_10_to_16': list(map(float, p)), 'expected_N': expected_n, **answer,
            'note': 'MC intervals condition on the supplied N probabilities and iid uniform spatial model; '
                    'uncertainty of fitted official N frequencies is not included.'}


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    dump(out / 'batch.json', {'status': 'running', 'planned_worlds': args.samples, 'completed_worlds': 0})
    route_source = ROOT / '第三问-算法3/bayes_tsp/open_routes.py'
    route_snapshot = out / 'open_routes_snapshot.py'
    shutil.copyfile(route_source, route_snapshot)
    shutil.copyfile(Path(__file__), out / 'analysis_snapshot.py')
    hashes = {str(route_source.relative_to(ROOT)): hashlib.sha256(route_source.read_bytes()).hexdigest(),
              Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              '全知纯移动基准.py': hashlib.sha256((ROOT / '全知纯移动基准.py').read_bytes()).hexdigest()}
    empirical = None
    if args.official_counts:
        counts_path = args.official_counts.resolve()
        raw = counts_path.read_bytes()
        data = json.loads(raw)
        counts = {r['source_count']: r['cases'] for r in data['frequencies']}
        assert sum(counts.values()) == data['official_Q3_practice_cases']
        empirical = [counts[n] / sum(counts.values()) for n in range(10, 17)]
        hashes[str(counts_path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    config = {'worlds': args.samples, 'workers': args.workers, 'seed': args.seed,
              'speed_m_per_s': 5, 'origin': [0, 0], 'return_to_origin': False,
              'primary_N_distribution': 'DiscreteUniform{10,...,16}',
              'positions_given_N': 'iid area-uniform in origin-centered disk radius 1800m',
              'position_sampler': 'theta=2*pi*U1, r=1800*sqrt(U2)',
              'random_generator': 'NumPy PCG64, SeedSequence([seed,world_index])',
              'TSP': 'Exact Held-Karp open path, N<=16; numerical float64 distances',
              'fees': 'Movement only; no switch, measurement, or clear fees',
              'spatial_model': 'User-specified model, not inferred from official spatial coordinates',
              'official_N_probability_sensitivity': empirical,
              'environment': {'python': sys.version, 'numpy': np.__version__}, 'sha256': hashes}
    dump(out / 'config.json', config)
    initialize(str(route_snapshot), args.seed)
    checks = selfcheck(ROUTER)
    ns = np.zeros(args.samples, dtype=np.int16)
    lengths = np.zeros(args.samples)
    coordinates = np.full((args.samples, 16, 2), np.nan)
    routes = np.full((args.samples, 16), -1, dtype=np.int16)
    completed = 0
    try:
        if args.workers == 1:
            jobs = map(solve_one, range(args.samples))
            pool = None
        else:
            pool = ProcessPoolExecutor(max_workers=args.workers,
                                       mp_context=multiprocessing.get_context('spawn'),
                                       initializer=initialize, initargs=(str(route_snapshot), args.seed))
            jobs = pool.map(solve_one, range(args.samples), chunksize=32)
        try:
            for index, n, xy, order, length in jobs:
                assert index == completed
                ns[index], lengths[index] = n, length
                coordinates[index, :n], routes[index, :n] = xy, order
                completed += 1
                if completed % 2000 == 0:
                    progress = {'status': 'running', 'planned_worlds': args.samples,
                                'completed_worlds': completed, 'elapsed_s': time.monotonic() - started}
                    dump(out / 'batch.json', progress)
                    print(json.dumps(progress), flush=True)
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        assert completed == args.samples
        np.savez_compressed(out / 'worlds_and_exact_routes.npz', source_counts=ns,
                            source_xy=coordinates, optimal_order=routes, optimal_length_m=lengths)
        times = lengths / 5
        per_source = times / ns
        lower = np.maximum(0., lengths - 20 * (2 * ns - 1) - .01) / 5 / ns
        ratio = float(times.sum() / ns.sum())
        ratio_se = float((times - ratio * ns).std(ddof=1) / ns.mean() / math.sqrt(args.samples))
        summary = {'model': config, 'all_worlds_completed': True, 'verification': checks,
                   'frequency_N_10_to_16': [int((ns == n).sum()) for n in range(10, 17)],
                   'sample_mean_N': float(ns.mean()),
                   'sample_mean_r_squared_div_1800_squared': float(np.nansum(coordinates ** 2) / ns.sum() / 1800 ** 2),
                   'mean_route_length_m': mean_stats(lengths), 'mean_pure_move_s_per_case': mean_stats(times),
                   'primary_expected_T_div_N_s_per_source': mean_stats(per_source),
                   'pooled_T_div_total_N_s_per_source': {'estimate': ratio, 'mc_standard_error': ratio_se,
                       'approx_mean_95': [ratio - 1.959963984540054 * ratio_se, ratio + 1.959963984540054 * ratio_se]},
                   'expected_20m_visit_bound_s_per_source': {'lower_bound_mean': float(lower.mean()),
                                                          'upper_bound_mean': float(per_source.mean()),
                                                          'continuous_optimum_solved': False},
                   'by_N': {str(n): {'worlds': int((ns == n).sum()),
                                     'length_m': mean_stats(lengths[ns == n]),
                                     's_per_source': mean_stats(per_source[ns == n])} for n in range(10, 17)},
                   'uniform_N_stratified_estimate': reweighted(ns, times, [1 / 7] * 7),
                   'official_frequency_N_sensitivity': reweighted(ns, times, empirical) if empirical else None,
                   'analysis_wall_time_s': time.monotonic() - started}
        dump(out / 'summary.json', summary)
        dump(out / 'batch.json', {'status': 'completed', 'planned_worlds': args.samples,
                                 'completed_worlds': completed, 'failed_worlds': 0})
        return summary
    except BaseException as exc:
        dump(out / 'batch.json', {'status': 'failed', 'planned_worlds': args.samples,
                                 'completed_worlds': completed, 'error': repr(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=20000)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20260912)
    parser.add_argument('--official-counts', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' /
                        ('oracle-tsp-mc-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
    args = parser.parse_args()
    if args.samples < 100 or args.workers < 1:
        parser.error('Need at least 100 maps and positive workers')
    result = run(args)
    print(json.dumps({k: v for k, v in result.items() if k not in ('model', 'by_N')},
                     ensure_ascii=False, indent=2), flush=True)
    print('Results:', args.output, flush=True)
