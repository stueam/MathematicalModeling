import json
import sys
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.insert(0, '/home/zypca/mathmodel/B题/第三问-算法3')
import run
from bayes_tsp.adaptive_search import AdaptiveSearchConfig, AdaptiveSearchPolicy
from bayes_tsp.sectors import SectorConfig

base_factory = run.make_policy


def make_policy(name, config):
    if name.startswith('adaptive'):
        options = {
            'adaptive2': dict(sector_rotations=2),
            'adaptive4': dict(sector_rotations=4),
            'adaptive-tail': dict(sector_rotations=4, rotate_with_sources=False),
            'adaptive-gain50': dict(sector_rotations=4, partition_min_saving_s=50.),
            'adaptive-gain100': dict(sector_rotations=4, partition_min_saving_s=100.)}
        return AdaptiveSearchPolicy(AdaptiveSearchConfig(**asdict(config), **options[name]))
    return base_factory(name, config)


run.make_policy = make_policy


def main():
    output = run.new_output()
    names = ['bayes-v6', 'adaptive-tail', 'adaptive-gain50', 'adaptive-gain100']
    jobs = [(seed, name, 'uniform', 'iid', None, None, asdict(SectorConfig()), 1200., 1500)
            for seed in range(10) for name in names]
    run.dump(output/'config.json', {'description': 'paired adaptive rotated-sector development experiment',
        'names': names, 'seeds': list(range(10)), 'workers': 16, 'source_hashes': run.code_manifest()})
    batch = {'status': 'running', 'planned': [run.job_key(job) for job in jobs], 'completed': []}
    run.dump(output/'batch.json', batch)
    summaries = []
    print(output, flush=True)
    with ProcessPoolExecutor(max_workers=16, mp_context=multiprocessing.get_context('fork')) as executor:
        pending = {executor.submit(run.run_case, job): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            summary, actions, decisions = future.result()
            key = run.job_key(job)
            run.dump(output/f'{key}-summary.json', summary)
            run.dump(output/f'{key}-actions.json', actions)
            run.dump(output/f'{key}-decisions.json', decisions)
            summaries.append(summary)
            batch['completed'].append(key)
            run.dump(output/'batch.json', batch)
            print(job[0], job[1], round(summary['virtual_time_s'], 2), summary['certified_complete'], flush=True)
    batch['status'] = 'completed'
    run.dump(output/'batch.json', batch)
    run.dump(output/'summary.json', summaries)
    for name in names:
        rows = [row for row in summaries if row['policy'] == name]
        print(name, {key: sum(row[key] for row in rows)/len(rows) for key in
            ['virtual_time_s', 'movement_m', 'measure_count', 'real_time_s', 'failed_clears']}, flush=True)


if __name__ == '__main__':
    main()
