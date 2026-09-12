"""Freeze and audit one existing LOCAL run; never call an official simulator.

Usage: python prepare_case.py --run-dir <directory-containing-config-and-run-files>
The default figure generator reads frozen files and does not need this import step.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def write_json(name, value):
    (HERE/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir/'config.json').read_text(encoding='utf-8'))
    assert config['mode'] == 'local' and config['policy'] == 'bayes-fast'
    key = f"{config['scenario']}-{config['error_mode']}-n{config['n']}-r{config['radius']}-{config['seed']}-{config['policy']}"
    summary_path = run_dir/f'{key}-summary.json'
    actions_path = run_dir/f'{key}-actions.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    actions = json.loads(actions_path.read_text(encoding='utf-8'))
    # Check the recorded code version before replaying or exposing ground truth.
    manifest = {k.replace('\\', '/'): v for k,v in config['code_sha256'].items()}
    for rel, sha in manifest.items():
        path = ROOT/'src'/rel
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha, rel

    sys.path.insert(0, str(ROOT/'src/Q3'))
    from bayes_tsp.shared import load
    simulator, core = load('simulator'), load('core')
    world = simulator.generate_world(config['seed'], config['n'], config['scenario'],
                                     config['radius'], config['error_mode'])
    # Truth is used only here for post-run audit and frozen visualization data.
    sources = list(world._sources.values())
    env = simulator.LocalSimulator(world)
    rows = []
    for index, entry in enumerate(actions, 1):
        action = core.Action(entry['action']['kind'], tuple(entry['action']['position']), entry['action']['channel'])
        actual = env.execute(action, f'figure-audit-{index}')
        expected = entry['response']
        for field in ('accepted', 'measure_result', 'svd_deg', 'clear_result', 'virtual_time_s'):
            assert actual.get(field) == expected.get(field), (index, field, actual, expected)
        rows.append(dict(index=index, kind=action.kind, x_m=action.position[0], y_m=action.position[1],
                         channel=action.channel, result=actual.get('measure_result', actual.get('clear_result')),
                         bearing_deg=actual.get('svd_deg',''), virtual_time_s=actual['virtual_time_s'],
                         planning_s=entry['planning_s']))
    for k, v in env.costs.items():
        assert math.isclose(v, summary['costs'][k], abs_tol=1e-6), k
    assert env.virtual_time == summary['virtual_time_s']
    assert world.score()['cleared_count'] == summary['cleared_count']
    assert summary['certified_complete'] and summary['all_cleared']
    assert summary['steps'] == len(rows)
    assert math.isclose(sum(summary['costs'].values()), summary['virtual_time_s'], abs_tol=1e-6)
    with (HERE/'actions.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (HERE/'sources.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['channel','x_m','y_m','radius_m'])
        for source in sources: writer.writerow([source.channel,*source.position,source.radius])
    write_json('summary.json', summary)
    write_json('provenance.json', {
        'case_kind':'local_simulation', 'run_id':run_dir.name, 'policy':config['policy'],
        'seed':config['seed'], 'scenario':config['scenario'], 'error_mode':config['error_mode'],
        'n_input':config['n'], 'radius_input':config['radius'],
        'selection':'The existing seed-0 verification case documented in src/Q3/VALIDATION.md; no best-case search.',
        'source_code_sha256':manifest,
        'original_log_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (summary_path,actions_path,run_dir/'config.json')},
        'audit':{'all_recorded_feedback_replayed_exactly':True,'replayed_actions':len(rows),
                 'time_decomposition_matches_summary':True,'code_files_matched':len(manifest)},
        'truth_usage':'Source coordinates are reconstructed only after the logged run, for replay audit and plotting.'})
    print(json.dumps({'actions_audited':len(rows),'code_files_matched':len(manifest),
                      'source_count':summary['source_count'],'total_s':summary['virtual_time_s']},ensure_ascii=False))


if __name__ == '__main__':
    main()
