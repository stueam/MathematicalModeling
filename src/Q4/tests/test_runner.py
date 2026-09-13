import json
import subprocess
import sys
from pathlib import Path

import pytest

from q4.compact import POLICIES, make_policy
from q4.policy import Config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', POLICIES)
def test_supported_policy_initializes(name):
    policy = make_policy(name, Config())
    assert policy.points
    assert callable(policy.choose)


def test_removed_experiment_is_rejected():
    with pytest.raises(ValueError, match='Unknown Q4 policy'):
        make_policy('mc-probes')


def test_incomplete_local_run_exits_unsuccessfully_and_keeps_logs(tmp_path):
    output = tmp_path / 'incomplete'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'run.py'), 'local', '--max-steps', '1', '--output', str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    summary = json.loads((output / 'summary.json').read_text())[0]
    assert summary['stop_reason'] == 'action_limit'
    assert not summary['certified_complete']
    assert not summary['all_cleared']
    assert list(output.glob('*-actions.jsonl'))


def test_manifest_contains_every_runtime_source():
    from run import manifest

    paths = {
        str(path.relative_to(ROOT.parent))
        for path in ROOT.rglob('*.py')
        if path.parent == ROOT or path.parent in (ROOT / 'q4', ROOT / 'vendor' / 'q3')
    }
    assert paths <= manifest().keys()


def test_s21_launcher_modes_cannot_conflict():
    from start_s21 import main

    with pytest.raises(SystemExit) as error:
        main(['--self-test', '--connect'])
    assert error.value.code == 2
