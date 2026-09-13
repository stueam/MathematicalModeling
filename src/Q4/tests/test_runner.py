import json
import subprocess
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


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


def test_certificate_cli_resolves_default_independently_of_cwd(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / 'check_s21_certificate.py')],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    certificate = json.loads(result.stdout)
    assert certificate['passed'] and certificate['boundary_covered']
    assert certificate['verified_cells'] == 3832


def test_practice_journal_preserves_coordinates_matching_team_digits(tmp_path):
    from practice_windows import LoggedSession

    with LoggedSession(tmp_path, '12345') as session:
        session.record({'request': {'robot_id': '12345', 'position': {'x': 12.12345, 'y': 0.0}}})
    record = json.loads((tmp_path / 'http-attempts.jsonl').read_text())
    assert record['request']['robot_id'] == '<已隐去>'
    assert record['request']['position'] == {'x': 12.12345, 'y': 0.0}
