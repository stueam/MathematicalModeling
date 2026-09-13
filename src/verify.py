"""Verify all three standalone deliveries without contacting an official simulator."""

import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
QUESTIONS = ('Q2', 'Q3', 'Q4')


def source_files(folder):
    return sorted(
        path
        for path in folder.rglob('*.py')
        if not {'.venv', '__pycache__', 'results'} & set(path.relative_to(folder).parts)
    )


def import_audit(folder):
    """Check local module reachability, including the explicit vendored load() API."""
    files = {path for path in source_files(folder) if 'tests' not in path.relative_to(folder).parts}
    edges = {path: set() for path in files}

    def add(source, target):
        candidates = [target.with_suffix('.py'), target / '__init__.py']
        candidates.extend(
            parent / '__init__.py' for parent in target.parents if parent.is_relative_to(folder)
        )
        edges[source].update(path for path in candidates if path in files)

    entries = {
        folder / name
        for name in (
            'run.py',
            'start_formal.py',
            'practice_windows.py',
            'start_s21.py',
            'check_s21_certificate.py',
        )
        if folder / name in files
    }
    for path in files:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(path, folder.joinpath(*alias.name.split('.')))
            elif isinstance(node, ast.ImportFrom):
                base = path.parent if node.level else folder
                for _ in range(max(0, node.level - 1)):
                    base = base.parent
                if node.module:
                    base = base.joinpath(*node.module.split('.'))
                add(path, base)
                for alias in node.names:
                    add(path, base / alias.name)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'load'
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                add(path, folder / 'vendor' / 'q3' / node.args[0].value)
    # Tests and stray script guards cannot make unused runtime modules look reachable.
    pending = list(entries)
    reached = set()
    while pending:
        path = pending.pop()
        if path not in reached:
            reached.add(path)
            pending.extend(edges[path] - reached)
    unused = sorted(str(path.relative_to(folder)) for path in files - reached)
    if unused:
        raise RuntimeError(f'{folder.name}: unreferenced modules: {unused}')
    return {'python_files': len(files), 'reachable_files': len(reached), 'unreferenced_files': unused}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New validation output directory')
    args = parser.parse_args()
    output = (
        args.output or ROOT / 'results' / ('validation-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUTF8='1')
    env.pop('PYTHONPATH', None)
    report = {'status': 'running', 'python': sys.version, 'steps': [], 'official_connection': False}
    code = [ROOT / 'verify.py'] + [path for question in QUESTIONS for path in source_files(ROOT / question)]
    report['code_sha256'] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code
    }

    def save():
        (output / 'summary.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
        )

    def run(label, arguments, cwd=ROOT):
        command = [sys.executable, *map(str, arguments)]
        print(f'[{label}] {" ".join(command)}', flush=True)
        started = time.monotonic()
        with (output / f'{label}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(
                command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False
            )
        report['steps'].append(
            {
                'name': label,
                'command': command,
                'returncode': result.returncode,
                'elapsed_s': time.monotonic() - started,
                'log': f'{label}.log',
            }
        )
        save()
        if result.returncode:
            raise RuntimeError(f'{label} failed; see {output / (label + ".log")}')
        print(f'[{label}] passed', flush=True)

    save()
    try:
        report['imports'] = {question: import_audit(ROOT / question) for question in QUESTIONS}
        run('lint', ['-m', 'ruff', 'check', '.'])
        run('format', ['-m', 'ruff', 'format', '--check', *code])
        run('unused', ['-m', 'vulture', *QUESTIONS, '--exclude', 'results,.venv', '--min-confidence', '100'])
        for question in QUESTIONS:
            run(f'{question}-tests', ['-m', 'pytest', '-q'], ROOT / question)
        for scenario in ('symmetric', 'asymmetric'):
            run(
                f'Q2-{scenario}',
                ['run.py', '--scenario', scenario, '--output', output / f'Q2-{scenario}'],
                ROOT / 'Q2',
            )
        run('Q3-local', ['run.py', 'local', '--seed', '0', '--output', output / 'Q3-local'], ROOT / 'Q3')
        run('Q4-coverage', ['start_s21.py', '--self-test'], ROOT / 'Q4')
        run('Q4-certificate-cli', ['check_s21_certificate.py'], ROOT / 'Q4')
        run('Q4-local', ['run.py', 'local', '--seed', '800', '--output', output / 'Q4-local'], ROOT / 'Q4')
        for question in ('Q3', 'Q4'):
            run(f'{question}-formal-help', ['start_formal.py', '--help'], ROOT / question)
            run(f'{question}-practice-help', ['practice_windows.py', '--help'], ROOT / question)
        paper_import = ROOT.parent / 'essay/data/q3_execution_case/prepare_case.py'
        paper_results = ROOT.parent / 'Q3/evidence/paper_results/verify.py'
        integrations = [
            ('Q3-figure-import', paper_import, ['--run-dir', output / 'Q3-local', '--check']),
            ('Q3-paper-results', paper_results, ['--paper', ROOT.parent / 'essay/essay.tex']),
        ]
        report['skipped_integrations'] = []
        report['integration_sha256'] = {}
        for label, script, arguments in integrations:
            if script.is_file():
                report['integration_sha256'][str(script.relative_to(ROOT.parent))] = hashlib.sha256(
                    script.read_bytes()
                ).hexdigest()
                run(label, [script, *arguments])
            else:
                report['skipped_integrations'].append(label)  # Standalone src copies remain supported.
        report['status'] = 'passed'
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        report.update(status='failed', error=str(error))
        print(str(error), file=sys.stderr)
    finally:
        save()
    print(f'Validation: {output}', flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
