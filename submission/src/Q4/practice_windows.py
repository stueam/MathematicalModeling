"""Windows UI-verified problem-four PRACTICE ONLY, using the paper S21 + probes algorithm.

Adapted from the existing algorithm-two practice guard. No formal controls,
generic confirmation clicks, hidden simulator data, or unpublished endpoints.
"""

import argparse
import base64
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import hashlib
import math
from pathlib import Path
import json
import re
import subprocess
import time
import os
import shutil

import requests

from q4.core import Belief
from q4.policy import Config, Policy
from q4.shared import SHARED_DIR, distance, load
from run import ROOT, dump, manifest
from formal_session import redact_log


def new_output():
    out = ROOT / 'results' / datetime.now().strftime('official-practice-%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    return out


def code_manifest():
    hashes = manifest()
    for path in (Path(__file__), SHARED_DIR / 'client.py'):
        hashes[str(path.relative_to(ROOT.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


POWERSHELL = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
if os.name == 'nt':
    POWERSHELL = shutil.which('powershell.exe') or str(
        Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    )
UI = r"""
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new()
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class Q4Window { [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h,int n); [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h); }'
$windows=@(Get-Process -Name 'jammers-simulator' -ErrorAction SilentlyContinue | Where-Object {$_.MainWindowHandle -ne 0})
if ($windows.Count -ne 1) {throw 'Expected exactly one visible simulator window'}
[Q4Window]::ShowWindow($windows[0].MainWindowHandle,9) | Out-Null
[Q4Window]::SetForegroundWindow($windows[0].MainWindowHandle) | Out-Null
Start-Sleep -Milliseconds 200
$root=[System.Windows.Automation.AutomationElement]::FromHandle($windows[0].MainWindowHandle)
"""
READ = r"""
$all=$root.FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.Condition]::TrueCondition)
$items=for ($i=0;$i -lt $all.Count;$i++) {
 $c=$all.Item($i).Current
 if ($c.Name -ne '') { [pscustomobject]@{name=$c.Name;type=$c.ControlType.ProgrammaticName;id=$c.AutomationId;enabled=$c.IsEnabled} }
}
ConvertTo-Json -InputObject @($items) -Depth 3 -Compress
"""


def powershell(script):
    encoded = base64.b64encode((UI + script).encode('utf-16-le')).decode('ascii')
    result = subprocess.run(
        [POWERSHELL, '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
        capture_output=True,
        timeout=20,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace')[:1500])
    return result.stdout.decode('utf-8-sig').strip()


def inspect_ui():
    for attempt in range(3):
        data = json.loads(powershell(READ))
        items = [i for i in (data or []) if isinstance(i.get('name'), str) and i['name']]
        if any(i.get('type') != 'ControlType.Pane' for i in items):
            return items
        if attempt < 2:
            time.sleep(0.5)  # WebView accessibility may initially expose only two panes.
    raise RuntimeError('Simulator UI is not readable; no test action sent')


def has_title(items, identifier):
    titles = [re.sub(r'\s+', '', i['name']) for i in items if i.get('id') == identifier]
    return bool(titles) and all(t == '问题4演练测试' for t in titles)


def practice_identity(items, awaiting=False):
    if not has_title(items, 'test-run-title'):
        raise RuntimeError('Not a verified problem-four PRACTICE screen')
    names = [i['name'] for i in items]
    if awaiting and '尚未进入' not in names:
        raise RuntimeError('Refusing to adopt a test that has already entered')
    ids = {m.group(1) for s in names if (m := re.fullmatch(r'队号\s+(\d+)', s))}
    cases = {s for s in names if re.fullmatch(r'[A-Z0-9]{4}(?:-[A-Z0-9]{4}){3}', s)}
    if len(ids) != 1 or len(cases) != 1:
        raise RuntimeError('Practice account and case are not unambiguously visible')
    return next(iter(ids)), next(iter(cases))


def redact_ui(items):
    return [dict(i, name=re.sub(r'队号\s+\d+', '队号 <已隐去>', i['name'])) for i in items]


def click_practice(name):
    if name not in ('开始问题4演练测试', '返回演练测试'):
        raise ValueError('Only the two explicit problem-four PRACTICE controls are allowed')
    guard = 'practice-4-title' if name.startswith('开始') else 'test-run-title'
    script = r"""
$guards=$root.FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::AutomationIdProperty,'GUARD'))
if ($guards.Count -eq 0) {throw 'Practice screen guard absent'}
foreach ($g in $guards) {if (($g.Current.Name -replace '\s','') -ne '问题4演练测试') {throw 'Practice screen guard failed'}}
$runs=$root.FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::AutomationIdProperty,'test-run-title'))
if ('GUARD' -eq 'practice-4-title' -and $runs.Count -ne 0) {throw 'Refusing to start over an existing test screen'}
if ('GUARD' -eq 'test-run-title') {
 $ended=$root.FindFirst([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::NameProperty,'测试已结束'))
 if (!$ended) {throw 'Refusing to leave an active test'}
}
$cond=[System.Windows.Automation.AndCondition]::new(
 [System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::NameProperty,'LABEL'),
 [System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::ControlTypeProperty,[System.Windows.Automation.ControlType]::Button))
$button=$root.FindFirst([System.Windows.Automation.TreeScope]::Descendants,$cond)
if (!$button -or !$button.Current.IsEnabled) {throw 'Explicit practice button is unavailable'}
$button.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
'practice-only button invoked'
"""
    return powershell(script.replace('GUARD', guard).replace('LABEL', name))


def start_practice(resume=False):
    items = inspect_ui()
    if resume:
        practice_identity(items, awaiting=True)
        return items
    if any(i.get('id') == 'test-run-title' for i in items):
        practice_identity(items)
        if not any(i['name'] == '测试已结束' for i in items):
            raise RuntimeError('An existing practice is active; not interrupting it')
        click_practice('返回演练测试')
        time.sleep(0.3)
    click_practice('开始问题4演练测试')
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        items = inspect_ui()
        if has_title(items, 'test-run-title') and any(
            re.search(r'接口.*就绪|等待机器狗进入|等待进入', i['name']) for i in items
        ):
            practice_identity(items, awaiting=True)
            return items
        time.sleep(1)
    raise RuntimeError('Practice readiness not established; no /enter sent')


def result_count(items):
    if not has_title(items, 'test-run-title') or not any(i['name'] == '测试已结束' for i in items):
        raise RuntimeError('Practice completion screen not verified')
    names = [i['name'] for i in items]
    try:
        start = names.index('本次演练测试干扰源数量')
        total = next(int(s) for s in names[start + 1 :] if re.fullmatch(r'\d+', s))
    except (ValueError, StopIteration):
        return None
    return total if 10 <= total <= 16 else None


def result_composition(items):
    """Read evaluator-only type counts after the verified practice has ended."""
    total = result_count(items)
    names = [i['name'] for i in items]
    matches = [re.search(r'本次案例含干扰源(\d+)个，其中全向(\d+)个、定向(\d+)个', s) for s in names]
    counts = {tuple(map(int, m.groups())) for m in matches if m}
    # The completion dialog may disappear while the permanent result panel remains.
    for i, name in enumerate(names):
        if name != '本次演练测试干扰源数量':
            continue
        section = re.sub(r'\s+', '', ''.join(names[i + 1 : i + 12]))
        match = re.search(r'共(\d+)个[，,]全向(\d+)个[，,]定向(\d+)个', section)
        if match:
            counts.add(tuple(map(int, match.groups())))
    if not counts:
        return {'omnidirectional_count': None, 'directional_count': None}
    if len(counts) != 1:
        raise RuntimeError('Conflicting official practice result panels')
    n, omni, directional = next(iter(counts))
    # Official Q4 practice can contain only directional sources (observed 0/14).
    # Type counts are evaluation metadata, never a controller assumption.
    if n != total or omni + directional != total or min(omni, directional) < 0:
        raise RuntimeError('Conflicting official practice type counts')
    return {'omnidirectional_count': omni, 'directional_count': directional}


class LoggedSession(requests.Session):
    """Persist each attempt before sending, including uncertain/malformed outcomes."""

    def __init__(self, folder, robot_id):
        super().__init__()
        self.trust_env = False
        self.path, self.robot_id = folder / 'http-attempts.jsonl', robot_id

    def record(self, data):
        data['recorded_at_utc'] = datetime.now(timezone.utc).isoformat()
        text = json.dumps(redact_log(data, self.robot_id), ensure_ascii=False, allow_nan=False)
        with self.path.open('a', encoding='utf-8') as log:
            log.write(text + '\n')
            log.flush()

    def post(self, url, **kwargs):
        payload = json.loads(kwargs['data'])
        self.record({'event': 'before_request', 'url': url, 'request': payload})
        started = time.monotonic()
        try:
            response = super().post(url, **kwargs)
        except Exception as exc:
            self.record(
                {
                    'event': 'request_error',
                    'request_id': payload['request_id'],
                    'error': f'{type(exc).__name__}: {exc}',
                    'elapsed_s': time.monotonic() - started,
                }
            )
            raise
        self.record(
            {
                'event': 'response',
                'request_id': payload['request_id'],
                'http_status': response.status_code,
                'body': response.text,
                'elapsed_s': time.monotonic() - started,
            }
        )
        return response


def run_once(folder, expected_case, config):
    policy = Policy(config)
    # This fresh UI check precedes client creation and every first /enter.
    items = inspect_ui()
    robot_id, case = practice_identity(items, awaiting=True)
    if case != expected_case:
        raise RuntimeError('Practice case changed before /enter')
    dump(folder / 'practice-before.json', redact_ui(items))
    session = LoggedSession(folder, robot_id)
    client = load('client').HttpClient(robot_id, 'http://127.0.0.1:2026', session=session)
    belief = Belief()  # Exactly the Q4 public state used in local validation.
    rows, costs = (
        [],
        {'move_s': 0.0, 'switch_s': 0.0, 'measure_s': 0.0, 'clear_success_s': 0.0, 'clear_failure_s': 0.0},
    )
    summary = {
        'mode': 'problem4_practice_only',
        'policy': 's21-probes',
        'case_code': case,
        'entered': False,
        'exited': False,
        'error': None,
        'source_count': None,
    }
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc)
    summary.update(started_at_utc=started_utc.isoformat(), implementation=policy.implementation)
    dump(folder / 'summary.json', summary)
    max_cost_difference = 0.0
    try:
        _, response = client.enter()
        summary['entered'] = True
        summary['remaining_real_duration_s'] = response['remaining_real_duration_s']
        remaining = float(response['remaining_real_duration_s'])
        virtual_limit = float(response['max_virtual_duration_s'])
        initial_time = float(response['virtual_time_s'])
        if not (
            math.isfinite(remaining)
            and 5 < remaining <= 1200
            and math.isfinite(virtual_limit)
            and 0 < virtual_limit <= 360000
            and initial_time == 0
        ):
            raise RuntimeError('Unexpected entry time budgets or nonzero initial virtual clock')
        summary['server_deadline_utc'] = (
            datetime.now(timezone.utc) + timedelta(seconds=client.deadline - time.monotonic())
        ).isoformat()
        dump(folder / 'summary.json', summary)
        belief.deadline = client.deadline - 5
        belief.virtual_limit = float(response['max_virtual_duration_s'])
        belief.virtual_time = float(response['virtual_time_s'])
        print(
            json.dumps(
                {
                    'case': case,
                    'entered': True,
                    'mode': 'problem4_practice_only',
                    'remaining_real_s': response['remaining_real_duration_s'],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        for index in range(6000):
            if belief.done():
                break
            if time.monotonic() >= belief.deadline or belief.virtual_time >= belief.virtual_limit:
                raise TimeoutError('Actual practice time budget reached before completion')
            tick = time.monotonic()
            action = policy.choose(belief)
            planning = time.monotonic() - tick
            if time.monotonic() >= belief.deadline:
                raise TimeoutError('Actual deadline reached during planning')
            request_id, response = client.execute(action)
            row = {
                'request_id': request_id,
                'action': asdict(action),
                'response': response,
                'planning_s': planning,
                'move_m': distance(belief.position, action.position),
            }
            rows.append(row)
            with (folder / 'actions.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(row, ensure_ascii=False) + '\n')
            move = round(distance(belief.position, action.position) / 5, 6)
            switch = int(action.kind == 'measure' and action.channel != belief.receiver)
            operation = 5 if action.kind == 'measure' or response.get('clear_result') == 'success' else 3
            predicted = round(belief.virtual_time + move + switch + operation, 6)
            max_cost_difference = max(max_cost_difference, abs(predicted - float(response['virtual_time_s'])))
            costs['move_s'] += move
            costs['switch_s'] += switch
            costs[
                'measure_s'
                if action.kind == 'measure'
                else 'clear_success_s'
                if operation == 5
                else 'clear_failure_s'
            ] += operation
            belief.apply(action, response, request_id)
            if max_cost_difference > 1e-4:
                raise RuntimeError('Official virtual cost differs from the model; stopping for review')
            if policy.records:
                with (folder / 'decisions.jsonl').open('a', encoding='utf-8') as log:
                    log.write(json.dumps(policy.records[-1], ensure_ascii=False) + '\n')
            if (index + 1) % 50 == 0:
                print(
                    json.dumps(
                        {
                            'case': case,
                            'actions': index + 1,
                            'cleared': len(belief.cleared),
                            'virtual_time_s': belief.virtual_time,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        if not belief.done():
            raise RuntimeError('Action limit reached without a complete public certificate')
        _, response = client.exit()
        summary['exited'] = response.get('exit_reason') == 'user_exit'
    except BaseException as exc:
        # No new request/exit is sent after an uncertain outcome or an error.
        summary['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        summary.update(
            certified_complete=belief.done(),
            cleared_count=len(belief.cleared),
            virtual_time_s=float(rows[-1]['response']['virtual_time_s']) if rows else belief.virtual_time,
            real_time_s=time.monotonic() - started,
            actions=len(rows),
            costs=costs,
            movement_m=5 * costs['move_s'],
            measure_count=int(costs['measure_s'] / 5),
            switch_count=int(costs['switch_s']),
            clear_failures=int(costs['clear_failure_s'] / 3),
            max_cost_difference_s=max_cost_difference,
            belief_implementation=type(belief).__name__,
        )
        summary.update(
            planning_time_s=sum(r['planning_s'] for r in rows),
            counters=policy.counters,
            channel_status={c: p.status for c, p in belief.channels.items()},
        )
        summary['average_per_cleared_s'] = (
            summary['virtual_time_s'] / len(belief.cleared) if belief.cleared else None
        )
        dump(folder / 'http-log.json', redact_log(client.log, robot_id))
        dump(folder / 'summary.json', summary)
        dump(folder / 'decisions.json', policy.records)
        session.close()
    if not summary['exited'] or summary['error']:
        raise RuntimeError(f'Practice incomplete; batch stopped. See {folder}/summary.json')
    time.sleep(0.5)
    items = inspect_ui()
    _, after_case = practice_identity(items)
    if after_case != case:
        raise RuntimeError('Completion screen belongs to a different case')
    dump(folder / 'practice-after.json', redact_ui(items))
    total = result_count(items)  # Evaluation only, after controller and /exit.
    summary.update(source_count=total, clearance_fraction=len(belief.cleared) / total if total else None)
    summary['official_result_verified'] = total is not None and total == len(belief.cleared)
    summary.update(result_composition(items))
    dump(folder / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if total is None or total != len(belief.cleared):
        raise RuntimeError('Practice total not verified or not all sources cleared; no next round')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connect', action='store_true')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--resume-ready-practice', action='store_true')
    args = parser.parse_args(argv)
    if not args.connect:
        print(json.dumps(redact_ui(inspect_ui()), ensure_ascii=False, indent=2))
        return
    if not 1 <= args.rounds <= 100:
        parser.error('rounds must be 1..100')
    out = new_output()
    config = Config()
    dump(
        out / 'config.json',
        {
            'mode': 'problem4_practice_only',
            'policy': 's21-probes',
            'rounds': args.rounds,
            'policy_config': asdict(config),
            'code_sha256': code_manifest(),
        },
    )
    manifest = {'status': 'running', 'planned_rounds': args.rounds, 'completed_rounds': 0}
    summaries = []
    dump(out / 'batch.json', manifest)
    print(f'Results: {out}', flush=True)
    try:
        for index in range(args.rounds):
            folder = out / f'practice-{index + 1}'
            folder.mkdir()
            items = start_practice(args.resume_ready_practice and index == 0)
            _, case = practice_identity(items, awaiting=True)
            dump(folder / 'practice-ready.json', redact_ui(items))
            summaries.append(run_once(folder, case, config))
            manifest['completed_rounds'] = len(summaries)
            dump(out / 'summary.json', summaries)
            dump(out / 'batch.json', manifest)
            print(f'Completed {len(summaries)}/{args.rounds} practice rounds', flush=True)
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest.update(status='stopped', error=f'{type(exc).__name__}: {exc}')
        try:
            dump(out / 'ui-on-stop.json', redact_ui(inspect_ui()))
        except Exception as ui_error:
            dump(out / 'ui-on-stop-error.json', {'error': str(ui_error)})
        raise
    finally:
        dump(out / 'batch.json', manifest)
        dump(out / 'summary.json', summaries)
        print(f'Results: {out}', flush=True)


if __name__ == '__main__':
    main()
