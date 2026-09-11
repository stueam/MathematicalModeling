"""Run only problem-three PRACTICE through the visible Windows simulator UI.

No formal-test controls, hidden simulator files, or unpublished endpoints are
used. The UI title is verified again immediately before the first /enter.
"""
import argparse
import base64
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time

_spec = importlib.util.spec_from_file_location('q3_algorithm_two_runner', Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)
from q3.client import HttpClient
from pomcp.belief import ParticleBelief
from pomcp.config import Config
from pomcp.planner import Planner
import requests

POWERSHELL = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
UI = r'''
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new()
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class Q3Window { [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h); [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h,int n); }'
$p=Get-Process -Name 'jammers-simulator' | Where-Object {$_.MainWindowHandle -ne 0} | Select-Object -First 1
if (!$p) {throw 'Simulator window not found'}
[Q3Window]::ShowWindow($p.MainWindowHandle,9) | Out-Null
[Q3Window]::SetForegroundWindow($p.MainWindowHandle) | Out-Null
Start-Sleep -Milliseconds 200
$root=[System.Windows.Automation.AutomationElement]::FromHandle($p.MainWindowHandle)
'''
READ = r'''
$all=$root.FindAll([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.Condition]::TrueCondition)
$items=for ($i=0;$i -lt $all.Count;$i++) {
 $c=$all.Item($i).Current
 if ($c.Name -ne '') { [pscustomobject]@{name=$c.Name;type=$c.ControlType.ProgrammaticName;id=$c.AutomationId;enabled=$c.IsEnabled} }
}
ConvertTo-Json -InputObject @($items) -Depth 3 -Compress
'''


def powershell(script):
    encoded = base64.b64encode((UI+script).encode('utf-16-le')).decode('ascii')
    result = subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
                            capture_output=True, timeout=20)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace')[:1500])
    return result.stdout.decode('utf-8-sig').strip()


def inspect_ui():
    data = json.loads(powershell(READ))
    data = [i for i in data if isinstance(i.get('name'), str) and i['name']]
    if not data:
        raise RuntimeError('No visible simulator controls; no test action was sent')
    return data


def is_practice_running_screen(items):
    return any(i.get('id') == 'test-run-title' and re.sub(r'\s+', '', i.get('name') or '') == '问题3演练测试'
               for i in items)


def click_practice(name):
    # Never accept an arbitrary label or a generic confirmation button.
    if name not in ('开始问题3演练测试', '返回演练测试'):
        raise ValueError('Only explicit problem-three practice controls are allowed')
    guard_id, guard_name = ('practice-3-title', '问题3演练测试') if name.startswith('开始') else ('test-run-title', '问题3演练测试')
    script = r'''
$guard=$root.FindFirst([System.Windows.Automation.TreeScope]::Descendants,[System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::AutomationIdProperty,'GUARD_ID'))
if (!$guard -or (($guard.Current.Name -replace '\s','') -ne 'GUARD_NAME')) {throw 'Practice screen guard failed'}
$cond=[System.Windows.Automation.AndCondition]::new(
 [System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::NameProperty,'BUTTON_NAME'),
 [System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::ControlTypeProperty,[System.Windows.Automation.ControlType]::Button))
$button=$root.FindFirst([System.Windows.Automation.TreeScope]::Descendants,$cond)
if (!$button -or !$button.Current.IsEnabled) {throw 'Practice control is unavailable'}
$button.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
'practice-only button invoked'
'''
    return powershell(script.replace('GUARD_ID', guard_id).replace('GUARD_NAME', guard_name).replace('BUTTON_NAME', name))


def redact_ui(items):
    return [dict(i, name=re.sub(r'队号\s+\d+', '队号 <已隐去>', i.get('name') or '')) for i in items]


def start_practice(resume_ready=False):
    items = inspect_ui()
    if resume_ready:
        if not is_practice_running_screen(items) or not any(i['name'] == '尚未进入' for i in items):
            raise RuntimeError('Resume requires a verified problem-three practice with no prior /enter')
        return items
    if is_practice_running_screen(items):
        if not any(i['name'] == '测试已结束' for i in items):
            raise RuntimeError('A practice is already active; refusing to disturb an existing run')
        click_practice('返回演练测试')
        time.sleep(.3)
    click_practice('开始问题3演练测试')
    deadline = time.monotonic()+45
    while time.monotonic() < deadline:
        items = inspect_ui()
        if is_practice_running_screen(items) and any(i['name'] == '2026' for i in items):
            if any(re.search(r'接口.*就绪|等待机器狗进入|等待进入', i['name']) for i in items):
                return items
        # Do not click any modal confirmations or other test controls.
        time.sleep(1)
    raise RuntimeError('Practice readiness was not established; no /enter sent. Inspect the saved UI.')


def run_once(cfg, folder, items):
    # Fresh UI check immediately before making the first mutating HTTP call.
    items = inspect_ui()
    if not is_practice_running_screen(items):
        raise RuntimeError('Not a problem-three practice; refusing /enter')
    if not any(i['name'] == '尚未进入' for i in items):
        raise RuntimeError('Practice is not awaiting its first /enter')
    names = [i['name'] for i in items]
    robot_ids = [m.group(1) for s in names if (m := re.fullmatch(r'队号\s+(\d+)', s))]
    codes = [s for s in names if re.fullmatch(r'[A-Z0-9]{4}(?:-[A-Z0-9]{4}){3}', s)]
    if len(robot_ids) != 1 or len(codes) != 1:
        raise RuntimeError('Practice account/case not unambiguously visible')
    runner.dump(folder/'practice-before.json', redact_ui(items))
    session = requests.Session()
    session.trust_env = False  # Windows/WSL mirrored loopback; never send through proxies.
    client = HttpClient(robot_ids[0], 'http://127.0.0.1:2026', session=session)
    b, summary = ParticleBelief(cfg.particles, cfg.seed), {}
    started = time.monotonic()
    try:
        _, entered = client.enter()
        b.deadline = client.deadline-5
        b.virtual_limit = float(entered['max_virtual_duration_s'])
        b.virtual_time = float(entered['virtual_time_s'])
        print(json.dumps({'case': codes[0], 'mode': 'problem3_practice', 'entered': True,
                          'remaining_real_s': entered['remaining_real_duration_s']}, ensure_ascii=False), flush=True)
        summary, _ = runner.control(b, Planner(cfg), lambda a, _: client.execute(a), folder, 4000)
        summary.update(exited=False, mode='problem3_practice', case_code=codes[0], cleared_count=len(b.cleared))
        if b.done() and summary['error'] is None and time.monotonic() < client.deadline:
            _, exited = client.exit()
            summary['exited'] = exited.get('exit_reason') == 'user_exit'
        summary['real_time_s_including_enter_exit'] = time.monotonic()-started
    except Exception as exc:
        summary.update(error=f'{type(exc).__name__}: {exc}', certified_complete=b.done(),
                       cleared_count=len(b.cleared), virtual_time_s=b.virtual_time)
    finally:
        # Practice logs remain useful for replay, but submission material must
        # not contain a hardcoded team number. Redact only our own JSON logs.
        for row in client.log:
            if 'request' in row:
                row['request']['robot_id'] = '<参赛队号已隐去>'
        runner.dump(folder/'http-log.json', client.log)
        runner.dump(folder/'summary.json', summary)
    if not summary.get('exited'):
        raise RuntimeError(f'Practice did not complete cleanly: {summary}; no new practice started')
    time.sleep(.5)
    items = inspect_ui()
    runner.dump(folder/'practice-after.json', redact_ui(items))
    names = [i['name'] for i in items]
    try:
        start = names.index('本次演练测试干扰源数量')
        total = next(int(s) for s in names[start+1:] if re.fullmatch(r'\d+', s))
    except (ValueError, StopIteration):
        total = None
    summary.update(source_count=total, clearance_fraction=len(b.cleared)/total if total else None,
                   average_per_cleared_s=b.virtual_time/len(b.cleared) if b.cleared else None)
    runner.dump(folder/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--connect', action='store_true', help='Start and run only problem-three practice')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--compute-scale', type=int, default=1,
                   help='Multiply particle/candidate/world/depth/planning budgets; never physical rules or official limits')
    p.add_argument('--resume-ready-practice', action='store_true', help='Adopt only a verified practice awaiting its first /enter')
    args = p.parse_args()
    if not args.connect:
        print(json.dumps(redact_ui(inspect_ui()), ensure_ascii=False, indent=2))
        return
    cfg = Config(workers=args.workers).scaled(args.compute_scale)
    out = Path(__file__).parent/'results'/datetime.now().strftime('official-practice-%Y%m%d-%H%M%S')
    out.mkdir()
    runner.dump(out/'config.json', {'mode': 'problem3_practice_only', 'planner': asdict(cfg),
                                  'rounds': args.rounds, 'compute_scale': args.compute_scale})
    root = Path(__file__).parent
    sources = list(root.glob('*.py'))+list((root/'pomcp').glob('*.py'))
    sources += list((root.parent/'第三问-算法1/q3').glob('*.py'))
    runner.dump(out/'source-hashes.json', {str(f.relative_to(root.parent)): hashlib.sha256(f.read_bytes()).hexdigest()
                                          for f in sources})
    results = []
    try:
        for index in range(args.rounds):
            folder = out/f'practice-{index+1}'
            folder.mkdir()
            items = start_practice(args.resume_ready_practice and index == 0)
            results.append(run_once(cfg, folder, items))
            runner.dump(out/'summary.json', results)
    except Exception as exc:
        runner.dump(out/'stopped.json', {'error': str(exc)})
        try:
            runner.dump(out/'ui-on-stop.json', redact_ui(inspect_ui()))
        except Exception as ui_error:
            runner.dump(out/'ui-on-stop-error.json', {'error': str(ui_error)})
        print(f'STOPPED: {exc}\nResults: {out}', flush=True)
        raise SystemExit(1)
    print(f'Results: {out}', flush=True)


if __name__ == '__main__':
    main()
