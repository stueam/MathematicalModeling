"""One already-open formal session. No UI clicks, test creation or truth lookup.

This transport shell is duplicated byte-for-byte in Q3 and Q4 so either package
can be copied independently. Policy, belief and HTTP client are injected.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import requests

CASE = re.compile(r'[A-Z0-9]{4}(?:-[A-Z0-9]{4}){3}')


def json_safe(value):
    # Preserve invalid protocol values as text in evidence, never as JSON NaN.
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def dump(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    # Windows indexers/antivirus may briefly hold an otherwise valid destination.
    # Retry only the local rename, never a robot action, for at most 0.62 seconds.
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.02 * 2**attempt)


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def redact(items):
    return [dict(i, name=re.sub(r'队号\s+\d+', '队号 <已隐去>', i['name'])) for i in items]


def identity(items, question, port, awaiting=True):
    titles = [''.join(i['name'].split()) for i in items if i.get('id') == 'test-run-title']
    if not titles or any(t != f'问题{question}正式测试' for t in titles):
        raise RuntimeError(f'Expected problem {question} FORMAL page; no request sent')
    names = [i['name'] for i in items]
    accounts = {m.group(1) for s in names if (m := re.fullmatch(r'队号\s+(\d+)', s))}
    cases = {s for s in names if CASE.fullmatch(s)}
    if len(accounts) != 1 or len(cases) != 1:
        raise RuntimeError('Account/case is not unambiguously visible')
    if awaiting:
        if '尚未进入' not in names or not any(re.search(r'接口.*就绪|等待机器狗进入|等待进入', s) for s in names):
            raise RuntimeError('Formal session is not ready and unentered')
        if any(s in names for s in ('测试已结束', '5秒倒计时', '正在准备测试数据')):
            raise RuntimeError('Formal session is not open')
        ports = [names[i+1] for i, s in enumerate(names[:-1]) if s == '接口端口']
        if not ports or any(s != str(port) for s in ports):
            raise RuntimeError('Configured port differs from the visible simulator port')
    return next(iter(accounts)), next(iter(cases))


@contextmanager
def process_lock(port):
    """Prevent two cooperating Q3/Q4 formal launchers using the same endpoint."""
    path = Path(tempfile.gettempdir())/f'cumcm-formal-robot-{port}.lock'
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another formal launcher is running on this port') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class LoggedSession(requests.Session):
    def __init__(self, folder, robot_id):
        super().__init__()
        self.trust_env = False
        self.folder, self.robot_id = folder, robot_id

    def record(self, value):
        value['recorded_at_utc'] = datetime.now(timezone.utc).isoformat()
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).replace(self.robot_id, '<已隐去>')
        append(self.folder/'http-attempts.jsonl', json.loads(encoded))

    def post(self, url, **kwargs):
        payload = json.loads(kwargs['data'])
        self.record(dict(event='before_request', url=url, request=payload))
        try:
            response = super().post(url, **kwargs)
        except BaseException as exc:
            self.record(dict(event='request_error', request_id=payload['request_id'], error=f'{type(exc).__name__}: {exc}'))
            raise
        self.record(dict(event='response', request_id=payload['request_id'], http_status=response.status_code, body=response.text))
        return response


def validate_action(action):
    if action.kind not in ('measure', 'clear') or type(action.channel) is not int or not 1 <= action.channel <= 20:
        raise ValueError('Invalid action type or channel')
    if len(action.position) != 2 or any(not math.isfinite(x) or abs(x) > 2000000 for x in action.position):
        raise ValueError('Invalid action position')


def run_once(folder, expected_case, question, build, inspect_ui, port=2026, max_actions=6000):
    # Initialize before the final fresh UI check; policy setup spends no session time.
    policy, belief, client_type = build()
    items = inspect_ui()
    robot_id, case = identity(items, question, port)
    if case != expected_case:
        raise RuntimeError('Case changed or does not match --case; no request sent')
    dump(folder/'ui-before.json', redact(items))
    dump(folder/'policy.json', dict(policy_class=type(policy).__name__, belief_class=type(belief).__name__,
        config=asdict(policy.config), implementation=getattr(policy, 'implementation', type(policy).__name__),
        station_count=len(policy.points) if hasattr(policy, 'points') else None))
    summary = dict(question=question, mode='formal_single', case_code=case, status='starting',
        entered=False, exited=False, algorithm_complete=False, official_all_cleared_verified=None,
        source_count=None, clearance_fraction=None, cleared_count=0, virtual_time_s=0.,
        average_per_cleared_s=None, actions=0, error=None, warnings=[],
        upload_status='not_checked', official_log_exported=False,
        started_at_utc=datetime.now(timezone.utc).isoformat())
    dump(folder/'summary.json', summary)
    session = LoggedSession(folder, robot_id)
    client = client_type(robot_id, f'http://127.0.0.1:{port}', session=session)
    started = time.monotonic()
    cleared = set()
    last_virtual = 0.
    entry_stamp = None
    exit_stamp = None
    costs = dict(move_s=0., switch_s=0., measure_s=0., clear_success_s=0., clear_failure_s=0.)
    max_error = 0.
    try:
        # An unanswered /enter may already have succeeded on the server.
        summary['entered'] = None
        dump(folder/'summary.json', summary)
        _, entry = client.enter()
        summary['entered'] = True
        dump(folder/'enter.json', entry)
        remaining, limit = float(entry['remaining_real_duration_s']), float(entry['max_virtual_duration_s'])
        if not (math.isfinite(remaining) and 0 < remaining <= 1200 and math.isfinite(limit) and 0 < limit <= 360000 and entry['virtual_time_s'] == 0):
            raise RuntimeError('Invalid entry budgets or nonzero initial time')
        summary['remaining_real_duration_s'] = remaining
        entry_stamp = entry.get('real_timestamp_ms')
        belief.deadline, belief.virtual_limit, belief.virtual_time = client.deadline-5, limit, 0.
        reason = 'action_limit'
        for index in range(max_actions):
            if belief.done():
                reason = 'algorithm_complete'
                break
            if time.monotonic() >= belief.deadline or belief.virtual_time >= limit:
                reason = 'budget_limit'
                break
            tick = time.monotonic()
            action = policy.choose(belief)
            planning = time.monotonic()-tick
            if time.monotonic() >= belief.deadline:
                reason = 'budget_limit'
                break
            validate_action(action)
            move = round(math.dist(belief.position, action.position)/5, 6)
            switch = int(action.kind == 'measure' and action.channel != belief.receiver)
            request_id, response = client.execute(action)
            append(folder/'actions.jsonl', dict(request_id=request_id, action=asdict(action), response=response, planning_s=planning))
            summary['actions'] += 1
            operation = 5 if action.kind == 'measure' or response.get('clear_result') == 'success' else 3
            observed_virtual = float(response['virtual_time_s'])
            if not math.isfinite(observed_virtual) or observed_virtual < last_virtual:
                raise RuntimeError('Invalid virtual time')
            last_virtual = observed_virtual
            if action.kind == 'clear' and response.get('clear_result') == 'success':
                if action.channel in cleared:
                    raise RuntimeError('Duplicate successful clear')
                cleared.add(action.channel)
            max_error = max(max_error, abs(round(belief.virtual_time+move+switch+operation, 6)-last_virtual))
            costs['move_s'] += move
            costs['switch_s'] += switch
            costs['measure_s' if action.kind == 'measure' else 'clear_success_s' if operation == 5 else 'clear_failure_s'] += operation
            belief.apply(action, response, request_id)
            if max_error > 1e-4:
                raise RuntimeError('Virtual cost differs from the verified model')
            if policy.records:
                append(folder/'decisions.jsonl', policy.records[-1])
            summary.update(status='running', cleared_count=len(cleared), virtual_time_s=last_virtual,
                           last_known_average_per_cleared_s=last_virtual/len(cleared) if cleared else None)
            dump(folder/'summary.json', summary)
            if (index+1) % 50 == 0:
                print(json.dumps(dict(case=case, actions=index+1, cleared=len(cleared), virtual_time_s=last_virtual)), flush=True)
        if belief.done():
            reason = 'algorithm_complete'
        summary['termination_reason'] = reason
        # All preceding outcomes are known here. No /exit is sent from an error handler.
        if time.monotonic() < client.deadline and last_virtual < limit:
            _, response = client.exit()
            dump(folder/'exit.json', response)
            exit_virtual = float(response['virtual_time_s'])
            if response.get('exit_reason') != 'user_exit' or not math.isfinite(exit_virtual) or abs(exit_virtual-last_virtual) > 1e-4:
                raise RuntimeError('Unexpected exit response')
            summary['exited'] = True
            exit_stamp = response.get('real_timestamp_ms')
        summary['status'] = 'completed' if reason == 'algorithm_complete' and summary['exited'] else 'incomplete'
    except BaseException as exc:
        summary.update(status='stopped', error=f'{type(exc).__name__}: {exc}')
    finally:
        summary.update(algorithm_complete=belief.done(), cleared_count=len(cleared),
            cleared_channels=sorted(cleared), virtual_time_s=last_virtual,
            metric_finalized=summary['exited'],
            average_per_cleared_s=last_virtual/len(cleared) if cleared and summary['exited'] else None,
            last_known_average_per_cleared_s=last_virtual/len(cleared) if cleared else None,
            client_elapsed_s=time.monotonic()-started,
            server_program_time_s=(exit_stamp-entry_stamp)/1000 if entry_stamp is not None and exit_stamp is not None else None,
            max_cost_difference_s=max_error, costs=costs, movement_m=5*costs['move_s'],
            ended_at_utc=datetime.now(timezone.utc).isoformat(),
            completion_evidence=dict(belief_done=belief.done(), channel_status={str(c):p.status for c,p in belief.channels.items()}))
        dump(folder/'summary.json', summary)
        safe_log=json.loads(json.dumps(client.log, ensure_ascii=False).replace(robot_id, '<已隐去>'))
        dump(folder/'http-log.json', safe_log)
        session.close()
    # Post-run UI is evidence only. Never read hidden source counts or send another request.
    try:
        after = inspect_ui()
        dump(folder/'ui-after.json', redact(after))
        _, after_case = identity(after, question, port, awaiting=False)
        if after_case != case:
            raise RuntimeError('Post-run page belongs to a different case')
    except Exception as exc:
        summary['warnings'].append(f'Post-run UI not verified: {exc}')
    dump(folder/'summary.json', summary)
    print(json.dumps(dict(case=case, status=summary['status'], cleared=len(cleared), average_per_cleared_s=summary['average_per_cleared_s'], output=str(folder)), ensure_ascii=False), flush=True)
    return summary


def main(question, root, build, inspect_ui, max_actions):
    parser=argparse.ArgumentParser(description=f'Q{question} formal ONE session; default reads UI only. Never starts a test.')
    parser.add_argument('--connect', action='store_true')
    parser.add_argument('--case', help='Exact currently open formal case code; required with --connect')
    parser.add_argument('--port', type=int, default=2026)
    parser.add_argument('--output', type=Path, help='Parent directory for a new case folder')
    args=parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Invalid port')
    if args.connect and (not args.case or not CASE.fullmatch(args.case)):
        parser.error('--connect requires --case XXXX-XXXX-XXXX-XXXX')
    if not args.connect:
        try:
            items=inspect_ui()
            _, case=identity(items, question, args.port)
            print(json.dumps(dict(network_used=False, ready=True, question=question, case=case)))
        except Exception as exc:
            print(json.dumps(dict(network_used=False, ready=False, reason=str(exc))))
        return 0
    with process_lock(args.port):
        items=inspect_ui()
        _, case=identity(items,question,args.port)
        if case != args.case:
            raise RuntimeError('Visible case differs from --case; no request sent')
        parent=args.output or root/'results'/'formal'
        folder=parent/(case+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        folder.mkdir(parents=True, exist_ok=False)
        files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.json')
               and not any(s in p.relative_to(root).parts for s in ('results','evidence','tests','__pycache__'))]
        dump(folder/'config.json',dict(question=question, mode='formal_single', case=case, port=args.port,
            python=sys.version, code_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
        print(f'Results: {folder}',flush=True)
        try:
            result=run_once(folder,case,question,build,inspect_ui,args.port,max_actions)
        except Exception as exc:
            # Includes preparation failures and unrecoverable local I/O failures.
            # Do not infer that no HTTP was sent; the attempt journal is evidence.
            dump(folder/'launcher-error.json', dict(error=f'{type(exc).__name__}: {exc}'))
            raise
        return 0 if result['status']=='completed' else 2
