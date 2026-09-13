"""No real network or simulator UI. Same test contract in standalone Q3 and Q4."""

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
import requests

import start_formal as entry
import formal_session as formal

QUESTION = int(Path(entry.__file__).parent.name[1:])
CASE = 'AB12-CD34-EF56-GH78'
OTHER = 'ZZ12-CD34-EF56-GH78'
Client = entry.practice.load('client').HttpClient
Action = entry.practice.load('core').Action


def screen(title=None, code=CASE):
    return [
        {'id': 'test-run-title', 'name': title or f'问题{QUESTION} 正式 测试'},
        *({'name': s} for s in ('队号 123456', code, '尚未进入', '等待机器狗进入', '接口端口', '2026')),
    ]


@pytest.fixture(autouse=True)
def no_external_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Real network/UI must never be used by these tests')

    monkeypatch.setattr(requests.Session, 'post', forbidden)
    monkeypatch.setattr(entry.practice, 'powershell', forbidden)


def response(body):
    result = requests.Response()
    result.status_code = 200
    result._content = json.dumps(body).encode()
    return result


@dataclass
class Config:
    unchanged: bool = True


class MiniBelief:
    def __init__(self):
        self.position = (0.0, 0.0)
        self.receiver = 1
        self.virtual_time = 0.0
        self.channels = {1: SimpleNamespace(status='unresolved')}

    def done(self):
        return self.channels[1].status == 'cleared'

    def apply(self, action, result, request_id):
        self.virtual_time = result['virtual_time_s']
        self.position = action.position
        self.channels[1].status = 'cleared' if action.kind == 'clear' else 'detected'


class MiniPolicy:
    config = Config()
    records = []

    def choose(self, belief):
        return Action('clear' if belief.virtual_time else 'measure', (0.0, 0.0), 1)


def mini_build():
    return MiniPolicy(), MiniBelief(), Client


def install_transport(monkeypatch, fault=None, remaining=1200):
    state = dict(paths=[], action_posts=[], virtual=0.0)

    def post(session, url, **kwargs):
        kind = url.rsplit('/', 1)[-1]
        state['paths'].append(kind)
        assert url.startswith('http://127.0.0.1:2026/')
        if kind == 'enter':
            return response(
                dict(
                    accepted=True,
                    virtual_time_s=0,
                    real_timestamp_ms=1000,
                    remaining_real_duration_s=remaining,
                    max_virtual_duration_s=360000,
                )
            )
        if kind == 'exit':
            return response(
                dict(
                    accepted=True,
                    virtual_time_s=state['virtual'],
                    exit_reason='user_exit',
                    real_timestamp_ms=2000,
                )
            )
        state['action_posts'].append(kwargs['data'])
        if kind == 'clear' and fault:
            if fault == 'timeout':
                raise requests.Timeout('deliberate loss')
            if fault == 'interrupt':
                raise KeyboardInterrupt('deliberate interrupt')
            if fault == 'rejected':
                return response(dict(accepted=False, virtual_time_s=0))
            if fault == 'malformed':
                result = response({})
                result._content = b'not JSON'
                return result
            if fault == 'nan':
                return response(dict(accepted=True, virtual_time_s=float('nan'), clear_result='success'))
            if fault == 'cost':
                return response(dict(accepted=True, virtual_time_s=100, clear_result='success'))
        state['virtual'] += 5
        return response(
            dict(
                accepted=True,
                virtual_time_s=state['virtual'],
                **({'measure_result': 'near'} if kind == 'measure' else {'clear_result': 'success'}),
            )
        )

    monkeypatch.setattr(requests.Session, 'post', post)
    return state


@pytest.mark.parametrize(
    'title', ['问题3演练测试', '问题4演练测试', f'问题{7 - QUESTION}正式测试', '正式测试']
)
def test_wrong_page_never_creates_transport(tmp_path, monkeypatch, title):
    monkeypatch.setattr(formal, 'LoggedSession', lambda *a: pytest.fail('No transport allowed'))
    with pytest.raises(RuntimeError):
        formal.run_once(tmp_path, CASE, QUESTION, mini_build, lambda: screen(title))


@pytest.mark.parametrize('change', ['countdown', 'entered', 'port', 'conflict', 'case'])
def test_invalid_identity_never_connects(tmp_path, change):
    items = screen()
    if change == 'countdown':
        items.append({'name': '5秒倒计时'})
    elif change == 'entered':
        items = [i for i in items if i['name'] != '尚未进入']
    elif change == 'port':
        items[-1]['name'] = '2027'
    elif change == 'conflict':
        items.append({'id': 'test-run-title', 'name': '问题3演练测试'})
    else:
        items.append({'name': OTHER})
    with pytest.raises(RuntimeError):
        formal.run_once(tmp_path, CASE, QUESTION, mini_build, lambda: items)


def test_default_is_read_only_even_on_ready_page(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['start_formal.py'])
    assert (
        formal.main(QUESTION, tmp_path, lambda: pytest.fail('Default must not build or connect'), screen, 10)
        == 0
    )
    assert '"network_used": false' in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('args', [['--connect'], ['--connect', '--case', 'bad'], ['--rounds', '20']])
def test_explicit_case_required_and_no_batch_switch(tmp_path, monkeypatch, args):
    monkeypatch.setattr(sys, 'argv', ['start_formal.py', *args])
    with pytest.raises(SystemExit) as caught:
        formal.main(QUESTION, tmp_path, mini_build, lambda: pytest.fail('No UI needed'), 10)
    assert caught.value.code == 2


def test_changed_case_after_initialization_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['start_formal.py', '--connect', '--case', CASE])
    calls = iter([screen(), screen(code=OTHER)])
    with pytest.raises(RuntimeError):
        formal.main(QUESTION, tmp_path, mini_build, lambda: next(calls), 10)
    assert len(list(tmp_path.rglob('launcher-error.json'))) == 1
    assert not list(tmp_path.rglob('http-attempts.jsonl'))


def test_single_session_result_without_truth_count(tmp_path, monkeypatch):
    state = install_transport(monkeypatch)
    result = formal.run_once(tmp_path, CASE, QUESTION, mini_build, screen)
    assert state['paths'] == ['enter', 'measure', 'clear', 'exit']
    assert result['status'] == 'completed' and result['metric_finalized']
    assert result['cleared_count'] == 1 and result['average_per_cleared_s'] == 10
    assert result['server_program_time_s'] == 1
    assert (
        result['source_count']
        is result['clearance_fraction']
        is result['official_all_cleared_verified']
        is None
    )
    assert result['upload_status'] == 'not_checked' and not result['official_log_exported']
    assert '123456' not in (tmp_path / 'http-attempts.jsonl').read_text(encoding='utf-8')


def test_log_redaction_preserves_numeric_collisions_and_transport_payload(tmp_path, monkeypatch):
    body = dict(
        accepted=True,
        measure_result='near',
        virtual_time_s=12.12345,
        real_timestamp_ms=12345000,
        robot_id='12345',
        request_id='public-12345',
    )
    posted = []

    def post(session, url, **kwargs):
        posted.append(json.loads(kwargs['data']))
        return response(body)

    monkeypatch.setattr(requests.Session, 'post', post)
    with formal.LoggedSession(tmp_path, '12345') as session:
        client = Client('12345', session=session)
        _, actual = client.execute(Action('measure', (12.12345, 0.0), 1))
    assert actual == body
    assert posted[0]['robot_id'] == '12345'
    assert posted[0]['position']['x'] == 12.12345
    journal = [json.loads(line) for line in (tmp_path / 'http-attempts.jsonl').read_text().splitlines()]
    assert journal[0]['request']['robot_id'] == '<已隐去>'
    assert journal[0]['request']['position']['x'] == 12.12345
    logged_body = json.loads(journal[1]['body'])
    assert logged_body == {**body, 'robot_id': '<已隐去>'}
    safe_log = formal.redact_log(client.log, '12345')
    assert safe_log[0]['response'] == logged_body
    assert client.log[0]['request']['robot_id'] == '12345'


@pytest.mark.parametrize(
    'body',
    [
        ' { "virtual_time_s" : 12.12345, "request_id": "public-12345" } ',
        'malformed response at 12.12345; request public-12345',
    ],
)
def test_log_redaction_keeps_nonidentity_response_text(body):
    assert formal.redact_log({'body': body}, '12345') == {'body': body}


def test_malformed_response_redacts_labelled_identity_only():
    body = 'robot_id="12345"; invalid x=12.12345; 队号 12345'
    safe = formal.redact_log({'body': body}, '12345')['body']
    assert safe == 'robot_id="<已隐去>"; invalid x=12.12345; 队号 <已隐去>'


@pytest.mark.parametrize('fault', ['timeout', 'interrupt', 'rejected', 'malformed', 'nan', 'cost'])
def test_uncertain_or_invalid_feedback_stops_and_preserves_evidence(tmp_path, monkeypatch, fault):
    state = install_transport(monkeypatch, fault)
    result = formal.run_once(tmp_path, CASE, QUESTION, mini_build, screen)
    assert result['status'] == 'stopped' and not result['exited']
    assert 'exit' not in state['paths']
    assert result['average_per_cleared_s'] is None and not result['metric_finalized']
    assert result['virtual_time_s'] == (100 if fault == 'cost' else 5)
    if fault == 'timeout':
        assert len(state['action_posts']) == 4
        assert len(set(state['action_posts'][1:])) == 1
    assert json.loads((tmp_path / 'summary.json').read_text(encoding='utf-8'))['error']
    assert (tmp_path / 'http-log.json').is_file()


def test_small_remaining_budget_exits_without_action(tmp_path, monkeypatch):
    state = install_transport(monkeypatch, remaining=4)
    result = formal.run_once(tmp_path, CASE, QUESTION, mini_build, screen)
    assert state['paths'] == ['enter', 'exit']
    assert result['status'] == 'incomplete' and result['termination_reason'] == 'budget_limit'
    assert result['average_per_cleared_s'] is None


def test_unanswered_enter_is_unknown_not_claimed_unentered(tmp_path, monkeypatch):
    attempts = []

    def lost_enter(session, url, **kwargs):
        assert url.endswith('/enter')
        attempts.append(kwargs['data'])
        raise requests.Timeout('Entry response lost')

    monkeypatch.setattr(requests.Session, 'post', lost_enter)
    result = formal.run_once(tmp_path, CASE, QUESTION, mini_build, screen)
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert result['entered'] is None and result['status'] == 'stopped'
    assert not result['exited'] and result['average_per_cleared_s'] is None


def test_action_limit_reports_incomplete(tmp_path, monkeypatch):
    state = install_transport(monkeypatch)
    result = formal.run_once(tmp_path, CASE, QUESTION, mini_build, screen, max_actions=1)
    assert state['paths'] == ['enter', 'measure', 'exit']
    assert result['status'] == 'incomplete' and not result['algorithm_complete']


def test_budget_expiring_during_planning_sends_no_action(tmp_path, monkeypatch):
    state = install_transport(monkeypatch)

    class SlowPolicy(MiniPolicy):
        def choose(self, belief):
            belief.deadline = time.monotonic() - 1
            return super().choose(belief)

    result = formal.run_once(tmp_path, CASE, QUESTION, lambda: (SlowPolicy(), MiniBelief(), Client), screen)
    assert state['paths'] == ['enter', 'exit']
    assert result['status'] == 'incomplete'


def test_second_formal_process_cannot_take_same_port():
    with formal.process_lock(64329):
        with pytest.raises(RuntimeError):
            with formal.process_lock(64329):
                pytest.fail('Lock must not be acquired twice')


def test_transient_windows_file_lock_retries_only_local_rename(tmp_path, monkeypatch):
    original = formal.os.replace
    attempts = []

    def rename(source, target):
        attempts.append(source)
        if len(attempts) < 3:
            raise PermissionError('Transient file reader')
        original(source, target)

    monkeypatch.setattr(formal.os, 'replace', rename)
    formal.dump(tmp_path / 'summary.json', {'saved': True})
    assert len(attempts) == 3
    assert json.loads((tmp_path / 'summary.json').read_text()) == {'saved': True}


def test_real_core_sequence_unchanged_with_lost_response_and_no_truth_ui(tmp_path, monkeypatch):
    """Compare full public-feedback policy run with formal HTTP wrapper, fixed world."""
    if QUESTION == 3:
        simulator = entry.practice.load('simulator')
    else:
        import q4.simulator as simulator
    world = simulator.generate_world(93013, 10)
    plain_env = simulator.LocalSimulator(world.clone())
    policy, belief, _ = entry.build()
    expected = []
    for index in range(6000):
        if belief.done():
            break
        action = policy.choose(belief)
        expected.append(action)
        result = plain_env.execute(action, str(index))
        belief.apply(action, result, str(index))
    assert belief.done()
    env = simulator.LocalSimulator(world)
    posted = []
    seen = []
    lost = [False]

    def fake_post(session, url, **kwargs):
        kind = url.rsplit('/', 1)[-1]
        payload = json.loads(kwargs['data'])
        if kind == 'enter':
            return response(
                dict(
                    accepted=True,
                    virtual_time_s=0,
                    remaining_real_duration_s=1200,
                    max_virtual_duration_s=360000,
                )
            )
        if kind == 'exit':
            return response(dict(accepted=True, virtual_time_s=env.virtual_time, exit_reason='user_exit'))
        action = Action(kind, (payload['position']['x'], payload['position']['y']), payload['channel'])
        if payload['request_id'] not in seen:
            seen.append(payload['request_id'])
            posted.append(action)
        body = env.execute(action, payload['request_id'])
        if not lost[0]:
            lost[0] = kwargs['data']
            raise requests.Timeout('Response lost AFTER execution')
        if len(posted) == 1:
            assert kwargs['data'] == lost[0]
        return response(body)

    monkeypatch.setattr(requests.Session, 'post', fake_post)
    result = formal.run_once(tmp_path, CASE, QUESTION, entry.build, screen)
    assert result['status'] == 'completed', result['error']
    assert posted == expected  # Formal shell must not alter policy, points or parameters.
    assert result['virtual_time_s'] == plain_env.virtual_time
    assert result['cleared_count'] == 10 and world.score()['all_cleared']
    assert result['source_count'] is None  # Truth used only by the test oracle above.
    assert result['max_cost_difference_s'] < 1e-4
