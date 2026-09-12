import json
import time

import pytest
import requests

import practice_windows as practice
from q4.policy import Config
from q4.shared import Action
from q4.simulator import LocalSimulator, generate_world


CASE = 'AB12-CD34-EF56-GH78'


def screen(title='问题4 演练 测试', state='尚未进入', code=CASE):
    return [{'id': 'test-run-title', 'name': title}, {'name': state},
            {'name': '队号 123456'}, {'name': code}]


@pytest.mark.parametrize('title', ['问题3演练测试', '问题3正式测试', '问题4正式测试', '演练测试', ''])
def test_non_q4_practice_titles_never_create_client(tmp_path, monkeypatch, title):
    monkeypatch.setattr(practice, 'inspect_ui', lambda: screen(title))
    monkeypatch.setattr(practice, 'load', lambda *a: pytest.fail('HTTP client must not be created'))
    with pytest.raises(RuntimeError):
        practice.run_once(tmp_path, CASE, Config())


@pytest.mark.parametrize('name', ['开始问题4正式测试', '开始问题3演练测试', '确认', '开始', '中止测试'])
def test_button_whitelist(name, monkeypatch):
    monkeypatch.setattr(practice, 'powershell', lambda *a: pytest.fail('No UI invocation allowed'))
    with pytest.raises(ValueError):
        practice.click_practice(name)


def test_conflicting_titles_changed_case_and_entered_rejected(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError):
        practice.practice_identity(screen()+[{'id': 'test-run-title', 'name': '问题4正式测试'}])
    with pytest.raises(RuntimeError):
        practice.practice_identity(screen(state='进行中'), awaiting=True)
    monkeypatch.setattr(practice, 'inspect_ui', screen)
    monkeypatch.setattr(practice, 'load', lambda *a: pytest.fail('No HTTP client allowed'))
    with pytest.raises(RuntimeError):
        practice.run_once(tmp_path, 'XX12-YY34-ZZ56-AA78', Config())


def test_no_start_over_other_test(monkeypatch):
    monkeypatch.setattr(practice, 'inspect_ui', lambda: screen('问题3演练测试', '测试已结束'))
    monkeypatch.setattr(practice, 'click_practice', lambda *a: pytest.fail('No click allowed'))
    with pytest.raises(RuntimeError):
        practice.start_practice()


def test_count_only_after_practice_exit():
    with pytest.raises(RuntimeError):
        practice.result_count(screen())
    assert practice.result_count(screen(state='测试已结束')+[
        {'name': '本次演练测试干扰源数量'}, {'name': '12'}]) == 12


@pytest.mark.parametrize('omni,directional', [(1, 11), (0, 12), (12, 0)])
def test_type_counts_are_evaluation_only_and_must_agree(omni, directional):
    items = screen(state='测试已结束')+[
        {'name': '本次演练测试干扰源数量'}, {'name': '12'},
        {'name': f'本次案例含干扰源12个，其中全向{omni}个、定向{directional}个。'}]
    assert practice.result_composition(items) == {'omnidirectional_count': omni, 'directional_count': directional}
    with pytest.raises(RuntimeError):
        practice.result_composition(screen())
    items[-1]['name'] = '本次案例含干扰源12个，其中全向2个、定向11个。'
    with pytest.raises(RuntimeError):
        practice.result_composition(items)


def test_webview_panes_retried_but_formal_still_rejected(monkeypatch):
    replies = iter([json.dumps([{'name': 'Web 内容', 'type': 'ControlType.Pane'}]),
                    json.dumps(screen('问题4正式测试'))])
    monkeypatch.setattr(practice, 'powershell', lambda *a: next(replies))
    monkeypatch.setattr(practice.time, 'sleep', lambda *a: None)
    with pytest.raises(RuntimeError):
        practice.practice_identity(practice.inspect_ui())


def test_permanent_type_panel_without_dialog_and_conflicting_panel():
    items = screen(state='测试已结束')+[
        {'name': s} for s in ('本次演练测试干扰源数量', '共', '12', '个， 全向', '0', '个， 定向', '12', '个')]
    assert practice.result_composition(items) == {'omnidirectional_count': 0, 'directional_count': 12}
    items.append({'name': '本次案例含干扰源12个，其中全向1个、定向11个。'})
    with pytest.raises(RuntimeError):
        practice.result_composition(items)


def response(body):
    result = requests.Response()
    result.status_code = 200
    result._content = json.dumps(body).encode()
    return result


@pytest.mark.parametrize('policy_name', ['mobile', 'adaptive', 'probes'])
def test_serial_practice_loop_against_local_protocol_with_retry(tmp_path, monkeypatch, policy_name):
    """No official service: exercise entry, public policy, same-ID retry and exit."""
    world = generate_world(2, 10)
    env = LocalSimulator(world)
    stage = {'entered': False, 'exited': False, 'retry_data': None, 'measure_posts': 0}
    deadline_before = time.monotonic()

    def ui():
        return screen(state='测试已结束')+[
            {'name': '本次演练测试干扰源数量'}, {'name': '10'}] if stage['exited'] else screen()

    def fake_post(session, url, **kwargs):
        payload = json.loads(kwargs['data'])
        kind = url.rsplit('/', 1)[-1]
        assert url.startswith('http://127.0.0.1:2026/')
        if kind == 'enter':
            assert not stage['entered']
            stage['entered'] = True
            return response({'accepted': True, 'virtual_time_s': 0,
                             'max_virtual_duration_s': 360000, 'remaining_real_duration_s': 1200})
        assert stage['entered'] and not stage['exited']
        if kind == 'exit':
            assert world.score()['all_cleared']
            stage['exited'] = True
            return response({'accepted': True, 'exit_reason': 'user_exit'})
        if kind == 'measure':
            stage['measure_posts'] += 1
            if stage['measure_posts'] == 1:
                stage['retry_data'] = kwargs['data']
                raise requests.Timeout('deliberate timeout')
            if stage['measure_posts'] == 2:
                assert kwargs['data'] == stage['retry_data']
        action = Action(kind, (payload['position']['x'], payload['position']['y']), payload['channel'])
        return response(env.execute(action, payload['request_id']))

    monkeypatch.setattr(practice, 'inspect_ui', ui)
    monkeypatch.setattr(requests.Session, 'post', fake_post)
    monkeypatch.setattr(practice.time, 'sleep', lambda *a: None)
    result = practice.run_once(tmp_path, CASE, Config(), policy_name)
    assert result['policy'] == policy_name
    assert result['official_result_verified'] and result['cleared_count'] == 10
    assert result['max_cost_difference_s'] < 1e-4
    assert result['certified_complete'] and result['exited']
    assert result['planning_time_s'] > 0 and time.monotonic() > deadline_before
    attempts = (tmp_path/'http-attempts.jsonl').read_text()
    assert 'deliberate timeout' in attempts and '123456' not in attempts
    assert result['server_deadline_utc']
