import pytest
import practice_windows as practice
from bayes_tsp.policy import Config


def screen(title='问题3 演练 测试', state='尚未进入', code='AB12-CD34-EF56-GH78'):
    return [{'id': 'test-run-title', 'name': title}, {'name': state}, {'name': '队号 123456'}, {'name': code}]


@pytest.mark.parametrize('title', ['问题3正式测试', '问题4演练测试', '问题4正式测试', '演练测试', ''])
def test_non_practice_or_ambiguous_titles_are_rejected(title):
    with pytest.raises(RuntimeError):
        practice.practice_identity(screen(title), awaiting=True)


@pytest.mark.parametrize('name', ['开始问题3正式测试', '确认', '开始', '问题3正式测试', '中止测试'])
def test_only_explicit_practice_buttons_are_callable(name, monkeypatch):
    monkeypatch.setattr(practice, 'powershell', lambda *a: pytest.fail('Must reject before UI invocation'))
    with pytest.raises(ValueError):
        practice.click_practice(name)


def test_entered_practice_is_not_adopted():
    with pytest.raises(RuntimeError):
        practice.practice_identity(screen(state='进行中'), awaiting=True)


def test_final_guard_precedes_http_client_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(practice, 'inspect_ui', lambda: screen('问题3正式测试'))
    monkeypatch.setattr(practice, 'load', lambda *a: pytest.fail('Must not create HTTP client'))
    with pytest.raises(RuntimeError):
        practice.run_once(tmp_path, 'AB12-CD34-EF56-GH78', Config())


def test_changed_case_is_rejected_before_enter(tmp_path, monkeypatch):
    monkeypatch.setattr(practice, 'inspect_ui', screen)
    monkeypatch.setattr(practice, 'load', lambda *a: pytest.fail('Must not create HTTP client'))
    with pytest.raises(RuntimeError):
        practice.run_once(tmp_path, 'XX12-YY34-ZZ56-AA78', Config())


def test_completed_count_is_only_read_on_practice_end_screen():
    items = screen(state='测试已结束') + [{'name': '本次演练测试干扰源数量'}, {'name': '12'}]
    assert practice.result_count(items) == 12
    with pytest.raises(RuntimeError):
        practice.result_count(screen())


def test_blank_snapshot_retried_but_formal_snapshot_is_not_accepted(monkeypatch):
    import json

    replies = iter(['[]', json.dumps(screen('问题3正式测试'))])
    monkeypatch.setattr(practice, 'powershell', lambda *a: next(replies))
    monkeypatch.setattr(practice.time, 'sleep', lambda *a: None)
    items = practice.inspect_ui()
    with pytest.raises(RuntimeError):
        practice.practice_identity(items, awaiting=True)
