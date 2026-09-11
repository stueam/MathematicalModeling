from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import practice_windows as practice


def test_only_problem_three_practice_title_is_accepted():
    assert practice.is_practice_running_screen([{'id': 'test-run-title', 'name': '问题3 演练 测试'}])
    for title in ('问题3正式测试', '问题4演练测试', '问题4正式测试', '演练测试'):
        assert not practice.is_practice_running_screen([{'id': 'test-run-title', 'name': title}])


@pytest.mark.parametrize('button', ['开始问题3正式测试', '开始问题4演练测试', '确认', '开始测试'])
def test_formal_or_ambiguous_buttons_cannot_be_invoked(button, monkeypatch):
    monkeypatch.setattr(practice, 'powershell', lambda _: pytest.fail('Must reject before calling Windows'))
    with pytest.raises(ValueError):
        practice.click_practice(button)


def test_formal_screen_blocks_enter_before_constructing_http_client(monkeypatch, tmp_path):
    monkeypatch.setattr(practice, 'inspect_ui', lambda: [{'id': 'test-run-title', 'name': '问题3正式测试'}])
    monkeypatch.setattr(practice, 'HttpClient', lambda *a, **kw: pytest.fail('No client on formal screen'))
    with pytest.raises(RuntimeError, match='refusing /enter'):
        practice.run_once(practice.Config(workers=1), tmp_path, [])


def test_transient_null_ui_names_are_filtered(monkeypatch):
    monkeypatch.setattr(practice, 'powershell', lambda _: '[{"name":null},{"name":"问题3 演练 测试","id":"test-run-title"}]')
    assert practice.is_practice_running_screen(practice.inspect_ui())
    assert practice.redact_ui([{'name': None}])[0]['name'] == ''


def test_resume_cannot_take_over_entered_or_formal_test(monkeypatch):
    for items in ([{'id': 'test-run-title', 'name': '问题3演练测试'}],
                  [{'id': 'test-run-title', 'name': '问题3正式测试'}, {'name': '尚未进入'}]):
        monkeypatch.setattr(practice, 'inspect_ui', lambda: items)
        with pytest.raises(RuntimeError, match='Resume requires'):
            practice.start_practice(resume_ready=True)
