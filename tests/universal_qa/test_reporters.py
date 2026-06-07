import pytest
from src.universal_qa.models import TestCase, StepTrace, TestResult
from src.universal_qa.reporters.terminal import TerminalReporter


def _make_result(title: str, passed: bool, type_: str = "functional",
                 failure_reason: str | None = None) -> TestResult:
    tc = TestCase(
        title=title, type=type_, priority="high",
        steps=["step"], expected_outcome="ok",
        source_url="https://example.com",
    )
    traces = [
        StepTrace(step="step", status="passed" if passed else "failed",
                  detail="detail", error=None if passed else failure_reason),
    ]
    return TestResult(
        test_case=tc, passed=passed, steps_trace=traces,
        failure_reason=failure_reason, duration_ms=1200,
    )


def test_terminal_reporter_formats_pass(capsys):
    reporter = TerminalReporter()
    result = _make_result("Login success", passed=True)
    reporter.report_one(result)
    captured = capsys.readouterr()
    assert "PASS" in captured.out
    assert "Login success" in captured.out
    assert "1.2s" in captured.out


def test_terminal_reporter_formats_fail_with_reason(capsys):
    reporter = TerminalReporter()
    result = _make_result("Login fail", passed=False,
                          failure_reason="Element not found within 30s")
    reporter.report_one(result)
    captured = capsys.readouterr()
    assert "FAIL" in captured.out
    assert "Element not found within 30s" in captured.out


def test_terminal_reporter_summary(capsys):
    reporter = TerminalReporter()
    results = [
        _make_result("T1", passed=True),
        _make_result("T2", passed=False, failure_reason="timeout"),
        _make_result("T3", passed=True),
    ]
    reporter.report_summary(results)
    captured = capsys.readouterr()
    assert "2" in captured.out   # passed count
    assert "1" in captured.out   # failed count


from src.universal_qa.reporters.html import HTMLReporter
import pathlib


def test_html_reporter_creates_file(tmp_path):
    reporter = HTMLReporter(output_dir=tmp_path)
    results = [
        _make_result("Login test", passed=True),
        _make_result("XSS test", passed=False, type_="security",
                     failure_reason="XSS payload executed"),
    ]
    path = reporter.generate(results)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Login test" in content
    assert "XSS test" in content
    assert "XSS payload executed" in content


def test_html_reporter_is_self_contained(tmp_path):
    reporter = HTMLReporter(output_dir=tmp_path)
    path = reporter.generate([_make_result("T", passed=True)])
    content = path.read_text(encoding="utf-8")
    assert "<style>" in content
    assert 'src="http' not in content  # no external http/https src links


def test_html_reporter_has_filter_controls(tmp_path):
    reporter = HTMLReporter(output_dir=tmp_path)
    path = reporter.generate([_make_result("T", passed=True)])
    content = path.read_text(encoding="utf-8")
    assert "functional" in content
    assert "accessibility" in content
    assert "security" in content
