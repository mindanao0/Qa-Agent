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
