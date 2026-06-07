import pytest
from src.universal_qa.models import TestCase, StepTrace, TestResult


def test_test_case_auto_id():
    tc = TestCase(
        title="Login with valid credentials",
        type="functional",
        priority="high",
        steps=["fill email", "fill password", "click login"],
        expected_outcome="redirected to dashboard",
        source_url="https://example.com/login",
    )
    assert len(tc.id) == 12
    assert tc.preconditions == []


def test_test_case_rejects_extra_field():
    with pytest.raises(Exception):
        TestCase(
            title="T", type="functional", priority="high",
            steps=["s"], expected_outcome="e",
            source_url="https://x.com", unknown_field="bad",
        )


def test_step_trace_default_error_is_none():
    st = StepTrace(step="click login", status="passed", detail="found by role")
    assert st.error is None


def test_test_result_failed_with_reason():
    tc = TestCase(
        title="XSS check", type="security", priority="high",
        steps=["inject payload"], expected_outcome="no execution",
        source_url="https://x.com",
    )
    result = TestResult(
        test_case=tc, passed=False,
        failure_reason="XSS payload executed", duration_ms=450,
    )
    assert not result.passed
    assert result.screenshot_path is None


def test_test_result_passed_no_failure_reason():
    tc = TestCase(
        title="T", type="accessibility", priority="low",
        steps=["check labels"], expected_outcome="all labels present",
        source_url="https://x.com",
    )
    result = TestResult(test_case=tc, passed=True, duration_ms=200)
    assert result.failure_reason is None
    assert result.steps_trace == []
