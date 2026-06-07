import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.universal_qa.models import TestCase, TestResult
from src.universal_qa.test_runner import UniversalTestRunner, _map_exception


def _make_tc(type_: str, url: str = "https://x.com") -> TestCase:
    return TestCase(
        title="Test", type=type_, priority="high",
        steps=["navigate", "check"], expected_outcome="ok",
        source_url=url,
    )


def test_map_exception_timeout():
    msg = _map_exception(TimeoutError("locator not found"))
    assert "ไม่พบ element" in msg


def test_map_exception_assertion():
    msg = _map_exception(AssertionError("expected True"))
    assert "ผลลัพธ์ไม่ตรงตามที่คาดหวัง" in msg


def test_map_exception_generic():
    msg = _map_exception(ValueError("something broke"))
    assert "something broke" in msg


@pytest.mark.asyncio
async def test_runner_routes_accessibility():
    runner = UniversalTestRunner.__new__(UniversalTestRunner)
    runner._screenshot_dir = None
    runner._terminal = MagicMock()
    tc = _make_tc("accessibility")

    with patch.object(runner, "_run_accessibility", new=AsyncMock(
        return_value=TestResult(test_case=tc, passed=True, duration_ms=100)
    )) as mock_a11y:
        result = await runner._dispatch(tc, AsyncMock())
    assert result.passed
    mock_a11y.assert_called_once()


@pytest.mark.asyncio
async def test_runner_returns_result_on_exception():
    runner = UniversalTestRunner.__new__(UniversalTestRunner)
    runner._screenshot_dir = None
    runner._terminal = MagicMock()
    tc = _make_tc("functional")

    with patch.object(runner, "_run_functional",
                      new=AsyncMock(side_effect=Exception("boom"))):
        result = await runner._dispatch(tc, AsyncMock())
    assert not result.passed
    assert result.failure_reason is not None
    assert "boom" in result.failure_reason
