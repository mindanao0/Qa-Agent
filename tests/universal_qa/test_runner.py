import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.universal_qa.models import TestCase, TestResult
from src.universal_qa.test_runner import UniversalTestRunner, _map_exception, _normalize_steps


def _make_tc(type_: str, url: str = "https://x.com") -> TestCase:
    return TestCase(
        title="Test", type=type_, priority="high",
        steps=["navigate", "check"], expected_outcome="ok",
        source_url=url,
    )


# ─── _normalize_steps ────────────────────────────────────────────────────────

def test_normalize_translates_thai_nav():
    result = _normalize_steps(["เปิดหน้า https://x.com/login"], "https://x.com/login")
    assert result == ["navigate to https://x.com/login"]


def test_normalize_translates_thai_fill():
    result = _normalize_steps(["กรอก \"Username\" ด้วย user"], "https://x.com/login")
    assert any("fill" in s for s in result)


def test_normalize_translates_thai_click():
    result = _normalize_steps(["คลิกปุ่ม \"Login\""], "https://x.com/login")
    assert any("click" in s for s in result)


def test_normalize_skips_login_click_on_post_login_page():
    """click 'Login' บน inventory page ต้องถูกตัดออก"""
    steps = ["คลิกปุ่ม \"Login\""]
    result = _normalize_steps(steps, "https://www.saucedemo.com/inventory.html")
    assert result == []


def test_normalize_keeps_login_click_on_login_page():
    """click 'Login' บนหน้า /login ต้องผ่าน"""
    steps = ["คลิกปุ่ม \"Login\""]
    result = _normalize_steps(steps, "https://example.com/login")
    assert any("click" in s for s in result)


def test_normalize_skips_fill_username_on_inventory():
    """fill 'Username' บน inventory page ต้องถูกตัดออก"""
    steps = ["กรอก \"Username\" ด้วย user"]
    result = _normalize_steps(steps, "https://www.saucedemo.com/inventory.html")
    assert result == []


def test_normalize_skips_fill_password_on_cart():
    """fill 'Password' บน cart page ต้องถูกตัดออก"""
    steps = ["fill \"Password\" with secret"]
    result = _normalize_steps(steps, "https://shop.com/cart")
    assert result == []


def test_normalize_converts_verify_to_verify_text():
    """ตรวจสอบ ... → verify text: 'quoted'"""
    steps = ["ตรวจสอบว่า \"Welcome\" แสดงขึ้นมา"]
    result = _normalize_steps(steps, "https://x.com")
    assert result == ['verify text: "Welcome"']


def test_normalize_drops_verify_without_quoted_text():
    """ตรวจสอบ ... ที่ไม่มี quoted text → ตัดออก"""
    steps = ["ตรวจสอบว่าหน้าโหลดได้"]
    result = _normalize_steps(steps, "https://x.com")
    assert result == []


def test_normalize_skips_click_path():
    steps = ['click "/login"']
    result = _normalize_steps(steps, "https://x.com")
    assert result == []


def test_normalize_skips_snake_case_placeholder():
    steps = ['click "select_product_page"']
    result = _normalize_steps(steps, "https://x.com")
    assert result == []


# ─── _map_exception ──────────────────────────────────────────────────────────

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


@pytest.mark.asyncio
async def test_runner_routes_e2e():
    """_dispatch ต้องเรียก _run_e2e สำหรับ type='e2e'."""
    runner = UniversalTestRunner.__new__(UniversalTestRunner)
    runner._screenshot_dir = None
    runner._terminal = MagicMock()
    tc = _make_tc("e2e")

    with patch.object(runner, "_run_e2e", new=AsyncMock(
        return_value=TestResult(test_case=tc, passed=True, duration_ms=200)
    )) as mock_e2e:
        result = await runner._dispatch(tc, AsyncMock())
    assert result.passed
    mock_e2e.assert_called_once()


@pytest.mark.asyncio
async def test_run_e2e_returns_passed_result():
    """_run_e2e ต้องคืน TestResult ที่ passed เมื่อ HypothesisExecutor pass."""
    from src.explorer.executor import HypothesisResult
    runner = UniversalTestRunner.__new__(UniversalTestRunner)
    runner._screenshot_dir = None
    runner._terminal = MagicMock()

    mock_exec = AsyncMock()
    mock_exec.execute = AsyncMock(return_value=HypothesisResult(
        hypothesis_id="abc123",
        passed=True,
        steps_executed=3,
        failure_reason=None,
    ))
    runner._hyp_executor = mock_exec

    tc = TestCase(
        title="E2E login → cart",
        type="e2e",
        priority="high",
        preconditions=["เข้าสู่ระบบแล้ว"],
        steps=[
            "เปิดหน้า https://x.com/login",
            "กรอก \"Username\" ด้วย user",
            "คลิกปุ่ม \"Login\"",
        ],
        expected_outcome="เข้าสู่ระบบสำเร็จ",
        source_url="https://x.com/login",
    )
    page = AsyncMock()
    page.url = "https://x.com/login"
    page.locator = MagicMock(return_value=AsyncMock())

    result = await runner._run_e2e(tc, page)
    assert result.passed is True
    assert result.test_case.type == "e2e"
    assert len(result.steps_trace) == len(tc.steps)


@pytest.mark.asyncio
async def test_run_e2e_returns_failed_result_on_hyp_failure():
    """_run_e2e ต้องคืน TestResult ที่ failed พร้อม failure_reason."""
    from src.explorer.executor import HypothesisResult
    runner = UniversalTestRunner.__new__(UniversalTestRunner)
    runner._screenshot_dir = None
    runner._terminal = MagicMock()

    mock_exec = AsyncMock()
    mock_exec.execute = AsyncMock(return_value=HypothesisResult(
        hypothesis_id="def456",
        passed=False,
        steps_executed=1,
        failure_reason="ไม่พบปุ่ม Login",
    ))
    runner._hyp_executor = mock_exec

    tc = TestCase(
        title="E2E fail test",
        type="e2e",
        priority="medium",
        steps=["เปิดหน้า https://x.com/login", "คลิกปุ่ม \"Login\""],
        expected_outcome="สำเร็จ",
        source_url="https://x.com/login",
    )
    page = AsyncMock()
    page.url = "https://x.com/login"
    page.locator = MagicMock(return_value=AsyncMock())

    result = await runner._run_e2e(tc, page)
    assert result.passed is False
    assert result.failure_reason == "ไม่พบปุ่ม Login"
