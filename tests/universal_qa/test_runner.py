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


def test_normalize_passes_through_click_path():
    """click '/login' ต้องผ่านมายัง executor (executor._resolve_step จะ resolve ด้วย LLM)"""
    steps = ['click "/login"']
    result = _normalize_steps(steps, "https://x.com")
    assert result == ['click "/login"']


def test_normalize_passes_through_snake_case_placeholder():
    """click snake_case ต้องผ่านมายัง executor (executor._resolve_step จะ resolve ด้วย LLM)"""
    steps = ['click "select_product_page"']
    result = _normalize_steps(steps, "https://x.com")
    assert result == ['click "select_product_page"']


# ─── Rule A: numbered prefix stripping ──────────────────────────────────────

def test_normalize_strips_numbered_prefix():
    result = _normalize_steps(['1. คลิกปุ่ม "Login"'], "https://x.com")
    assert any("click" in s for s in result)


def test_normalize_strips_numbered_prefix_english():
    result = _normalize_steps(['2. fill "Email" with "test@test.com"'], "https://x.com")
    assert result == ['fill "Email" with "test@test.com"']


# ─── Rule B: extended Thai verify variants ───────────────────────────────────

def test_normalize_thai_verify_variants():
    result = _normalize_steps(['เช็คว่า "Success" ปรากฏ'], "https://x.com")
    assert result == ['verify text: "Success"']


# ─── Rule C: scroll → skip ───────────────────────────────────────────────────

def test_normalize_skips_scroll():
    result = _normalize_steps(["scroll down to footer", "เลื่อนลงดู footer"], "https://x.com")
    assert result == []


# ─── Rule D: wait → skip ─────────────────────────────────────────────────────

def test_normalize_skips_wait():
    result = _normalize_steps(["wait 2 seconds", "รอ 500ms"], "https://x.com")
    assert result == []


# ─── Rule E: เลือก/select → select option: ───────────────────────────────────

def test_normalize_select_thai():
    result = _normalize_steps(['เลือก "Thailand"'], "https://x.com")
    assert result == ['select option: "Thailand"']


# ─── Rule F: fill in / type in → fill ────────────────────────────────────────

def test_normalize_fill_in_to_fill():
    result = _normalize_steps(['fill in "Email" with "test@test.com"'], "https://x.com")
    assert result == ['fill "Email" with "test@test.com"']


def test_normalize_type_in_to_fill():
    result = _normalize_steps(['type in "Search" with "shoes"'], "https://x.com")
    assert result == ['fill "Search" with "shoes"']


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


# ─── Sprint 15 BrowserWorkerPool wiring (opt-in, default off) ───────────────


def _runner_with_pool(use_worker_pool: bool, max_workers: int = 3) -> UniversalTestRunner:
    runner = UniversalTestRunner.__new__(UniversalTestRunner)
    runner._screenshot_dir = None
    runner._terminal = MagicMock()
    runner._use_worker_pool = use_worker_pool
    runner._max_workers = max_workers
    return runner


class _FakeContext:
    def __init__(self, browser, state: str = "seeded") -> None:
        self.browser = browser
        self._state = state

    async def storage_state(self):
        return {"cookies": [{"name": "session", "value": self._state}]}

    async def new_page(self):
        return _FakePage(self)

    async def close(self):
        pass


class _FakePage:
    def __init__(self, context) -> None:
        self.context = context


class _FakeBrowser:
    def __init__(self) -> None:
        self.new_contexts: list[tuple] = []

    async def new_context(self, storage_state=None):
        ctx = _FakeContext(self, state=str(storage_state))
        self.new_contexts.append((ctx, storage_state))
        return ctx


class _NoBrowserContext:
    browser = None


class _NoBrowserPage:
    context = _NoBrowserContext()


@pytest.mark.asyncio
async def test_run_uses_sequential_loop_when_flag_off():
    """Default behaviour: run() must NOT touch BrowserWorkerPool when the
    opt-in flag is off — this is the main regression risk of this wiring."""
    runner = _runner_with_pool(use_worker_pool=False)
    tc = _make_tc("functional")
    dispatch_calls = []

    async def fake_dispatch(t, page):
        dispatch_calls.append((t, page))
        return TestResult(test_case=t, passed=True, duration_ms=1)

    runner._dispatch = fake_dispatch
    with patch("src.universal_qa.test_runner.BrowserWorkerPool") as mock_pool_cls:
        results = await runner.run([tc], AsyncMock())
    mock_pool_cls.assert_not_called()
    assert len(results) == 1
    assert len(dispatch_calls) == 1


@pytest.mark.asyncio
async def test_run_parallel_dispatches_all_cases_in_original_order():
    runner = _runner_with_pool(use_worker_pool=True, max_workers=3)
    seen_pages = []

    async def fake_dispatch(tc, page):
        seen_pages.append(page)
        passed = tc.title != "fail-me"
        return TestResult(test_case=tc, passed=passed, duration_ms=1)

    runner._dispatch = fake_dispatch

    browser = _FakeBrowser()
    main_page = _FakePage(_FakeContext(browser, state="authed"))
    test_cases = [
        _make_tc("functional") if i != 2 else
        TestCase(title="fail-me", type="functional", priority="high",
                  steps=["x"], expected_outcome="ok", source_url="https://x.com")
        for i in range(6)
    ]

    results = await runner._run_parallel(test_cases, main_page)

    assert len(results) == 6
    # order preserved even though the pool returns worker-major order
    assert [r.test_case.id for r in results] == [tc.id for tc in test_cases]
    assert sum(r.passed for r in results) == 5
    assert not results[2].passed
    # genuine fan-out: more than one worker page was used
    assert len({id(p) for p in seen_pages}) >= 2
    # every worker context was seeded with the caller's authenticated session
    assert browser.new_contexts
    for _ctx, storage_state in browser.new_contexts:
        assert storage_state == {"cookies": [{"name": "session", "value": "authed"}]}


@pytest.mark.asyncio
async def test_run_parallel_falls_back_to_sequential_without_browser():
    """No reachable Browser handle on page.context -> falls back to the
    existing sequential loop instead of raising."""
    runner = _runner_with_pool(use_worker_pool=True)
    dispatch_calls = []

    async def fake_dispatch(tc, page):
        dispatch_calls.append(tc.id)
        return TestResult(test_case=tc, passed=True, duration_ms=1)

    runner._dispatch = fake_dispatch
    test_cases = [_make_tc("functional") for _ in range(3)]

    results = await runner._run_parallel(test_cases, _NoBrowserPage())

    assert len(results) == 3
    assert len(dispatch_calls) == 3


@pytest.mark.asyncio
async def test_run_dispatches_to_run_parallel_when_flag_on():
    runner = _runner_with_pool(use_worker_pool=True)
    tc = _make_tc("functional")
    expected = [TestResult(test_case=tc, passed=True, duration_ms=1)]

    with patch.object(runner, "_run_parallel", new=AsyncMock(return_value=expected)) as mock_rp:
        results = await runner.run([tc], AsyncMock())

    mock_rp.assert_called_once()
    assert results == expected
