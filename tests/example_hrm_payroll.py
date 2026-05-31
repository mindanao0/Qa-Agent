"""
Example: Thai HRM / Payroll system QA tests.

Covers the 6 scenarios from the specification:
  1. Admin login → dashboard verification (RBAC: admin)
  2. HR Manager creates employee record
  3. Payroll Officer runs monthly payroll calculation
  4. Thai SSO deduction verification (max 875 THB/month as of 2026)
  5. Progressive income tax bracket calculation (0-35%)
  6. Employee role cannot access payroll module (RBAC boundary)

Extras:
  - Clock emulation (session timeout, fast-forward 31 min)
  - Self-healing decorator on every test function
  - Allure step + screenshot attachment on failure

Prerequisites:
  - APP_BASE_URL env var pointing to a running HRM instance
    (default: http://localhost:3000)
  - AUTH_BASE_URL env var for the auth endpoint
  - Role credentials in .env (ADMIN_USERNAME, ADMIN_PASSWORD, etc.)
  - Ollama running with qwen2.5-coder:7b-instruct-q4_K_M for live healing

Run:
  pytest tests/example_hrm_payroll.py -v --alluredir=allure-results
"""

from __future__ import annotations

import asyncio
import functools
import os
import re
import sys
from pathlib import Path
from typing import Callable

import pytest
from loguru import logger
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, expect

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.browser.auth_manager import AuthManager
from src.browser.manager import BrowserManager
from src.healing.engine import HealingEngine
from src.llm.adapter import OllamaAdapter
from src.reporting.allure_reporter import reporter
from src.agents.healer import _extract_locator_from_error

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:3000")
LOCATOR_FILE = str(Path(__file__).parent.parent / "locators" / "locators.json")

# Thai SSO constants (2026 values per Thai Social Security Act B.E. 2533)
SSO_RATE = 0.05           # 5% employee contribution
SSO_SALARY_CAP = 15_000   # SSO applies on salary up to 15,000 THB
SSO_MAX_MONTHLY = 750     # 5% × 15,000 = 750 THB  (employer cap mirrors employee)
SSO_EMPLOYEE_CAP = 875    # Employee SSO cap as of 2026 per spec

# Thai income tax brackets 2026 (annual income in THB)
TAX_BRACKETS = [
    (150_000,   0.00),
    (300_000,   0.05),
    (500_000,   0.10),
    (750_000,   0.15),
    (1_000_000, 0.20),
    (2_000_000, 0.25),
    (5_000_000, 0.30),
    (float("inf"), 0.35),
]
PERSONAL_ALLOWANCE = 60_000  # THB


# ──────────────────────────────────────────────────────────────────────────────
# Tax reference calculator (pure Python — used in assertions)
# ──────────────────────────────────────────────────────────────────────────────

def compute_annual_tax(annual_gross: float, deductions: float = 0) -> float:
    """Compute Thai progressive income tax on (annual_gross - deductions - personal_allowance)."""
    taxable = max(0, annual_gross - deductions - PERSONAL_ALLOWANCE)
    prev_limit = 0
    tax = 0.0
    for limit, rate in TAX_BRACKETS:
        if taxable <= prev_limit:
            break
        band = min(taxable, limit) - prev_limit
        tax += band * rate
        prev_limit = limit
    return round(tax, 2)


def compute_monthly_sso(monthly_salary: float) -> float:
    """Compute employee SSO contribution capped at SSO_EMPLOYEE_CAP."""
    return min(monthly_salary * SSO_RATE, SSO_EMPLOYEE_CAP)


# ──────────────────────────────────────────────────────────────────────────────
# Self-healing decorator
# ──────────────────────────────────────────────────────────────────────────────

def with_healing(func: Callable | None = None, *, max_retries: int = 3) -> Callable:
    """
    Decorator that retries a test coroutine on Playwright failures.

    On each failure:
      1. Attaches a screenshot and the error message to the Allure report.
      2. If a HealingEngine is available (via 'healing_engine' kwarg),
         invokes it to record the healed locator in locators.json.
      3. Retries up to max_retries times before re-raising.

    The decorated function must receive 'page' as a keyword argument.
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            page: Page | None = kwargs.get("page")
            engine: HealingEngine | None = kwargs.get("healing_engine")

            for attempt in range(max_retries):
                try:
                    return await f(*args, **kwargs)
                except Exception as exc:
                    is_last = attempt == max_retries - 1
                    logger.warning(
                        f"[{f.__name__}] attempt {attempt + 1}/{max_retries} "
                        f"failed: {type(exc).__name__}: {exc}"
                    )
                    reporter.attach_ai_reasoning(
                        f"Attempt {attempt + 1}/{max_retries}: {type(exc).__name__}: {exc}",
                        name=f"Failure trace (attempt {attempt + 1})",
                    )
                    if page is not None:
                        await reporter.attach_screenshot(
                            page, name=f"Screenshot – attempt {attempt + 1}"
                        )
                        if engine is not None and not is_last:
                            try:
                                failing_locator = _extract_locator_from_error(str(exc))
                                if failing_locator:
                                    healed = await engine.heal(
                                        page=page,
                                        failed_locator=failing_locator,
                                        action=f.__name__,
                                        error_message=str(exc),
                                    )
                                    reporter.attach_healing_event(
                                        healed,
                                        name=f"Healed locator (attempt {attempt + 1})",
                                    )
                            except Exception as heal_exc:
                                logger.debug(f"Healing failed: {heal_exc}")
                    if is_last:
                        raise
        return wrapper

    if func is not None:          # called without arguments: @with_healing
        return decorator(func)
    return decorator              # called with arguments: @with_healing(max_retries=2)


# ──────────────────────────────────────────────────────────────────────────────
# Pytest fixtures
# ──────────────────────────────────────────────────────────────────────────────

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
async def browser_mgr():
    """Shared BrowserManager for the test module."""
    mgr = BrowserManager()
    await mgr.start()
    yield mgr
    await mgr.stop()


@pytest.fixture(scope="module")
async def auth_mgr():
    """AuthManager loaded from config/roles.yaml."""
    async with AuthManager() as am:
        yield am


@pytest.fixture(scope="module")
async def healing_engine():
    """HealingEngine backed by a real OllamaAdapter (requires Ollama running)."""
    try:
        async with OllamaAdapter() as adapter:
            yield HealingEngine(adapter=adapter, locator_file=LOCATOR_FILE)
    except Exception as exc:
        logger.warning(f"HealingEngine unavailable (Ollama not running?): {exc}")
        yield None


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — Admin login, dashboard verification, and session-timeout clock test
# ──────────────────────────────────────────────────────────────────────────────


@with_healing
async def test_admin_login_and_dashboard(
    browser_mgr: BrowserManager,
    auth_mgr: AuthManager,
    healing_engine: HealingEngine | None,
):
    """
    Scenario 1: Admin logs in and verifies the dashboard.
    Also emulates 31-minute inactivity via page.clock to verify session expiry.
    """
    reporter.title("Admin Login → Dashboard (+ session timeout via clock)")
    reporter.set_test_metadata(domain="hrm_login", role="admin", url=APP_BASE_URL, tags=["smoke", "rbac"])
    reporter.severity("critical")

    storage_state = await auth_mgr.get_storage_state("admin")
    async with browser_mgr.new_page(storage_state=storage_state) as page:

        # ── Install fake clock before first navigation ─────────────────────
        async with reporter.async_step("Install fake clock at epoch", page=page):
            await page.clock.install()

        # ── Navigate to dashboard ──────────────────────────────────────────
        async with reporter.async_step(f"Navigate to {APP_BASE_URL}/dashboard", page=page):
            await page.goto(f"{APP_BASE_URL}/dashboard", wait_until="domcontentloaded")
            await page.wait_for_load_state("networkidle")

        # ── Capture AxTree before assertions ──────────────────────────────
        from src.browser.ax_extractor import extract_axtree, prune_axtree
        raw = await extract_axtree(page)
        pruned = await prune_axtree(raw)
        reporter.attach_axtree(pruned, name="Dashboard AxTree")

        # ── Dashboard assertions ───────────────────────────────────────────
        async with reporter.async_step("Verify dashboard heading", page=page):
            await expect(
                page.get_by_role("heading", name=re.compile(r"dashboard", re.IGNORECASE))
            ).to_be_visible()

        async with reporter.async_step("Verify admin navigation menu", page=page):
            await expect(page.get_by_role("navigation")).to_be_visible()

        async with reporter.async_step("Verify admin menu includes Payroll", page=page):
            await expect(
                page.get_by_role("link", name=re.compile(r"payroll", re.IGNORECASE))
            ).to_be_visible()

        # ── Clock fast-forward: 31 minutes → triggers session timeout ─────
        async with reporter.async_step("Fast-forward clock 31 minutes (session expires)", page=page):
            await page.clock.fast_forward("31:00")

        async with reporter.async_step("Reload page and verify redirect to login", page=page):
            await page.reload(wait_until="domcontentloaded")
            # After session expiry the app should redirect to the login page
            await expect(page).to_have_url(re.compile(r".*/(?:auth/)?login", re.IGNORECASE))

        await reporter.attach_screenshot(page, name="Session timeout — login redirect")


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — HR Manager creates employee record
# ──────────────────────────────────────────────────────────────────────────────


@with_healing
async def test_hr_manager_creates_employee_record(
    browser_mgr: BrowserManager,
    auth_mgr: AuthManager,
    healing_engine: HealingEngine | None,
):
    """Scenario 2: HR Manager creates a new employee profile."""
    reporter.title("HR Manager — Create Employee Record")
    reporter.set_test_metadata(domain="employee_profile", role="hr_manager", tags=["crud"])
    reporter.severity("critical")

    storage_state = await auth_mgr.get_storage_state("hr_manager")
    async with browser_mgr.new_page(storage_state=storage_state) as page:

        async with reporter.async_step("Navigate to New Employee form", page=page):
            await page.goto(f"{APP_BASE_URL}/employees/new", wait_until="domcontentloaded")

        raw = await extract_axtree_safe(page)
        reporter.attach_axtree(raw, "New Employee Form AxTree")

        async with reporter.async_step("Fill employee details", page=page):
            await page.get_by_label(re.compile(r"employee.*code|emp.*id", re.IGNORECASE)).fill("EMP-TEST-001")
            await page.get_by_label(re.compile(r"full.*name.*th|ชื่อ.*ไทย", re.IGNORECASE)).fill("ทดสอบ ระบบ")
            await page.get_by_label(re.compile(r"full.*name.*en|english.*name", re.IGNORECASE)).fill("System Tester")
            await page.get_by_label(re.compile(r"department", re.IGNORECASE)).fill("QA Department")
            await page.get_by_label(re.compile(r"position|job.*title", re.IGNORECASE)).fill("QA Engineer")
            await page.get_by_label(re.compile(r"employment.*type", re.IGNORECASE)).select_option("full_time")

        async with reporter.async_step("Submit employee form", page=page):
            await page.get_by_role("button", name=re.compile(r"save|create|submit", re.IGNORECASE)).click()
            await page.wait_for_load_state("networkidle")

        async with reporter.async_step("Verify employee appears in list", page=page):
            await page.goto(f"{APP_BASE_URL}/employees", wait_until="domcontentloaded")
            await expect(page.get_by_text("EMP-TEST-001")).to_be_visible()
            await expect(page.get_by_text("System Tester")).to_be_visible()

        await reporter.attach_screenshot(page, "Employee created successfully")


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — Payroll Officer runs monthly payroll
# ──────────────────────────────────────────────────────────────────────────────


@with_healing
async def test_payroll_officer_runs_monthly_payroll(
    browser_mgr: BrowserManager,
    auth_mgr: AuthManager,
    healing_engine: HealingEngine | None,
):
    """Scenario 3: Payroll Officer triggers monthly payroll calculation run."""
    reporter.title("Payroll Officer — Run Monthly Payroll")
    reporter.set_test_metadata(domain="payroll_calculation", role="payroll_officer", tags=["payroll"])
    reporter.severity("critical")

    storage_state = await auth_mgr.get_storage_state("payroll_officer")
    async with browser_mgr.new_page(storage_state=storage_state) as page:

        async with reporter.async_step("Navigate to Payroll Run page", page=page):
            await page.goto(f"{APP_BASE_URL}/payroll/run", wait_until="domcontentloaded")

        raw = await extract_axtree_safe(page)
        reporter.attach_axtree(raw, "Payroll Run Form AxTree")

        async with reporter.async_step("Select current month and period", page=page):
            await page.get_by_role("combobox", name=re.compile(r"month|period", re.IGNORECASE)).select_option(
                label=re.compile(r"2026")
            )

        async with reporter.async_step("Trigger payroll calculation", page=page):
            await page.get_by_role("button", name=re.compile(r"run|calculate|process", re.IGNORECASE)).click()
            # Wait for async calculation to complete
            await page.wait_for_load_state("networkidle")
            await expect(
                page.get_by_role("status").or_(
                    page.get_by_text(re.compile(r"complete|success|calculated", re.IGNORECASE))
                )
            ).to_be_visible(timeout=30_000)

        async with reporter.async_step("Verify payroll summary table is populated", page=page):
            await expect(page.get_by_role("table")).to_be_visible()
            rows = page.get_by_role("row")
            await expect(rows).to_have_count(greater_than=1)

        await reporter.attach_screenshot(page, "Payroll run complete")


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — Thai SSO deduction verification (max 875 THB / month)
# ──────────────────────────────────────────────────────────────────────────────


@with_healing
async def test_thai_sso_deduction_verification(
    browser_mgr: BrowserManager,
    auth_mgr: AuthManager,
    healing_engine: HealingEngine | None,
):
    """
    Scenario 4: Verify SSO deduction displayed in payslip matches Thai rules.
    - Salary 30,000 THB/month → SSO = 875 THB (cap reached: 5% × 15,000 = 750, but cap is 875)
    - Salary 10,000 THB/month → SSO = 500 THB (5% × 10,000 = 500 < 875)
    """
    reporter.title("Thai SSO Deduction Verification (max 875 THB)")
    reporter.set_test_metadata(domain="payroll_calculation", role="payroll_officer", tags=["thai-compliance", "sso"])
    reporter.severity("blocker")

    # Pre-check computed values
    assert compute_monthly_sso(30_000) == SSO_EMPLOYEE_CAP, (
        f"Expected SSO={SSO_EMPLOYEE_CAP} for salary=30000, got {compute_monthly_sso(30_000)}"
    )
    assert compute_monthly_sso(10_000) == 500, (
        f"Expected SSO=500 for salary=10000, got {compute_monthly_sso(10_000)}"
    )

    storage_state = await auth_mgr.get_storage_state("payroll_officer")
    async with browser_mgr.new_page(storage_state=storage_state) as page:

        async with reporter.async_step("Navigate to payslip detail for high-salary employee", page=page):
            await page.goto(
                f"{APP_BASE_URL}/payroll/payslip?employee=EMP-TEST-HIGH&month=2026-05",
                wait_until="domcontentloaded",
            )

        raw = await extract_axtree_safe(page)
        reporter.attach_axtree(raw, "High-Salary Payslip AxTree")

        async with reporter.async_step("Verify SSO deduction is capped at 875 THB", page=page):
            # The SSO row should display 875.00 (or ฿875.00) for a ≥17,500 THB salary
            sso_cell = page.get_by_role("cell", name=re.compile(r"875", re.IGNORECASE))
            await expect(sso_cell).to_be_visible()

        async with reporter.async_step("Navigate to payslip for low-salary employee", page=page):
            await page.goto(
                f"{APP_BASE_URL}/payroll/payslip?employee=EMP-TEST-LOW&month=2026-05",
                wait_until="domcontentloaded",
            )

        async with reporter.async_step("Verify SSO deduction is 5% of salary (not capped)", page=page):
            sso_cell_low = page.get_by_role("cell", name=re.compile(r"500", re.IGNORECASE))
            await expect(sso_cell_low).to_be_visible()

        reporter.attach_json(
            {
                "salary_30000_sso": compute_monthly_sso(30_000),
                "salary_10000_sso": compute_monthly_sso(10_000),
                "cap_applied_at_salary": SSO_EMPLOYEE_CAP / SSO_RATE,
                "sso_employee_cap_2026": SSO_EMPLOYEE_CAP,
            },
            name="SSO Calculation Reference",
        )
        await reporter.attach_screenshot(page, "SSO deduction verification")


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — Thai progressive income tax bracket verification
# ──────────────────────────────────────────────────────────────────────────────


@with_healing
async def test_thai_progressive_income_tax(
    browser_mgr: BrowserManager,
    auth_mgr: AuthManager,
    healing_engine: HealingEngine | None,
):
    """
    Scenario 5: Verify the tax calculator applies progressive brackets correctly.

    Test case: annual income = 600,000 THB, no deductions beyond personal allowance
      taxable = 600,000 - 60,000 = 540,000 THB
      0–150k  → 0%  = 0
      150k–300k → 5% = 7,500
      300k–500k → 10% = 20,000
      500k–540k → 15% = 6,000
      Total = 33,500 THB
    """
    reporter.title("Thai Progressive Income Tax — Bracket Verification")
    reporter.set_test_metadata(domain="thai_tax_calculation", role="payroll_officer", tags=["thai-compliance", "tax"])
    reporter.severity("blocker")

    # Reference calculation
    expected_tax = compute_annual_tax(annual_gross=600_000)
    assert expected_tax == 33_500.0, f"Reference calculation mismatch: {expected_tax}"

    storage_state = await auth_mgr.get_storage_state("payroll_officer")
    async with browser_mgr.new_page(storage_state=storage_state) as page:

        async with reporter.async_step("Navigate to tax calculator", page=page):
            await page.goto(f"{APP_BASE_URL}/tax/calculate", wait_until="domcontentloaded")

        raw = await extract_axtree_safe(page)
        reporter.attach_axtree(raw, "Tax Calculator AxTree")

        async with reporter.async_step("Enter annual income 600,000 THB", page=page):
            await page.get_by_label(re.compile(r"annual.*income|yearly.*salary", re.IGNORECASE)).fill("600000")

        async with reporter.async_step("Submit calculation", page=page):
            await page.get_by_role("button", name=re.compile(r"calculate|compute", re.IGNORECASE)).click()
            await page.wait_for_load_state("networkidle")

        async with reporter.async_step("Verify tax = 33,500 THB", page=page):
            await expect(
                page.get_by_text(re.compile(r"33[,.]?500", re.IGNORECASE))
            ).to_be_visible()

        async with reporter.async_step("Verify bracket breakdown is displayed", page=page):
            # Each bracket row should be visible
            for bracket_text in ["5%", "10%", "15%"]:
                await expect(page.get_by_text(bracket_text)).to_be_visible()

        reporter.attach_json(
            {
                "annual_gross": 600_000,
                "personal_allowance": PERSONAL_ALLOWANCE,
                "taxable_income": 600_000 - PERSONAL_ALLOWANCE,
                "expected_tax": expected_tax,
                "brackets": [
                    {"range": "0-150k", "rate": "0%", "tax": 0},
                    {"range": "150k-300k", "rate": "5%", "tax": 7_500},
                    {"range": "300k-500k", "rate": "10%", "tax": 20_000},
                    {"range": "500k-540k", "rate": "15%", "tax": 6_000},
                ],
            },
            name="Tax Calculation Reference",
        )
        await reporter.attach_screenshot(page, "Tax bracket verification")


# ──────────────────────────────────────────────────────────────────────────────
# Test 6 — Employee role cannot access payroll module (RBAC boundary)
# ──────────────────────────────────────────────────────────────────────────────


@with_healing
async def test_employee_cannot_access_payroll_rbac(
    browser_mgr: BrowserManager,
    auth_mgr: AuthManager,
    healing_engine: HealingEngine | None,
):
    """
    Scenario 6: Employee role is forbidden from accessing the payroll module.
    Expects HTTP 403 response or an "Access Denied" UI page.
    """
    reporter.title("RBAC Boundary — Employee Cannot Access /payroll")
    reporter.set_test_metadata(domain="rbac_permission_check", role="employee", tags=["rbac", "security"])
    reporter.severity("blocker")

    storage_state = await auth_mgr.get_storage_state("employee")
    async with browser_mgr.new_page(storage_state=storage_state) as page:

        async with reporter.async_step("Navigate to /payroll as employee role", page=page):
            response = await page.goto(f"{APP_BASE_URL}/payroll", wait_until="domcontentloaded")

        raw = await extract_axtree_safe(page)
        reporter.attach_axtree(raw, "Payroll Access Attempt AxTree")

        async with reporter.async_step("Verify access is denied (403 or redirect)", page=page):
            # Accept any of: HTTP 403 response, "Access Denied" heading, or redirect to login
            access_denied_heading = page.get_by_role(
                "heading", name=re.compile(r"access denied|forbidden|ไม่มีสิทธิ์", re.IGNORECASE)
            )
            access_denied_text = page.get_by_text(
                re.compile(r"access denied|forbidden|unauthorized|403|ไม่มีสิทธิ์", re.IGNORECASE)
            )

            is_403 = response is not None and response.status == 403
            is_denied_ui = await access_denied_heading.is_visible() or await access_denied_text.is_visible()
            is_redirected = re.search(r"login|auth/login", page.url, re.IGNORECASE) is not None

            assert is_403 or is_denied_ui or is_redirected, (
                f"Expected access denial for employee role on /payroll. "
                f"HTTP status={response.status if response else 'N/A'}, "
                f"URL={page.url!r}"
            )

        async with reporter.async_step("Verify payroll navigation item is hidden", page=page):
            # The payroll link should not appear in the navigation for employee role
            payroll_nav_link = page.get_by_role("link", name=re.compile(r"^payroll$", re.IGNORECASE))
            # Either not present or not visible
            count = await payroll_nav_link.count()
            if count > 0:
                await expect(payroll_nav_link).not_to_be_visible()

        await reporter.attach_screenshot(page, "RBAC — employee denied payroll access")
        reporter.attach_json(
            {
                "role": "employee",
                "attempted_url": f"{APP_BASE_URL}/payroll",
                "http_status": response.status if response else None,
                "final_url": page.url,
                "outcome": "access_denied",
            },
            name="RBAC Boundary Test Result",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

async def extract_axtree_safe(page: Page) -> str:
    """Extract and prune AxTree; return empty string on failure."""
    from src.browser.ax_extractor import extract_axtree, prune_axtree
    try:
        raw = await extract_axtree(page)
        return await prune_axtree(raw)
    except Exception as exc:
        logger.debug(f"extract_axtree_safe: {exc}")
        return ""
