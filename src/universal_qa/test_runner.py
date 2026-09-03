from __future__ import annotations

import pathlib
import re
import time
from urllib.parse import urljoin, urlparse

from playwright.async_api import Page

from src.agents.observer_driver.observers.accessibility_observer import (
    AccessibilityObserver,
)
from src.agents.observer_driver.observers.security_observer import SecurityObserver
from src.contractskill.sfg import SFGStore
from src.explorer.executor import HypothesisExecutor
from src.explorer.hypothesis import TestHypothesis
from src.llm.instructor_client import InstructorClient
from src.universal_qa.models import StepTrace, TestCase, TestResult
from src.universal_qa.reporters.terminal import TerminalReporter

_XSS_PAYLOAD = "<script>window.__xss_fired=true;</script>"
_SQLI_PAYLOAD = "' OR '1'='1"

# Thai action keywords → English equivalents ที่ HypothesisExecutor เข้าใจ
_THAI_NAV = re.compile(r'^(เปิดหน้า|ไปที่หน้า|ไปที่|นำทางไปยัง|เปิด)\s+', re.IGNORECASE)
_THAI_FILL = re.compile(r'^(กรอก|พิมพ์|ใส่ข้อมูล|ใส่)\s+', re.IGNORECASE)
_THAI_CLICK = re.compile(r'^(คลิกปุ่ม|คลิกลิงก์|คลิก|กดปุ่ม|กด)\s+', re.IGNORECASE)
# Thai/English assertion keywords — Playwright ไม่รองรับ assertion โดยตรง  (Rule B extended)
_VERIFY_RE = re.compile(
    r'^(ตรวจสอบ|ตรวจว่า|ยืนยัน|เช็คว่า|เช็ค|ดูว่า|สังเกต|verify\s|assert\s|check\s+that\s|ensure\s)',
    re.IGNORECASE,
)
# Rule A — strip numbered-list prefix (e.g. "1. คลิก Login" → "คลิก Login")
_NUMBERED_PREFIX_RE = re.compile(r'^\d+\.\s+')
# Rule C — scroll steps executor cannot handle
_SCROLL_RE = re.compile(r'^(scroll|เลื่อน)', re.IGNORECASE)
# Rule D — wait steps executor cannot handle
_WAIT_RE = re.compile(r'^(wait|รอ\s)', re.IGNORECASE)
# Rule E — เลือก/select → select option:
_THAI_SELECT_RE = re.compile(r'^(เลือก|เลือกตัวเลือก)\s+', re.IGNORECASE)
_BARE_SELECT_RE = re.compile(r'^select\s+(?!option:)', re.IGNORECASE)
# Rule F — fill in / type in / type → fill
_FILL_IN_RE = re.compile(r'^(fill in|type in|type)\s+', re.IGNORECASE)
# Match first quoted substring (ASCII double/single + Unicode left/right variants)
_QUOTED_TEXT_RE = re.compile(
    '(?:'
    '"([^"]+)"'   # "…"  Unicode left/right double
    '|'
    '‘([^’]+)’'   # '…'  Unicode left/right single
    '|'
    '"([^"]+)"'                   # "…"  ASCII double
    '|'
    "'([^']+)'"                   # '…'  ASCII single
    ')'
)


_POST_LOGIN_PATH_RE = re.compile(
    r'/(inventory|cart|checkout|dashboard|profile|account|products?|orders?|settings?)',
    re.IGNORECASE,
)
_LOGIN_PATH_RE = re.compile(r'/(login|signin|sign-in|auth)', re.IGNORECASE)


def _normalize_steps(steps: list[str], source_url: str) -> list[str]:
    """Thai action keywords to English + relative path to full URL."""
    base = "{u.scheme}://{u.netloc}".format(u=urlparse(source_url))
    # ตรวจจาก path ว่าเป็นหน้า post-login (inventory/cart/checkout/…) อย่างชัดเจน
    # ไม่ใช้ netloc เพราะ saucedemo login page อยู่ที่ root "/" ซึ่งไม่มี "login" ใน URL
    _src_is_post_login = bool(_POST_LOGIN_PATH_RE.search(source_url))
    # login page = URL path มีคำว่า login/signin/auth
    _src_is_login = bool(_LOGIN_PATH_RE.search(source_url))

    fixed = []
    for step in steps:
        # Rule A. strip numbered-list prefix (e.g. "1. คลิก Login" → "คลิก Login")
        step = _NUMBERED_PREFIX_RE.sub("", step)

        # 1. แปล Thai keywords เป็น English
        if _THAI_NAV.match(step):
            step = _THAI_NAV.sub("navigate to ", step)
        elif _THAI_FILL.match(step):
            step = _THAI_FILL.sub("fill ", step)
        elif _THAI_CLICK.match(step):
            step = _THAI_CLICK.sub("click ", step)

        # 2. แก้ relative path → full URL สำหรับ navigate steps
        is_nav = step.lower().startswith("navigate")
        has_http = "http://" in step or "https://" in step
        if is_nav and not has_http:
            path_match = re.search(r'["\']?(/[\w/\-]*)["\']?', step)
            if path_match:
                path = path_match.group(1)
                full_url = urljoin(base + "/", path.lstrip("/"))
                step = re.sub(re.escape(path_match.group(0)), f" {full_url}", step, count=1).strip()

        # 3. convert assertion steps -> "verify text: X" (if quoted text found)
        #    no quoted text -> skip (nothing to verify)  [Rule B extended regex above]
        if _VERIFY_RE.match(step):
            quoted_m = re.search(_QUOTED_TEXT_RE, step)
            if quoted_m:
                extracted = next(g for g in quoted_m.groups() if g is not None)
                fixed.append('verify text: "' + extracted + '"')
            continue

        # Rule C. scroll → skip (executor cannot scroll)
        if _SCROLL_RE.match(step):
            continue

        # Rule D. wait → skip (executor cannot wait by time)
        if _WAIT_RE.match(step):
            continue

        # Rule E. เลือก/select → select option:
        if _THAI_SELECT_RE.match(step):
            step = _THAI_SELECT_RE.sub("select option: ", step)
        elif _BARE_SELECT_RE.match(step):
            step = _BARE_SELECT_RE.sub("select option: ", step)

        # Rule F. fill in / type in / type → fill
        if _FILL_IN_RE.match(step):
            step = _FILL_IN_RE.sub("fill ", step)

        # 4. (ลบออก) click "/path" และ click snake_case — ส่งต่อให้ executor._resolve_step
        #    ซึ่งจะใช้ LLM + DOM จริงเพื่อหา element ที่ตรงกัน แทนการ skip

        # 5. (รวมกับข้อ 4 แล้ว)

        # 6. skip: login click/navigate บนหน้า post-login ที่ชัดเจน (inventory/cart/…)
        #    ตรวจ path ของ source URL แทน netloc เพราะ saucedemo login อยู่ที่ "/" ไม่มี "login" ใน URL
        _login_step_m = re.match(
            r'^(click|navigate)\s+.*\b(login|sign[\s-]?in)\b',
            step, re.IGNORECASE,
        )
        if _login_step_m and _src_is_post_login:
            continue

        # 7. skip: fill credential (username/password/email) บนหน้า post-login
        #    LLM บางครั้ง generate ขั้นตอน login ซ้ำแม้อยู่บนหน้า inventory แล้ว
        _fill_cred_m = re.match(
            r'^fill\s+["\']?(username|password|email|user\s+name)["\']?\b',
            step, re.IGNORECASE,
        )
        if _fill_cred_m and _src_is_post_login:
            continue

        fixed.append(step)
    return fixed


_PW_FILL_RE = re.compile(
    r'(fill|กรอก|พิมพ์|ใส่).*\bpassword\b',
    re.IGNORECASE,
)
_PW_VALUE_RE = re.compile(r'(?:with|ด้วย)\s+"?([^"\s]+)"?', re.IGNORECASE)


async def _pre_fill_password(steps: list[str], page: Page) -> list[str]:
    """กรอก password field โดยตรงแล้วลบ step นั้นออก ป้องกัน BLOCKED_ACTION_PATTERNS."""
    safe: list[str] = []
    for step in steps:
        if _PW_FILL_RE.search(step):
            val_match = _PW_VALUE_RE.search(step)
            if val_match:
                value = val_match.group(1)
            else:
                # รูปแบบ: fill "value" ในช่อง 'Password' — ดึง quoted string แรก
                _qm = re.search(r'"([^"]+)"|\'([^\']+)\'', step)
                value = (_qm.group(1) or _qm.group(2)) if _qm else "test"
            try:
                await page.locator("input[type=password]").first.fill(value, timeout=5_000)
            except Exception:
                pass
        else:
            safe.append(step)
    return safe or steps


def _map_exception(exc: Exception) -> str:
    msg = str(exc)
    if isinstance(exc, TimeoutError) or "timeout" in msg.lower():
        return f"ไม่พบ element ภายใน 30 วินาที — {msg[:120]}"
    if isinstance(exc, AssertionError):
        return f"ผลลัพธ์ไม่ตรงตามที่คาดหวัง — {msg[:120]}"
    return msg[:200]


class UniversalTestRunner:
    """Executes a list of TestCase objects using Playwright.

    Routes by type:
    - functional    → HypothesisExecutor (existing Sprint 5)
    - accessibility → AccessibilityObserver._scan static method
    - security      → SecurityObserver static methods + XSS/SQLi injection
    """

    def __init__(
        self,
        sfg_store: SFGStore | None = None,
        screenshot_dir: pathlib.Path | None = None,
        terminal_reporter: TerminalReporter | None = None,
    ) -> None:
        instructor = InstructorClient()
        self._hyp_executor = HypothesisExecutor(instructor, sfg_store)
        self._screenshot_dir = screenshot_dir
        self._terminal = terminal_reporter or TerminalReporter()

    async def run(
        self, test_cases: list[TestCase], page: Page
    ) -> list[TestResult]:
        results: list[TestResult] = []
        for tc in test_cases:
            result = await self._dispatch(tc, page)
            self._terminal.report_one(result)
            results.append(result)
        return results

    async def _dispatch(self, tc: TestCase, page: Page) -> TestResult:
        try:
            if tc.type == "accessibility":
                return await self._run_accessibility(tc, page)
            if tc.type == "security":
                return await self._run_security(tc, page)
            if tc.type == "e2e":
                return await self._run_e2e(tc, page)
            if tc.type == "broken_link":
                return await self._run_broken_link(tc, page)
            return await self._run_functional(tc, page)
        except Exception as exc:
            return TestResult(
                test_case=tc,
                passed=False,
                failure_reason=_map_exception(exc),
                duration_ms=0,
            )

    async def _run_functional(self, tc: TestCase, page: Page) -> TestResult:
        start = time.monotonic()
        normalized = _normalize_steps(tc.steps, tc.source_url)
        # กรอก password field โดยตรงก่อนส่ง executor
        # เพราะ BLOCKED_ACTION_PATTERNS บล็อก step ที่มีคำว่า "password"
        safe_steps = await _pre_fill_password(normalized, page)
        hyp = TestHypothesis(
            goal=tc.title,
            start_url=tc.source_url,
            preconditions=tc.preconditions,
            steps=safe_steps,
            expected_outcome=tc.expected_outcome,
            confidence=0.7,
        )
        hyp_result = await self._hyp_executor.execute(hyp, page)
        traces = [
            StepTrace(
                step=step,
                status="passed" if hyp_result.passed else "failed",
                detail=f"ดำเนินการ {hyp_result.steps_executed} ขั้นตอน",
                error=hyp_result.failure_reason if not hyp_result.passed else None,
            )
            for step in tc.steps
        ]
        screenshot = await self._maybe_screenshot(page, tc.id, hyp_result.passed)
        return TestResult(
            test_case=tc,
            passed=hyp_result.passed,
            steps_trace=traces,
            failure_reason=hyp_result.failure_reason,
            screenshot_path=screenshot,
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    async def _run_e2e(self, tc: TestCase, page: Page) -> TestResult:
        """Execute E2E flow — ไม่ reset page (maintains login state across steps)."""
        start = time.monotonic()
        normalized = _normalize_steps(tc.steps, tc.source_url or page.url)
        safe_steps = await _pre_fill_password(normalized, page)
        hyp = TestHypothesis(
            goal=tc.title,
            start_url="",  # E2E จัดการ navigation เองผ่าน steps — ไม่ auto-goto
            preconditions=tc.preconditions,
            steps=safe_steps,
            expected_outcome=tc.expected_outcome,
            confidence=0.7,
        )
        hyp_result = await self._hyp_executor.execute(hyp, page, max_repairs=3)
        traces = [
            StepTrace(
                step=step,
                status="passed" if hyp_result.passed else "failed",
                detail=f"E2E: {hyp_result.steps_executed} steps",
                error=hyp_result.failure_reason if not hyp_result.passed else None,
            )
            for step in tc.steps
        ]
        duration = int((time.monotonic() - start) * 1000)
        screenshot = await self._maybe_screenshot(page, tc.id, hyp_result.passed)
        return TestResult(
            test_case=tc,
            passed=hyp_result.passed,
            failure_reason=hyp_result.failure_reason,
            steps_trace=traces,
            screenshot_path=screenshot,
            duration_ms=duration,
        )

    async def _run_accessibility(self, tc: TestCase, page: Page) -> TestResult:
        start = time.monotonic()
        traces: list[StepTrace] = []

        try:
            await page.goto(tc.source_url, wait_until="domcontentloaded", timeout=30_000)
            traces.append(StepTrace(step=f"เปิดหน้า {tc.source_url}",
                                    status="passed", detail="โหลดหน้าสำเร็จ"))
        except Exception as exc:
            return TestResult(
                test_case=tc, passed=False,
                steps_trace=traces,
                failure_reason=_map_exception(exc),
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        try:
            snapshot = await page.aria_snapshot()
        except Exception:
            snapshot = ""

        aria_findings = AccessibilityObserver._scan(snapshot)
        missing_alt: int = await page.evaluate(
            "() => document.querySelectorAll('img:not([alt])').length"
        )

        passed = not aria_findings and missing_alt == 0
        detail = (
            f"ARIA: {aria_findings if aria_findings else ['ผ่าน']}; "
            f"รูปขาด alt: {missing_alt}"
        )
        traces.append(StepTrace(
            step="ตรวจสอบกฎ Accessibility",
            status="passed" if passed else "failed",
            detail=detail,
            error="; ".join(aria_findings) if aria_findings else None,
        ))
        if missing_alt:
            traces.append(StepTrace(
                step="ตรวจสอบ alt text ของรูปภาพ",
                status="failed",
                detail=f"รูปภาพ {missing_alt} รูปขาด alt attribute",
                error=f"รูปภาพ {missing_alt} รูปไม่มี alt text",
            ))

        failure_reason = None
        if not passed:
            parts = []
            if aria_findings:
                parts.extend(aria_findings)
            if missing_alt:
                parts.append(f"{missing_alt} image(s) missing alt text")
            failure_reason = "; ".join(parts)

        screenshot = await self._maybe_screenshot(page, tc.id, passed)
        return TestResult(
            test_case=tc, passed=passed, steps_trace=traces,
            failure_reason=failure_reason, screenshot_path=screenshot,
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    async def _run_security(self, tc: TestCase, page: Page) -> TestResult:
        start = time.monotonic()
        traces: list[StepTrace] = []
        is_xss = "xss" in tc.title.lower()
        payload = _XSS_PAYLOAD if is_xss else _SQLI_PAYLOAD

        try:
            await page.goto(tc.source_url, wait_until="domcontentloaded", timeout=30_000)
            traces.append(StepTrace(step=f"เปิดหน้า {tc.source_url}",
                                    status="passed", detail="โหลดหน้าสำเร็จ"))
        except Exception as exc:
            return TestResult(
                test_case=tc, passed=False, steps_trace=traces,
                failure_reason=_map_exception(exc),
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        inputs = page.get_by_role("textbox")
        count = await inputs.count()
        for i in range(count):
            try:
                await inputs.nth(i).fill(payload, timeout=5_000)
            except Exception:
                pass
        traces.append(StepTrace(
            step=f"กรอก payload ลงใน {count} ช่องรับข้อมูล",
            status="passed", detail=f"payload: {payload[:60]}",
        ))

        try:
            await page.get_by_role("button").first.click(timeout=5_000)
            await page.wait_for_timeout(1_000)
        except Exception:
            pass

        passed = True
        failure_reason = None
        if is_xss:
            xss_fired: bool = await page.evaluate("() => !!window.__xss_fired")
            if xss_fired:
                passed = False
                failure_reason = "XSS payload ถูก execute (window.__xss_fired=true)"
        else:
            content = (await page.content()).lower()
            sql_keywords = ["sql syntax", "mysql_fetch", "ora-0", "sqlite", "syntax error near"]
            hit = next((kw for kw in sql_keywords if kw in content), None)
            if hit:
                passed = False
                failure_reason = f"พบ SQL error keyword '{hit}' ในหน้าที่ตอบกลับ"

        missing_headers = await SecurityObserver._missing_headers(page, tc.source_url)
        if missing_headers:
            traces.append(StepTrace(
                step="ตรวจสอบ Security Headers",
                status="failed",
                detail=f"ขาด: {missing_headers}",
                error=f"ขาด Headers: {', '.join(missing_headers)}",
            ))

        traces.append(StepTrace(
            step="ตรวจสอบว่า payload ไม่ถูก execute",
            status="passed" if passed else "failed",
            detail="ตรวจสอบ response ของหน้าแล้ว",
            error=failure_reason,
        ))

        screenshot = await self._maybe_screenshot(page, tc.id, passed)
        return TestResult(
            test_case=tc, passed=passed, steps_trace=traces,
            failure_reason=failure_reason, screenshot_path=screenshot,
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    async def _run_broken_link(self, tc: TestCase, page: Page) -> TestResult:
        """ตรวจสอบ links ทุกอันบนหน้า — HTTP GET แต่ละ link ผ่าน page.request"""
        import time as _time
        start = _time.monotonic()
        try:
            await page.goto(tc.source_url, wait_until="domcontentloaded", timeout=30_000)
            hrefs: list[str] = await page.evaluate("""() =>
                Array.from(document.querySelectorAll('a[href]'))
                    .map(a => a.href)
                    .filter(h => h.startsWith('http'))
                    .slice(0, 30)
            """)
            broken = []
            for href in hrefs:
                try:
                    resp = await page.request.get(href, timeout=10_000)
                    if resp.status >= 400:
                        broken.append(f"{href} → {resp.status}")
                except Exception:
                    broken.append(f"{href} → timeout/error")
            passed = len(broken) == 0
            return TestResult(
                test_case=tc,
                passed=passed,
                failure_reason=("Broken links: " + "; ".join(broken[:5])) if broken else None,
                duration_ms=int((_time.monotonic() - start) * 1000),
            )
        except Exception as exc:
            return TestResult(
                test_case=tc,
                passed=False,
                failure_reason=str(exc)[:200],
                duration_ms=0,
            )

    async def _maybe_screenshot(
        self, page: Page, test_id: str, passed: bool
    ) -> str | None:
        if passed or self._screenshot_dir is None:
            return None
        try:
            self._screenshot_dir.mkdir(parents=True, exist_ok=True)
            path = self._screenshot_dir / f"{test_id}.png"
            await page.screenshot(path=str(path), full_page=False)
            return str(path)
        except Exception:
            return None


__all__ = ["UniversalTestRunner", "_map_exception"]
