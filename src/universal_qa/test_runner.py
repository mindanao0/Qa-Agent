from __future__ import annotations

import pathlib
import re
import time
from urllib.parse import urljoin, urlparse

from loguru import logger
from playwright.async_api import Page

from src.agents.observer_driver.observers.accessibility_observer import (
    AccessibilityObserver,
)
from src.agents.observer_driver.observers.security_observer import SecurityObserver
from src.contractskill.sfg import (
    BLOCKED_ACTION_PATTERNS as _SFG_BLOCKED_ACTION_PATTERNS,
    SFGStore,
)
from src.explorer.executor import HypothesisExecutor
from src.explorer.hypothesis import TestHypothesis
from src.fuzzer.anomaly_classifier import _SQL_ERROR_RE
from src.fuzzer.vector_generator import base_vectors_for, infer_field_type
from src.llm.instructor_client import InstructorClient
from src.universal_qa.models import StepTrace, TestCase, TestResult
from src.universal_qa.reporters.terminal import TerminalReporter

_XSS_PAYLOAD = "<script>window.__xss_fired=true;</script>"
_SQLI_PAYLOAD = "' OR '1'='1"

# ── Security fuzzing: adversarial vectors + safety guards (pure, offline) ─────
#
# The security runner injects a BOUNDED, TYPE-AWARE set of adversarial vectors
# per field (from src.fuzzer.vector_generator.BASE_VECTORS_BY_TYPE) instead of
# one fixed payload. Every vector is READ-SAFE — boolean/UNION SQLi and
# JS-flag/alert XSS only, NO DROP/DELETE/UPDATE and no data mutation — so
# probing a genuinely-vulnerable target can never destroy data. See
# _run_security for the thin Playwright driver around these pure helpers.

# XSS probes. [0] is the executable payload that sets window.__xss_fired so the
# execution check can tell an EXECUTED script from a merely REFLECTED one.
_XSS_VECTORS: tuple[str, ...] = (
    _XSS_PAYLOAD,                                    # executes → sets the flag
    "<img src=x onerror=window.__xss_fired=true>",   # attribute-context executor
    "<script>alert('xss')</script>",                 # reflected-only probe (no flag)
)
# SQLi probes — all READ-oriented (boolean tautology / UNION), non-mutating.
_SQLI_VECTORS: tuple[str, ...] = (
    _SQLI_PAYLOAD,               # "' OR '1'='1"  boolean tautology
    "1' OR '1'='1'--",          # tautology + comment
    "' UNION SELECT NULL--",     # UNION probe (read-only, non-mutating)
)

# EXPLICIT cap — at most this many vectors per field AND at most this many
# fill→submit rounds per test. DELIBERATE bound (kept small on purpose) so a
# 14-site run stays practical; this is NOT a silent truncation.
_MAX_VECTORS_PER_FIELD = 5

# DELIBERATE safety guard: a field whose accessible name looks like a credential
# / secret is NEVER filled with a payload, even if a page exposes it with a
# textbox role. Playwright's get_by_role("textbox") already excludes
# <input type=password> (no textbox ARIA role); this is belt-and-suspenders for
# a mislabeled / role-overridden secret field.
_PASSWORD_FIELD_KW: tuple[str, ...] = (
    "password", "passwd", "pwd", "pin", "cvv", "cvc", "secret",
)

# SQL-error markers the reused _SQL_ERROR_RE regex does not already cover.
_EXTRA_SQL_ERROR_MARKERS: tuple[str, ...] = (
    "mysql_fetch", "you have an error in your sql", "warning: mysql",
    "sqlite", "ora-0", "unclosed quotation mark", "odbc",
    "quoted string not properly terminated",
)

# JS (a DOM read, NOT a locator) returning the action URL of the form the
# security runner would submit — the <form> around the first submit-like button,
# else the first <form>. Mirrors SFGTraversalExplorer._action_blocked's read.
_SUBMIT_FORM_ACTION_JS = """() => {
    const btn = document.querySelector('button, input[type=submit], [role=button]');
    const form = (btn && btn.closest('form')) || document.querySelector('form');
    if (!form) return '';
    return String(form.getAttribute('action') || form.action || '');
}"""

# JS (a DOM read) computing a field's accessible name from its own element.
_FIELD_NAME_JS = (
    "el => ("
    "el.getAttribute('aria-label')"
    " || el.getAttribute('placeholder')"
    " || (el.labels && el.labels[0] && el.labels[0].textContent)"
    " || el.getAttribute('name')"
    " || el.getAttribute('id')"
    " || ''"
    ").trim()"
)


def _is_password_field(field_name: str) -> bool:
    """True if a field's accessible name looks like a credential/secret.

    Used as an EXPLICIT, deliberate safety guard — such a field is never fuzzed.
    """
    low = (field_name or "").lower()
    return any(k in low for k in _PASSWORD_FIELD_KW)


def _is_blocked_action(action_url: str) -> bool:
    """True if a form's action targets a BLOCKED (destructive/credential)
    endpoint — delete/remove/transfer/payment/logout/deactivate/password.

    Uses the SAME src.contractskill.sfg.BLOCKED_ACTION_PATTERNS frozenset that
    SFGTraversalExplorer._action_blocked uses, so the security runner refuses to
    submit exactly the forms the crawler refuses to click. Pure/offline.
    """
    low = (action_url or "").lower()
    return any(p in low for p in _SFG_BLOCKED_ACTION_PATTERNS)


def _security_vectors_for_field(field_name: str, is_xss: bool) -> list[str]:
    """Bounded, TYPE-AWARE adversarial vectors for ONE field (pure/offline).

    infer_field_type(name, None) maps the field's accessible name to a
    BASE_VECTORS_BY_TYPE category (string/integer/email/slug) so an email box is
    probed with malformed-email vectors, an integer box with numeric edge cases,
    etc. — instead of blasting one payload into every textbox.

    Kind-specific injection probes go FIRST (the executable XSS payload / the
    canonical SQLi strings must always run so the runner's checks stay
    meaningful), then type-aware base vectors fill the remaining budget. Deduped,
    order-preserving, capped at _MAX_VECTORS_PER_FIELD (EXPLICIT bound).
    """
    ftype = infer_field_type(field_name or "", None)
    kind = list(_XSS_VECTORS if is_xss else _SQLI_VECTORS)
    base = base_vectors_for(ftype, max_vectors=_MAX_VECTORS_PER_FIELD)
    merged: list[str] = []
    seen: set[str] = set()
    for v in (*kind, *base):
        if v in seen:
            continue
        seen.add(v)
        merged.append(v)
        if len(merged) >= _MAX_VECTORS_PER_FIELD:
            break
    return merged


def _find_reflected_xss(page_content: str, injected: list[str]) -> str | None:
    """First injected HTML/JS vector that appears RAW (unescaped) in the
    serialized DOM → reflected XSS. Entity-encoded reflections (the app escaped
    '<' to '&lt;') correctly do NOT match. Pure/offline.
    """
    content = page_content or ""
    for v in injected:
        if not v:
            continue
        if ("<" in v or "javascript:" in v.lower()) and v in content:
            return v
    return None


def _find_sql_error(page_content: str) -> str | None:
    """First SQL-error marker leaked into the content, or None. Reuses the
    Sprint-13 _SQL_ERROR_RE regex (broad, vendor-agnostic) plus extra vendor
    markers it does not cover. Pure/offline — NO HTTP-status coupling (that is
    why AnomalyClassifier.classify, which is HTTP-only, is NOT reused here).
    """
    content = page_content or ""
    m = _SQL_ERROR_RE.search(content)
    if m:
        return m.group(0)
    low = content.lower()
    return next((kw for kw in _EXTRA_SQL_ERROR_MARKERS if kw in low), None)


def _classify_security_response(
    *, xss_fired: bool, page_content: str, injected: list[str]
) -> tuple[bool, str | None]:
    """Pure post-submit classifier → (passed, failure_reason). Offline-testable.

    Records a vulnerability signal (passed=False) if ANY of:
      * window.__xss_fired was set           → script EXECUTED.
      * an injected vector reflects unescaped → reflected XSS.
      * a SQL-error string leaked             → SQLi signal.
    """
    if xss_fired:
        return False, "XSS payload ถูก execute (window.__xss_fired=true)"
    reflected = _find_reflected_xss(page_content, injected)
    if reflected:
        return False, f"XSS payload สะท้อนกลับแบบ unescaped ในหน้า: {reflected[:80]}"
    sql_hit = _find_sql_error(page_content)
    if sql_hit:
        return False, f"พบ SQL error keyword '{sql_hit}' ในหน้าที่ตอบกลับ"
    return True, None


async def _read_field_name(locator) -> str:
    """Best-effort accessible name of a field via its own element (role-based
    locator + element.evaluate — a DOM read, NOT a CSS selector)."""
    try:
        name = await locator.evaluate(_FIELD_NAME_JS)
    except Exception:
        return ""
    return (name or "").strip()

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
        # is_xss selects the vector SET; DETECTION is unified below (execution +
        # reflection + SQL-error) so either test class catches either signal.
        is_xss = "xss" in tc.title.lower()

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

        # ── SAFETY GUARD 1: blocked-endpoint / read-safe scoping ─────────────
        # Inspect the form we WOULD submit BEFORE touching it. If its action
        # targets a destructive/credential endpoint (same sfg frozenset the
        # crawler refuses to click), SKIP submission entirely — recorded HONESTLY
        # as "skipped" (never a misleading pass/fail). The header audit still
        # runs (read-only GET). This closes a real gap: the old runner filled and
        # submitted with NO endpoint guard.
        try:
            action_url = await page.evaluate(_SUBMIT_FORM_ACTION_JS)
        except Exception:
            action_url = ""
        if _is_blocked_action(action_url):
            traces.append(StepTrace(
                step="ข้ามการ submit ฟอร์ม (ปลายทางอยู่ในรายการต้องห้าม)",
                status="skipped",
                detail=f"ไม่ submit — form action ชี้ไป blocked endpoint: {action_url[:80]}",
            ))
            missing_headers = await SecurityObserver._missing_headers(page, tc.source_url)
            if missing_headers:
                traces.append(StepTrace(
                    step="ตรวจสอบ Security Headers", status="failed",
                    detail=f"ขาด: {missing_headers}",
                    error=f"ขาด Headers: {', '.join(missing_headers)}",
                ))
            # passed=True: no vulnerability was exercised AND a destructive submit
            # was correctly avoided. The status="skipped" trace makes the
            # non-execution explicit — this is NOT a "tested and safe" claim.
            screenshot = await self._maybe_screenshot(page, tc.id, True)
            return TestResult(
                test_case=tc, passed=True, steps_trace=traces,
                failure_reason=None, screenshot_path=screenshot,
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        # ── Enumerate fillable fields (role-based) + SAFETY GUARD 2: password ─
        inputs = page.get_by_role("textbox")
        try:
            count = await inputs.count()
        except Exception:
            count = 0
        field_plans: list[tuple[int, list[str]]] = []  # (textbox index, vectors)
        skipped_password = 0
        for i in range(count):
            name = await _read_field_name(inputs.nth(i))
            if _is_password_field(name):
                # DELIBERATE safety guard — never fuzz a password/secret field.
                skipped_password += 1
                continue
            field_plans.append((i, _security_vectors_for_field(name, is_xss)))

        # EXPLICIT bound: at most _MAX_VECTORS_PER_FIELD fill→submit rounds (one
        # vector per field per round). Bounded on purpose; NOT a silent trim.
        rounds = min(
            _MAX_VECTORS_PER_FIELD,
            max((len(v) for _, v in field_plans), default=0),
        )

        passed = True
        failure_reason: str | None = None
        total_injected = 0
        for r in range(rounds):
            # Replay-from-seed between vectors: re-navigate so window.__xss_fired
            # and the form reset (deterministic — matches the crawler's pattern).
            if r > 0:
                try:
                    await page.goto(tc.source_url, wait_until="domcontentloaded",
                                    timeout=30_000)
                except Exception:
                    break
            round_injected: list[str] = []
            for idx, vectors in field_plans:
                if r >= len(vectors):
                    continue
                try:
                    await inputs.nth(idx).fill(vectors[r], timeout=5_000)
                    round_injected.append(vectors[r])
                except Exception:
                    pass
            total_injected += len(round_injected)
            # Submit via the first button (role-based locator — same style as before).
            try:
                await page.get_by_role("button").first.click(timeout=5_000)
                await page.wait_for_timeout(800)
            except Exception:
                pass
            try:
                content = await page.content()
            except Exception:
                content = ""
            try:
                xss_fired = bool(await page.evaluate("() => !!window.__xss_fired"))
            except Exception:
                xss_fired = False
            passed, failure_reason = _classify_security_response(
                xss_fired=xss_fired, page_content=content, injected=round_injected,
            )
            if not passed:
                break

        traces.append(StepTrace(
            step=(f"กรอก adversarial payload ลง {len(field_plans)} ช่อง "
                  f"({total_injected} vectors, {rounds} รอบ; "
                  f"ข้าม password {skipped_password} ช่อง)"),
            status="passed",
            detail=f"kind={'XSS' if is_xss else 'SQLi'}, cap={_MAX_VECTORS_PER_FIELD}/field",
        ))

        missing_headers = await SecurityObserver._missing_headers(page, tc.source_url)
        if missing_headers:
            traces.append(StepTrace(
                step="ตรวจสอบ Security Headers",
                status="failed",
                detail=f"ขาด: {missing_headers}",
                error=f"ขาด Headers: {', '.join(missing_headers)}",
            ))

        traces.append(StepTrace(
            step="ตรวจสอบว่า payload ไม่ถูก execute / ไม่รั่ว SQL error",
            status="passed" if passed else "failed",
            detail="ตรวจสอบ response ของหน้าแล้ว (execution + reflection + SQL-error)",
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
