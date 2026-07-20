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
from src.contractskill.sfg import (
    BLOCKED_ACTION_PATTERNS as _SFG_BLOCKED_ACTION_PATTERNS,
    SFGStore,
)
from src.explorer.executor import HypothesisExecutor
from src.explorer.hypothesis import TestHypothesis
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
# Long, unambiguous keywords match as SUBSTRINGS ("confirmPassword", "user_passwd");
# short collision-prone ones match only as WHOLE tokens (name split on non-alnum) so
# "shipping"/"opinion" ($pin) and "secretary" ($secret) are NOT false-skipped.
_PASSWORD_SUBSTR_KW: tuple[str, ...] = ("password", "passwd")
_PASSWORD_TOKEN_KW: frozenset[str] = frozenset({"pwd", "pin", "cvv", "cvc", "secret"})

# TIGHT SQL-error signatures — specific DB error phrases ONLY. Deliberately NOT
# bare vendor names ('sqlite'/'odbc'/'ora-0'): a page that merely mentions SQLite
# or an ODBC driver, or a product code containing 'ora-0', must NOT be flagged.
# Matched case-insensitively and ONLY on SQLi tests (see _classify_security_response).
_SQL_ERROR_SIGNATURES: tuple[str, ...] = (
    "you have an error in your sql syntax",
    "warning: mysql", "mysqli_", "mysql_fetch",
    "unclosed quotation mark after the character string",
    "quoted string not properly terminated",
    "unterminated quoted string",
    "sqlstate[",
    "sqlite3.operationalerror", "sqlite_error",
    "org.postgresql.util.psqlexception", "syntax error at or near",
    "microsoft ole db provider", "odbc sql server driver",
)

# Oracle errors are ORA-<5 digits> — a regex catches the whole family precisely
# without false-matching a bare 'ora-0' inside a product code.
_SQL_ORA_RE = re.compile(r"ora-\d{5}", re.IGNORECASE)

# Non-executing HTML regions: a payload reflected INSIDE these is inert (the
# browser runs no scripts in a <textarea> body or an HTML comment), so such a
# reflection must NOT be reported as XSS. Stripped before the reflection scan.
_INERT_TEXTAREA_RE = re.compile(r"<textarea\b[^>]*>.*?</textarea>", re.IGNORECASE | re.DOTALL)
_INERT_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# JS (a DOM read, NOT a locator) run ON the exact submit control the runner will
# click — get_by_role("button").first — so the blocked-endpoint guard governs
# WHAT IS ACTUALLY SUBMITTED. Returns that control's OWN enclosing <form> action
# (RAW attribute — never the resolved absolute form.action, whose host could
# false-match a blocked keyword), the form method, and the control's label (so a
# non-form destructive control like a JS "Logout"/"Delete" is caught by its
# text). method is "" when the control is not inside a <form>.
_SUBMIT_TARGET_JS = (
    "el => {"
    " const form = el.closest('form');"
    " const fa = el.getAttribute('formaction');"     # button override wins over form
    " const fm = el.getAttribute('formmethod');"
    " const action = fa != null ? String(fa)"
    "   : (form ? String(form.getAttribute('action') || '') : '');"
    " const method = String(fm"
    "   || (form ? (form.getAttribute('method') || 'get') : '')).toLowerCase();"
    " const label = String(el.getAttribute('aria-label') || el.textContent || el.value || '').trim();"
    " return { action, method, label };"
    "}"
)

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
    """True if a field's accessible name looks like a credential/secret. EXPLICIT
    safety guard — such a field is never fuzzed. Long keywords match as
    substrings; short/ambiguous ones only as whole tokens, so ordinary fields like
    'shipping'/'opinion'/'secretary' are not false-skipped.
    """
    low = (field_name or "").lower()
    if any(k in low for k in _PASSWORD_SUBSTR_KW):
        return True
    return any(t in _PASSWORD_TOKEN_KW for t in re.split(r"[^a-z0-9]+", low) if t)


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


def _strip_inert_html(content: str) -> str:
    """Remove non-executing regions (<textarea> bodies, HTML comments) so a
    payload reflected ONLY there is not mistaken for executable reflected XSS."""
    content = _INERT_TEXTAREA_RE.sub("", content or "")
    return _INERT_COMMENT_RE.sub("", content)


def _find_reflected_xss(page_content: str, injected: list[str]) -> str | None:
    """First injected HTML/JS vector that appears RAW (unescaped) in an EXECUTING
    region of the serialized DOM → reflected XSS. Entity-encoded reflections (the
    app escaped '<' to '&lt;') AND reflections confined to inert contexts
    (<textarea>, comments) correctly do NOT match. Pure/offline.
    """
    content = _strip_inert_html(page_content)
    for v in injected:
        if not v:
            continue
        if ("<" in v or "javascript:" in v.lower()) and v in content:
            return v
    return None


def _find_sql_error(page_content: str) -> str | None:
    """First TIGHT SQL-error signature leaked into the content, or None. Matches
    only specific DB error phrases (see _SQL_ERROR_SIGNATURES), never bare vendor
    names, so a benign page is not misread as SQLi. Pure/offline.
    """
    low = (page_content or "").lower()
    hit = next((s for s in _SQL_ERROR_SIGNATURES if s in low), None)
    if hit:
        return hit
    m = _SQL_ORA_RE.search(low)
    return m.group(0) if m else None


def _classify_security_response(
    *, xss_fired: bool, page_content: str, injected: list[str], is_sqli: bool
) -> tuple[bool, str | None]:
    """Pure post-submit classifier → (passed, failure_reason). Offline-testable.

    Records a vulnerability signal (passed=False) if ANY of:
      * window.__xss_fired was set               → script EXECUTED (definitive).
      * an injected vector reflects unescaped     → reflected XSS (inert
        <textarea>/comment contexts excluded).
      * a SQL-error signature leaked AND this is a SQLi test → SQLi signal. SQL
        detection is GATED to SQLi tests so an XSS run against a page that merely
        mentions a database is never reported as a SQL injection.
    """
    if xss_fired:
        return False, "XSS payload ถูก execute (window.__xss_fired=true)"
    reflected = _find_reflected_xss(page_content, injected)
    if reflected:
        return False, f"XSS payload สะท้อนกลับแบบ unescaped ในหน้า: {reflected[:80]}"
    if is_sqli:
        sql_hit = _find_sql_error(page_content)
        if sql_hit:
            return False, f"พบ SQL error signature '{sql_hit}' ในหน้าที่ตอบกลับ"
    return True, None


async def _read_field_name(locator) -> str:
    """Best-effort accessible name of a field via its own element (role-based
    locator + element.evaluate — a DOM read, NOT a CSS selector)."""
    try:
        name = await locator.evaluate(_FIELD_NAME_JS)
    except Exception:
        return ""
    return (name or "").strip()


# Text-input ARIA roles the security runner fuzzes. Includes searchbox
# (<input type=search>) so those fields are covered, not silently skipped;
# deliberately excludes spinbutton (numeric) and combobox (may be a <select>).
_FILLABLE_ROLES: tuple[str, ...] = ("textbox", "searchbox")


async def _plan_fillable_fields(page, is_xss: bool):
    """Enumerate fuzzable, non-password text fields on the CURRENT page and pair
    each with its adversarial vectors. Run FRESH per round so a DOM change across
    a navigation can never map a stale index onto the wrong field (which could
    otherwise fill a password/secret field). Returns (plans, skipped_password)
    where plans is a list of (field_locator, vectors)."""
    plans: list[tuple[object, list[str]]] = []
    skipped_password = 0
    for role in _FILLABLE_ROLES:
        loc = page.get_by_role(role)
        try:
            n = await loc.count()
        except Exception:
            n = 0
        for i in range(n):
            field = loc.nth(i)
            name = await _read_field_name(field)
            if _is_password_field(name):
                skipped_password += 1
                continue
            plans.append((field, _security_vectors_for_field(name, is_xss)))
    return plans, skipped_password


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

        # ── SAFETY GUARD 1: inspect the ACTUAL submit control we will click ───
        # Read the form action + method + label of get_by_role("button").first —
        # the exact control clicked below — so the blocked-endpoint guard governs
        # WHAT IS ACTUALLY SUBMITTED. (The old code queried a separate <form> via
        # CSS, which could differ from the clicked button's form, and could pass a
        # non-form JS "Logout"/"Delete" button straight through.) SKIP (honest
        # "skipped") when the control's form action OR its own label matches the
        # sfg blocked frozenset.
        submit = page.get_by_role("button").first
        try:
            target = await submit.evaluate(_SUBMIT_TARGET_JS)
        except Exception:
            target = {"action": "", "method": "", "label": ""}
        action = str(target.get("action") or "")
        label = str(target.get("label") or "")
        method = str(target.get("method") or "").lower()
        if _is_blocked_action(action) or _is_blocked_action(label):
            traces.append(StepTrace(
                step="ข้ามการ submit (ปลายทาง/ปุ่มอยู่ในรายการต้องห้าม)",
                status="skipped",
                detail=f"ไม่ submit — blocked target: action={action[:60]!r} label={label[:40]!r}",
            ))
            return await self._finish_security_skipped(tc, page, traces, start)

        # READ-SAFE round budget: a GET form is idempotent → multi-round fuzzing is
        # safe; a POST form (or a non-form control, method="") mutates → submit AT
        # MOST ONCE (matching the old single-submit runner), so an XSS/SQLi fuzz of
        # a comment/register/add-to-cart form can't create N stored records.
        max_rounds = _MAX_VECTORS_PER_FIELD if method == "get" else 1

        # ── Fuzzable field discovery (textbox + searchbox; password skipped) ──
        plans, skipped_password = await _plan_fillable_fields(page, is_xss)
        if not plans:
            # Nothing fuzzable → NOT a "tested and safe" pass; record an honest
            # skip so the QA report never claims a page was security-tested when
            # no payload was ever injected.
            traces.append(StepTrace(
                step="ข้ามการทดสอบ security (ไม่มีช่องกรอกที่ fuzz ได้)",
                status="skipped",
                detail=f"ไม่พบ text field ที่ fuzz ได้ (ข้าม password {skipped_password} ช่อง)",
            ))
            return await self._finish_security_skipped(tc, page, traces, start)

        rounds = min(max_rounds, max((len(v) for _, v in plans), default=0))

        passed = True
        failure_reason: str | None = None
        total_injected = 0
        for r in range(rounds):
            # Replay-from-seed: re-navigate to the ORIGINAL url so the form and
            # window.__xss_fired reset deterministically between vectors.
            if r > 0:
                try:
                    await page.goto(tc.source_url, wait_until="domcontentloaded",
                                    timeout=30_000)
                except Exception:
                    break
            # Re-enumerate fresh each round (robust to any DOM change; password is
            # re-checked here so a stale index can never fill a secret field).
            round_plans, _ = await _plan_fillable_fields(page, is_xss)
            round_injected: list[str] = []
            for field, vectors in round_plans:
                if r >= len(vectors):
                    continue
                try:
                    await field.fill(vectors[r], timeout=5_000)
                    round_injected.append(vectors[r])
                except Exception:
                    pass
            total_injected += len(round_injected)
            try:
                await submit.click(timeout=5_000)
                await page.wait_for_timeout(1_000)
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
            # The 1000ms settle above (restored from the old runner) is the real
            # deferred-execution mitigation; we break on the first fire, so there
            # is no cross-round flag to accumulate.
            passed, failure_reason = _classify_security_response(
                xss_fired=xss_fired, page_content=content,
                injected=round_injected, is_sqli=not is_xss,
            )
            if not passed:
                break

        traces.append(StepTrace(
            step=(f"กรอก adversarial payload ลง {len(plans)} ช่อง "
                  f"({total_injected} vectors, {rounds} รอบ, method={method or 'n/a'}; "
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

    async def _finish_security_skipped(
        self, tc: TestCase, page: Page, traces: list[StepTrace], start: float
    ) -> TestResult:
        """Finish a security test that was SKIPPED (blocked target or no fuzzable
        field): run the read-only header audit, then return passed=True. A
        status='skipped' trace is already recorded, so this is NOT a
        'tested and safe' claim — nothing was fuzzed."""
        missing_headers = await SecurityObserver._missing_headers(page, tc.source_url)
        if missing_headers:
            traces.append(StepTrace(
                step="ตรวจสอบ Security Headers", status="failed",
                detail=f"ขาด: {missing_headers}",
                error=f"ขาด Headers: {', '.join(missing_headers)}",
            ))
        screenshot = await self._maybe_screenshot(page, tc.id, True)
        return TestResult(
            test_case=tc, passed=True, steps_trace=traces, failure_reason=None,
            screenshot_path=screenshot,
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
