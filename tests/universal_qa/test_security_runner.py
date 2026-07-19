"""Enhanced security-runner tests — typed vectors + safety guards, hardened after
the 2026-07-19 code review. Two layers, both offline (no live browser / LLM / net):
  * PURE helpers (vector selection, blocked-target decision, password guard,
    tight SQL signatures, inert-context-aware reflection, post-submit classifier).
  * RUNNER integration via a minimal fake Playwright page — asserts the guards
    actually gate real submission, that GET vs POST bounds the submit count, and
    that a page with nothing fuzzable is SKIPPED (not a vacuous pass).
"""
import pytest

from src.universal_qa.models import TestCase
from src.universal_qa.test_runner import (
    UniversalTestRunner,
    _MAX_VECTORS_PER_FIELD,
    _XSS_PAYLOAD,
    _classify_security_response,
    _find_reflected_xss,
    _find_sql_error,
    _is_blocked_action,
    _is_password_field,
    _read_field_name,
    _security_vectors_for_field,
    _strip_inert_html,
)


# ─────────────────────────── PURE HELPERS ───────────────────────────────────

def test_password_field_guard():
    for name in ("Password", "Confirm Password", "passwd", "pwd", "PIN", "CVV", "secret key"):
        assert _is_password_field(name), name
    for name in ("Username", "Email", "Search", "Comment", "First name", ""):
        assert not _is_password_field(name), name


def test_blocked_action_uses_sfg_frozenset():
    for token in ("/account/delete", "/transfer/funds", "https://x/logout",
                  "/user/deactivate", "/do-payment", "/remove-item", "Logout", "Delete account"):
        assert _is_blocked_action(token), token
    for token in ("/search?q=1", "/login", "/api/articles", "", "/submit-review", "Search", "Post comment"):
        assert not _is_blocked_action(token), token


def test_vectors_keep_executable_xss_first_when_xss():
    for name in ("Search", "Email", "limit", "slug", "Comment"):
        v = _security_vectors_for_field(name, is_xss=True)
        assert v[0] == _XSS_PAYLOAD, name


def test_vectors_are_bounded_and_unique():
    for is_xss in (True, False):
        for name in ("Search", "Email", "limit", "slug", "username", ""):
            v = _security_vectors_for_field(name, is_xss=is_xss)
            assert 0 < len(v) <= _MAX_VECTORS_PER_FIELD
            assert len(v) == len(set(v))


def test_vectors_are_read_safe_no_data_mutation():
    banned = ("drop ", "delete ", "update ", "insert ", "truncate")
    for is_xss in (True, False):
        for name in ("Search", "Email", "limit", "slug"):
            for vec in _security_vectors_for_field(name, is_xss=is_xss):
                assert not any(b in vec.lower() for b in banned), vec


def test_type_aware_selection_picks_the_right_set():
    email = _security_vectors_for_field("Email Address", is_xss=False)
    integer = _security_vectors_for_field("limit", is_xss=False)
    slug = _security_vectors_for_field("slug", is_xss=False)
    assert "notanemail" in email
    assert "notanemail" not in integer
    assert "0" in integer
    assert "0" not in email
    assert email != integer and integer != slug and email != slug


# ── reflection detection (inert-context aware) ──────────────────────────────

def test_find_reflected_xss_only_matches_unescaped():
    payload = _XSS_PAYLOAD
    assert _find_reflected_xss(f"<div>{payload}</div>", [payload]) == payload
    # entity-encoded reflection = escaped = NOT a vuln
    assert _find_reflected_xss("&lt;script&gt;window.__xss_fired=true;&lt;/script&gt;",
                               [payload]) is None
    # a non-HTML (SQLi) vector never counts as reflected XSS
    assert _find_reflected_xss("' OR '1'='1 reflected here", ["' OR '1'='1"]) is None
    assert _find_reflected_xss("clean page", [payload]) is None


def test_find_reflected_xss_ignores_inert_contexts():
    """A payload echoed ONLY inside a <textarea> or an HTML comment does not
    execute, so it must not be flagged (was a false positive before the fix)."""
    payload = _XSS_PAYLOAD
    assert _find_reflected_xss(f"<textarea>{payload}</textarea>", [payload]) is None
    assert _find_reflected_xss(f"<!-- {payload} -->", [payload]) is None
    # but the SAME payload also present in an executing region IS flagged
    assert _find_reflected_xss(
        f"<textarea>{payload}</textarea><div>{payload}</div>", [payload]) == payload


def test_strip_inert_html_removes_textarea_and_comments():
    stripped = _strip_inert_html("A<textarea>SECRET</textarea>B<!--HIDDEN-->C")
    assert "SECRET" not in stripped and "HIDDEN" not in stripped
    assert "A" in stripped and "B" in stripped and "C" in stripped


# ── SQL-error detection (tight, gated) ──────────────────────────────────────

def test_find_sql_error_matches_specific_signatures():
    assert _find_sql_error("You have an error in your SQL syntax near '1'")
    assert _find_sql_error('SQLITE_ERROR: near "SELECT": syntax error')
    assert _find_sql_error("Warning: mysql_fetch_array() expects")
    assert _find_sql_error("ORA-00933: SQL command not properly ended")
    assert _find_sql_error("all good, welcome home") is None


def test_find_sql_error_ignores_bare_vendor_names():
    """The old loose markers ('sqlite'/'odbc'/'ora-0'/'near \"') false-flagged
    benign pages. The tight signatures must NOT match any of these."""
    for benign in ("Powered by SQLite", "install the ODBC driver",
                   "product code Aurora-0 in stock", 'click near "here"',
                   "our database uses PostgreSQL"):
        assert _find_sql_error(benign) is None, benign


# ── unified classifier (SQL gated to SQLi) ──────────────────────────────────

def test_classify_prioritizes_execution_then_reflection_then_sql():
    fired, reason = _classify_security_response(
        xss_fired=True, page_content="", injected=[], is_sqli=False)
    assert not fired and "execute" in reason

    fired, reason = _classify_security_response(
        xss_fired=False, page_content=f"<p>{_XSS_PAYLOAD}</p>",
        injected=[_XSS_PAYLOAD], is_sqli=False)
    assert not fired and "unescaped" in reason

    fired, reason = _classify_security_response(
        xss_fired=False, page_content="you have an error in your SQL syntax",
        injected=["' OR '1'='1"], is_sqli=True)
    assert not fired and "SQL" in reason

    ok, reason = _classify_security_response(
        xss_fired=False, page_content="welcome", injected=["' OR '1'='1"], is_sqli=True)
    assert ok and reason is None


def test_classify_sql_detection_gated_to_sqli():
    """An XSS run (is_sqli=False) against a page that merely mentions a DB error
    string must NOT be reported as a vulnerability."""
    ok, reason = _classify_security_response(
        xss_fired=False, page_content="you have an error in your sql syntax",
        injected=[_XSS_PAYLOAD], is_sqli=False)
    assert ok and reason is None
    # the SAME content on a SQLi run IS flagged
    bad, reason = _classify_security_response(
        xss_fired=False, page_content="you have an error in your sql syntax",
        injected=["' OR '1'='1"], is_sqli=True)
    assert not bad and "SQL" in reason


# ─────────────────────── FAKE PLAYWRIGHT PAGE ───────────────────────────────

class _FakeField:
    """A single fuzzable field: reports its accessible name, records fills."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.filled: list[str] = []

    async def evaluate(self, _js):          # _read_field_name(_FIELD_NAME_JS)
        return self._name

    async def fill(self, value, timeout=None):
        self.filled.append(value)


class _FakeLocatorList:
    def __init__(self, fields):
        self._fields = fields

    async def count(self):
        return len(self._fields)

    def nth(self, i):
        return self._fields[i]


class _FakeButton:
    """The submit control. evaluate() returns the {action, method, label} dict
    that _SUBMIT_TARGET_JS would — this is the SAME element that gets clicked, so
    the guard governs the real submit."""

    def __init__(self, action="", method="get", label="Submit"):
        self._t = {"action": action, "method": method, "label": label}
        self.clicks = 0

    async def evaluate(self, _js):          # submit.evaluate(_SUBMIT_TARGET_JS)
        return dict(self._t)

    async def click(self, timeout=None):
        self.clicks += 1


class _FakeButtonLocator:
    def __init__(self, button):
        self.first = button


class _FakeRequest:
    async def get(self, url, timeout=None):
        raise RuntimeError("no network in tests")  # _missing_headers -> []


class FakePage:
    """Implements ONLY what _run_security / _finish_security_skipped touch."""

    def __init__(self, *, textboxes=None, searchboxes=None, button=None,
                 xss_fired=False, content=""):
        self._roles = {
            "textbox": _FakeLocatorList(textboxes or []),
            "searchbox": _FakeLocatorList(searchboxes or []),
        }
        self.button = button if button is not None else _FakeButton()
        self._buttons = _FakeButtonLocator(self.button)
        self._xss_fired = xss_fired
        self._content = content
        self.url = "https://x.com/form"
        self.gotos: list[str] = []
        self.request = _FakeRequest()

    async def goto(self, url, **kwargs):
        self.gotos.append(url)

    def get_by_role(self, role, **kwargs):
        if role == "button":
            return self._buttons
        if role in self._roles:
            return self._roles[role]
        raise AssertionError(f"unexpected role {role!r}")

    async def evaluate(self, js, *args):     # only the __xss_fired probe now
        if "__xss_fired" in js:
            return self._xss_fired
        return None

    async def content(self):
        return self._content

    async def wait_for_timeout(self, ms):
        return None


def _runner():
    r = UniversalTestRunner.__new__(UniversalTestRunner)
    r._screenshot_dir = None
    r._terminal = None
    return r


def _sec_tc(title: str) -> TestCase:
    return TestCase(
        title=title, type="security", priority="high",
        steps=["เปิดหน้า https://x.com/form", "กด submit form"],
        expected_outcome="ปลอดภัย", source_url="https://x.com/form",
    )


def _skipped(result) -> bool:
    return any(t.status == "skipped" for t in result.steps_trace)


# ─────────────────────── RUNNER INTEGRATION ─────────────────────────────────

@pytest.mark.asyncio
async def test_field_name_reads_accessible_name():
    assert await _read_field_name(_FakeField("Email")) == "Email"


@pytest.mark.asyncio
async def test_xss_execution_is_flagged():
    page = FakePage(textboxes=[_FakeField("Search")],
                    button=_FakeButton(action="/search", method="get"), xss_fired=True)
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    assert result.passed is False
    assert "execute" in (result.failure_reason or "")
    assert page.button.clicks >= 1


@pytest.mark.asyncio
async def test_reflected_xss_is_flagged_without_execution():
    page = FakePage(textboxes=[_FakeField("Comment")],
                    button=_FakeButton(action="/search", method="get"),
                    xss_fired=False, content=f"<div>results for {_XSS_PAYLOAD}</div>")
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    assert result.passed is False
    assert "unescaped" in (result.failure_reason or "")


@pytest.mark.asyncio
async def test_reflected_xss_in_textarea_is_not_flagged():
    """Payload echoed only inside a <textarea> is inert → must pass (no false pos)."""
    page = FakePage(textboxes=[_FakeField("Comment")],
                    button=_FakeButton(action="/search", method="get"),
                    xss_fired=False, content=f"<textarea>{_XSS_PAYLOAD}</textarea>")
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    assert result.passed is True


@pytest.mark.asyncio
async def test_sql_error_in_content_is_flagged_on_sqli():
    page = FakePage(textboxes=[_FakeField("id")],
                    button=_FakeButton(action="/search", method="get"),
                    xss_fired=False, content="You have an error in your SQL syntax near ''1''")
    result = await _runner()._run_security(_sec_tc("ทดสอบ SQL injection"), page)
    assert result.passed is False
    assert "SQL" in (result.failure_reason or "")


@pytest.mark.asyncio
async def test_xss_run_against_sqlite_mention_is_not_flagged():
    """XSS run: SQL-error detection is gated OFF, so a page mentioning a DB error
    string is not misreported (was a false SQLi positive before the fix)."""
    page = FakePage(textboxes=[_FakeField("Search")],
                    button=_FakeButton(action="/search", method="get"),
                    xss_fired=False, content="Powered by SQLite. you have an error in your sql syntax")
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    assert result.passed is True


@pytest.mark.asyncio
async def test_blocked_form_action_is_skipped_not_submitted():
    field = _FakeField("Search")
    page = FakePage(textboxes=[field],
                    button=_FakeButton(action="/account/delete-profile", method="post", label="Delete"))
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    assert page.button.clicks == 0
    assert field.filled == []
    assert _skipped(result)
    assert result.passed is True and result.failure_reason is None


@pytest.mark.asyncio
async def test_blocked_by_non_form_button_label_is_skipped():
    """A JS Logout/Delete control with NO form (method="") is caught by its label
    — the old guard, which only read a <form> action, let it through and clicked."""
    field = _FakeField("Search")
    page = FakePage(textboxes=[field],
                    button=_FakeButton(action="", method="", label="Logout"))
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    assert page.button.clicks == 0
    assert field.filled == []
    assert _skipped(result)
    assert result.passed is True


@pytest.mark.asyncio
async def test_password_field_is_never_fuzzed():
    pw = _FakeField("Password")
    search = _FakeField("Search")
    page = FakePage(textboxes=[pw, search],
                    button=_FakeButton(action="/search", method="get"), content="ok")
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    assert pw.filled == []
    assert search.filled
    assert result.passed is True
    assert "ข้าม password 1 ช่อง" in " ".join(t.step for t in result.steps_trace)


@pytest.mark.asyncio
async def test_searchbox_field_is_fuzzed():
    """<input type=search> exposes role searchbox — must be fuzzed, not skipped."""
    sb = _FakeField("Site search")
    page = FakePage(textboxes=[], searchboxes=[sb],
                    button=_FakeButton(action="/search", method="get"), content="welcome")
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    assert sb.filled          # the searchbox was actually fuzzed
    assert result.passed is True
    assert not _skipped(result)


@pytest.mark.asyncio
async def test_no_fuzzable_field_is_skipped_not_vacuous_pass():
    """No textbox/searchbox (only password, here modelled as none) → the test is
    SKIPPED, never a silent passed=True with no payload injected."""
    page = FakePage(textboxes=[], searchboxes=[],
                    button=_FakeButton(action="/search", method="get"))
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    assert page.button.clicks == 0          # nothing submitted
    assert _skipped(result)
    assert "ไม่มีช่องกรอก" in " ".join(t.step for t in result.steps_trace)


@pytest.mark.asyncio
async def test_post_form_submitted_at_most_once():
    """READ-SAFE: a POST (mutating) form is submitted ONCE even though the field
    has multiple vectors — no 5× stored records."""
    field = _FakeField("Comment")
    assert len(_security_vectors_for_field("Comment", is_xss=True)) > 1  # would be >1 round if GET
    page = FakePage(textboxes=[field],
                    button=_FakeButton(action="/post-comment", method="post"), content="thanks")
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    assert page.button.clicks == 1
    assert len(field.filled) == 1
    assert result.passed is True


@pytest.mark.asyncio
async def test_get_form_runs_multiple_rounds():
    """A GET (idempotent) form is fuzzed across multiple rounds."""
    field = _FakeField("Search")
    n_vectors = len(_security_vectors_for_field("Search", is_xss=True))
    assert n_vectors > 1
    page = FakePage(textboxes=[field],
                    button=_FakeButton(action="/search", method="get"), content="welcome")
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    assert page.button.clicks == n_vectors == len(field.filled)
    assert page.gotos.count("https://x.com/form") == n_vectors  # initial + re-navs
    assert result.passed is True


@pytest.mark.asyncio
async def test_clean_form_passes_and_submits():
    field = _FakeField("Search")
    page = FakePage(textboxes=[field],
                    button=_FakeButton(action="/search", method="get"),
                    xss_fired=False, content="welcome home")
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    assert result.passed is True and result.failure_reason is None
    assert page.button.clicks >= 1
    assert field.filled
