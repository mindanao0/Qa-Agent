"""Enhanced security-runner tests — richer type-aware vectors + safety guards.

Two layers, both offline (no live browser, no LLM, no network):
  * PURE helpers (vector selection, blocked-endpoint decision, password guard,
    post-submit classification) — the load-bearing logic, unit-tested directly.
  * RUNNER integration via a minimal fake Playwright page exposing only the
    methods _run_security uses (goto / get_by_role / evaluate / content /
    wait_for_timeout). Confirms the guards actually gate real submission.
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
)


# ─────────────────────────── PURE HELPERS ───────────────────────────────────

def test_password_field_guard():
    for name in ("Password", "Confirm Password", "passwd", "pwd", "PIN", "CVV", "secret key"):
        assert _is_password_field(name), name
    for name in ("Username", "Email", "Search", "Comment", "First name", ""):
        assert not _is_password_field(name), name


def test_blocked_action_uses_sfg_frozenset():
    # Same delete/remove/transfer/payment/logout/deactivate/password patterns
    # that SFGTraversalExplorer._action_blocked uses.
    for action in ("/account/delete", "/transfer/funds", "https://x/logout",
                   "/user/deactivate", "/do-payment", "/remove-item", "/change-password"):
        assert _is_blocked_action(action), action
    for action in ("/search?q=1", "/login", "/api/articles", "", "/submit-review"):
        assert not _is_blocked_action(action), action


def test_vectors_keep_executable_xss_first_when_xss():
    # The executable payload MUST be present (and first) so the window.__xss_fired
    # execution check still fires.
    for name in ("Search", "Email", "limit", "slug", "Comment"):
        v = _security_vectors_for_field(name, is_xss=True)
        assert v[0] == _XSS_PAYLOAD, name


def test_vectors_are_bounded_and_unique():
    for is_xss in (True, False):
        for name in ("Search", "Email", "limit", "slug", "username", ""):
            v = _security_vectors_for_field(name, is_xss=is_xss)
            assert 0 < len(v) <= _MAX_VECTORS_PER_FIELD  # EXPLICIT cap honored
            assert len(v) == len(set(v))                 # deduped


def test_vectors_are_read_safe_no_data_mutation():
    # No DROP/DELETE/UPDATE/INSERT/TRUNCATE — probing a vulnerable target must
    # never destroy data.
    banned = ("drop ", "delete ", "update ", "insert ", "truncate")
    for is_xss in (True, False):
        for name in ("Search", "Email", "limit", "slug"):
            for vec in _security_vectors_for_field(name, is_xss=is_xss):
                low = vec.lower()
                assert not any(b in low for b in banned), vec


def test_type_aware_selection_picks_the_right_set():
    # Email box → malformed-email vectors; integer box → numeric edge cases.
    email = _security_vectors_for_field("Email Address", is_xss=False)
    integer = _security_vectors_for_field("limit", is_xss=False)
    slug = _security_vectors_for_field("slug", is_xss=False)
    assert "notanemail" in email          # from BASE_VECTORS_BY_TYPE["email"]
    assert "notanemail" not in integer    # not blasted everywhere
    assert "0" in integer                 # from BASE_VECTORS_BY_TYPE["integer"]
    assert "0" not in email
    # the three field types yield genuinely different vector sets
    assert email != integer and integer != slug and email != slug


def test_find_reflected_xss_only_matches_unescaped():
    payload = _XSS_PAYLOAD
    assert _find_reflected_xss(f"<div>{payload}</div>", [payload]) == payload
    # entity-encoded reflection = the app escaped it = NOT a vuln
    assert _find_reflected_xss("&lt;script&gt;window.__xss_fired=true;&lt;/script&gt;",
                               [payload]) is None
    # a non-HTML (SQLi) vector never counts as reflected XSS
    assert _find_reflected_xss("' OR '1'='1 reflected here", ["' OR '1'='1"]) is None
    assert _find_reflected_xss("clean page", [payload]) is None


def test_find_sql_error_regex_and_extra_markers():
    assert _find_sql_error("You have an error in your SQL syntax near '1'")   # regex
    assert _find_sql_error('SQLITE_ERROR: near "SELECT": syntax error')       # regex
    assert _find_sql_error("Warning: mysql_fetch_array() expects")            # extra marker
    assert _find_sql_error("ORA-00933: SQL command not properly ended")       # regex ORA-\\d
    assert _find_sql_error("all good, welcome home") is None


def test_classify_prioritizes_execution_then_reflection_then_sql():
    fired, reason = _classify_security_response(xss_fired=True, page_content="", injected=[])
    assert not fired and "execute" in reason

    fired, reason = _classify_security_response(
        xss_fired=False, page_content=f"<p>{_XSS_PAYLOAD}</p>", injected=[_XSS_PAYLOAD])
    assert not fired and "unescaped" in reason

    fired, reason = _classify_security_response(
        xss_fired=False, page_content="error in your SQL syntax", injected=["' OR '1'='1"])
    assert not fired and "SQL" in reason

    ok, reason = _classify_security_response(
        xss_fired=False, page_content="welcome", injected=["' OR '1'='1"])
    assert ok and reason is None


# ─────────────────────── FAKE PLAYWRIGHT PAGE ───────────────────────────────

class _FakeField:
    """A single textbox: reports its accessible name, records fills."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.filled: list[str] = []

    async def evaluate(self, _js):        # _read_field_name(_FIELD_NAME_JS)
        return self._name

    async def fill(self, value, timeout=None):
        self.filled.append(value)


class _FakeTextboxLocator:
    def __init__(self, fields):
        self._fields = fields

    async def count(self):
        return len(self._fields)

    def nth(self, i):
        return self._fields[i]


class _FakeButton:
    def __init__(self):
        self.clicks = 0

    async def click(self, timeout=None):
        self.clicks += 1


class _FakeButtonLocator:
    def __init__(self, button):
        self.first = button


class _FakeRequest:
    async def get(self, url, timeout=None):
        # No network in tests → SecurityObserver._missing_headers catches this
        # and returns [] (no header trace), keeping the test hermetic.
        raise RuntimeError("no network in tests")


class FakePage:
    """Implements ONLY what _run_security touches."""

    def __init__(self, *, fields, action_url="", xss_fired=False, content=""):
        self._textboxes = _FakeTextboxLocator(fields)
        self.button = _FakeButton()
        self._buttons = _FakeButtonLocator(self.button)
        self._action_url = action_url
        self._xss_fired = xss_fired
        self._content = content
        self.url = "https://x.com/form"
        self.gotos: list[str] = []
        self.request = _FakeRequest()

    async def goto(self, url, **kwargs):
        self.gotos.append(url)

    def get_by_role(self, role, **kwargs):
        if role == "textbox":
            return self._textboxes
        if role == "button":
            return self._buttons
        raise AssertionError(f"unexpected role {role!r}")

    async def evaluate(self, js, *args):
        if "__xss_fired" in js:
            return self._xss_fired
        return self._action_url          # _SUBMIT_FORM_ACTION_JS

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


# ─────────────────────── RUNNER INTEGRATION ─────────────────────────────────

@pytest.mark.asyncio
async def test_field_name_reads_accessible_name():
    assert await _read_field_name(_FakeField("Email")) == "Email"


@pytest.mark.asyncio
async def test_xss_execution_is_flagged():
    page = FakePage(fields=[_FakeField("Search")], action_url="", xss_fired=True)
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    assert result.passed is False
    assert "execute" in (result.failure_reason or "")
    assert page.button.clicks >= 1                    # a submit did happen


@pytest.mark.asyncio
async def test_reflected_xss_is_flagged_without_execution():
    page = FakePage(
        fields=[_FakeField("Comment")], action_url="",
        xss_fired=False, content=f"<div>results for {_XSS_PAYLOAD}</div>",
    )
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    assert result.passed is False
    assert "unescaped" in (result.failure_reason or "")


@pytest.mark.asyncio
async def test_sql_error_in_content_is_flagged():
    page = FakePage(
        fields=[_FakeField("Search")], action_url="/search",
        xss_fired=False, content="You have an error in your SQL syntax near ''1''",
    )
    result = await _runner()._run_security(_sec_tc("ทดสอบ SQL injection"), page)
    assert result.passed is False
    assert "SQL" in (result.failure_reason or "")


@pytest.mark.asyncio
async def test_blocked_form_is_skipped_not_submitted():
    field = _FakeField("Search")
    page = FakePage(fields=[field], action_url="/account/delete-profile")
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    # blocked → NEVER submitted, NEVER filled; honest "skipped" trace, not a
    # misleading pass/fail (passed=True == destructive submit correctly avoided).
    assert page.button.clicks == 0
    assert field.filled == []
    assert any(t.status == "skipped" for t in result.steps_trace)
    assert result.passed is True
    assert result.failure_reason is None


@pytest.mark.asyncio
async def test_password_field_is_never_fuzzed():
    pw = _FakeField("Password")
    search = _FakeField("Search")
    page = FakePage(fields=[pw, search], action_url="", xss_fired=False, content="ok")
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    assert pw.filled == []              # DELIBERATE guard: password never filled
    assert search.filled               # the non-secret field WAS fuzzed
    assert result.passed is True
    assert "ข้าม password 1 ช่อง" in " ".join(t.step for t in result.steps_trace)


@pytest.mark.asyncio
async def test_clean_form_passes_and_submits():
    field = _FakeField("Search")
    page = FakePage(fields=[field], action_url="", xss_fired=False, content="welcome home")
    result = await _runner()._run_security(_sec_tc("ทดสอบ XSS injection"), page)
    assert result.passed is True
    assert result.failure_reason is None
    assert page.button.clicks >= 1
    assert field.filled                # fields were actually fuzzed
