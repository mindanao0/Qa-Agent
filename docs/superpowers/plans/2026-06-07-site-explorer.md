# SiteExplorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an interaction-based exploration phase to the Universal QA Agent that systematically clicks every element on every discovered page, building a NavigationMap of real action→page paths, then feeds those real paths to the test planner so generated test steps are correct by construction.

**Architecture:** A new `SiteExplorer` runs after `SiteDiscovery`. For each discovered URL it scans interactive elements (ElementScanner), clicks each sequentially, fills forms with dummy/LLM data (FormFiller), recovers from session loss (SessionGuard), records each successful action as an `ExploredAction`, detects multi-step flows, and returns a `NavigationMap`. `UniversalTestPlanner` consumes the NavigationMap to produce per-page + flow-based test cases.

**Tech Stack:** Python 3.13, Playwright async, Pydantic V2, InstructorClient (Ollama), loguru, uv. Reuses `SFGCrawler._visit_node`, `AuthManager`, `is_safe_action`/`BLOCKED_ACTION_PATTERNS`.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/universal_qa/explorer/__init__.py` | Create | Package marker |
| `src/universal_qa/explorer/nav_map.py` | Create | ElementCandidate, ExploredAction, ExploredPage, NavigationFlow, NavigationMap, ExplorerConfig models |
| `src/universal_qa/explorer/element_scanner.py` | Create | Scan + classify interactive elements (incl. iframe/shadow DOM) |
| `src/universal_qa/explorer/form_filler.py` | Create | Dummy data + LLM fallback for forms |
| `src/universal_qa/explorer/session_guard.py` | Create | Detect session expiry + re-auth |
| `src/universal_qa/explorer/site_explorer.py` | Create | SiteExplorer orchestrator |
| `src/universal_qa/test_planner.py` | Modify | Accept NavigationMap; add `_plan_flows()` + `_action_to_step()` |
| `src/universal_qa/agent.py` | Modify | Insert Phase 3 (SiteExplorer); pass NavigationMap to planner |
| `src/universal_qa/__main__.py` | Modify | Add `--explore-timeout`, `--max-depth`, `--allow-destructive` |
| `tests/universal_qa/test_nav_map.py` | Create | Model validation tests |
| `tests/universal_qa/test_element_scanner.py` | Create | Scanner classification tests |
| `tests/universal_qa/test_form_filler.py` | Create | Dummy-data selection tests |
| `tests/universal_qa/test_session_guard.py` | Create | Session detection tests |
| `tests/universal_qa/test_site_explorer.py` | Create | Explorer logic tests (mocked page) |
| `tests/universal_qa/test_planner.py` | Modify | Update for NavigationMap input |

---

## Task 1: NavigationMap Data Models

**Files:**
- Create: `src/universal_qa/explorer/__init__.py`
- Create: `src/universal_qa/explorer/nav_map.py`
- Create: `tests/universal_qa/test_nav_map.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/universal_qa/test_nav_map.py
import pytest
from pydantic import ValidationError
from src.universal_qa.explorer.nav_map import (
    ElementCandidate, ExploredAction, ExploredPage, NavigationFlow,
    NavigationMap, ExplorerConfig,
)


def test_element_candidate_minimal():
    ec = ElementCandidate(label="Add to cart", priority=1)
    assert ec.role is None
    assert ec.is_in_iframe is False
    assert ec.is_in_shadow is False


def test_explored_action_defaults():
    a = ExploredAction(page_url="https://x.com/a", action_label="Cart")
    assert a.leads_to_url is None
    assert a.leads_to_modal is False
    assert a.state_change is None
    assert a.is_destructive is False


def test_explored_action_rejects_extra():
    with pytest.raises(ValidationError):
        ExploredAction(page_url="https://x.com", action_label="X", bogus=1)


def test_explored_page_defaults():
    p = ExploredPage(url="https://x.com/p", title="P", pam_content="...", actions=[])
    assert p.state_snapshot is None
    assert p.requires_path == []


def test_navigation_flow_holds_steps():
    a = ExploredAction(page_url="https://x.com/a", action_label="Go")
    flow = NavigationFlow(flow_id="f1", name="checkout", steps=[a],
                          start_url="https://x.com/a", end_url="https://x.com/b")
    assert len(flow.steps) == 1


def test_navigation_map_aggregates():
    nm = NavigationMap(base_url="https://x.com", pages=[], flows=[],
                       explored_at_iso="2026-06-07T00:00:00Z")
    assert nm.pages == []
    assert nm.flows == []


def test_explorer_config_defaults():
    cfg = ExplorerConfig()
    assert cfg.max_pages == 50
    assert cfg.explore_timeout_min == 5
    assert cfg.max_depth == 4
    assert cfg.allow_destructive is False
    assert cfg.max_visits_per_url == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/universal_qa/test_nav_map.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.universal_qa.explorer'`

- [ ] **Step 3: Create the package marker**

```python
# src/universal_qa/explorer/__init__.py
# (empty)
```

- [ ] **Step 4: Implement the models**

```python
# src/universal_qa/explorer/nav_map.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ElementCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    role: str | None = None
    name: str | None = None
    selector: str | None = None
    is_in_iframe: bool = False
    is_in_shadow: bool = False
    priority: int = 2  # 0=nav, 1=form-button, 2=other


class ExploredAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_url: str
    action_label: str
    element_role: str | None = None
    element_name: str | None = None
    element_selector: str | None = None
    leads_to_url: str | None = None
    leads_to_modal: bool = False
    state_change: dict | None = None
    is_destructive: bool = False


class ExploredPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    title: str
    pam_content: str
    actions: list[ExploredAction]
    state_snapshot: str | None = None
    requires_path: list[ExploredAction] = Field(default_factory=list)


class NavigationFlow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flow_id: str
    name: str
    steps: list[ExploredAction]
    start_url: str
    end_url: str


class NavigationMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    pages: list[ExploredPage]
    flows: list[NavigationFlow]
    explored_at_iso: str


class ExplorerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_pages: int = 50
    explore_timeout_min: int = 5
    max_depth: int = 4
    allow_destructive: bool = False
    max_visits_per_url: int = 2


__all__ = [
    "ElementCandidate", "ExploredAction", "ExploredPage",
    "NavigationFlow", "NavigationMap", "ExplorerConfig",
]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/universal_qa/test_nav_map.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```
git add src/universal_qa/explorer/__init__.py src/universal_qa/explorer/nav_map.py tests/universal_qa/test_nav_map.py
git commit -m "feat(explorer): add NavigationMap data models"
```

---

## Task 2: Element Scanner

**Files:**
- Create: `src/universal_qa/explorer/element_scanner.py`
- Create: `tests/universal_qa/test_element_scanner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/universal_qa/test_element_scanner.py
import pytest
from unittest.mock import AsyncMock
from src.universal_qa.explorer.element_scanner import ElementScanner
from src.universal_qa.explorer.nav_map import ElementCandidate


def test_classify_priority_nav():
    assert ElementScanner._classify_priority("nav", "Home") == 0
    assert ElementScanner._classify_priority("banner", "Logo") == 0


def test_classify_priority_form_button():
    assert ElementScanner._classify_priority("button", "Submit") == 1


def test_classify_priority_other():
    assert ElementScanner._classify_priority("link", "Random") == 2


def test_raw_to_candidates_sorts_by_priority():
    scanner = ElementScanner()
    raw = [
        {"label": "Random link", "role": "link", "name": "Random",
         "selector": "a.x", "container": "main", "is_in_iframe": False,
         "is_in_shadow": False},
        {"label": "Home", "role": "link", "name": "Home",
         "selector": "a.home", "container": "nav", "is_in_iframe": False,
         "is_in_shadow": False},
    ]
    cands = scanner._raw_to_candidates(raw)
    assert isinstance(cands[0], ElementCandidate)
    # nav (priority 0) must sort before other (priority 2)
    assert cands[0].label == "Home"


@pytest.mark.asyncio
async def test_scan_returns_candidates():
    scanner = ElementScanner()
    page = AsyncMock()
    page.evaluate = AsyncMock(return_value=[
        {"label": "Buy", "role": "button", "name": "Buy", "selector": "button.buy",
         "container": "form", "is_in_iframe": False, "is_in_shadow": False},
    ])
    page.frames = []
    cands = await scanner.scan(page)
    assert len(cands) == 1
    assert cands[0].label == "Buy"
    assert cands[0].priority == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/universal_qa/test_element_scanner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the scanner**

```python
# src/universal_qa/explorer/element_scanner.py
from __future__ import annotations

from loguru import logger
from playwright.async_api import Page

from src.universal_qa.explorer.nav_map import ElementCandidate

# Roles/containers that indicate primary navigation.
_NAV_CONTAINERS = frozenset({"nav", "banner", "header", "navigation", "menu"})
_FORM_ROLES = frozenset({"button"})

# JS that collects interactive elements from main doc + shadow roots.
_SCAN_JS = """
() => {
    const out = [];
    const seen = new Set();
    function describe(el, container) {
        const role = el.getAttribute('role')
            || ({BUTTON:'button', A:'link', INPUT:'textbox', SELECT:'combobox'})[el.tagName] || null;
        const name = (el.getAttribute('aria-label')
            || el.textContent || el.value || '').trim().slice(0, 60);
        const label = name || role || el.tagName.toLowerCase();
        if (!label || seen.has(container + '|' + label)) return;
        seen.add(container + '|' + label);
        const sel = el.id ? '#' + el.id
            : (el.className && typeof el.className === 'string' && el.className.trim()
                ? el.tagName.toLowerCase() + '.' + el.className.trim().split(/\\s+/)[0]
                : el.tagName.toLowerCase());
        out.push({label, role, name: name || null, selector: sel,
                  container, is_in_iframe: false, is_in_shadow: container === 'shadow'});
    }
    const SEL = 'a, button, [role=button], [role=link], input[type=submit], select, [onclick]';
    function containerOf(el) {
        let p = el;
        while (p) {
            const tag = (p.tagName || '').toLowerCase();
            const role = p.getAttribute && p.getAttribute('role');
            if (tag === 'nav' || tag === 'header' || role === 'navigation' || role === 'banner') return 'nav';
            if (tag === 'form') return 'form';
            p = p.parentElement;
        }
        return 'main';
    }
    document.querySelectorAll(SEL).forEach(el => describe(el, containerOf(el)));
    // shadow DOM piercing
    document.querySelectorAll('*').forEach(host => {
        if (host.shadowRoot) {
            host.shadowRoot.querySelectorAll(SEL).forEach(el => describe(el, 'shadow'));
        }
    });
    return out.slice(0, 40);
}
"""


class ElementScanner:
    """Scans a page for interactive elements (main doc + shadow + same-origin iframes)."""

    @staticmethod
    def _classify_priority(container_or_role: str, label: str) -> int:
        c = (container_or_role or "").lower()
        if c in _NAV_CONTAINERS:
            return 0
        if c in _FORM_ROLES or c == "form":
            return 1
        return 2

    def _raw_to_candidates(self, raw: list[dict]) -> list[ElementCandidate]:
        cands: list[ElementCandidate] = []
        for r in raw:
            container = r.get("container", "main")
            role = r.get("role")
            # priority by container first, then role
            prio = self._classify_priority(container, r.get("label", ""))
            if prio == 2 and role:
                prio = self._classify_priority(role, r.get("label", ""))
            cands.append(ElementCandidate(
                label=r.get("label", ""),
                role=role,
                name=r.get("name"),
                selector=r.get("selector"),
                is_in_iframe=bool(r.get("is_in_iframe")),
                is_in_shadow=bool(r.get("is_in_shadow")),
                priority=prio,
            ))
        cands.sort(key=lambda c: c.priority)
        return cands

    async def scan(self, page: Page) -> list[ElementCandidate]:
        try:
            raw: list[dict] = await page.evaluate(_SCAN_JS)
        except Exception as exc:
            logger.warning(f"ElementScanner: main scan failed — {exc!r}")
            raw = []

        # same-origin iframes
        for frame in getattr(page, "frames", []):
            try:
                if frame == page.main_frame:
                    continue
                furl = frame.url
                if not furl or not furl.startswith("http"):
                    continue
                fraw = await frame.evaluate(_SCAN_JS)
                for item in fraw:
                    item["is_in_iframe"] = True
                raw.extend(fraw)
            except Exception:
                continue

        return self._raw_to_candidates(raw)


__all__ = ["ElementScanner"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/universal_qa/test_element_scanner.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```
git add src/universal_qa/explorer/element_scanner.py tests/universal_qa/test_element_scanner.py
git commit -m "feat(explorer): add ElementScanner — interactive element scan with priority + shadow/iframe"
```

---

## Task 3: Form Filler

**Files:**
- Create: `src/universal_qa/explorer/form_filler.py`
- Create: `tests/universal_qa/test_form_filler.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/universal_qa/test_form_filler.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.universal_qa.explorer.form_filler import FormFiller, _dummy_for_type


def test_dummy_for_email():
    assert "@" in _dummy_for_type("email")


def test_dummy_for_password():
    assert _dummy_for_type("password") == "QaTest123!"


def test_dummy_for_number():
    assert _dummy_for_type("number").isdigit()


def test_dummy_for_unknown_returns_text():
    assert _dummy_for_type("color") == "Test Input"


@pytest.mark.asyncio
async def test_fill_with_dummy_fills_each_input():
    filler = FormFiller()
    page = AsyncMock()
    # two inputs: email + text
    email_input = AsyncMock()
    text_input = AsyncMock()
    email_input.get_attribute = AsyncMock(return_value="email")
    text_input.get_attribute = AsyncMock(return_value="text")
    page.query_selector_all = AsyncMock(return_value=[email_input, text_input])

    filled = await filler.fill_with_dummy(page)
    assert filled == 2
    email_input.fill.assert_awaited_once()
    text_input.fill.assert_awaited_once()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/universal_qa/test_form_filler.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the form filler**

```python
# src/universal_qa/explorer/form_filler.py
from __future__ import annotations

from typing import Literal

from loguru import logger
from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict

from src.llm.instructor_client import InstructorClient, StructuredGenerationError

_DUMMY_BY_TYPE = {
    "email": "qa_test@mailinator.com",
    "password": "QaTest123!",
    "text": "Test Input",
    "number": "12345",
    "tel": "0812345678",
    "date": "2026-01-01",
    "search": "test",
    "url": "https://example.com",
}


def _dummy_for_type(input_type: str) -> str:
    return _DUMMY_BY_TYPE.get((input_type or "text").lower(), "Test Input")


class _FieldValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    value: str


class _FieldValues(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fields: list[_FieldValue]


class FormFiller:
    """Fills form inputs with dummy data; falls back to LLM-suggested values."""

    def __init__(self, client: InstructorClient | None = None) -> None:
        self._client = client

    async def fill_with_dummy(self, page: Page) -> int:
        """Fill every text-like input with dummy data. Returns count filled."""
        inputs = await page.query_selector_all(
            "input:not([type=hidden]):not([type=submit]):not([type=button]), textarea"
        )
        filled = 0
        for el in inputs:
            try:
                itype = (await el.get_attribute("type")) or "text"
                await el.fill(_dummy_for_type(itype), timeout=3_000)
                filled += 1
            except Exception:
                continue
        return filled

    async def fill_with_llm(self, page: Page, field_labels: list[str]) -> int:
        """Ask the LLM for plausible values when dummy data fails validation."""
        if self._client is None or not field_labels:
            return 0
        prompt = (
            "Provide realistic test values (JSON) for these form fields. "
            "Return field 'fields': list of {label, value}.\n"
            + "\n".join(f"- {lbl}" for lbl in field_labels)
        )
        try:
            resp: _FieldValues = await self._client.create_structured(
                prompt, _FieldValues, temperature=0.1
            )
        except (StructuredGenerationError, Exception) as exc:
            logger.warning(f"FormFiller: LLM fill failed — {exc!r}")
            return 0

        filled = 0
        for fv in resp.fields:
            try:
                loc = page.get_by_label(fv.label)
                if await loc.count():
                    await loc.first.fill(fv.value, timeout=3_000)
                    filled += 1
            except Exception:
                continue
        return filled


__all__ = ["FormFiller", "_dummy_for_type"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/universal_qa/test_form_filler.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```
git add src/universal_qa/explorer/form_filler.py tests/universal_qa/test_form_filler.py
git commit -m "feat(explorer): add FormFiller — dummy data + LLM fallback"
```

---

## Task 4: Session Guard

**Files:**
- Create: `src/universal_qa/explorer/session_guard.py`
- Create: `tests/universal_qa/test_session_guard.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/universal_qa/test_session_guard.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.universal_qa.explorer.session_guard import SessionGuard


@pytest.mark.asyncio
async def test_is_session_lost_true_when_password_visible():
    guard = SessionGuard(auth=MagicMock())
    page = AsyncMock()
    page.query_selector = AsyncMock(return_value=object())  # password input exists
    page.title = AsyncMock(return_value="Some Page")
    assert await guard.is_session_lost(page) is True


@pytest.mark.asyncio
async def test_is_session_lost_false_when_no_login_signals():
    guard = SessionGuard(auth=MagicMock())
    page = AsyncMock()
    page.query_selector = AsyncMock(return_value=None)
    page.title = AsyncMock(return_value="Dashboard")
    assert await guard.is_session_lost(page) is False


@pytest.mark.asyncio
async def test_recover_calls_auth_setup_and_counts():
    auth = MagicMock()
    auth.setup = AsyncMock(return_value=("u", "p"))
    guard = SessionGuard(auth=auth, max_attempts=3)
    page = AsyncMock()
    ok = await guard.recover(page)
    assert ok is True
    auth.setup.assert_awaited_once()


@pytest.mark.asyncio
async def test_recover_stops_after_max_attempts():
    auth = MagicMock()
    auth.setup = AsyncMock(return_value=None)
    guard = SessionGuard(auth=auth, max_attempts=2)
    page = AsyncMock()
    await guard.recover(page)
    await guard.recover(page)
    third = await guard.recover(page)
    assert third is False  # exceeded max_attempts
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/universal_qa/test_session_guard.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the session guard**

```python
# src/universal_qa/explorer/session_guard.py
from __future__ import annotations

from loguru import logger
from playwright.async_api import Page

from src.universal_qa.auth_manager import AuthManager

_LOGIN_TITLE_KEYWORDS = ("login", "sign in", "signin", "เข้าสู่ระบบ")


class SessionGuard:
    """Detects when a session has been lost (kicked back to login) and re-authenticates."""

    def __init__(self, auth: AuthManager, max_attempts: int = 3) -> None:
        self._auth = auth
        self._max_attempts = max_attempts
        self._attempts = 0

    async def is_session_lost(self, page: Page) -> bool:
        try:
            has_password = await page.query_selector("input[type=password]")
            if has_password is not None:
                return True
        except Exception:
            pass
        try:
            title = (await page.title()) or ""
            if any(k in title.lower() for k in _LOGIN_TITLE_KEYWORDS):
                return True
        except Exception:
            pass
        return False

    async def recover(self, page: Page) -> bool:
        """Attempt re-auth. Returns False if attempts exhausted."""
        if self._attempts >= self._max_attempts:
            logger.warning("SessionGuard: max re-auth attempts reached")
            return False
        self._attempts += 1
        try:
            await self._auth.setup(page)
            logger.info(f"SessionGuard: re-authenticated (attempt {self._attempts})")
            return True
        except Exception as exc:
            logger.warning(f"SessionGuard: re-auth failed — {exc!r}")
            return False


__all__ = ["SessionGuard"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/universal_qa/test_session_guard.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```
git add src/universal_qa/explorer/session_guard.py tests/universal_qa/test_session_guard.py
git commit -m "feat(explorer): add SessionGuard — detect session loss + re-auth"
```

---

## Task 5: SiteExplorer Orchestrator

**Files:**
- Create: `src/universal_qa/explorer/site_explorer.py`
- Create: `tests/universal_qa/test_site_explorer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/universal_qa/test_site_explorer.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.universal_qa.explorer.site_explorer import SiteExplorer, _element_key, _is_destructive
from src.universal_qa.explorer.nav_map import ExplorerConfig, ExploredAction, NavigationMap


def test_element_key_stable():
    k1 = _element_key("https://x.com/a", "button", "Buy")
    k2 = _element_key("https://x.com/a", "button", "Buy")
    assert k1 == k2


def test_is_destructive_matches_blocked():
    assert _is_destructive("Delete account") is True
    assert _is_destructive("Logout") is True
    assert _is_destructive("View products") is False


def test_build_flow_from_path():
    explorer = SiteExplorer.__new__(SiteExplorer)
    steps = [
        ExploredAction(page_url="https://x.com/a", action_label="Add",
                       leads_to_url="https://x.com/a"),
        ExploredAction(page_url="https://x.com/a", action_label="Cart",
                       leads_to_url="https://x.com/cart"),
    ]
    flow = explorer._build_flow(steps, idx=0)
    assert flow.start_url == "https://x.com/a"
    assert flow.end_url == "https://x.com/cart"
    assert len(flow.steps) == 2


@pytest.mark.asyncio
async def test_explore_records_navigation(tmp_path):
    cfg = ExplorerConfig(max_pages=1, max_depth=1)
    auth = MagicMock()
    explorer = SiteExplorer(auth=auth, config=cfg)

    page = AsyncMock()
    page.url = "https://x.com/start"
    page.goto = AsyncMock()
    page.title = AsyncMock(return_value="Start")
    page.evaluate = AsyncMock(return_value="{}")  # localStorage snapshot
    page.wait_for_load_state = AsyncMock()

    # No interactive elements found → page recorded with no actions
    with patch.object(explorer._scanner, "scan", new=AsyncMock(return_value=[])), \
         patch.object(explorer, "_record_page_node", new=AsyncMock(
             return_value=("Start", "pam content"))), \
         patch.object(explorer._guard, "is_session_lost", new=AsyncMock(return_value=False)):
        nav_map = await explorer.explore(page, ["https://x.com/start"])

    assert isinstance(nav_map, NavigationMap)
    assert len(nav_map.pages) == 1
    assert nav_map.pages[0].url == "https://x.com/start"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/universal_qa/test_site_explorer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the explorer**

```python
# src/universal_qa/explorer/site_explorer.py
from __future__ import annotations

import hashlib
import time
from collections import deque
from urllib.parse import urlparse

from loguru import logger
from playwright.async_api import Page

from src.contractskill.crawler import CrawlerConfig, SFGCrawler
from src.contractskill.sfg import BLOCKED_ACTION_PATTERNS, SFGStore
from src.perception.grounder import Grounder
from src.universal_qa.auth_manager import AuthManager
from src.universal_qa.explorer.element_scanner import ElementScanner
from src.universal_qa.explorer.form_filler import FormFiller
from src.universal_qa.explorer.nav_map import (
    ElementCandidate, ExploredAction, ExploredPage, ExplorerConfig,
    NavigationFlow, NavigationMap,
)
from src.universal_qa.explorer.session_guard import SessionGuard


def _element_key(page_url: str, role: str | None, name: str | None) -> str:
    raw = f"{page_url}|{role or ''}|{name or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _is_destructive(label: str) -> bool:
    low = (label or "").lower()
    return any(p in low for p in BLOCKED_ACTION_PATTERNS)


def _clean_url(url: str) -> str:
    return url.split("?")[0].split("#")[0]


class SiteExplorer:
    """Phase 3: interaction-based exploration that builds a NavigationMap.

    For each discovered URL, clicks every interactive element sequentially,
    records action→page transitions, fills forms, recovers from session loss,
    and auto-detects multi-step flows.
    """

    def __init__(
        self,
        auth: AuthManager,
        config: ExplorerConfig | None = None,
        client=None,
    ) -> None:
        self._auth = auth
        self._cfg = config or ExplorerConfig()
        self._scanner = ElementScanner()
        self._filler = FormFiller(client=client)
        self._guard = SessionGuard(auth=auth)
        # SFG infrastructure for PAM grounding (reuses _visit_node)
        self._sfg_store: SFGStore | None = None
        self._crawler: SFGCrawler | None = None

    async def explore(self, page: Page, discovered_urls: list[str]) -> NavigationMap:
        import tempfile, pathlib
        sfg_path = pathlib.Path(tempfile.mkdtemp(prefix="uqa_explore_")) / "sfg.db"
        self._sfg_store = SFGStore(db_path=sfg_path)
        self._crawler = SFGCrawler(self._sfg_store, Grounder(), CrawlerConfig())

        base_domain = urlparse(page.url).netloc
        base_url = f"{urlparse(page.url).scheme}://{base_domain}"
        start = time.monotonic()

        pages: list[ExploredPage] = []
        flows: list[NavigationFlow] = []
        visited_actions: set[str] = set()
        visit_count: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque((u, 0) for u in discovered_urls)

        while queue and len(pages) < self._cfg.max_pages:
            url, depth = queue.popleft()
            curl = _clean_url(url)
            if depth > self._cfg.max_depth:
                continue
            if visit_count.get(curl, 0) >= self._cfg.max_visits_per_url:
                continue
            if (time.monotonic() - start) / 60.0 >= self._cfg.explore_timeout_min:
                logger.info("SiteExplorer: timeout reached")
                break
            visit_count[curl] = visit_count.get(curl, 0) + 1

            try:
                explored, new_urls, page_flows = await self._explore_page(
                    page, url, depth, base_domain, visited_actions
                )
            except Exception as exc:
                logger.warning(f"SiteExplorer: page {url} failed — {exc!r}")
                continue

            pages.append(explored)
            flows.extend(page_flows)
            logger.info(
                f"  Pages found: {len(pages)} | Actions: {len(explored.actions)} | Flows: {len(flows)}"
            )
            for nu in new_urls:
                ncu = _clean_url(nu)
                if urlparse(ncu).netloc == base_domain and visit_count.get(ncu, 0) == 0:
                    queue.append((nu, depth + 1))

        return NavigationMap(
            base_url=base_url, pages=pages, flows=flows,
            explored_at_iso="",  # stamped by caller if needed
        )

    async def _explore_page(
        self, page: Page, url: str, depth: int, base_domain: str,
        visited_actions: set[str],
    ) -> tuple[ExploredPage, list[str], list[NavigationFlow]]:
        # Navigate (skip if already there)
        if _clean_url(page.url) != _clean_url(url):
            await page.goto(url, wait_until="networkidle", timeout=30_000)
        else:
            await page.wait_for_load_state("networkidle", timeout=10_000)

        # Session check
        if await self._guard.is_session_lost(page):
            await self._guard.recover(page)
            await page.goto(url, wait_until="networkidle", timeout=30_000)

        logger.info(f"[EXPLORE] {url}")

        title, pam = await self._record_page_node(page)
        snapshot = await self._snapshot_state(page)

        candidates = await self._scanner.scan(page)
        actions: list[ExploredAction] = []
        new_urls: list[str] = []

        for cand in candidates:
            key = _element_key(url, cand.role, cand.name or cand.label)
            if key in visited_actions:
                continue
            destructive = _is_destructive(cand.label)
            if destructive and not self._cfg.allow_destructive:
                actions.append(ExploredAction(
                    page_url=url, action_label=cand.label,
                    element_role=cand.role, element_name=cand.name,
                    element_selector=cand.selector, is_destructive=True,
                ))
                visited_actions.add(key)
                continue

            action = await self._try_click(page, url, cand, base_domain)
            if action is not None:
                actions.append(action)
                visited_actions.add(key)
                if action.leads_to_url:
                    new_urls.append(action.leads_to_url)
            # always return to the page we are exploring
            if _clean_url(page.url) != _clean_url(url):
                try:
                    await page.goto(url, wait_until="networkidle", timeout=15_000)
                except Exception:
                    pass

        explored = ExploredPage(
            url=url, title=title, pam_content=pam,
            actions=actions, state_snapshot=snapshot,
        )
        flows = self._detect_flows(actions)
        return explored, new_urls, flows

    async def _try_click(
        self, page: Page, url: str, cand: ElementCandidate, base_domain: str,
    ) -> ExploredAction | None:
        pre_url = _clean_url(page.url)
        pre_badge = await self._snapshot_state(page)
        try:
            loc = self._locate(page, cand)
            await loc.click(timeout=3_000)
            await page.wait_for_load_state("networkidle", timeout=1_000)
        except Exception as exc:
            logger.debug(f"  click '{cand.label}' skipped: {exc!r}")
            return None

        post_url = _clean_url(page.url)
        if post_url != pre_url and urlparse(post_url).netloc == base_domain:
            logger.info(f"  → คลิก \"{cand.label}\" ──► navigate: {post_url} ✓")
            return ExploredAction(
                page_url=url, action_label=cand.label,
                element_role=cand.role, element_name=cand.name,
                element_selector=cand.selector, leads_to_url=page.url,
            )

        # modal/state detection
        post_badge = await self._snapshot_state(page)
        if post_badge != pre_badge:
            logger.info(f"  → คลิก \"{cand.label}\" ──► state เปลี่ยน")
            return ExploredAction(
                page_url=url, action_label=cand.label,
                element_role=cand.role, element_name=cand.name,
                element_selector=cand.selector,
                state_change={"snapshot": "changed"},
            )
        return None

    @staticmethod
    def _locate(page: Page, cand: ElementCandidate):
        # ARIA primary, selector fallback
        if cand.role and cand.name:
            return page.get_by_role(cand.role, name=cand.name).first
        if cand.name:
            return page.get_by_text(cand.name).first
        if cand.selector:
            return page.locator(cand.selector).first
        return page.get_by_text(cand.label).first

    async def _snapshot_state(self, page: Page) -> str:
        try:
            return await page.evaluate(
                "() => JSON.stringify({ls: {...localStorage}, ss: {...sessionStorage}})"
            )
        except Exception:
            return "{}"

    async def _record_page_node(self, page: Page) -> tuple[str, str]:
        """Ground the page via SFGCrawler._visit_node; return (title, pam_content)."""
        try:
            node, _ = await self._crawler._visit_node(page, None)
            return node.page_title, node.pam_content
        except Exception as exc:
            logger.warning(f"SiteExplorer: grounding failed — {exc!r}")
            try:
                return (await page.title()) or "", ""
            except Exception:
                return "", ""

    def _detect_flows(self, actions: list[ExploredAction]) -> list[NavigationFlow]:
        nav_actions = [a for a in actions if a.leads_to_url]
        flows: list[NavigationFlow] = []
        if len(nav_actions) >= 2:
            flows.append(self._build_flow(nav_actions, idx=0))
        return flows

    def _build_flow(self, steps: list[ExploredAction], idx: int) -> NavigationFlow:
        start = steps[0].page_url
        end = steps[-1].leads_to_url or steps[-1].page_url
        name = "flow_" + hashlib.sha256(f"{start}{end}{idx}".encode()).hexdigest()[:8]
        return NavigationFlow(
            flow_id=name, name=name, steps=steps,
            start_url=start, end_url=end,
        )


__all__ = ["SiteExplorer"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/universal_qa/test_site_explorer.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```
git add src/universal_qa/explorer/site_explorer.py tests/universal_qa/test_site_explorer.py
git commit -m "feat(explorer): add SiteExplorer orchestrator — interaction loop + flow detection"
```

---

## Task 6: TestPlanner accepts NavigationMap

**Files:**
- Modify: `src/universal_qa/test_planner.py`
- Modify: `tests/universal_qa/test_planner.py`

This task adds a new `plan_from_map()` method (keeping the old `plan()` for backward compatibility) plus `_plan_flows()` and `_action_to_step()`.

- [ ] **Step 1: Write the failing test (append to existing test_planner.py)**

```python
# Append to tests/universal_qa/test_planner.py
from src.universal_qa.explorer.nav_map import (
    ExploredAction, ExploredPage, NavigationFlow, NavigationMap,
)


def test_action_to_step_navigate():
    planner = UniversalTestPlanner.__new__(UniversalTestPlanner)
    a = ExploredAction(page_url="https://x.com/a", action_label="Home",
                       element_role="link", element_name="Home",
                       leads_to_url="https://x.com/home")
    step = planner._action_to_step(a)
    assert "https://x.com/home" in step
    assert step.startswith("เปิดหน้า")


def test_action_to_step_button_click():
    planner = UniversalTestPlanner.__new__(UniversalTestPlanner)
    a = ExploredAction(page_url="https://x.com/a", action_label="Buy",
                       element_role="button", element_name="Checkout")
    step = planner._action_to_step(a)
    assert "Checkout" in step
    assert "คลิกปุ่ม" in step


def test_plan_flows_builds_test_case():
    planner = UniversalTestPlanner.__new__(UniversalTestPlanner)
    steps = [
        ExploredAction(page_url="https://x.com/inv", action_label="Add to cart",
                       element_role="button", element_name="Add to cart"),
        ExploredAction(page_url="https://x.com/inv", action_label="Cart",
                       element_role="link", element_name="Cart",
                       leads_to_url="https://x.com/cart"),
    ]
    flow = NavigationFlow(flow_id="f1", name="checkout_flow", steps=steps,
                          start_url="https://x.com/inv", end_url="https://x.com/cart")
    cases = planner._plan_flows([flow])
    assert len(cases) == 1
    assert cases[0].type == "functional"
    assert cases[0].priority == "high"
    assert len(cases[0].steps) == 2
    assert "https://x.com/cart" in cases[0].expected_outcome
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/universal_qa/test_planner.py -v -k "action_to_step or plan_flows"`
Expected: FAIL with `AttributeError: ... has no attribute '_action_to_step'`

- [ ] **Step 3: Add the methods to test_planner.py**

Add this import near the top of `src/universal_qa/test_planner.py` (after the existing imports):

```python
from src.universal_qa.explorer.nav_map import (
    ExploredAction, ExploredPage, NavigationFlow, NavigationMap,
)
```

Add these methods inside the `UniversalTestPlanner` class (after `_sort_by_priority`):

```python
    async def plan_from_map(self, nav_map: NavigationMap) -> list[TestCase]:
        """Generate test cases from a NavigationMap (preferred over plan())."""
        per_page = await self._plan_from_pages(nav_map.pages)
        flows = self._plan_flows(nav_map.flows)
        return self._sort_by_priority(per_page + flows)

    async def _plan_from_pages(self, pages: list[ExploredPage]) -> list[TestCase]:
        results: list[TestCase] = []
        for page in pages:
            # Accessibility — one per page
            results.append(TestCase(
                title=f"ตรวจสอบ Accessibility: {page.title}",
                type="accessibility", priority="medium",
                preconditions=[f"อยู่ที่หน้า {page.url}"],
                steps=[
                    f"เปิดหน้า {page.url}",
                    "ตรวจสอบว่า element ที่โต้ตอบได้ทุกตัวมี accessible name",
                    "ตรวจสอบว่ารูปภาพทุกรูปมี alt text",
                    "ตรวจสอบว่าไม่มี decorative role บน element ที่โต้ตอบได้",
                ],
                expected_outcome="ไม่พบการละเมิดกฎ WCAG",
                source_url=page.url,
            ))
            # Security — XSS + SQLi if page has form-like actions
            if any(a.element_role == "button" for a in page.actions):
                for kind, payload in (("XSS", _XSS_PAYLOAD), ("SQL", _SQLI_PAYLOAD)):
                    results.append(TestCase(
                        title=f"ทดสอบ {kind} injection: {page.title}",
                        type="security", priority="high",
                        preconditions=[f"อยู่ที่หน้า {page.url}"],
                        steps=[
                            f"เปิดหน้า {page.url}",
                            f"กรอก {kind} payload ลงทุกช่องรับข้อมูล: {payload}",
                            "กด submit form",
                            f"ตรวจสอบว่า payload ไม่ทำงาน",
                        ],
                        expected_outcome=f"หน้าเว็บไม่ได้รับผลกระทบจาก {kind} payload",
                        source_url=page.url,
                    ))
        return results

    def _plan_flows(self, flows: list[NavigationFlow]) -> list[TestCase]:
        cases: list[TestCase] = []
        for flow in flows:
            steps = [self._action_to_step(a) for a in flow.steps]
            cases.append(TestCase(
                title=f"ทดสอบ flow {flow.name} ตั้งแต่ต้นจนจบ",
                type="functional", priority="high",
                preconditions=[f"เริ่มที่หน้า {flow.start_url}"],
                steps=steps or [f"เปิดหน้า {flow.start_url}"],
                expected_outcome=f"เข้าหน้า {flow.end_url} สำเร็จ",
                source_url=flow.start_url,
            ))
        return cases

    @staticmethod
    def _action_to_step(action: ExploredAction) -> str:
        if action.leads_to_url:
            return f"เปิดหน้า {action.leads_to_url}"
        name = action.element_name or action.action_label
        if action.element_role == "button":
            return f'คลิกปุ่ม "{name}"'
        if action.element_role == "link":
            return f'คลิก "{name}"'
        return f'คลิก "{name}"'
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/universal_qa/test_planner.py -v`
Expected: all pass (existing 4 + new 3 = 7)

- [ ] **Step 5: Commit**

```
git add src/universal_qa/test_planner.py tests/universal_qa/test_planner.py
git commit -m "feat(planner): add plan_from_map — per-page + flow-based test cases from NavigationMap"
```

---

## Task 7: Integrate into Agent + CLI

**Files:**
- Modify: `src/universal_qa/agent.py`
- Modify: `src/universal_qa/__main__.py`

- [ ] **Step 1: Modify agent.py constructor to accept new params**

Replace the `__init__` of `UniversalQAAgent` in `src/universal_qa/agent.py` with:

```python
    def __init__(
        self,
        url: str,
        *,
        username: str | None = None,
        password: str | None = None,
        max_pages: int = 50,
        explore_timeout: int = 5,
        max_depth: int = 4,
        allow_destructive: bool = False,
        headless: bool = True,
        output_dir: pathlib.Path | None = None,
    ) -> None:
        self._url = url
        self._max_pages = max_pages
        self._headless = headless
        self._output_dir = output_dir or pathlib.Path("reports")
        self._auth = AuthManager(username=username, password=password)
        self._discovery = SiteDiscovery(max_pages=max_pages)
        self._planner = UniversalTestPlanner()
        self._terminal = TerminalReporter()
        # explorer config
        from src.universal_qa.explorer.nav_map import ExplorerConfig
        self._explorer_cfg = ExplorerConfig(
            max_pages=max_pages,
            explore_timeout_min=explore_timeout,
            max_depth=max_depth,
            allow_destructive=allow_destructive,
        )
```

- [ ] **Step 2: Modify agent.py run() to add Phase 3**

In `src/universal_qa/agent.py`, replace the Phase 2 + Phase 3 (Discover + Plan) block inside `run()` with:

```python
                # Phase 2: Discover URLs (fast)
                discover_url = page.url if page.url != self._url else self._url
                logger.info(f"Phase 2: site discovery from {discover_url}")
                sfg_store = await self._discovery.discover(page, discover_url)
                discovered_urls = [
                    n.url for n in sfg_store.get_nodes_by_url_prefix(
                        f"{urlparse(discover_url).scheme}://{urlparse(discover_url).netloc}"
                    )
                ]
                if discover_url not in discovered_urls:
                    discovered_urls.insert(0, discover_url)

                # Phase 3: Explore (thorough) → NavigationMap
                logger.info("Phase 3: interaction-based exploration")
                from src.universal_qa.explorer.site_explorer import SiteExplorer
                explorer = SiteExplorer(
                    auth=self._auth, config=self._explorer_cfg,
                    client=self._planner._client,
                )
                nav_map = await explorer.explore(page, discovered_urls)
                logger.info(
                    f"  Explored {len(nav_map.pages)} pages, {len(nav_map.flows)} flows"
                )

                # Phase 4: Plan from NavigationMap
                logger.info("Phase 4: generating test cases")
                test_cases = await self._planner.plan_from_map(nav_map)
                logger.info(f"  {len(test_cases)} test cases generated")
```

Make sure `from urllib.parse import urlparse` is imported at the top of `agent.py` (add it if missing).

- [ ] **Step 3: Add new CLI args to __main__.py**

Replace the body of `main()` in `src/universal_qa/__main__.py` with:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Universal QA Agent — test any website")
    parser.add_argument("--url", required=True, help="Website URL to test")
    parser.add_argument("--username", default=None, help="Login username/email (optional)")
    parser.add_argument("--password", default=None, help="Login password (optional)")
    parser.add_argument("--max-pages", type=int, default=50,
                        help="Max pages to crawl (default 50)")
    parser.add_argument("--explore-timeout", type=int, default=5,
                        help="Exploration timeout in minutes (default 5)")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="Max navigation depth (default 4)")
    parser.add_argument("--allow-destructive", action="store_true", default=False,
                        help="Allow clicking delete/remove/payment actions")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run browser in headless mode (default)")
    parser.add_argument("--no-headless", dest="headless", action="store_false",
                        help="Show browser window")
    args = parser.parse_args()

    agent = UniversalQAAgent(
        url=args.url,
        username=args.username,
        password=args.password,
        max_pages=args.max_pages,
        explore_timeout=args.explore_timeout,
        max_depth=args.max_depth,
        allow_destructive=args.allow_destructive,
        headless=args.headless,
    )

    results = asyncio.run(agent.run())
    passed = sum(1 for r in results if r.passed)
    print(json.dumps({
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
    }, indent=2))
```

- [ ] **Step 4: Verify the universal_qa unit suite still passes**

Run: `uv run pytest tests/universal_qa/ -v`
Expected: all tests pass

- [ ] **Step 5: Verify CLI help shows new flags**

Run: `uv run python -m src.universal_qa --help`
Expected: output contains `--explore-timeout`, `--max-depth`, `--allow-destructive`

- [ ] **Step 6: Commit**

```
git add src/universal_qa/agent.py src/universal_qa/__main__.py
git commit -m "feat(agent): wire SiteExplorer as Phase 3 + new CLI args (explore-timeout/max-depth/allow-destructive)"
```

---

## Task 8: Live Integration Test

- [ ] **Step 1: Run full universal_qa unit suite**

Run: `uv run pytest tests/universal_qa/ -v`
Expected: all tests pass. Report exact count.

- [ ] **Step 2: Live test on saucedemo (SPA with multi-step flows)**

Run:
```
uv run python -m src.universal_qa --url "https://www.saucedemo.com/" --username "standard_user" --password "secret_sauce" --max-pages 8 --explore-timeout 4
```
Expected:
- Phase 3 logs `[EXPLORE]` lines with clicked elements
- Discovers more than 1 page (inventory + item + cart + checkout)
- At least one flow detected
- `reports/qa_report_<ts>.html` created
- Final JSON summary printed

Report: number of pages explored, flows found, total test cases, pass rate.

- [ ] **Step 3: Live test on automationexercise (multi-page static site)**

Run:
```
uv run python -m src.universal_qa --url "https://automationexercise.com" --max-pages 12 --explore-timeout 4
```
Expected: runs end to end, HTML report created, JSON summary printed.

- [ ] **Step 4: Open the HTML report and verify flow-based test cases appear**

Run: `Get-ChildItem reports\qa_report_*.html | Sort-Object LastWriteTime -Descending | Select-Object -First 1`
Verify: report contains flow test cases with multi-step Thai steps.

- [ ] **Step 5: Final commit**

```
git add -A
git commit -m "feat(explorer): complete SiteExplorer integration — interaction-based exploration end-to-end"
```
