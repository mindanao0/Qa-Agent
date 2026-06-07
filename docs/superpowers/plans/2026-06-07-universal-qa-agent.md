# Universal QA Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a general-purpose web QA agent that discovers any website's capabilities, generates functional/accessibility/security test cases, executes them via Playwright, and reports results as real-time terminal output + self-contained HTML.

**Architecture:** One shared Playwright browser context handles auth detection, BFS site discovery (via SFGCrawler._visit_node), test execution, and screenshot capture. InstructorClient generates functional test cases from SFG node PAM content; accessibility and security tests are rule-based/template-based (no LLM). HypothesisExecutor handles functional test execution; observers' static methods handle accessibility and security checks.

**Tech Stack:** Python 3.13, Playwright, Pydantic V2, InstructorClient (Ollama), SFGCrawler/_visit_node, AccessibilityObserver._scan, SecurityObserver static methods, loguru, uv

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/universal_qa/__init__.py` | Create | Package marker |
| `src/universal_qa/models.py` | Create | TestCase, StepTrace, TestResult Pydantic models |
| `src/universal_qa/site_discovery.py` | Create | BFS crawl over shared auth'd page using SFGCrawler._visit_node |
| `src/universal_qa/auth_manager.py` | Create | Login with credentials OR auto-register |
| `src/universal_qa/test_planner.py` | Create | LLM functional + rule-based accessibility + template security |
| `src/universal_qa/test_runner.py` | Create | Routes TestCase by type → execute → StepTrace + screenshot |
| `src/universal_qa/reporters/__init__.py` | Create | Package marker |
| `src/universal_qa/reporters/terminal.py` | Create | Real-time loguru pass/fail output |
| `src/universal_qa/reporters/html.py` | Create | Self-contained HTML with filter + screenshots |
| `src/universal_qa/agent.py` | Create | UniversalQAAgent orchestrator (4 phases) |
| `src/universal_qa/__main__.py` | Create | CLI: --url, --username, --password, --max-pages, --headless |
| `src/continuous/loop_controller.py` | Modify | Add SiteProfile + --profile CLI arg; default "todomvc" |
| `tests/universal_qa/__init__.py` | Create | Package marker |
| `tests/universal_qa/test_models.py` | Create | Unit tests for all three models |
| `tests/universal_qa/test_site_discovery.py` | Create | Unit tests for SiteDiscovery |
| `tests/universal_qa/test_auth_manager.py` | Create | Unit tests for AuthManager strategy selection |
| `tests/universal_qa/test_planner.py` | Create | Unit tests for planner (functional/accessibility/security output) |
| `tests/universal_qa/test_reporters.py` | Create | Unit tests for TerminalReporter and HTMLReporter |
| `tests/universal_qa/test_runner.py` | Create | Unit tests for UniversalTestRunner routing |

---

## Task 1: Data Models

**Files:**
- Create: `src/universal_qa/__init__.py`
- Create: `src/universal_qa/models.py`
- Create: `tests/universal_qa/__init__.py`
- Create: `tests/universal_qa/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/universal_qa/test_models.py
import pytest
from src.universal_qa.models import TestCase, StepTrace, TestResult


def test_test_case_auto_id():
    tc = TestCase(
        title="Login with valid credentials",
        type="functional",
        priority="high",
        steps=["fill email", "fill password", "click login"],
        expected_outcome="redirected to dashboard",
        source_url="https://example.com/login",
    )
    assert len(tc.id) == 12
    assert tc.preconditions == []


def test_test_case_rejects_extra_field():
    with pytest.raises(Exception):
        TestCase(
            title="T", type="functional", priority="high",
            steps=["s"], expected_outcome="e",
            source_url="https://x.com", unknown_field="bad",
        )


def test_step_trace_default_error_is_none():
    st = StepTrace(step="click login", status="passed", detail="found by role")
    assert st.error is None


def test_test_result_failed_with_reason():
    tc = TestCase(
        title="XSS check", type="security", priority="high",
        steps=["inject payload"], expected_outcome="no execution",
        source_url="https://x.com",
    )
    result = TestResult(
        test_case=tc, passed=False,
        failure_reason="XSS payload executed", duration_ms=450,
    )
    assert not result.passed
    assert result.screenshot_path is None


def test_test_result_passed_no_failure_reason():
    tc = TestCase(
        title="T", type="accessibility", priority="low",
        steps=["check labels"], expected_outcome="all labels present",
        source_url="https://x.com",
    )
    result = TestResult(test_case=tc, passed=True, duration_ms=200)
    assert result.failure_reason is None
    assert result.steps_trace == []
```

- [ ] **Step 2: Run failing tests**

```
uv run pytest tests/universal_qa/test_models.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.universal_qa'`

- [ ] **Step 3: Create package markers**

```python
# src/universal_qa/__init__.py
# (empty)
```

```python
# tests/universal_qa/__init__.py
# (empty)
```

- [ ] **Step 4: Implement models.py**

```python
# src/universal_qa/models.py
from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str
    type: Literal["functional", "accessibility", "security"]
    priority: Literal["high", "medium", "low"]
    preconditions: list[str] = Field(default_factory=list)
    steps: list[str]
    expected_outcome: str
    source_url: str


class StepTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    status: Literal["passed", "failed", "skipped"]
    detail: str
    error: str | None = None


class TestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_case: TestCase
    passed: bool
    steps_trace: list[StepTrace] = Field(default_factory=list)
    failure_reason: str | None = None
    screenshot_path: str | None = None
    duration_ms: int = 0


__all__ = ["TestCase", "StepTrace", "TestResult"]
```

- [ ] **Step 5: Run passing tests**

```
uv run pytest tests/universal_qa/test_models.py -v
```
Expected: `5 passed`

- [ ] **Step 6: Commit**

```
git add src/universal_qa/__init__.py src/universal_qa/models.py tests/universal_qa/__init__.py tests/universal_qa/test_models.py
git commit -m "feat(universal_qa): add TestCase/StepTrace/TestResult models"
```

---

## Task 2: Site Discovery

**Files:**
- Create: `src/universal_qa/site_discovery.py`
- Create: `tests/universal_qa/test_site_discovery.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/universal_qa/test_site_discovery.py
import pathlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.universal_qa.site_discovery import SiteDiscovery


def test_site_discovery_default_limits():
    sd = SiteDiscovery()
    assert sd.max_pages == 50
    assert sd.max_depth == 6


def test_site_discovery_custom_limits():
    sd = SiteDiscovery(max_pages=20, max_depth=3)
    assert sd.max_pages == 20


@pytest.mark.asyncio
async def test_discover_returns_sfg_store(tmp_path):
    sd = SiteDiscovery(max_pages=2, max_depth=1)

    mock_page = AsyncMock()
    mock_page.url = "https://example.com"
    mock_page.goto = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value=[])  # no links found

    mock_node = MagicMock()
    mock_node.node_id = "abc123"

    with patch.object(sd, "_visit_and_record", return_value=mock_node):
        store = await sd.discover(mock_page, "https://example.com",
                                   db_path=tmp_path / "sfg.db")

    assert store is not None
```

- [ ] **Step 2: Run failing tests**

```
uv run pytest tests/universal_qa/test_site_discovery.py -v
```
Expected: `ImportError: cannot import name 'SiteDiscovery'`

- [ ] **Step 3: Implement site_discovery.py**

```python
# src/universal_qa/site_discovery.py
from __future__ import annotations

import pathlib
import tempfile
from collections import deque
from urllib.parse import urlparse

from loguru import logger
from playwright.async_api import Page

from src.contractskill.crawler import CrawlerConfig, SFGCrawler
from src.contractskill.sfg import SFGStore
from src.perception.grounder import Grounder


class SiteDiscovery:
    """BFS site crawler using an already-authenticated shared Page.

    Uses SFGCrawler._visit_node directly (not crawl()) so the caller's
    authenticated BrowserContext is preserved throughout.
    """

    def __init__(self, max_pages: int = 50, max_depth: int = 6) -> None:
        self.max_pages = max_pages
        self.max_depth = max_depth

    async def discover(
        self,
        page: Page,
        start_url: str,
        db_path: pathlib.Path | None = None,
    ) -> SFGStore:
        path = db_path or pathlib.Path(tempfile.mkdtemp(prefix="uqa_sfg_")) / "sfg.db"
        store = SFGStore(db_path=path)
        grounder = Grounder()
        config = CrawlerConfig(max_pages=self.max_pages, max_depth=self.max_depth)
        crawler = SFGCrawler(store, grounder, config)

        base_domain = urlparse(start_url).netloc
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start_url, 0)])
        pages_visited = 0

        while queue and pages_visited < self.max_pages:
            url, depth = queue.popleft()
            if url in visited or depth > self.max_depth:
                continue

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                visited.add(url)
                await self._visit_and_record(crawler, page)
                pages_visited += 1
                logger.info(f"SiteDiscovery: visited {url} ({pages_visited}/{self.max_pages})")

                if depth < self.max_depth:
                    links: list[str] = await page.evaluate(
                        "() => Array.from(document.querySelectorAll('a[href]'))"
                        ".map(a => a.href).filter(h => h.startsWith('http'))"
                    )
                    for link in links:
                        clean = link.split("?")[0].split("#")[0]
                        if urlparse(clean).netloc == base_domain and clean not in visited:
                            queue.append((clean, depth + 1))
            except Exception as exc:
                logger.warning(f"SiteDiscovery: {url} skipped — {exc!r}")

        logger.info(f"SiteDiscovery: done — {pages_visited} pages, {store.node_count()} nodes")
        return store

    @staticmethod
    async def _visit_and_record(crawler: SFGCrawler, page: Page):
        node, _ = await crawler._visit_node(page, None)
        return node


__all__ = ["SiteDiscovery"]
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/universal_qa/test_site_discovery.py -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```
git add src/universal_qa/site_discovery.py tests/universal_qa/test_site_discovery.py
git commit -m "feat(universal_qa): add SiteDiscovery — BFS over shared auth'd page"
```

---

## Task 3: Auth Manager

**Files:**
- Create: `src/universal_qa/auth_manager.py`
- Create: `tests/universal_qa/test_auth_manager.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/universal_qa/test_auth_manager.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.universal_qa.auth_manager import AuthManager


def test_auth_manager_stores_credentials():
    am = AuthManager(username="user@test.com", password="pass123")
    assert am._username == "user@test.com"
    assert am._password == "pass123"


def test_auth_manager_no_credentials():
    am = AuthManager()
    assert am._username is None


def test_choose_strategy_with_credentials():
    am = AuthManager(username="u@x.com", password="p")
    assert am._strategy() == "login"


def test_choose_strategy_without_credentials():
    am = AuthManager()
    assert am._strategy() == "auto_register"


@pytest.mark.asyncio
async def test_setup_skips_when_no_form_found():
    am = AuthManager()
    page = AsyncMock()
    page.query_selector = AsyncMock(return_value=None)
    page.goto = AsyncMock()

    result = await am.setup(page)
    assert result is None  # no auth form found → unauthenticated
```

- [ ] **Step 2: Run failing tests**

```
uv run pytest tests/universal_qa/test_auth_manager.py -v
```
Expected: `ImportError: cannot import name 'AuthManager'`

- [ ] **Step 3: Implement auth_manager.py**

```python
# src/universal_qa/auth_manager.py
from __future__ import annotations

import random
import string

from loguru import logger
from playwright.async_api import Page


class AuthManager:
    """Handles authentication for any website.

    Strategy:
    1. If username + password provided → find login form → fill → submit
    2. Else → find register form → create qa_test_<rand>@mailinator.com → login
    3. If no auth form found → return None (unauthenticated)
    """

    _LOGIN_FORM_SELECTORS = [
        'form[action*="login"]', 'form[action*="signin"]',
        'form[id*="login"]', 'form[id*="signin"]',
        'input[type="password"]',
    ]
    _REGISTER_FORM_SELECTORS = [
        'form[action*="register"]', 'form[action*="signup"]',
        'form[id*="register"]', 'form[id*="signup"]',
        'a[href*="register"]', 'a[href*="signup"]',
    ]

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._username = username
        self._password = password

    def _strategy(self) -> str:
        return "login" if (self._username and self._password) else "auto_register"

    async def setup(self, page: Page) -> tuple[str, str] | None:
        """Authenticate on the current page. Returns (username, password) or None."""
        if self._strategy() == "login":
            return await self._do_login(page)
        return await self._do_auto_register(page)

    async def _do_login(self, page: Page) -> tuple[str, str] | None:
        has_form = await self._has_selector(page, self._LOGIN_FORM_SELECTORS)
        if not has_form:
            logger.info("AuthManager: no login form found — skipping auth")
            return None
        try:
            await page.get_by_role("textbox", name="email").fill(
                self._username or "", timeout=5_000
            )
        except Exception:
            try:
                inputs = page.get_by_role("textbox")
                if await inputs.count() >= 1:
                    await inputs.first.fill(self._username or "", timeout=5_000)
            except Exception:
                pass
        try:
            await page.get_by_label("password").fill(self._password or "", timeout=5_000)
        except Exception:
            try:
                await page.locator("input[type=password]").first.fill(
                    self._password or "", timeout=5_000
                )
            except Exception:
                pass
        try:
            await page.get_by_role("button", name="login").click(timeout=5_000)
        except Exception:
            try:
                await page.get_by_role("button", name="sign in").click(timeout=5_000)
            except Exception:
                pass
        await page.wait_for_load_state("domcontentloaded", timeout=10_000)
        logger.info(f"AuthManager: login attempted for {self._username}")
        return (self._username or "", self._password or "")

    async def _do_auto_register(self, page: Page) -> tuple[str, str] | None:
        has_register = await self._has_selector(page, self._REGISTER_FORM_SELECTORS)
        if not has_register:
            logger.info("AuthManager: no register form found — skipping auth")
            return None
        rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        email = f"qa_test_{rand}@mailinator.com"
        password = "QaTest123!"
        name = f"QA Test {rand[:4]}"

        try:
            reg_link = page.get_by_role("link", name="signup")
            if not await reg_link.count():
                reg_link = page.get_by_role("link", name="register")
            if await reg_link.count():
                await reg_link.first.click(timeout=5_000)
                await page.wait_for_load_state("domcontentloaded", timeout=10_000)
        except Exception:
            pass

        try:
            name_input = page.get_by_role("textbox", name="name")
            if await name_input.count():
                await name_input.first.fill(name, timeout=5_000)
        except Exception:
            pass
        try:
            await page.get_by_role("textbox", name="email").fill(email, timeout=5_000)
        except Exception:
            pass
        try:
            for pw_input in await page.locator("input[type=password]").all():
                await pw_input.fill(password, timeout=5_000)
        except Exception:
            pass
        try:
            await page.get_by_role("button", name="signup").click(timeout=5_000)
        except Exception:
            try:
                await page.get_by_role("button", name="register").click(timeout=5_000)
            except Exception:
                pass
        await page.wait_for_load_state("domcontentloaded", timeout=10_000)
        logger.info(f"AuthManager: auto-registered as {email}")
        self._username = email
        self._password = password
        return (email, password)

    @staticmethod
    async def _has_selector(page: Page, selectors: list[str]) -> bool:
        for sel in selectors:
            try:
                el = await page.query_selector(sel)
                if el is not None:
                    return True
            except Exception:
                pass
        return False


__all__ = ["AuthManager"]
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/universal_qa/test_auth_manager.py -v
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```
git add src/universal_qa/auth_manager.py tests/universal_qa/test_auth_manager.py
git commit -m "feat(universal_qa): add AuthManager — credentials or auto-register"
```

---

## Task 4: Test Planner

**Files:**
- Create: `src/universal_qa/test_planner.py`
- Create: `tests/universal_qa/test_planner.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/universal_qa/test_planner.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.universal_qa.models import TestCase
from src.universal_qa.test_planner import UniversalTestPlanner


def _make_node(url: str, title: str, pam: str, tags: list[str]):
    node = MagicMock()
    node.url = url
    node.page_title = title
    node.pam_content = pam
    node.coverage_tags = tags
    return node


def test_plan_accessibility_one_per_node():
    planner = UniversalTestPlanner.__new__(UniversalTestPlanner)
    nodes = [
        _make_node("https://x.com/login", "Login", "login form", ["form"]),
        _make_node("https://x.com/about", "About", "static text", []),
    ]
    cases = planner._plan_accessibility(nodes, "https://x.com")
    assert len(cases) == 2
    assert all(tc.type == "accessibility" for tc in cases)
    assert all(isinstance(tc, TestCase) for tc in cases)


def test_plan_security_only_form_nodes():
    planner = UniversalTestPlanner.__new__(UniversalTestPlanner)
    nodes = [
        _make_node("https://x.com/login", "Login", "login form", ["form"]),
        _make_node("https://x.com/about", "About", "static text", []),
    ]
    cases = planner._plan_security(nodes, "https://x.com")
    assert len(cases) == 2  # XSS + SQLi only for the form node
    assert all(tc.type == "security" for tc in cases)


def test_plan_sorts_by_priority():
    planner = UniversalTestPlanner.__new__(UniversalTestPlanner)
    cases = [
        TestCase(title="low", type="functional", priority="low",
                 steps=["s"], expected_outcome="e", source_url="https://x.com"),
        TestCase(title="high", type="functional", priority="high",
                 steps=["s"], expected_outcome="e", source_url="https://x.com"),
        TestCase(title="med", type="functional", priority="medium",
                 steps=["s"], expected_outcome="e", source_url="https://x.com"),
    ]
    sorted_cases = planner._sort_by_priority(cases)
    assert [tc.priority for tc in sorted_cases] == ["high", "medium", "low"]
```

- [ ] **Step 2: Run failing tests**

```
uv run pytest tests/universal_qa/test_planner.py -v
```
Expected: `ImportError: cannot import name 'UniversalTestPlanner'`

- [ ] **Step 3: Implement test_planner.py**

```python
# src/universal_qa/test_planner.py
from __future__ import annotations

from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.contractskill.sfg import SFGNode, SFGStore
from src.llm.instructor_client import InstructorClient, StructuredGenerationError
from src.universal_qa.models import TestCase

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_XSS_PAYLOAD = "<script>alert('xss')</script>"
_SQLI_PAYLOAD = "' OR '1'='1"


class _FuncItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    priority: Literal["high", "medium", "low"] = "medium"
    preconditions: list[str] = Field(default_factory=list)
    steps: list[str]
    expected_outcome: str


class _FuncResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    test_cases: list[_FuncItem]


class UniversalTestPlanner:
    """Generates TestCase list from SFGStore nodes.

    Functional: LLM analyses each form-node's PAM content.
    Accessibility: rule-based, 1 TestCase per node.
    Security: template XSS + SQLi per form-node.
    """

    def __init__(self) -> None:
        self._client = InstructorClient()

    async def plan(self, sfg_store: SFGStore, start_url: str) -> list[TestCase]:
        nodes = sfg_store.get_nodes_by_url_prefix(start_url)
        if not nodes:
            logger.warning("UniversalTestPlanner: no SFG nodes found for %s", start_url)
            return []

        functional = await self._plan_functional(nodes, start_url)
        accessibility = self._plan_accessibility(nodes, start_url)
        security = self._plan_security(nodes, start_url)

        all_cases = functional + accessibility + security
        return self._sort_by_priority(all_cases)

    async def _plan_functional(
        self, nodes: list[SFGNode], start_url: str
    ) -> list[TestCase]:
        results: list[TestCase] = []
        form_nodes = [n for n in nodes if "form" in n.coverage_tags]
        if not form_nodes:
            form_nodes = nodes[:5]  # fallback: use first 5 pages

        for node in form_nodes[:10]:  # cap at 10 to avoid LLM overload
            prompt = (
                f"You are a QA engineer. Given this web page, write 2-3 test cases "
                f"(at least 1 happy-path + 1 negative) as JSON.\n\n"
                f"URL: {node.url}\n"
                f"Title: {node.page_title}\n"
                f"Page content summary:\n{node.pam_content[:800]}\n\n"
                f"Return JSON with field 'test_cases': list of objects each having "
                f"title, priority (high/medium/low), preconditions (list), "
                f"steps (list), expected_outcome."
            )
            try:
                response: _FuncResponse = await self._client.create_structured(
                    prompt, _FuncResponse, temperature=0.1
                )
                for item in response.test_cases:
                    results.append(TestCase(
                        title=item.title,
                        type="functional",
                        priority=item.priority,
                        preconditions=item.preconditions,
                        steps=item.steps,
                        expected_outcome=item.expected_outcome,
                        source_url=node.url,
                    ))
            except (StructuredGenerationError, Exception) as exc:
                logger.warning(f"UniversalTestPlanner: LLM failed for {node.url}: {exc!r}")
                results.append(TestCase(
                    title=f"Verify {node.page_title} loads",
                    type="functional",
                    priority="medium",
                    steps=[f"Navigate to {node.url}", "Verify page title is present"],
                    expected_outcome="Page loads without error",
                    source_url=node.url,
                ))
        return results

    def _plan_accessibility(
        self, nodes: list[SFGNode], start_url: str
    ) -> list[TestCase]:
        return [
            TestCase(
                title=f"Accessibility: {node.page_title}",
                type="accessibility",
                priority="medium",
                preconditions=[f"user is on {node.url}"],
                steps=[
                    f"Navigate to {node.url}",
                    "Check all interactive elements have accessible names",
                    "Check all images have alt text",
                    "Check no decorative roles on interactive elements",
                ],
                expected_outcome="No WCAG violations found",
                source_url=node.url,
            )
            for node in nodes
        ]

    def _plan_security(
        self, nodes: list[SFGNode], start_url: str
    ) -> list[TestCase]:
        results: list[TestCase] = []
        for node in nodes:
            if "form" not in node.coverage_tags:
                continue
            results.append(TestCase(
                title=f"XSS injection: {node.page_title}",
                type="security",
                priority="high",
                preconditions=[f"user is on {node.url}"],
                steps=[
                    f"Navigate to {node.url}",
                    f"Fill all text inputs with XSS payload: {_XSS_PAYLOAD}",
                    "Submit the form",
                    "Verify payload is not executed",
                ],
                expected_outcome="Page does not execute the script payload",
                source_url=node.url,
            ))
            results.append(TestCase(
                title=f"SQL injection: {node.page_title}",
                type="security",
                priority="high",
                preconditions=[f"user is on {node.url}"],
                steps=[
                    f"Navigate to {node.url}",
                    f"Fill all text inputs with SQLi payload: {_SQLI_PAYLOAD}",
                    "Submit the form",
                    "Verify no SQL error is exposed",
                ],
                expected_outcome="Page does not expose SQL errors or unintended data",
                source_url=node.url,
            ))
        return results

    @staticmethod
    def _sort_by_priority(cases: list[TestCase]) -> list[TestCase]:
        return sorted(cases, key=lambda tc: _PRIORITY_ORDER.get(tc.priority, 1))


__all__ = ["UniversalTestPlanner"]
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/universal_qa/test_planner.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```
git add src/universal_qa/test_planner.py tests/universal_qa/test_planner.py
git commit -m "feat(universal_qa): add UniversalTestPlanner — LLM functional + rule-based a11y/security"
```

---

## Task 5: Terminal Reporter

**Files:**
- Create: `src/universal_qa/reporters/__init__.py`
- Create: `src/universal_qa/reporters/terminal.py`
- Create: `tests/universal_qa/test_reporters.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/universal_qa/test_reporters.py
import pytest
from src.universal_qa.models import TestCase, StepTrace, TestResult
from src.universal_qa.reporters.terminal import TerminalReporter


def _make_result(title: str, passed: bool, type_: str = "functional",
                 failure_reason: str | None = None) -> TestResult:
    tc = TestCase(
        title=title, type=type_, priority="high",
        steps=["step"], expected_outcome="ok",
        source_url="https://example.com",
    )
    traces = [
        StepTrace(step="step", status="passed" if passed else "failed",
                  detail="detail", error=None if passed else failure_reason),
    ]
    return TestResult(
        test_case=tc, passed=passed, steps_trace=traces,
        failure_reason=failure_reason, duration_ms=1200,
    )


def test_terminal_reporter_formats_pass(capsys):
    reporter = TerminalReporter()
    result = _make_result("Login success", passed=True)
    reporter.report_one(result)
    captured = capsys.readouterr()
    assert "PASS" in captured.out
    assert "Login success" in captured.out
    assert "1.2s" in captured.out


def test_terminal_reporter_formats_fail_with_reason(capsys):
    reporter = TerminalReporter()
    result = _make_result("Login fail", passed=False,
                          failure_reason="Element not found within 30s")
    reporter.report_one(result)
    captured = capsys.readouterr()
    assert "FAIL" in captured.out
    assert "Element not found within 30s" in captured.out


def test_terminal_reporter_summary(capsys):
    reporter = TerminalReporter()
    results = [
        _make_result("T1", passed=True),
        _make_result("T2", passed=False, failure_reason="timeout"),
        _make_result("T3", passed=True),
    ]
    reporter.report_summary(results)
    captured = capsys.readouterr()
    assert "2" in captured.out   # passed
    assert "1" in captured.out   # failed
```

- [ ] **Step 2: Run failing tests (only terminal reporter tests)**

```
uv run pytest tests/universal_qa/test_reporters.py::test_terminal_reporter_formats_pass tests/universal_qa/test_reporters.py::test_terminal_reporter_formats_fail_with_reason tests/universal_qa/test_reporters.py::test_terminal_reporter_summary -v
```
Expected: `ImportError`

- [ ] **Step 3: Create reporters package and implement terminal.py**

```python
# src/universal_qa/reporters/__init__.py
# (empty)
```

```python
# src/universal_qa/reporters/terminal.py
from __future__ import annotations

from src.universal_qa.models import TestResult


class TerminalReporter:
    """Prints real-time pass/fail output to stdout."""

    def report_one(self, result: TestResult) -> None:
        status = "PASS" if result.passed else "FAIL"
        duration_s = result.duration_ms / 1000
        tc = result.test_case
        print(f"[{status}] {tc.type:<13} {tc.title:<55} ({duration_s:.1f}s)")
        if not result.passed:
            if result.failure_reason:
                print(f"       → {result.failure_reason}")
            for trace in result.steps_trace:
                if trace.status == "failed":
                    print(f"       → Step failed: {trace.step}")
                    if trace.error:
                        print(f"         Error: {trace.error}")

    def report_summary(self, results: list[TestResult]) -> None:
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        total = len(results)
        print("\n" + "=" * 70)
        print(f"  SUMMARY  total={total}  passed={passed}  failed={failed}")
        if failed > 0:
            print(f"\n  Failed tests:")
            for r in results:
                if not r.passed:
                    print(f"    - [{r.test_case.type}] {r.test_case.title}")
                    if r.failure_reason:
                        print(f"      {r.failure_reason}")
        print("=" * 70)


__all__ = ["TerminalReporter"]
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/universal_qa/test_reporters.py::test_terminal_reporter_formats_pass tests/universal_qa/test_reporters.py::test_terminal_reporter_formats_fail_with_reason tests/universal_qa/test_reporters.py::test_terminal_reporter_summary -v
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```
git add src/universal_qa/reporters/__init__.py src/universal_qa/reporters/terminal.py tests/universal_qa/test_reporters.py
git commit -m "feat(universal_qa): add TerminalReporter — real-time pass/fail output"
```

---

## Task 6: HTML Reporter

**Files:**
- Create: `src/universal_qa/reporters/html.py`
- Extend: `tests/universal_qa/test_reporters.py`

- [ ] **Step 1: Write failing tests (add to existing test_reporters.py)**

```python
# Append to tests/universal_qa/test_reporters.py

from src.universal_qa.reporters.html import HTMLReporter
import pathlib


def test_html_reporter_creates_file(tmp_path):
    reporter = HTMLReporter(output_dir=tmp_path)
    results = [
        _make_result("Login test", passed=True),
        _make_result("XSS test", passed=False, type_="security",
                     failure_reason="XSS payload executed"),
    ]
    path = reporter.generate(results)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Login test" in content
    assert "XSS test" in content
    assert "XSS payload executed" in content


def test_html_reporter_is_self_contained(tmp_path):
    reporter = HTMLReporter(output_dir=tmp_path)
    path = reporter.generate([_make_result("T", passed=True)])
    content = path.read_text(encoding="utf-8")
    assert "<style>" in content          # embedded CSS
    assert "http" not in content.split("<style>")[0].split("</style>")[0]  # no external links in CSS
```

- [ ] **Step 2: Run failing tests**

```
uv run pytest tests/universal_qa/test_reporters.py::test_html_reporter_creates_file tests/universal_qa/test_reporters.py::test_html_reporter_is_self_contained -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement html.py**

```python
# src/universal_qa/reporters/html.py
from __future__ import annotations

import base64
import datetime
import pathlib

from src.universal_qa.models import TestResult

_CSS = """
body{font-family:system-ui,sans-serif;margin:0;padding:20px;background:#f5f5f5}
h1{color:#333}.summary{display:flex;gap:16px;margin-bottom:24px}
.badge{padding:6px 16px;border-radius:6px;font-weight:700;font-size:14px}
.pass{background:#d4edda;color:#155724}.fail{background:#f8d7da;color:#721c24}
.info{background:#d1ecf1;color:#0c5460}
.filters{margin-bottom:16px}
.filters button{margin-right:8px;padding:4px 12px;cursor:pointer;border:1px solid #ccc;border-radius:4px}
.test-card{background:#fff;border-radius:8px;padding:16px;margin-bottom:12px;border-left:4px solid #ccc}
.test-card.passed{border-left-color:#28a745}.test-card.failed{border-left-color:#dc3545}
.test-title{font-size:16px;font-weight:600;margin-bottom:8px}
.test-meta{font-size:12px;color:#888;margin-bottom:8px}
.step{padding:4px 0;font-size:13px}
.step.passed::before{content:"✓ ";color:#28a745}
.step.failed::before{content:"✗ ";color:#dc3545}
.failure-reason{background:#fff3cd;padding:8px;border-radius:4px;font-size:13px;margin-top:8px}
.screenshot{margin-top:8px}
.screenshot img{max-width:100%;border:1px solid #ddd;border-radius:4px}
"""

_JS = """
function filter(type){
  document.querySelectorAll('.test-card').forEach(c=>{
    c.style.display=(type==='all'||c.dataset.type===type||c.dataset.status===type)?'':'none';
  });
}
"""


class HTMLReporter:
    """Generates a self-contained HTML QA report."""

    def __init__(self, output_dir: pathlib.Path | None = None) -> None:
        self._output_dir = output_dir or pathlib.Path("reports")

    def generate(self, results: list[TestResult]) -> pathlib.Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._output_dir / f"qa_report_{ts}.html"

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        cards = "\n".join(self._render_card(r) for r in results)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>QA Report {ts}</title>
<style>{_CSS}</style></head>
<body>
<h1>QA Report — {ts}</h1>
<div class="summary">
  <span class="badge info">Total: {len(results)}</span>
  <span class="badge pass">Passed: {passed}</span>
  <span class="badge fail">Failed: {failed}</span>
</div>
<div class="filters">
  <button onclick="filter('all')">All</button>
  <button onclick="filter('functional')">Functional</button>
  <button onclick="filter('accessibility')">Accessibility</button>
  <button onclick="filter('security')">Security</button>
  <button onclick="filter('passed')">Passed</button>
  <button onclick="filter('failed')">Failed</button>
</div>
{cards}
<script>{_JS}</script>
</body></html>"""
        path.write_text(html, encoding="utf-8")
        return path

    def _render_card(self, result: TestResult) -> str:
        tc = result.test_case
        status = "passed" if result.passed else "failed"
        steps_html = "".join(
            f'<div class="step {t.status}">{t.step}'
            + (f' — <em>{t.error}</em>' if t.error else "")
            + f' <small style="color:#aaa">({t.detail})</small></div>'
            for t in result.steps_trace
        )
        failure_html = (
            f'<div class="failure-reason">⚠ {result.failure_reason}</div>'
            if result.failure_reason else ""
        )
        screenshot_html = ""
        if result.screenshot_path:
            try:
                img_bytes = pathlib.Path(result.screenshot_path).read_bytes()
                b64 = base64.b64encode(img_bytes).decode()
                screenshot_html = (
                    f'<div class="screenshot">'
                    f'<img src="data:image/png;base64,{b64}" alt="screenshot"/>'
                    f'</div>'
                )
            except Exception:
                pass
        precond_html = (
            "<ul>" + "".join(f"<li>{p}</li>" for p in tc.preconditions) + "</ul>"
            if tc.preconditions else ""
        )
        return f"""<div class="test-card {status}" data-type="{tc.type}" data-status="{status}">
  <div class="test-title">{tc.title}</div>
  <div class="test-meta">{tc.type.upper()} | priority: {tc.priority} | {result.duration_ms}ms | {tc.source_url}</div>
  {precond_html}
  {steps_html}
  {failure_html}
  {screenshot_html}
</div>"""


__all__ = ["HTMLReporter"]
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/universal_qa/test_reporters.py -v
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```
git add src/universal_qa/reporters/html.py tests/universal_qa/test_reporters.py
git commit -m "feat(universal_qa): add HTMLReporter — self-contained HTML with filter + screenshots"
```

---

## Task 7: Test Runner

**Files:**
- Create: `src/universal_qa/test_runner.py`
- Create: `tests/universal_qa/test_runner.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/universal_qa/test_runner.py
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
    assert "not found within" in msg


def test_map_exception_assertion():
    msg = _map_exception(AssertionError("expected True"))
    assert "Expected outcome not met" in msg


def test_map_exception_generic():
    msg = _map_exception(ValueError("something"))
    assert "something" in msg


@pytest.mark.asyncio
async def test_runner_routes_accessibility():
    runner = UniversalTestRunner.__new__(UniversalTestRunner)
    tc = _make_tc("accessibility")
    page = AsyncMock()
    page.goto = AsyncMock()
    page.aria_snapshot = AsyncMock(return_value="- textbox \"Email\" []")
    page.evaluate = AsyncMock(return_value=0)

    with patch.object(runner, "_run_accessibility", new=AsyncMock(
        return_value=TestResult(test_case=tc, passed=True, duration_ms=100)
    )):
        result = await runner._dispatch(tc, page)
    assert result.passed


@pytest.mark.asyncio
async def test_runner_returns_result_on_exception():
    runner = UniversalTestRunner.__new__(UniversalTestRunner)
    runner._hyp_executor = AsyncMock()
    runner._screenshot_dir = None
    tc = _make_tc("functional")
    page = AsyncMock()

    with patch.object(runner, "_run_functional",
                      new=AsyncMock(side_effect=Exception("boom"))):
        result = await runner._dispatch(tc, page)
    assert not result.passed
    assert result.failure_reason is not None
```

- [ ] **Step 2: Run failing tests**

```
uv run pytest tests/universal_qa/test_runner.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement test_runner.py**

```python
# src/universal_qa/test_runner.py
from __future__ import annotations

import pathlib
import time

from loguru import logger
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


def _map_exception(exc: Exception) -> str:
    msg = str(exc)
    if isinstance(exc, TimeoutError) or "timeout" in msg.lower():
        return f"Element not found within 30s — {msg[:120]}"
    if isinstance(exc, AssertionError):
        return f"Expected outcome not met — {msg[:120]}"
    return msg[:200]


class UniversalTestRunner:
    """Executes a list of TestCase objects using Playwright.

    Routes by type:
    - functional   → HypothesisExecutor (Sprint 5, existing)
    - accessibility → AccessibilityObserver._scan static method
    - security     → SecurityObserver static methods + XSS/SQLi injection
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
        hyp = TestHypothesis(
            goal=tc.title,
            start_url=tc.source_url,
            preconditions=tc.preconditions,
            steps=tc.steps,
            expected_outcome=tc.expected_outcome,
            confidence=0.7,
        )
        hyp_result = await self._hyp_executor.execute(hyp, page)
        traces = [
            StepTrace(
                step=step,
                status="passed" if hyp_result.passed else "failed",
                detail=f"executed {hyp_result.steps_executed} step(s)",
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

    async def _run_accessibility(self, tc: TestCase, page: Page) -> TestResult:
        start = time.monotonic()
        traces: list[StepTrace] = []

        try:
            await page.goto(tc.source_url, wait_until="domcontentloaded", timeout=30_000)
            traces.append(StepTrace(step=f"Navigate to {tc.source_url}",
                                    status="passed", detail="page loaded"))
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
            f"ARIA: {aria_findings if aria_findings else ['ok']}; "
            f"missing alt: {missing_alt}"
        )
        traces.append(StepTrace(
            step="Check accessibility rules",
            status="passed" if passed else "failed",
            detail=detail,
            error="; ".join(aria_findings) if aria_findings else None,
        ))
        if missing_alt:
            traces.append(StepTrace(
                step="Check image alt text",
                status="failed",
                detail=f"{missing_alt} image(s) missing alt attribute",
                error=f"{missing_alt} images missing alt",
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
            traces.append(StepTrace(step=f"Navigate to {tc.source_url}",
                                    status="passed", detail="page loaded"))
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
            step=f"Fill {count} input(s) with payload",
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
                failure_reason = "XSS payload was executed (window.__xss_fired=true)"
        else:
            content = (await page.content()).lower()
            sql_keywords = ["sql syntax", "mysql_fetch", "ora-0", "sqlite", "syntax error near"]
            hit = next((kw for kw in sql_keywords if kw in content), None)
            if hit:
                passed = False
                failure_reason = f"SQL error keyword '{hit}' found in page response"

        # Also run SecurityObserver header checks
        missing_headers = await SecurityObserver._missing_headers(page, tc.source_url)
        if missing_headers:
            traces.append(StepTrace(
                step="Check security headers",
                status="failed",
                detail=f"missing: {missing_headers}",
                error=f"Missing headers: {', '.join(missing_headers)}",
            ))

        traces.append(StepTrace(
            step="Verify payload not executed",
            status="passed" if passed else "failed",
            detail="checked page response",
            error=failure_reason,
        ))

        screenshot = await self._maybe_screenshot(page, tc.id, passed)
        return TestResult(
            test_case=tc, passed=passed, steps_trace=traces,
            failure_reason=failure_reason, screenshot_path=screenshot,
            duration_ms=int((time.monotonic() - start) * 1000),
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
```

- [ ] **Step 4: Run tests**

```
uv run pytest tests/universal_qa/test_runner.py -v
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```
git add src/universal_qa/test_runner.py tests/universal_qa/test_runner.py
git commit -m "feat(universal_qa): add UniversalTestRunner — functional/accessibility/security dispatch"
```

---

## Task 8: Orchestrator + CLI

**Files:**
- Create: `src/universal_qa/agent.py`
- Create: `src/universal_qa/__main__.py`

- [ ] **Step 1: Implement agent.py**

```python
# src/universal_qa/agent.py
from __future__ import annotations

import pathlib

from loguru import logger
from playwright.async_api import async_playwright

from src.universal_qa.auth_manager import AuthManager
from src.universal_qa.models import TestResult
from src.universal_qa.reporters.html import HTMLReporter
from src.universal_qa.reporters.terminal import TerminalReporter
from src.universal_qa.site_discovery import SiteDiscovery
from src.universal_qa.test_planner import UniversalTestPlanner
from src.universal_qa.test_runner import UniversalTestRunner

_LAUNCH_ARGS = ["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"]


class UniversalQAAgent:
    """Orchestrates all 4 phases: Discover → Plan → Execute → Report."""

    def __init__(
        self,
        url: str,
        *,
        username: str | None = None,
        password: str | None = None,
        max_pages: int = 50,
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

    async def run(self) -> list[TestResult]:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=self._headless, args=_LAUNCH_ARGS
            )
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                )
            )
            page = await context.new_page()

            try:
                logger.info(f"UniversalQAAgent: starting run on {self._url}")

                # Phase 1: Navigate + Auth
                await page.goto(self._url, wait_until="domcontentloaded", timeout=30_000)
                await self._auth.setup(page)

                # Phase 2: Discover
                logger.info("Phase 2: site discovery")
                sfg_store = await self._discovery.discover(page, self._url)

                # Phase 3: Plan
                logger.info("Phase 3: generating test cases")
                test_cases = await self._planner.plan(sfg_store, self._url)
                logger.info(f"  {len(test_cases)} test cases generated")

                # Phase 4: Execute + Report
                logger.info("Phase 4: executing test cases")
                screenshot_dir = self._output_dir / "screenshots"
                runner = UniversalTestRunner(
                    sfg_store=sfg_store,
                    screenshot_dir=screenshot_dir,
                    terminal_reporter=self._terminal,
                )
                results = await runner.run(test_cases, page)

                # Final reports
                self._terminal.report_summary(results)
                html_reporter = HTMLReporter(output_dir=self._output_dir)
                report_path = html_reporter.generate(results)
                logger.info(f"HTML report: {report_path}")

                return results

            finally:
                await browser.close()


__all__ = ["UniversalQAAgent"]
```

- [ ] **Step 2: Implement __main__.py**

```python
# src/universal_qa/__main__.py
import argparse
import asyncio
import json

from src.universal_qa.agent import UniversalQAAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal QA Agent — test any website")
    parser.add_argument("--url", required=True, help="Website URL to test")
    parser.add_argument("--username", default=None, help="Login username/email (optional)")
    parser.add_argument("--password", default=None, help="Login password (optional)")
    parser.add_argument("--max-pages", type=int, default=50,
                        help="Max pages to crawl (default 50)")
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


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke test the CLI**

```
uv run python -m src.universal_qa --help
```
Expected output contains: `--url`, `--username`, `--password`, `--max-pages`

- [ ] **Step 4: Commit**

```
git add src/universal_qa/agent.py src/universal_qa/__main__.py
git commit -m "feat(universal_qa): add UniversalQAAgent orchestrator + CLI entry point"
```

---

## Task 9: SiteProfile in ContinuousLoopController

**Files:**
- Modify: `src/continuous/loop_controller.py`

- [ ] **Step 1: Read current __main__ block in loop_controller.py (line 631+)**

Confirm the `__main__` block at the bottom of the file parses `--url` and `--max-cycles`.

- [ ] **Step 2: Add SiteProfile and --profile argument**

In `src/continuous/loop_controller.py`, make these changes:

After the existing imports, add:
```python
from typing import Literal

SiteProfile = Literal["todomvc", "generic"]
```

In the `__main__` block, add `--profile` argument:
```python
parser.add_argument(
    "--profile",
    default="todomvc",
    choices=["todomvc", "generic"],
    help="Site profile: 'todomvc' uses hardcoded plans (default), "
         "'generic' uses SFGCrawler discovery",
)
```

In the `_main()` async function, branch on profile:
```python
async def _main() -> None:
    if args.profile == "generic":
        # Delegate to UniversalQAAgent for generic sites
        from src.universal_qa.agent import UniversalQAAgent
        agent = UniversalQAAgent(args.url, max_pages=50)
        results = await agent.run()
        print(json.dumps({
            "profile": "generic",
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
        }, indent=2))
        return

    # Original todomvc path — unchanged
    controller = ContinuousLoopController(args.url, max_cycles=args.max_cycles)
    state = await controller.run()
    print(json.dumps({
        "profile": "todomvc",
        "run_id": state.get("run_id", ""),
        "cycles_completed": state.get("cycle", 0),
        "tests_generated": len(state.get("tests_generated", [])),
        "tests_passed": len(state.get("tests_passed", [])),
        "tests_failed": len(state.get("tests_failed", [])),
        "stop_reason": state.get("stop_reason"),
        "otel_spans_emitted": controller.otel_spans_emitted,
    }, indent=2))
```

- [ ] **Step 3: Verify existing tests still pass**

```
uv run pytest tests/continuous -v
```
Expected: all tests pass (no regression)

- [ ] **Step 4: Test the new --profile flag**

```
uv run python -m src.continuous.loop_controller --help
```
Expected: shows `--profile {todomvc,generic}`

- [ ] **Step 5: Commit**

```
git add src/continuous/loop_controller.py
git commit -m "feat(loop_controller): add SiteProfile + --profile flag; generic delegates to UniversalQAAgent"
```

---

## Task 10: Integration Test

- [ ] **Step 1: Run full unit test suite for universal_qa**

```
uv run pytest tests/universal_qa/ -v
```
Expected: all tests pass

- [ ] **Step 2: Run live integration test**

```
uv run python -m src.universal_qa --url "https://automationexercise.com" --max-pages 15 --no-headless
```
Expected:
- Browser opens, navigates to site
- Terminal shows `[PASS]`/`[FAIL]` lines in real time
- `reports/qa_report_<ts>.html` created
- Final JSON summary printed

- [ ] **Step 3: Open HTML report**

```
start reports\qa_report_*.html
```
Verify: filter buttons work, failed tests show step trace and screenshot

- [ ] **Step 4: Test via loop_controller generic profile**

```
uv run python -m src.continuous.loop_controller --url "https://automationexercise.com" --max-cycles 1 --profile generic
```
Expected: delegates to UniversalQAAgent, JSON result shows `"profile": "generic"`

- [ ] **Step 5: Final commit**

```
git add -A
git commit -m "feat(universal_qa): complete Universal QA Agent — discover/plan/execute/report any website"
```
