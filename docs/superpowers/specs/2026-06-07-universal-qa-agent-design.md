# Universal QA Agent — Design Spec
**Date:** 2026-06-07
**Status:** Approved

---

## Goal

Build a general-purpose web QA agent that works on **any website** — not just TodoMVC.
Given a URL it automatically:
1. Discovers what the site can do (crawl)
2. Generates test cases covering functional, accessibility, and security dimensions
3. Executes those test cases via Playwright
4. Reports results in real-time terminal output + a self-contained HTML report

First target: `https://automationexercise.com`

---

## Architecture

```
python -m src.universal_qa \
  --url https://automationexercise.com \
  [--username test@mail.com] [--password 1234] \
  [--max-pages 80]

┌─────────────────────────────────────────────────────┐
│                  UniversalQAAgent                   │
│                                                     │
│  Phase 1: DISCOVER                                  │
│    SiteDiscovery ──► SFGCrawler (existing)          │
│    AuthManager   ──► credentials OR auto-register   │
│                                                     │
│  Phase 2: PLAN                                      │
│    UniversalTestPlanner                             │
│      ├─ Functional test cases  (LLM analyses SFG)  │
│      ├─ Accessibility checks   (WCAG rules)         │
│      └─ Security probes        (XSS/SQLi)           │
│                                                     │
│  Phase 3: EXECUTE                                   │
│    UniversalTestRunner                              │
│      ├─ HypothesisExecutor     (existing)           │
│      ├─ AccessibilityObserver  (existing)           │
│      └─ SecurityObserver       (existing)           │
│                                                     │
│  Phase 4: REPORT                                    │
│    TerminalReporter  ──► real-time loguru           │
│    HTMLReporter      ──► self-contained .html       │
└─────────────────────────────────────────────────────┘
```

### ContinuousLoopController — future-safe refactor
Add `SiteProfile` concept to `loop_controller.py`:
- `"todomvc"` profile → uses existing hardcoded `_CYCLE_PLANS` (no sprint regression)
- `"generic"` profile → uses `SFGCrawler` discovery automatically
- Default remains `"todomvc"` until explicitly changed so all existing sprint audits pass unchanged

---

## Data Models

```python
class TestCase(BaseModel):
    id: str
    title: str
    type: Literal["functional", "accessibility", "security"]
    priority: Literal["high", "medium", "low"]
    preconditions: list[str]
    steps: list[str]
    expected_outcome: str
    source_url: str

class StepTrace(BaseModel):
    step: str
    status: Literal["passed", "failed", "skipped"]
    detail: str        # human-readable: "Found button by role, clicked successfully"
    error: str | None  # populated on failure

class TestResult(BaseModel):
    test_case: TestCase
    passed: bool
    steps_trace: list[StepTrace]
    failure_reason: str | None   # plain-English: UniversalTestRunner maps exception types
                                 # (TimeoutError→"Element not found within 30s",
                                 #  AssertionError→"Expected outcome not met", etc.)
    screenshot_path: str | None  # captured immediately on failure
    duration_ms: int
```

All models use `extra="forbid"` (Pydantic V2).

---

## Phase Details

### Phase 1 — Discover

**SiteDiscovery** (`src/universal_qa/site_discovery.py`)
- Wraps existing `SFGCrawler` with `CrawlerConfig(max_pages=max_pages, max_depth=6)`
- Returns `SFGStore` containing every discovered page, form, and interactive element

**AuthManager** (`src/universal_qa/auth_manager.py`)
- Detects login/register forms via AOM role analysis
- Strategy:
  1. If `--username` + `--password` provided → fill login form and proceed
  2. Else → locate register form → create `qa_test_<8-char-rand>@mailinator.com` account → login
  3. If no auth form found → continue unauthenticated (public pages only)
- Auth state persisted in `BrowserContext` for the full run

### Phase 2 — Plan

**UniversalTestPlanner** (`src/universal_qa/test_planner.py`)
- Reads SFG nodes from `SFGStore`
- Calls Ollama (Semaphore(1), temperature=0.1) to generate test cases per page cluster
- Produces three test lists:

| Type | Source | Rule |
|------|--------|------|
| Functional | LLM analyses forms + flows in SFG | ≥1 happy-path + ≥1 negative per form |
| Accessibility | Rule-based from AOM | 1 TestCase per page — checks all 3 WCAG rules in sequence: missing labels, missing alt text, low contrast; reports each as a separate StepTrace |
| Security | Template-based | XSS `<script>alert(1)</script>` + SQLi `' OR '1'='1` on every input field |

- Output: `list[TestCase]` sorted by priority (high → medium → low)

### Phase 3 — Execute

**UniversalTestRunner** (`src/universal_qa/test_runner.py`)
- Iterates `list[TestCase]`, routes by `type`:
  - `functional` → `HypothesisExecutor` (existing, Sprint 5)
  - `accessibility` → `AccessibilityObserver` (existing, Sprint 5)
  - `security` → `SecurityObserver` (existing, Sprint 5)
- Per test case:
  - Records `StepTrace` for every step
  - On failure → captures screenshot to `reports/screenshots/<id>.png`
  - On failure → `RepairEngine` retries once
  - Timeout: 30s per test case
  - Never fabricates pass; records actual result
- Emits OTel span per test case

### Phase 4 — Report

**TerminalReporter** (`src/universal_qa/reporters/terminal.py`)
- Prints real-time as each test completes:
  ```
  [PASS] functional  Login with valid credentials          (2.1s)
  [FAIL] functional  Login with invalid credentials        (5.3s)
         → Step 3 failed: Expected error message not found
         → Tried: get_by_role("alert") — TimeoutError after 5s
  ```

**HTMLReporter** (`src/universal_qa/reporters/html.py`)
- Generates one self-contained `reports/qa_report_<timestamp>.html`
- Sections:
  - Summary bar: total / pass / fail / skip, grouped by type
  - Filter controls: by type (functional/accessibility/security), by status
  - Per test card: title, priority, preconditions, step-by-step trace with status badges,
    failure reason in plain English, inline screenshot (base64) on failure
- No external CSS/JS dependencies (fully self-contained)

---

## File Layout

```
src/universal_qa/
    __init__.py
    __main__.py          # CLI entry point
    agent.py             # UniversalQAAgent orchestrator
    site_discovery.py    # SiteDiscovery (wraps SFGCrawler)
    auth_manager.py      # AuthManager (credentials + auto-register)
    test_planner.py      # UniversalTestPlanner
    test_runner.py       # UniversalTestRunner
    models.py            # TestCase, StepTrace, TestResult
    reporters/
        __init__.py
        terminal.py      # TerminalReporter
        html.py          # HTMLReporter

reports/                 # generated at runtime (gitignored)
    qa_report_<ts>.html
    screenshots/
        <test_id>.png
```

### Existing files touched
| File | Change |
|------|--------|
| `src/continuous/loop_controller.py` | Add `SiteProfile` + `--profile` CLI arg; default `"todomvc"` |

### Existing files NOT touched
All Sprint 1–15 audit files, measure scripts, and test suites remain unchanged.

---

## Testing

- Unit tests in `tests/universal_qa/`:
  - `test_models.py` — TestCase / StepTrace / TestResult validation
  - `test_planner.py` — LLM output parsing, type routing
  - `test_auth_manager.py` — credential strategy selection
  - `test_reporters.py` — HTML structure, terminal output format
- Integration: `uv run python -m src.universal_qa --url https://automationexercise.com --max-pages 30`

---

## Constraints

- LLM: Ollama localhost:11434 only, Semaphore(1), temperature=0.1
- Locators: `get_by_role`, `get_by_label`, `get_by_text`, `get_by_test_id` only — no CSS/XPath
- VRAM: ≤5800MB hard limit (existing VRAMMonitor)
- Security probes: read-path only — no account deletion, no data destruction
- Auth auto-register: uses `@mailinator.com` throwaway addresses only
