# SiteExplorer — Design Spec
**Date:** 2026-06-07
**Status:** Approved

---

## Goal

Replace the current single-phase SiteDiscovery (BFS link extraction only) with a **two-phase approach** that adds a systematic interaction-based exploration phase. The result is a `NavigationMap` — a structured record of every page, every action, and every multi-step flow the agent discovered by actually clicking through the site. `UniversalTestPlanner` then generates test cases whose steps come directly from the exploration, making them correct by construction.

---

## Architecture

```
python -m src.universal_qa \
  --url https://www.saucedemo.com/ \
  --username standard_user --password secret_sauce \
  --max-pages 20 --explore-timeout 5 --max-depth 4 \
  --allow-destructive   # optional

┌─────────────────────────────────────────────────────────────┐
│                    UniversalQAAgent                         │
│                                                             │
│  Phase 1: AUTH                                              │
│    AuthManager  ──► login or auto-register                  │
│                                                             │
│  Phase 2: DISCOVERY (fast)                                  │
│    SiteDiscovery ──► BFS + interaction ──► list[URL]        │
│                                                             │
│  Phase 3: EXPLORATION (thorough)  ← NEW                     │
│    SiteExplorer ──► clicks every element per page           │
│                 ──► builds NavigationMap                    │
│                 ──► records real paths + steps              │
│                                                             │
│  Phase 4: PLAN                                              │
│    UniversalTestPlanner ──► receives NavigationMap          │
│      ├─ per-page test cases (functional / a11y / security)  │
│      └─ flow-based test cases (end-to-end from real paths)  │
│                                                             │
│  Phase 5: EXECUTE + REPORT (unchanged)                      │
└─────────────────────────────────────────────────────────────┘
```

**Terminal output during exploration:**
```
[EXPLORE] https://www.saucedemo.com/inventory.html
  → คลิก "Add to cart" ──► state เปลี่ยน (cart_badge: 0→1)
  → คลิก "Cart" ──► navigate: /cart.html ✓
  → พบ flow: inventory → add_to_cart → cart  (3 steps)
  Pages found: 4 | Actions recorded: 12 | Flows: 2

[EXPLORE] https://www.saucedemo.com/cart.html
  → คลิก "Checkout" ──► navigate: /checkout-step-one.html ✓
```

---

## Data Models (`src/universal_qa/explorer/nav_map.py`)

```python
class ExploredAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_url: str                    # page where action was taken
    action_label: str                # human label e.g. "Add to cart"
    element_role: str | None         # ARIA role e.g. "button"
    element_name: str | None         # ARIA name e.g. "Checkout"
    element_selector: str | None     # fallback CSS/XPath selector
    leads_to_url: str | None         # URL navigated to (None = modal/state)
    leads_to_modal: bool = False     # True if action opens a modal
    state_change: dict | None = None # e.g. {"cart_badge": "0→1"}
    is_destructive: bool = False     # matched BLOCKED_ACTION_PATTERNS


class ExploredPage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    title: str
    pam_content: str                         # from SFGCrawler._visit_node
    actions: list[ExploredAction]
    state_snapshot: str | None = None        # localStorage/sessionStorage JSON
    requires_path: list[ExploredAction] = Field(default_factory=list)


class NavigationFlow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    flow_id: str
    name: str                        # e.g. "add_to_cart_and_checkout"
    steps: list[ExploredAction]      # sequence of real successful actions
    start_url: str
    end_url: str


class NavigationMap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str
    pages: list[ExploredPage]
    flows: list[NavigationFlow]
    explored_at_iso: str
```

---

## SiteExplorer Algorithm (`src/universal_qa/explorer/site_explorer.py`)

**Input:** `page: Page`, `discovered_urls: list[str]`, `config: ExplorerConfig`

**Output:** `NavigationMap`

### Per-page exploration loop

For each URL from SiteDiscovery:

1. **NAVIGATE** — goto URL; replay `requires_path` if state is needed; restore `state_snapshot` if available. If session expired → SessionGuard re-authenticates and retries.

2. **FIND ELEMENTS** — ElementScanner runs JavaScript to collect interactive elements:
   - Tags: `<button>`, `<a href="#">`, `[role=button]`, `[role=link]`, `input[type=submit]`, `select`, `[onclick]`
   - Same-origin iframes: pierce and scan inside
   - Shadow DOM: pierce and scan inside
   - Priority order: nav/header elements → form buttons → everything else

3. **FOR EACH ELEMENT (sequential, 3s timeout):**
   - Check BLOCKED patterns → skip if dangerous (unless `--allow-destructive`)
   - Check `visited_actions` (URL + element key) → skip if already tried
   - Save `pre_url` and `pre_state` (localStorage snapshot)
   - Click element
   - Wait for `networkidle` (max 1s)
   - Classify result:
     - **URL changed** → `ExploredAction(leads_to_url=new_url)`, add to BFS queue
     - **Modal opened** → `ExploredAction(leads_to_modal=True)`, explore modal recursively
     - **State changed** → record `state_change` dict
     - **Nothing** → skip
   - Navigate back to original URL

4. **FORM FILLING** — if element is a form submit button:
   - Try dummy data first: `"Test"`, `"test@test.com"`, `"12345"`, `"Test Address"`
   - If validation error → FormFiller asks LLM to generate data from field labels
   - If still failing → skip this form

5. **CYCLE DETECTION** (option D):
   - Track visited URL+action pairs → skip if already done
   - Each URL visited max 2 times
   - Detect cycles from navigation graph — stop branch if path returns to already-visited URL

6. **FLOW DETECTION** — if a path has ≥2 steps and `end_url ≠ start_url`:
   - Auto-create `NavigationFlow` with the step sequence

### Stop conditions (any one triggers stop)
- `pages_visited >= max_pages`
- `elapsed >= explore_timeout` minutes
- `depth > max_depth`

---

## Element Scanner (`src/universal_qa/explorer/element_scanner.py`)

Returns `list[ElementCandidate]` per page:

```python
class ElementCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str                   # human-readable label
    role: str | None             # ARIA role (primary selector)
    name: str | None             # ARIA name (primary selector)
    selector: str | None         # fallback: id / class / nth-of-type
    is_in_iframe: bool = False
    is_in_shadow: bool = False
    priority: int                # 0=nav, 1=form-button, 2=other
```

JavaScript evaluation uses `document.querySelectorAll` + `element.shadowRoot` piercing + `frame.contentDocument` for same-origin iframes.

---

## Form Filler (`src/universal_qa/explorer/form_filler.py`)

```python
_DUMMY_BY_TYPE = {
    "email":    "qa_test@mailinator.com",
    "password": "QaTest123!",
    "text":     "Test Input",
    "number":   "12345",
    "tel":      "0812345678",
    "date":     "2026-01-01",
}
```

If dummy data causes a validation error → call `InstructorClient.create_structured()` with field labels to generate plausible values. Temperature=0.1, Semaphore(1).

---

## Session Guard (`src/universal_qa/explorer/session_guard.py`)

After each navigation, checks if the page has returned to the login form:
- Looks for `input[type=password]` OR login-related title keywords
- If detected → calls `AuthManager.setup(page)` again → continues exploration
- Max 3 re-auth attempts per run

---

## TestPlanner Integration

`UniversalTestPlanner.plan()` signature changes from `(sfg_store, start_url)` → `(nav_map: NavigationMap)`.

**Per-page tests** — same as before but richer: LLM sees `pam_content` + `actions` list instead of just PAM content.

**Flow-based tests** — new `_plan_flows()` method converts each `NavigationFlow` directly to a `TestCase` without LLM:
```python
TestCase(
    title=f"ทดสอบ {flow.name} ตั้งแต่ต้นจนจบ",
    type="functional",
    priority="high",
    steps=[_action_to_step(a) for a in flow.steps],
    expected_outcome=f"เข้าหน้า {flow.end_url} สำเร็จ",
    source_url=flow.start_url,
)
```

`_action_to_step(action)` converts `ExploredAction` → Thai step string:
- navigate → `เปิดหน้า {url}`
- button click → `คลิกปุ่ม "{element_name}"`
- link click → `คลิก "{element_name}"`
- fill → `กรอก "{element_name}" ด้วย "{value}"`

---

## File Layout

```
src/universal_qa/
    explorer/                          ← NEW
        __init__.py
        site_explorer.py               ← SiteExplorer orchestrator
        nav_map.py                     ← NavigationMap + all sub-models
        element_scanner.py             ← scan + classify interactive elements
        form_filler.py                 ← dummy data + LLM fallback
        session_guard.py               ← detect session expiry + re-auth

    models.py                          ← unchanged
    site_discovery.py                  ← unchanged (Phase 2)
    auth_manager.py                    ← unchanged
    test_planner.py                    ← MODIFIED: accepts NavigationMap
    test_runner.py                     ← unchanged
    agent.py                           ← MODIFIED: add Phase 3
    __main__.py                        ← MODIFIED: new CLI args

tests/universal_qa/
    test_nav_map.py                    ← NEW
    test_element_scanner.py            ← NEW
    test_form_filler.py                ← NEW
    test_session_guard.py              ← NEW
    test_site_explorer.py              ← NEW
    test_planner.py                    ← MODIFIED: update for NavigationMap
```

### CLI changes
```
--explore-timeout INT   exploration timeout in minutes (default 5)
--max-depth INT         max navigation depth (default 4)
--allow-destructive     allow clicking delete/remove/payment actions
```

---

## Constraints

- LLM: Ollama localhost:11434, Semaphore(1), temperature=0.1 (FormFiller only)
- Locators: ARIA role+name primary; multi-selector fallback
- Sequential execution (no parallelism) for stability
- BLOCKED_ACTION_PATTERNS from Sprint 4 still apply unless `--allow-destructive`
- Session re-auth: max 3 attempts per run
- All Pydantic models use `extra="forbid"`
- Sprint 1–15 audit files and measure scripts: untouched
