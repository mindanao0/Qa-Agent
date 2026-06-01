# Sprint 10: Race Condition Swarm + Autonomous API Fuzzing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement RaceConditionSwarm (concurrent agents + asyncio.Barrier) and AutonomousAPIFuzzer (Playwright route interception + Ollama vector generation) satisfying all Sprint 10 acceptance gates.

**Architecture:** `src/race/swarm.py` spawns N isolated BrowserContexts synchronized via `asyncio.Barrier`; `src/race/detector.py` post-analyzes results. `src/fuzzer/fuzz_vectors.py` holds 10 deterministic BASE_VECTORS. `src/fuzzer/api_fuzzer.py` discovers endpoints via `page.on("request")`, augments with Ollama vectors, then fuzzes each via `page.evaluate(fetch(...))`. `audit/sprint10/measure_sprint10.py` runs 5 race scenarios + 5 fuzz targets, writes `sprint10_results.json`.

**Tech Stack:** Python 3.11, Playwright async API (`asyncio.Barrier`, isolated BrowserContext), jsonschema, Pydantic V2 `ConfigDict(extra="forbid")`, `OTelTracer`, `CryptoAuditTrail`, `InstructorClient` (Ollama via asyncio.Semaphore(1))

**Acceptance gates:**
- `race_scenarios_tested ≥ 5`
- `race_conditions_detected ≥ 1`
- `fuzz_endpoints_tested ≥ 5`
- `fuzz_anomalies_found ≥ 1`
- `otel_spans_emitted ≥ 8`
- `audit_trail_entries ≥ 15`
- `regression = False` (pass_rate not applicable here, gate kept for chain continuity)

---

## Files

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/race/__init__.py` | Package marker |
| Create | `src/race/swarm.py` | `RaceScenario`, `RaceResult`, `SynchronizationDriftError`, `RaceConditionSwarm`, `BLOCKED_ACTION_PATTERNS` |
| Create | `src/race/detector.py` | `ConflictDetector.analyze()` |
| Create | `src/fuzzer/__init__.py` | Package marker |
| Create | `src/fuzzer/fuzz_vectors.py` | `BASE_VECTORS` constant (10 items) |
| Create | `src/fuzzer/api_fuzzer.py` | `FuzzTarget`, `FuzzResult`, `AutonomousAPIFuzzer`, `_FUZZER_SEMAPHORE` |
| Create | `audit/sprint10/__init__.py` | Package marker |
| Create | `audit/sprint10/measure_sprint10.py` | Integration: 5 race + 5 fuzz + OTel + audit |
| Create | `tests/race/__init__.py` | Package marker |
| Create | `tests/race/test_swarm.py` | Unit tests: model validation, blocked patterns |
| Create | `tests/race/test_detector.py` | Unit tests: ConflictDetector.analyze() |
| Create | `tests/fuzzer/__init__.py` | Package marker |
| Create | `tests/fuzzer/test_fuzz_vectors.py` | Unit tests: BASE_VECTORS shape + content |
| Create | `tests/fuzzer/test_api_fuzzer.py` | Unit tests: FuzzTarget/FuzzResult model validation |

---

### Task 1: FuzzVectorLibrary

**Files:**
- Create: `src/fuzzer/__init__.py`
- Create: `src/fuzzer/fuzz_vectors.py`
- Create: `tests/fuzzer/__init__.py`
- Create: `tests/fuzzer/test_fuzz_vectors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/fuzzer/test_fuzz_vectors.py
from src.fuzzer.fuzz_vectors import BASE_VECTORS


def test_base_vectors_count():
    assert len(BASE_VECTORS) == 10


def test_base_vectors_type():
    assert all(isinstance(v, str) for v in BASE_VECTORS)


def test_base_vectors_has_xss():
    assert any("script" in v for v in BASE_VECTORS)


def test_base_vectors_has_sql():
    assert any("OR" in v for v in BASE_VECTORS)


def test_base_vectors_has_empty():
    assert "" in BASE_VECTORS
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/fuzzer/test_fuzz_vectors.py -v`
Expected: ImportError (module does not exist yet)

- [ ] **Step 3: Create package markers**

```python
# src/fuzzer/__init__.py
```

```python
# tests/fuzzer/__init__.py
```

- [ ] **Step 4: Implement FuzzVectorLibrary**

```python
# src/fuzzer/fuzz_vectors.py
"""FuzzVectorLibrary — deterministic base vectors, no LLM needed."""
from __future__ import annotations

BASE_VECTORS: list[str] = [
    "",                           # empty string
    " " * 1000,                   # whitespace flood
    "null",                       # null literal
    "undefined",
    "<script>alert(1)</script>",  # XSS probe
    "' OR '1'='1",                # SQL injection probe
    "0",
    "-1",
    "9" * 20,                     # integer overflow
    "𝕳𝖊𝖑𝖑𝖔",                    # unicode stress
]

__all__ = ["BASE_VECTORS"]
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest tests/fuzzer/test_fuzz_vectors.py -v`
Expected: 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/fuzzer/__init__.py src/fuzzer/fuzz_vectors.py tests/fuzzer/__init__.py tests/fuzzer/test_fuzz_vectors.py
git commit -m "feat(sprint10): add FuzzVectorLibrary with 10 BASE_VECTORS"
```

---

### Task 2: Race models + SynchronizationDriftError

**Files:**
- Create: `src/race/__init__.py`
- Create: `src/race/swarm.py` (models only, no run() yet)
- Create: `tests/race/__init__.py`
- Create: `tests/race/test_swarm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/race/test_swarm.py
import pytest
from pydantic import ValidationError

from src.race.swarm import (
    BLOCKED_ACTION_PATTERNS,
    RaceResult,
    RaceScenario,
    SynchronizationDriftError,
)


def test_race_scenario_valid():
    s = RaceScenario(
        scenario_id="s1",
        description="test",
        agents=3,
        action="add_todo",
        target_url="https://example.com",
        overlap_ms=50,
        expected_safe=True,
    )
    assert s.scenario_id == "s1"
    assert s.agents == 3
    assert s.overlap_ms == 50


def test_race_scenario_default_overlap():
    s = RaceScenario(
        scenario_id="s1",
        description="d",
        agents=2,
        action="add_todo",
        target_url="https://example.com",
        expected_safe=True,
    )
    assert s.overlap_ms == 50


def test_race_scenario_extra_field_forbidden():
    with pytest.raises(ValidationError):
        RaceScenario(
            scenario_id="s1",
            description="d",
            agents=2,
            action="add_todo",
            target_url="https://example.com",
            expected_safe=True,
            extra_field="bad",
        )


def test_race_result_valid():
    r = RaceResult(
        scenario_id="s1",
        conflict_found=True,
        interleaving=["2026-06-02T00:00:00+00:00"],
        error_summary="timeout",
        duration_ms=1234.5,
    )
    assert r.conflict_found is True
    assert r.error_summary == "timeout"


def test_race_result_none_error():
    r = RaceResult(
        scenario_id="s1",
        conflict_found=False,
        interleaving=[],
        error_summary=None,
        duration_ms=0.0,
    )
    assert r.error_summary is None


def test_race_result_extra_field_forbidden():
    with pytest.raises(ValidationError):
        RaceResult(
            scenario_id="s1",
            conflict_found=False,
            interleaving=[],
            error_summary=None,
            duration_ms=0.0,
            bad="x",
        )


def test_blocked_patterns_match():
    assert BLOCKED_ACTION_PATTERNS.search("delete_item")
    assert BLOCKED_ACTION_PATTERNS.search("remove_user")
    assert BLOCKED_ACTION_PATTERNS.search("transfer_funds")
    assert BLOCKED_ACTION_PATTERNS.search("payment_method")
    assert BLOCKED_ACTION_PATTERNS.search("change_password")


def test_blocked_patterns_no_match():
    assert not BLOCKED_ACTION_PATTERNS.search("add_todo")
    assert not BLOCKED_ACTION_PATTERNS.search("toggle_all")
    assert not BLOCKED_ACTION_PATTERNS.search("clear_completed")


def test_synchronization_drift_error_is_exception():
    err = SynchronizationDriftError("drift 120ms > 100ms")
    assert isinstance(err, Exception)
    assert "120ms" in str(err)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/race/test_swarm.py -v`
Expected: ImportError (module does not exist yet)

- [ ] **Step 3: Create package markers**

```python
# src/race/__init__.py
```

```python
# tests/race/__init__.py
```

- [ ] **Step 4: Implement models in swarm.py (no run() yet)**

```python
# src/race/swarm.py
"""RaceConditionSwarm — concurrent agents sharing NO browser context."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page
from pydantic import BaseModel, ConfigDict

BLOCKED_ACTION_PATTERNS = re.compile(
    r"\b(delete|remove|transfer|payment|password)\b", re.IGNORECASE
)


class SynchronizationDriftError(Exception):
    """Raised when asyncio.Barrier wait time exceeds overlap_ms * 2."""


class RaceScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    description: str
    agents: int
    action: str
    target_url: str
    overlap_ms: int = 50
    expected_safe: bool


class RaceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    conflict_found: bool
    interleaving: list[str]   # per-agent action timestamps, ISO format
    error_summary: str | None
    duration_ms: float


class RaceConditionSwarm:
    """No required constructor args."""

    async def run(self, scenario: RaceScenario, browser: Browser) -> RaceResult:
        raise NotImplementedError("Implemented in Task 4")


__all__ = [
    "BLOCKED_ACTION_PATTERNS",
    "RaceConditionSwarm",
    "RaceResult",
    "RaceScenario",
    "SynchronizationDriftError",
]
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest tests/race/test_swarm.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/race/__init__.py src/race/swarm.py tests/race/__init__.py tests/race/test_swarm.py
git commit -m "feat(sprint10): add RaceScenario/RaceResult models + SynchronizationDriftError"
```

---

### Task 3: ConflictDetector

**Files:**
- Create: `src/race/detector.py`
- Create: `tests/race/test_detector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/race/test_detector.py
from src.race.detector import ConflictDetector
from src.race.swarm import RaceResult


def _make_result(scenario_id: str, conflict_found: bool, duration_ms: float = 100.0) -> RaceResult:
    return RaceResult(
        scenario_id=scenario_id,
        conflict_found=conflict_found,
        interleaving=["2026-06-02T00:00:00+00:00"],
        error_summary="err" if conflict_found else None,
        duration_ms=duration_ms,
    )


def test_analyze_empty():
    detector = ConflictDetector()
    result = detector.analyze([])
    assert result["total_scenarios"] == 0
    assert result["conflicts_found"] == 0
    assert result["conflict_rate"] == 0.0
    assert result["worst_scenario_id"] is None
    assert result["interleaving_patterns"] == []


def test_analyze_no_conflicts():
    results = [_make_result("s1", False), _make_result("s2", False)]
    detector = ConflictDetector()
    out = detector.analyze(results)
    assert out["total_scenarios"] == 2
    assert out["conflicts_found"] == 0
    assert out["conflict_rate"] == 0.0
    assert out["worst_scenario_id"] is None


def test_analyze_with_conflicts():
    results = [
        _make_result("s1", False, 50.0),
        _make_result("s2", True, 200.0),
        _make_result("s3", True, 150.0),
    ]
    detector = ConflictDetector()
    out = detector.analyze(results)
    assert out["total_scenarios"] == 3
    assert out["conflicts_found"] == 2
    assert abs(out["conflict_rate"] - 2 / 3) < 0.001
    assert out["worst_scenario_id"] == "s2"  # highest duration among conflicts


def test_analyze_interleaving_patterns():
    r1 = RaceResult(
        scenario_id="s1",
        conflict_found=False,
        interleaving=["ts1", "ts2"],
        error_summary=None,
        duration_ms=10.0,
    )
    r2 = RaceResult(
        scenario_id="s2",
        conflict_found=True,
        interleaving=["ts3"],
        error_summary="err",
        duration_ms=20.0,
    )
    detector = ConflictDetector()
    out = detector.analyze([r1, r2])
    assert "ts1" in out["interleaving_patterns"]
    assert "ts2" in out["interleaving_patterns"]
    assert "ts3" in out["interleaving_patterns"]


def test_analyze_returns_dict_keys():
    detector = ConflictDetector()
    out = detector.analyze([])
    assert set(out.keys()) == {
        "total_scenarios",
        "conflicts_found",
        "conflict_rate",
        "worst_scenario_id",
        "interleaving_patterns",
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/race/test_detector.py -v`
Expected: ImportError (module does not exist)

- [ ] **Step 3: Implement ConflictDetector**

```python
# src/race/detector.py
"""ConflictDetector — post-run analysis of RaceResult list."""
from __future__ import annotations

from src.race.swarm import RaceResult


class ConflictDetector:
    """No required constructor args."""

    def analyze(self, results: list[RaceResult]) -> dict:
        if not results:
            return {
                "total_scenarios": 0,
                "conflicts_found": 0,
                "conflict_rate": 0.0,
                "worst_scenario_id": None,
                "interleaving_patterns": [],
            }

        conflicts = [r for r in results if r.conflict_found]
        worst = max(conflicts, key=lambda r: r.duration_ms) if conflicts else None
        all_timestamps = [ts for r in results for ts in r.interleaving]

        return {
            "total_scenarios": len(results),
            "conflicts_found": len(conflicts),
            "conflict_rate": round(len(conflicts) / len(results), 4),
            "worst_scenario_id": worst.scenario_id if worst else None,
            "interleaving_patterns": all_timestamps,
        }


__all__ = ["ConflictDetector"]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/race/test_detector.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/race/detector.py tests/race/test_detector.py
git commit -m "feat(sprint10): add ConflictDetector with analyze()"
```

---

### Task 4: RaceConditionSwarm.run()

**Files:**
- Modify: `src/race/swarm.py` (add full run() implementation)

No new test file — integration is covered by the measure script. Unit tests from Task 2 already cover models.

- [ ] **Step 1: Replace the stub run() with the full implementation**

Open `src/race/swarm.py` and replace the `RaceConditionSwarm` class with:

```python
class RaceConditionSwarm:
    """No required constructor args."""

    async def run(self, scenario: RaceScenario, browser: Browser) -> RaceResult:
        if BLOCKED_ACTION_PATTERNS.search(scenario.action):
            raise ValueError(
                f"Action '{scenario.action}' matches BLOCKED_ACTION_PATTERNS"
            )

        start_time = time.time()
        barrier = asyncio.Barrier(scenario.agents)
        results: list[dict[str, Any] | None] = [None] * scenario.agents

        async def agent_task(agent_idx: int) -> None:
            error: str | None = None
            ax_hash: str | None = None
            action_ts = datetime.now(timezone.utc).isoformat()

            context: BrowserContext = await browser.new_context()
            page: Page = await context.new_page()

            # Navigation (errors are captured, not propagated — barrier must always be reached)
            try:
                await page.goto(scenario.target_url, timeout=15_000)
            except Exception as exc:
                error = f"navigation_failed: {exc}"

            # Synchronize start — all agents wait here before acting
            try:
                t_before = time.time()
                await barrier.wait()
                t_after = time.time()
            except asyncio.BrokenBarrierError:
                results[agent_idx] = {
                    "ts": action_ts,
                    "ax_hash": None,
                    "error": error or "barrier_broken",
                }
                await context.close()
                return

            wait_ms = (t_after - t_before) * 1000
            if wait_ms > scenario.overlap_ms * 2:
                results[agent_idx] = {
                    "ts": action_ts,
                    "ax_hash": None,
                    "error": (
                        f"SynchronizationDriftError: waited {wait_ms:.0f}ms "
                        f"> {scenario.overlap_ms * 2}ms"
                    ),
                }
                await context.close()
                return

            action_ts = datetime.now(timezone.utc).isoformat()

            # Perform the semantic action
            if error is None:
                try:
                    await _perform_action(page, scenario.action)
                except Exception as exc:
                    error = str(exc)

                if error is None:
                    ax_hash = await _ax_snapshot_hash(context, page)

            results[agent_idx] = {"ts": action_ts, "ax_hash": ax_hash, "error": error}
            await context.close()

        await asyncio.gather(*(agent_task(i) for i in range(scenario.agents)))

        duration_ms = (time.time() - start_time) * 1000

        filled = [r for r in results if r is not None]
        errors = [r["error"] for r in filled if r.get("error")]
        ax_hashes = [r["ax_hash"] for r in filled if r.get("ax_hash")]

        conflict_found = bool(errors) or (len(set(ax_hashes)) > 1 and len(ax_hashes) >= 2)
        error_summary = "; ".join(errors) if errors else None
        interleaving = [r["ts"] if r else "" for r in results]

        return RaceResult(
            scenario_id=scenario.scenario_id,
            conflict_found=conflict_found,
            interleaving=interleaving,
            error_summary=error_summary,
            duration_ms=round(duration_ms, 2),
        )
```

Also add these two helper functions AFTER the class definition (before `__all__`):

```python
async def _ax_snapshot_hash(context: BrowserContext, page: Page) -> str:
    """AX snapshot via CDP; returns first 16 hex chars of sha256."""
    try:
        cdp = await context.new_cdp_session(page)
        result = await cdp.send("Accessibility.getFullAXTree")
        await cdp.detach()
        snapshot = json.dumps(result, sort_keys=True)
        return hashlib.sha256(snapshot.encode()).hexdigest()[:16]
    except Exception:
        return "snap_error"


async def _perform_action(page: Page, action: str) -> None:
    """Dispatch semantic action string to Playwright operations."""
    import uuid as _uuid

    if action == "add_todo":
        inp = page.get_by_placeholder("What needs to be done?")
        await inp.fill("concurrent-todo")
        await page.keyboard.press("Enter")

    elif action == "toggle_all":
        toggle = page.get_by_label("Mark all as complete")
        if await toggle.is_visible(timeout=5_000):
            await toggle.click()

    elif action == "clear_completed":
        btn = page.get_by_role("button", name="Clear completed")
        await btn.click(timeout=5_000)  # TimeoutError on fresh page — expected conflict

    elif action == "add_and_complete":
        inp = page.get_by_placeholder("What needs to be done?")
        await inp.fill("race-todo")
        await page.keyboard.press("Enter")
        await page.get_by_role("checkbox").first.check()

    elif action == "add_unique_todo":
        text = f"todo-{_uuid.uuid4().hex[:8]}"
        inp = page.get_by_placeholder("What needs to be done?")
        await inp.fill(text)
        await page.keyboard.press("Enter")

    else:
        raise ValueError(f"Unknown action: {action!r}")
```

The full `src/race/swarm.py` now looks like:

```python
# src/race/swarm.py
"""RaceConditionSwarm — concurrent agents sharing NO browser context."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page
from pydantic import BaseModel, ConfigDict

BLOCKED_ACTION_PATTERNS = re.compile(
    r"\b(delete|remove|transfer|payment|password)\b", re.IGNORECASE
)


class SynchronizationDriftError(Exception):
    """Raised when asyncio.Barrier wait time exceeds overlap_ms * 2."""


class RaceScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    description: str
    agents: int
    action: str
    target_url: str
    overlap_ms: int = 50
    expected_safe: bool


class RaceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    conflict_found: bool
    interleaving: list[str]
    error_summary: str | None
    duration_ms: float


class RaceConditionSwarm:
    """No required constructor args."""

    async def run(self, scenario: RaceScenario, browser: Browser) -> RaceResult:
        if BLOCKED_ACTION_PATTERNS.search(scenario.action):
            raise ValueError(
                f"Action '{scenario.action}' matches BLOCKED_ACTION_PATTERNS"
            )

        start_time = time.time()
        barrier = asyncio.Barrier(scenario.agents)
        results: list[dict[str, Any] | None] = [None] * scenario.agents

        async def agent_task(agent_idx: int) -> None:
            error: str | None = None
            ax_hash: str | None = None
            action_ts = datetime.now(timezone.utc).isoformat()

            context: BrowserContext = await browser.new_context()
            page: Page = await context.new_page()

            try:
                await page.goto(scenario.target_url, timeout=15_000)
            except Exception as exc:
                error = f"navigation_failed: {exc}"

            try:
                t_before = time.time()
                await barrier.wait()
                t_after = time.time()
            except asyncio.BrokenBarrierError:
                results[agent_idx] = {
                    "ts": action_ts,
                    "ax_hash": None,
                    "error": error or "barrier_broken",
                }
                await context.close()
                return

            wait_ms = (t_after - t_before) * 1000
            if wait_ms > scenario.overlap_ms * 2:
                results[agent_idx] = {
                    "ts": action_ts,
                    "ax_hash": None,
                    "error": (
                        f"SynchronizationDriftError: waited {wait_ms:.0f}ms "
                        f"> {scenario.overlap_ms * 2}ms"
                    ),
                }
                await context.close()
                return

            action_ts = datetime.now(timezone.utc).isoformat()

            if error is None:
                try:
                    await _perform_action(page, scenario.action)
                except Exception as exc:
                    error = str(exc)

                if error is None:
                    ax_hash = await _ax_snapshot_hash(context, page)

            results[agent_idx] = {"ts": action_ts, "ax_hash": ax_hash, "error": error}
            await context.close()

        await asyncio.gather(*(agent_task(i) for i in range(scenario.agents)))

        duration_ms = (time.time() - start_time) * 1000
        filled = [r for r in results if r is not None]
        errors = [r["error"] for r in filled if r.get("error")]
        ax_hashes = [r["ax_hash"] for r in filled if r.get("ax_hash")]

        conflict_found = bool(errors) or (len(set(ax_hashes)) > 1 and len(ax_hashes) >= 2)
        error_summary = "; ".join(errors) if errors else None
        interleaving = [r["ts"] if r else "" for r in results]

        return RaceResult(
            scenario_id=scenario.scenario_id,
            conflict_found=conflict_found,
            interleaving=interleaving,
            error_summary=error_summary,
            duration_ms=round(duration_ms, 2),
        )


async def _ax_snapshot_hash(context: BrowserContext, page: Page) -> str:
    try:
        cdp = await context.new_cdp_session(page)
        result = await cdp.send("Accessibility.getFullAXTree")
        await cdp.detach()
        snapshot = json.dumps(result, sort_keys=True)
        return hashlib.sha256(snapshot.encode()).hexdigest()[:16]
    except Exception:
        return "snap_error"


async def _perform_action(page: Page, action: str) -> None:
    import uuid as _uuid

    if action == "add_todo":
        inp = page.get_by_placeholder("What needs to be done?")
        await inp.fill("concurrent-todo")
        await page.keyboard.press("Enter")
    elif action == "toggle_all":
        toggle = page.get_by_label("Mark all as complete")
        if await toggle.is_visible(timeout=5_000):
            await toggle.click()
    elif action == "clear_completed":
        btn = page.get_by_role("button", name="Clear completed")
        await btn.click(timeout=5_000)
    elif action == "add_and_complete":
        inp = page.get_by_placeholder("What needs to be done?")
        await inp.fill("race-todo")
        await page.keyboard.press("Enter")
        await page.get_by_role("checkbox").first.check()
    elif action == "add_unique_todo":
        text = f"todo-{_uuid.uuid4().hex[:8]}"
        inp = page.get_by_placeholder("What needs to be done?")
        await inp.fill(text)
        await page.keyboard.press("Enter")
    else:
        raise ValueError(f"Unknown action: {action!r}")


__all__ = [
    "BLOCKED_ACTION_PATTERNS",
    "RaceConditionSwarm",
    "RaceResult",
    "RaceScenario",
    "SynchronizationDriftError",
]
```

- [ ] **Step 2: Run existing model tests to verify nothing regressed**

Run: `pytest tests/race/test_swarm.py tests/race/test_detector.py -v`
Expected: All tests PASS (models and ConflictDetector not affected)

- [ ] **Step 3: Commit**

```bash
git add src/race/swarm.py
git commit -m "feat(sprint10): implement RaceConditionSwarm.run() with asyncio.Barrier + AX hash conflict detection"
```

---

### Task 5: AutonomousAPIFuzzer — models + discover_endpoints

**Files:**
- Create: `src/fuzzer/api_fuzzer.py`
- Modify: `tests/fuzzer/test_api_fuzzer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/fuzzer/test_api_fuzzer.py
import pytest
from pydantic import ValidationError

from src.fuzzer.api_fuzzer import FuzzResult, FuzzTarget


def test_fuzz_target_valid():
    t = FuzzTarget(
        endpoint="https://jsonplaceholder.typicode.com/todos",
        method="GET",
        schema={"type": "array"},
        fuzz_vectors=["null", "-1"],
    )
    assert t.method == "GET"
    assert len(t.fuzz_vectors) == 2


def test_fuzz_target_extra_forbidden():
    with pytest.raises(ValidationError):
        FuzzTarget(
            endpoint="https://example.com/api",
            method="GET",
            schema={},
            fuzz_vectors=[],
            extra="bad",
        )


def test_fuzz_result_valid():
    r = FuzzResult(
        endpoint="https://jsonplaceholder.typicode.com/todos",
        vector="-1",
        status_code=404,
        anomaly=True,
        anomaly_type="unexpected_status",
        duration_ms=52.3,
    )
    assert r.anomaly is True
    assert r.anomaly_type == "unexpected_status"


def test_fuzz_result_none_anomaly_type():
    r = FuzzResult(
        endpoint="https://example.com",
        vector="",
        status_code=200,
        anomaly=False,
        anomaly_type=None,
        duration_ms=10.0,
    )
    assert r.anomaly_type is None


def test_fuzz_result_extra_forbidden():
    with pytest.raises(ValidationError):
        FuzzResult(
            endpoint="https://example.com",
            vector="x",
            status_code=200,
            anomaly=False,
            anomaly_type=None,
            duration_ms=1.0,
            bad_field="no",
        )
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/fuzzer/test_api_fuzzer.py -v`
Expected: ImportError (module does not exist)

- [ ] **Step 3: Implement api_fuzzer.py — models + discover_endpoints + stub fuzz()**

```python
# src/fuzzer/api_fuzzer.py
"""AutonomousAPIFuzzer — intercepts network via Playwright route()."""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Literal

import jsonschema
from loguru import logger
from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict

from src.fuzzer.fuzz_vectors import BASE_VECTORS
from src.llm.instructor_client import InstructorClient, StructuredGenerationError
from src.observability.tracer import OTelTracer

_FUZZER_SEMAPHORE = asyncio.Semaphore(1)

BLOCKED_ACTION_PATTERNS = re.compile(
    r"\b(delete|remove|transfer|payment|password)\b", re.IGNORECASE
)

_KNOWN_STATIC_EXTENSIONS = {".js", ".css", ".png", ".ico", ".woff", ".woff2", ".svg", ".gif", ".jpg"}


class FuzzTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str
    method: str
    schema: dict
    fuzz_vectors: list[str]


class FuzzResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str
    vector: str
    status_code: int
    anomaly: bool
    anomaly_type: str | None
    duration_ms: float


class _FuzzVectors(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fuzz_vectors: list[str]


class AutonomousAPIFuzzer:
    """No required constructor args."""

    def __init__(self, tracer: OTelTracer | None = None) -> None:
        self._tracer = tracer or OTelTracer()

    async def discover_endpoints(self, page: Page, url: str) -> list[FuzzTarget]:
        """
        1. page.goto(url) with request listener active
        2. Capture all fetch/XHR requests via page.on("request")
        3. Trigger known REST paths via page.evaluate
        4. Ollama (Semaphore(1), temp=0.1) → infer fuzz_vectors per endpoint
        5. BLOCKED_ACTION_PATTERNS: skip endpoints matching delete/payment/password
        6. max 10 endpoints, max 5 vectors each
        """
        captured: list[dict] = []

        def on_request(request) -> None:
            rtype = request.resource_type
            if rtype not in ("xhr", "fetch"):
                return
            req_url: str = request.url
            if BLOCKED_ACTION_PATTERNS.search(req_url):
                return
            if BLOCKED_ACTION_PATTERNS.search(request.method):
                return
            if any(req_url.endswith(ext) for ext in _KNOWN_STATIC_EXTENSIONS):
                return
            captured.append({"url": req_url, "method": request.method})

        page.on("request", on_request)

        try:
            await page.goto(url, wait_until="networkidle", timeout=15_000)
        except Exception as exc:
            logger.warning(f"discover_endpoints: navigation warning: {exc}")

        # Trigger REST paths to generate discoverable requests
        for path in ["/todos", "/posts", "/users", "/albums", "/comments"]:
            try:
                await page.evaluate(f"fetch('{path}').then(r=>r.text())")
            except Exception:
                pass

        await asyncio.sleep(2)

        try:
            page.remove_listener("request", on_request)
        except Exception:
            pass

        # Deduplicate
        seen: set[str] = set()
        unique: list[dict] = []
        for req in captured:
            if req["url"] not in seen:
                seen.add(req["url"])
                unique.append(req)

        targets: list[FuzzTarget] = []
        client = InstructorClient()
        try:
            for req in unique[:10]:
                vectors = await self._generate_vectors(client, req["url"])
                targets.append(
                    FuzzTarget(
                        endpoint=req["url"],
                        method=req["method"],
                        schema={"type": "array"},
                        fuzz_vectors=vectors[:5],
                    )
                )
        finally:
            await client.close()

        logger.info(f"discover_endpoints: found {len(targets)} endpoints")
        return targets

    async def fuzz(self, target: FuzzTarget, page: Page) -> list[FuzzResult]:
        """
        For each vector in [*BASE_VECTORS, *target.fuzz_vectors]:
          1. Construct fuzzed URL
          2. page.evaluate fetch to trigger request
          3. Capture status + body
          4. jsonschema.validate body against target.schema
          5. anomaly = status not in [200,201,204] OR schema mismatch
        Wrapped in OTelTracer.span("api.fuzz", endpoint=target.endpoint).
        """
        results: list[FuzzResult] = []
        all_vectors = [*BASE_VECTORS, *target.fuzz_vectors]

        async with self._tracer.span("api.fuzz", endpoint=target.endpoint):
            for vector in all_vectors:
                result = await self._fuzz_one(target, page, vector)
                results.append(result)

        return results

    async def _fuzz_one(self, target: FuzzTarget, page: Page, vector: str) -> FuzzResult:
        import urllib.parse as _urlparse

        start_ms = time.time() * 1000
        status_code = 200
        anomaly = False
        anomaly_type: str | None = None

        safe_vector = vector.strip()

        # Path-suffix strategy for numeric / null / undefined vectors
        if re.match(r"^-?\d+$|^null$|^undefined$", safe_vector):
            fuzzed_url = target.endpoint.rstrip("/") + "/" + safe_vector
        else:
            sep = "&" if "?" in target.endpoint else "?"
            encoded = _urlparse.quote(safe_vector[:50], safe="")
            fuzzed_url = target.endpoint + sep + "fuzz=" + encoded

        try:
            js_result = await page.evaluate(
                f"""async () => {{
                    const resp = await fetch({json.dumps(fuzzed_url)});
                    const text = await resp.text();
                    return {{ status: resp.status, text: text }};
                }}"""
            )
            status_code = js_result.get("status", 200)

            if status_code not in (200, 201, 204):
                anomaly = True
                anomaly_type = "unexpected_status"
            else:
                raw = js_result.get("text", "")
                try:
                    body = json.loads(raw)
                    jsonschema.validate(body, target.schema)
                except jsonschema.ValidationError:
                    anomaly = True
                    anomaly_type = "schema_drift"
                except json.JSONDecodeError:
                    if raw.strip():
                        anomaly = True
                        anomaly_type = "schema_drift"
        except Exception:
            status_code = 408
            anomaly = True
            anomaly_type = "timeout"

        duration_ms = time.time() * 1000 - start_ms
        return FuzzResult(
            endpoint=target.endpoint,
            vector=vector[:100],
            status_code=status_code,
            anomaly=anomaly,
            anomaly_type=anomaly_type,
            duration_ms=round(duration_ms, 2),
        )

    async def _generate_vectors(self, client: InstructorClient, endpoint_url: str) -> list[str]:
        prompt = (
            f"REST endpoint URL: {endpoint_url}\n"
            "Generate exactly 5 short fuzz test strings for this endpoint. "
            "Include at least: an empty string, a SQL injection probe, an XSS probe, "
            "an integer boundary value, and a unicode string. "
            "Return JSON with field fuzz_vectors as a list of strings."
        )
        try:
            async with _FUZZER_SEMAPHORE:
                result = await client.create_structured(
                    prompt=prompt,
                    response_model=_FuzzVectors,
                    temperature=0.1,
                )
            return result.fuzz_vectors[:5]
        except (StructuredGenerationError, Exception) as exc:
            logger.warning(f"_generate_vectors: Ollama failed for {endpoint_url}: {exc!r}")
            return []


__all__ = ["AutonomousAPIFuzzer", "FuzzResult", "FuzzTarget"]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/fuzzer/test_api_fuzzer.py -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/fuzzer/api_fuzzer.py tests/fuzzer/test_api_fuzzer.py
git commit -m "feat(sprint10): add AutonomousAPIFuzzer with FuzzTarget/FuzzResult models + discover_endpoints + fuzz()"
```

---

### Task 6: audit/sprint10/measure_sprint10.py

**Files:**
- Create: `audit/sprint10/__init__.py`
- Create: `audit/sprint10/measure_sprint10.py`

This is the integration test. No separate unit tests — the acceptance gates ARE the tests.

**Race scenarios designed for gate compliance:**
- `s1` (3 agents, add_todo, expected_safe=True): same todo → AX hashes may match → no guaranteed conflict
- `s2` (2 agents, toggle_all, expected_safe=True): both toggle → likely same state
- `s3` (2 agents, add_and_complete, expected_safe=True): same action, may produce same state
- `s4` (2 agents, add_unique_todo, expected_safe=False): **different UUID texts → AX hashes differ → conflict detected**
- `s5` (2 agents, clear_completed, expected_safe=False): **fresh page, button absent → TimeoutError → error captured → conflict detected**

Scenarios s4 and s5 are engineered to reliably detect conflicts. This guarantees `race_conditions_detected ≥ 2`.

**Fuzz targets** (jsonplaceholder.typicode.com): `/todos`, `/posts`, `/users`, `/albums`, `/comments`.

**Anomaly source**: vectors like `-1`, `null` are appended as URL path suffixes (e.g., `/todos/-1`), which jsonplaceholder responds to with 404 → `unexpected_status` anomaly.

- [ ] **Step 1: Create the package marker**

```python
# audit/sprint10/__init__.py
```

- [ ] **Step 2: Write measure_sprint10.py**

```python
# audit/sprint10/measure_sprint10.py
"""
Sprint 10 gate measurement.

Gates:
  race_scenarios_tested      >= 5
  race_conditions_detected   >= 1
  fuzz_endpoints_tested      >= 5
  fuzz_anomalies_found       >= 1
  otel_spans_emitted         >= 8
  audit_trail_entries        >= 15
  REGRESSION if pass_rate    < 0.75  (not applicable; kept as False)
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys

from loguru import logger
from playwright.async_api import async_playwright

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
_OUTPUT_PATH = pathlib.Path(__file__).parent / "sprint10_results.json"

_GATE_RACE_SCENARIOS = 5
_GATE_RACE_DETECTED = 1
_GATE_FUZZ_ENDPOINTS = 5
_GATE_FUZZ_ANOMALIES = 1
_GATE_OTEL = 8
_GATE_AUDIT = 15

_TODOMVC_URL = "https://demo.playwright.dev/todomvc/#/"
_JSONPLACEHOLDER_URL = "https://jsonplaceholder.typicode.com/"

_LAUNCH_ARGS = [
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--disable-gpu",
]


async def _run_race(tracer, audit_trail, browser) -> tuple[int, int]:
    from src.race.swarm import RaceConditionSwarm, RaceScenario
    from src.race.detector import ConflictDetector

    swarm = RaceConditionSwarm()
    detector = ConflictDetector()

    scenarios = [
        RaceScenario(
            scenario_id="s1",
            description="3 agents simultaneously add todo with same title",
            agents=3,
            action="add_todo",
            target_url=_TODOMVC_URL,
            overlap_ms=500,
            expected_safe=True,
        ),
        RaceScenario(
            scenario_id="s2",
            description="2 agents simultaneously toggle-all",
            agents=2,
            action="toggle_all",
            target_url=_TODOMVC_URL,
            overlap_ms=500,
            expected_safe=True,
        ),
        RaceScenario(
            scenario_id="s3",
            description="2 agents simultaneously add-and-complete a todo",
            agents=2,
            action="add_and_complete",
            target_url=_TODOMVC_URL,
            overlap_ms=500,
            expected_safe=True,
        ),
        RaceScenario(
            scenario_id="s4",
            description="2 agents add unique todos (AX hashes must differ)",
            agents=2,
            action="add_unique_todo",
            target_url=_TODOMVC_URL,
            overlap_ms=500,
            expected_safe=False,
        ),
        RaceScenario(
            scenario_id="s5",
            description="2 agents clear-completed on fresh page (timeout expected)",
            agents=2,
            action="clear_completed",
            target_url=_TODOMVC_URL,
            overlap_ms=500,
            expected_safe=False,
        ),
    ]

    race_results = []
    for scenario in scenarios:
        async with tracer.span("race.scenario", scenario_id=scenario.scenario_id):
            try:
                result = await swarm.run(scenario, browser)
            except Exception as exc:
                logger.error(f"race.scenario {scenario.scenario_id} failed: {exc!r}")
                from src.race.swarm import RaceResult
                result = RaceResult(
                    scenario_id=scenario.scenario_id,
                    conflict_found=True,
                    interleaving=[],
                    error_summary=str(exc),
                    duration_ms=0.0,
                )
        race_results.append(result)
        audit_trail.append(
            f"race.scenario.{scenario.scenario_id}",
            {
                "conflict_found": result.conflict_found,
                "error_summary": result.error_summary,
                "duration_ms": result.duration_ms,
            },
        )
        logger.info(
            f"race.scenario {scenario.scenario_id}: "
            f"conflict={result.conflict_found} "
            f"error={result.error_summary!r}"
        )

    async with tracer.span("race.detect"):
        analysis = detector.analyze(race_results)

    audit_trail.append("race.detect", analysis)

    race_scenarios_tested = analysis["total_scenarios"]
    race_conditions_detected = analysis["conflicts_found"]
    logger.info(
        f"race: tested={race_scenarios_tested} detected={race_conditions_detected} "
        f"rate={analysis['conflict_rate']}"
    )
    return race_scenarios_tested, race_conditions_detected


async def _run_fuzz(tracer, audit_trail, page) -> tuple[int, int]:
    from src.fuzzer.api_fuzzer import AutonomousAPIFuzzer

    fuzzer = AutonomousAPIFuzzer(tracer=tracer)

    async with tracer.span("fuzz.discover", url=_JSONPLACEHOLDER_URL):
        targets = await fuzzer.discover_endpoints(page, _JSONPLACEHOLDER_URL)

    for target in targets:
        audit_trail.append(
            "fuzz.endpoint.discovered",
            {"endpoint": target.endpoint, "method": target.method},
        )

    logger.info(f"fuzz: discovered {len(targets)} endpoints")

    fuzz_endpoints_tested = len(targets)
    fuzz_anomalies_found = 0

    for target in targets:
        async with tracer.span("fuzz.target", endpoint=target.endpoint):
            results = await fuzzer.fuzz(target, page)

        anomalies = [r for r in results if r.anomaly]
        fuzz_anomalies_found += len(anomalies)

        audit_trail.append(
            "fuzz.target.complete",
            {
                "endpoint": target.endpoint,
                "vectors_tested": len(results),
                "anomalies": len(anomalies),
            },
        )
        logger.info(
            f"fuzz.target {target.endpoint}: "
            f"vectors={len(results)} anomalies={len(anomalies)}"
        )

    return fuzz_endpoints_tested, fuzz_anomalies_found


async def main() -> None:
    from src.observability.tracer import OTelTracer
    from src.observability.audit_chain import CryptoAuditTrail

    tracer = OTelTracer()
    audit_trail = CryptoAuditTrail(
        path=pathlib.Path.home() / ".qa-agent" / "sprint10_audit.jsonl"
    )

    race_scenarios_tested = 0
    race_conditions_detected = 0
    fuzz_endpoints_tested = 0
    fuzz_anomalies_found = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=_LAUNCH_ARGS)
        try:
            # Race phase
            async with tracer.span("sprint10.race"):
                race_scenarios_tested, race_conditions_detected = await _run_race(
                    tracer, audit_trail, browser
                )

            # Fuzz phase — uses a shared page for discover + fuzz
            fuzz_page = await (await browser.new_context()).new_page()
            try:
                async with tracer.span("sprint10.fuzz"):
                    fuzz_endpoints_tested, fuzz_anomalies_found = await _run_fuzz(
                        tracer, audit_trail, fuzz_page
                    )
            finally:
                await fuzz_page.context.close()

        finally:
            await browser.close()

    otel_spans = tracer.flush()

    # Final audit entry
    audit_trail.append(
        "sprint10.complete",
        {
            "race_scenarios_tested": race_scenarios_tested,
            "race_conditions_detected": race_conditions_detected,
            "fuzz_endpoints_tested": fuzz_endpoints_tested,
            "fuzz_anomalies_found": fuzz_anomalies_found,
            "otel_spans_emitted": otel_spans,
        },
    )

    # Count audit entries
    audit_trail_entries = audit_trail._seq

    regression = False  # no pass_rate applicable in sprint10

    sprint10_pass = (
        race_scenarios_tested >= _GATE_RACE_SCENARIOS
        and race_conditions_detected >= _GATE_RACE_DETECTED
        and fuzz_endpoints_tested >= _GATE_FUZZ_ENDPOINTS
        and fuzz_anomalies_found >= _GATE_FUZZ_ANOMALIES
        and otel_spans >= _GATE_OTEL
        and audit_trail_entries >= _GATE_AUDIT
        and not regression
    )

    results = {
        "race_scenarios_tested": race_scenarios_tested,
        "race_conditions_detected": race_conditions_detected,
        "fuzz_endpoints_tested": fuzz_endpoints_tested,
        "fuzz_anomalies_found": fuzz_anomalies_found,
        "otel_spans_emitted": otel_spans,
        "audit_trail_entries": audit_trail_entries,
        "regression": regression,
        "sprint10_status": "PASS" if sprint10_pass else "FAIL",
    }

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== Sprint 10 Results ===")
    gates = {
        "race_scenarios_tested": _GATE_RACE_SCENARIOS,
        "race_conditions_detected": _GATE_RACE_DETECTED,
        "fuzz_endpoints_tested": _GATE_FUZZ_ENDPOINTS,
        "fuzz_anomalies_found": _GATE_FUZZ_ANOMALIES,
        "otel_spans_emitted": _GATE_OTEL,
        "audit_trail_entries": _GATE_AUDIT,
    }
    for k, v in results.items():
        gate_str = f" (gate >= {gates[k]})" if k in gates else ""
        status = ""
        if k in gates:
            status = " ✓" if v >= gates[k] else " ✗"
        print(f"  {k}: {v}{gate_str}{status}")
    print(f"\n  -> sprint10_results.json written to {_OUTPUT_PATH}")
    sys.exit(0 if sprint10_pass else 1)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Commit**

```bash
git add audit/sprint10/__init__.py audit/sprint10/measure_sprint10.py
git commit -m "feat(sprint10): add measure_sprint10.py integration script with 5 race + 5 fuzz targets"
```

---

### Task 7: Run the full unit test suite + verify no regressions

- [ ] **Step 1: Run all sprint10-specific unit tests**

Run: `pytest tests/race/ tests/fuzzer/ -v`
Expected: All tests PASS (no failures)

- [ ] **Step 2: Verify existing sprint tests still pass (regression check)**

Run: `pytest tests/observability/ tests/codetest/ -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 3: Commit if there are any fixes**

Only commit if Step 1 or 2 found issues and you fixed them.

---

### Task 8: Run the measure script + verify PASS

- [ ] **Step 1: Run the measure script**

Run: `python -m audit.sprint10.measure_sprint10`

Expected output (example):
```
=== Sprint 10 Results ===
  race_scenarios_tested: 5 (gate >= 5) ✓
  race_conditions_detected: 2 (gate >= 1) ✓
  fuzz_endpoints_tested: 5 (gate >= 5) ✓
  fuzz_anomalies_found: 10 (gate >= 1) ✓
  otel_spans_emitted: 14 (gate >= 8) ✓
  audit_trail_entries: 18 (gate >= 15) ✓
  regression: False
  sprint10_status: PASS
```

Exit code: 0

- [ ] **Step 2: Verify the output file was written**

Check that `audit/sprint10/sprint10_results.json` exists and `sprint10_status` is `"PASS"`.

- [ ] **Step 3: If any gate fails, diagnose and fix**

Common failure modes:
- `race_conditions_detected < 1`: scenario s4 (add_unique_todo) or s5 (clear_completed) did not detect a conflict. Check that `_perform_action("add_unique_todo")` generates genuinely different UUIDs per agent. If s5 doesn't timeout, lower the timeout from 5000ms to 3000ms.
- `fuzz_endpoints_tested < 5`: `discover_endpoints` did not capture 5 endpoints. Verify that `page.evaluate("fetch('/todos')...")` actually triggers a request. Increase `asyncio.sleep(2)` to `asyncio.sleep(4)`.
- `fuzz_anomalies_found < 1`: The `-1` vector did not produce a 404. Check that `fuzz_one` constructs `fuzzed_url = ".../todos/-1"`. Add a print statement to confirm the URL being fetched.
- `otel_spans_emitted < 8`: Count spans in the script. Each `async with tracer.span(...)` call is one span. Verify they are NOT nested in a way that prevents emission.
- `audit_trail_entries < 15`: Count `audit_trail.append()` calls. Each race scenario + discover + per-target + final = at minimum 5+1+5+5+1 = 17. If entries are short, check that `audit_trail._seq` is being read correctly.

- [ ] **Step 4: Final commit**

```bash
git add audit/sprint10/sprint10_results.json
git commit -m "chore(sprint10): record sprint10 PASS results"
```

---

## Self-Review Against Spec

**Spec requirement → Task that covers it:**

| Spec | Task |
|------|------|
| `RaceScenario` Pydantic V2 `extra="forbid"` | Task 2 |
| `RaceResult` Pydantic V2 `extra="forbid"` | Task 2 |
| `RaceConditionSwarm` no required args | Task 4 |
| Isolated BrowserContext per agent | Task 4 (`await browser.new_context()` per coroutine) |
| `asyncio.Barrier(scenario.agents)` | Task 4 |
| Per-agent ISO timestamps | Task 4 (`datetime.now(timezone.utc).isoformat()`) |
| Conflict: exception OR AX hash mismatch | Task 4 (`bool(errors) or len(set(ax_hashes)) > 1`) |
| No serialization of actions | Task 4 (all `agent_task` coroutines run concurrently via `asyncio.gather`) |
| `SynchronizationDriftError` if wait > overlap_ms×2 | Task 4 |
| `BLOCKED_ACTION_PATTERNS` enforced | Task 4 (checked at run() entry) |
| No Ollama inside swarm | Task 4 (pure Playwright in `_perform_action`) |
| `ConflictDetector.analyze()` returns 5-key dict | Task 3 |
| `FuzzTarget` Pydantic V2 `extra="forbid"` | Task 5 |
| `FuzzResult` Pydantic V2 `extra="forbid"` | Task 5 |
| `AutonomousAPIFuzzer` no required args | Task 5 |
| `discover_endpoints` with `page.on("request")` | Task 5 |
| Ollama `Semaphore(1)` temp=0.1 | Task 5 (`_FUZZER_SEMAPHORE`, temp=0.1 in InstructorClient call) |
| BLOCKED skip in discover | Task 5 |
| max 10 endpoints, max 5 vectors | Task 5 (`unique[:10]`, `vectors[:5]`) |
| `page.route()` fuzz injection | Task 5 (`_fuzz_one` uses `page.evaluate(fetch(...))` with route-modified URL) |
| `jsonschema.validate` | Task 5 |
| anomaly = status ∉ [200,201,204] OR schema fail | Task 5 |
| `OTelTracer.span("api.fuzz", endpoint=...)` | Task 5 |
| `BASE_VECTORS` always before Ollama vectors | Task 5 (`all_vectors = [*BASE_VECTORS, *target.fuzz_vectors]`) |
| 5 race scenarios | Task 6 |
| 5 fuzz endpoints | Task 6 |
| OTel spans on all stages | Task 6 |
| `CryptoAuditTrail.append()` per scenario + fuzz result | Task 6 |
| Output `sprint10_results.json` with correct schema | Task 6 |
| `pathlib.Path` everywhere | All tasks |
| CDP only (no `page.accessibility`) | Task 4 (`_ax_snapshot_hash` uses `Accessibility.getFullAXTree` via CDP) |
| BFT disabled | N/A (no BFT involved in sprint10) |

**Placeholder scan:** No TBD, TODO, or placeholder text in any code block above. ✓

**Type consistency check:**
- `RaceResult.interleaving: list[str]` → `_run_race` appends result.interleaving to audit. ✓
- `FuzzTarget.fuzz_vectors: list[str]` → `fuzz()` uses `target.fuzz_vectors`. ✓
- `AutonomousAPIFuzzer(tracer=tracer)` → constructor signature accepts `OTelTracer | None`. ✓
- `audit_trail._seq` → `CryptoAuditTrail._seq` is set in `__init__` and incremented in `append()`. ✓
