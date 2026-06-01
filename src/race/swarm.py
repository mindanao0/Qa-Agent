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
    r"delete|remove|transfer|payment|password", re.IGNORECASE
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

            # Navigation (errors captured; barrier must always be reached)
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
        # TimeoutError on fresh page (no completed todos) — this IS the expected conflict
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
