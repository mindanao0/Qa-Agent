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
        semantic_hashes = [r["ax_hash"] for r in filled if r.get("ax_hash")]

        # Identical application state across agents → identical semantic_hash → no
        # conflict. A genuine race surfaces either as an action error or as
        # divergent semantic state (different hashes). Raw CDP nodeId/backendDOMNodeId
        # are deliberately excluded (see _semantic_hash) so that nodeId drift between
        # isolated BrowserContexts can no longer manufacture a false positive.
        conflict_found = bool(errors) or (len(set(semantic_hashes)) > 1)
        error_summary = "; ".join(errors) if errors else None
        interleaving = [r["ts"] if r else "" for r in results]

        return RaceResult(
            scenario_id=scenario.scenario_id,
            conflict_found=conflict_found,
            interleaving=interleaving,
            error_summary=error_summary,
            duration_ms=round(duration_ms, 2),
        )


def _semantic_hash(ax_tree: dict) -> str:
    """Hash only the SEMANTIC state of interactive nodes — role / name / checked —
    ignoring CDP nodeId / backendDOMNodeId / childIds.

    Those raw identifiers differ by construction between isolated BrowserContexts,
    so hashing the full AX tree (the old behaviour) guaranteed a different hash for
    every agent and therefore a false 'conflict' even when the rendered application
    state was byte-for-byte identical. Projecting onto (role, name, checked) for the
    interactive roles lets identical app state produce an identical hash.
    """
    nodes = ax_tree.get("nodes", [])
    semantic = sorted(
        [
            {
                "role": (n.get("role") or {}).get("value", ""),
                "name": (n.get("name") or {}).get("value", ""),
                "checked": (n.get("checked") or {}).get("value", ""),
            }
            for n in nodes
            if (n.get("role") or {}).get("value", "")
            in ("checkbox", "textbox", "button", "listitem")
        ],
        key=lambda x: (x["role"], x["name"]),
    )
    return hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()


async def _ax_snapshot_hash(context: BrowserContext, page: Page) -> str:
    """Capture the AX tree via CDP and reduce it to a semantic-state hash.

    Returns the sha256 of the interactive-node semantic projection (see
    _semantic_hash) instead of a hash of the raw CDP payload.
    """
    try:
        cdp = await context.new_cdp_session(page)
        result = await cdp.send("Accessibility.getFullAXTree")
        await cdp.detach()
        return _semantic_hash(result)
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

    elif action == "read_only_view":
        # Pure read — no state mutation, no localStorage write. Counting the todo
        # list items is a read-only AOM query that returns 0 on a fresh page and
        # never raises, so concurrent agents always observe identical state.
        # Demonstrates that read operations cannot produce a race conflict.
        await page.get_by_role("listitem").count()

    else:
        raise ValueError(f"Unknown action: {action!r}")


__all__ = [
    "BLOCKED_ACTION_PATTERNS",
    "RaceConditionSwarm",
    "RaceResult",
    "RaceScenario",
    "SynchronizationDriftError",
]
