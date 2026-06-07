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

from src.race.interleaving_recorder import AgentEvent, InterleavingRecorder

BLOCKED_ACTION_PATTERNS = re.compile(
    r"delete|remove|transfer|payment|password", re.IGNORECASE
)

# HTTP status codes that are NOT a conflict for the spec's conflict rule.
_OK_STATUSES: frozenset[int] = frozenset({200, 201, 204})


class SynchronizationDriftError(Exception):
    """Raised when asyncio.Barrier wait time exceeds overlap_ms * 2."""


def _skips_navigation(action: str) -> bool:
    """HTTP / conduit actions use ``page.request`` and need no ``page.goto``.

    Skipping the pre-barrier navigation removes the navigation-time variance that
    manufactured Sprint 10 ``SynchronizationDriftError`` false-positives — the
    conflict signal then comes purely from the real HTTP response.
    """
    return action.startswith("http_") or action.startswith("conduit_")


def _compute_conflict(
    statuses: list[int],
    semantic_hashes: list[str],
    errors: list[str],
) -> bool:
    """Spec conflict rule: a conflict exists if any agent saw a non-OK HTTP
    status, OR agents observed divergent semantic state, OR any agent errored.

    CDP nodeId/backendDOMNodeId/childIds are excluded from the hash upstream
    (see ``_semantic_hash``), so identical real state collapses to one hash.
    """
    if any(s not in _OK_STATUSES for s in statuses):
        return True
    if len(set(semantic_hashes)) > 1:
        return True
    if errors:
        return True
    return False


class RaceScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    description: str
    agents: int
    action: str
    target_url: str
    overlap_ms: int = 50
    expected_safe: bool
    # Optional per-scenario data (e.g. the SHARED username for a conduit_register
    # collision). Declared+optional → backward compatible with Sprint 10 scenarios.
    payload: dict | None = None


class RaceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    conflict_found: bool
    interleaving: list[str]   # per-agent action timestamps, ISO format
    error_summary: str | None
    duration_ms: float


class RaceConditionSwarm:
    """No required constructor args."""

    async def run(
        self,
        scenario: RaceScenario,
        browser: Browser,
        recorder: InterleavingRecorder | None = None,
    ) -> RaceResult:
        if BLOCKED_ACTION_PATTERNS.search(scenario.action):
            raise ValueError(
                f"Action '{scenario.action}' matches BLOCKED_ACTION_PATTERNS"
            )

        start_time = time.time()
        barrier = asyncio.Barrier(scenario.agents)
        results: list[dict[str, Any] | None] = [None] * scenario.agents
        skip_nav = _skips_navigation(scenario.action)

        async def agent_task(agent_idx: int) -> None:
            error: str | None = None
            ax_hash: str | None = None
            status: int | None = None
            action_ts = datetime.now(timezone.utc).isoformat()

            context: BrowserContext = await browser.new_context()
            page: Page = await context.new_page()

            # Navigation only for UI actions; HTTP/conduit actions use page.request.
            if not skip_nav:
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
                    "ts": action_ts, "ax_hash": None, "status": None,
                    "error": error or "barrier_broken",
                }
                await context.close()
                return

            wait_ms = (t_after - t_before) * 1000
            if wait_ms > scenario.overlap_ms * 2:
                results[agent_idx] = {
                    "ts": action_ts, "ax_hash": None, "status": None,
                    "error": (
                        f"SynchronizationDriftError: waited {wait_ms:.0f}ms "
                        f"> {scenario.overlap_ms * 2}ms"
                    ),
                }
                await context.close()
                return

            action_ts = datetime.now(timezone.utc).isoformat()
            started_ms = time.monotonic() * 1000

            # Perform the semantic action
            if error is None:
                state_override: str | None = None
                try:
                    state_override, status = await _perform_action(
                        page, scenario.action, scenario.target_url, scenario.payload
                    )
                except Exception as exc:
                    error = str(exc)

                if error is None:
                    if state_override is not None:
                        # HTTP action: hash the normalized response signature so
                        # identical responses across agents collapse to one hash.
                        ax_hash = hashlib.sha256(state_override.encode()).hexdigest()
                    else:
                        # Browser action: hash the semantic AX projection.
                        ax_hash = await _ax_snapshot_hash(context, page)

            ended_ms = time.monotonic() * 1000
            results[agent_idx] = {
                "ts": action_ts, "ax_hash": ax_hash, "status": status, "error": error,
            }

            if recorder is not None:
                try:
                    recorder.record(AgentEvent(
                        agent_id=f"agent{agent_idx}",
                        action=scenario.action,
                        started_at=round(started_ms, 3),
                        ended_at=round(ended_ms, 3),
                        status_code=status,
                        response_hash=(ax_hash[:8] if ax_hash else ""),
                    ))
                except Exception:
                    pass

            await context.close()

        await asyncio.gather(*(agent_task(i) for i in range(scenario.agents)))

        duration_ms = (time.time() - start_time) * 1000
        filled = [r for r in results if r is not None]
        errors = [r["error"] for r in filled if r.get("error")]
        semantic_hashes = [r["ax_hash"] for r in filled if r.get("ax_hash")]
        statuses = [r["status"] for r in filled if r.get("status") is not None]

        # Spec conflict rule: non-OK status OR divergent semantic state OR error.
        # Raw CDP nodeId/backendDOMNodeId are excluded (see _semantic_hash) so
        # nodeId drift between isolated BrowserContexts cannot manufacture a false
        # positive — the same Sprint 10 protection, now status-aware.
        conflict_found = _compute_conflict(statuses, semantic_hashes, errors)
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


def _normalize_http_response(status: int, body_text: str) -> str:
    """Reduce an HTTP response to a logical signature for semantic comparison.

    Strips the server-assigned ``id`` (volatile — jsonplaceholder echoes a fresh
    one per request) so that N agents issuing the SAME request produce an
    IDENTICAL signature and therefore do NOT manufacture a false race conflict.
    Genuinely different status codes or bodies still diverge.
    """
    try:
        parsed = json.loads(body_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return f"{status}|{body_text}"
    if isinstance(parsed, dict):
        parsed = {k: v for k, v in parsed.items() if k != "id"}
    return f"{status}|{json.dumps(parsed, sort_keys=True)}"


async def _perform_action(
    page: Page,
    action: str,
    target_url: str = "",
    payload: dict | None = None,
) -> tuple[str | None, int | None]:
    """Dispatch a semantic action string to Playwright operations.

    Returns ``(signature, status)``:
      * UI actions → ``(None, None)`` (caller hashes the AX tree).
      * HTTP / conduit actions → ``(normalized_signature, http_status)`` so
        identical responses across agents collapse to one hash AND the status is
        available to the conflict rule.

    HTTP/conduit actions exercise a real shared REST backend so race testing is
    not confined to localStorage-only targets.
    """
    import uuid as _uuid

    base = target_url.rstrip("/")

    # ── Generic HTTP actions (Sprint 10 — jsonplaceholder) ───────────────────
    if action == "http_get":
        resp = await page.request.get(target_url, timeout=15_000)
        return _normalize_http_response(resp.status, await resp.text()), resp.status

    if action == "http_post":
        resp = await page.request.post(
            target_url,
            data={"title": "race-todo", "completed": False, "userId": 1},
            timeout=15_000,
        )
        return _normalize_http_response(resp.status, await resp.text()), resp.status

    if action == "http_put":
        resp = await page.request.put(
            target_url,
            data={"title": "race-todo", "completed": True, "userId": 1},
            timeout=15_000,
        )
        return _normalize_http_response(resp.status, await resp.text()), resp.status

    # ── Conduit (RealWorld) real-backend actions (Sprint 12) ─────────────────
    if action == "conduit_register":
        # All agents in a register-collision scenario share the SAME username
        # (from payload) → real DB unique-constraint race: one 201/200, rest 422.
        username = (payload or {}).get("username") or f"race_{_uuid.uuid4().hex[:10]}"
        body = {
            "user": {
                "username": username,
                "email": f"{username}@example.com",
                "password": "Passw0rd123",
            }
        }
        resp = await page.request.post(f"{base}/api/users", data=body, timeout=5_000)
        return _normalize_http_response(resp.status, await resp.text()), resp.status

    if action == "conduit_read_tags":
        resp = await page.request.get(f"{base}/api/tags", timeout=5_000)
        return _normalize_http_response(resp.status, await resp.text()), resp.status

    if action == "conduit_read_articles":
        resp = await page.request.get(f"{base}/api/articles?limit=5", timeout=5_000)
        return _normalize_http_response(resp.status, await resp.text()), resp.status

    # ── UI actions (return (None, None); caller hashes the AX tree) ──────────
    if action == "add_todo":
        inp = page.get_by_placeholder("What needs to be done?")
        await inp.fill("concurrent-todo")
        await page.keyboard.press("Enter")
        return None, None

    if action == "toggle_all":
        toggle = page.get_by_label("Mark all as complete")
        if await toggle.is_visible(timeout=5_000):
            await toggle.click()
        return None, None

    if action == "clear_completed":
        # TimeoutError on fresh page (no completed todos) — this IS the expected conflict
        btn = page.get_by_role("button", name="Clear completed")
        await btn.click(timeout=5_000)
        return None, None

    if action == "add_and_complete":
        inp = page.get_by_placeholder("What needs to be done?")
        await inp.fill("race-todo")
        await page.keyboard.press("Enter")
        await page.get_by_role("checkbox").first.check()
        return None, None

    if action == "add_unique_todo":
        text = f"todo-{_uuid.uuid4().hex[:8]}"
        inp = page.get_by_placeholder("What needs to be done?")
        await inp.fill(text)
        await page.keyboard.press("Enter")
        return None, None

    if action == "read_only_view":
        # Pure read — no state mutation. Concurrent agents observe identical state.
        await page.get_by_role("listitem").count()
        return None, None

    raise ValueError(f"Unknown action: {action!r}")


__all__ = [
    "BLOCKED_ACTION_PATTERNS",
    "RaceConditionSwarm",
    "RaceResult",
    "RaceScenario",
    "SynchronizationDriftError",
]
