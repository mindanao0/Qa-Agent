"""Driver agent: capture AXTree → plan action via Ollama → execute via Playwright.

The Driver owns ALL state-mutating browser interactions. After every action
it broadcasts a :class:`~src.agents.observer_driver.trace_bus.TraceEvent` so
the Observer fleet can audit the side effects without sharing the Page handle.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from loguru import logger
from playwright.async_api import Page

from src.agents.observer_driver.state import DriverAction, DriverFeedback
from src.agents.observer_driver.trace_bus import TraceBus, TraceEvent
from src.llm.adapter import OllamaAdapter

DRIVER_MODEL = os.getenv("OBSERVER_DRIVER_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
AGENT_STATE_DIR = Path("agent_state")
AX_SNAPSHOT_MAX_BYTES = 8_000
PLAN_TIMEOUT_SEC = 60
ACTION_TIMEOUT_MS = 15_000

_ALLOWED_ACTIONS = {"click", "fill", "navigate", "assert", "noop"}


def _now_ms() -> int:
    return int(time.time() * 1000)


async def capture_axtree(page: Page, step: int) -> Path:
    """Persist the page's accessibility tree as YAML and return its file path.

    Uses Playwright's ``page.aria_snapshot()`` (already proven in
    :mod:`src.browser.ax_extractor`); the snapshot is the Observer-Driver's
    canonical view of the DOM.
    """
    AGENT_STATE_DIR.mkdir(exist_ok=True)
    snapshot_path = AGENT_STATE_DIR / f"ax_snapshot_{step}.yml"

    try:
        snapshot = await page.aria_snapshot()
    except Exception as exc:  # noqa: BLE001 — defensive on flaky pages
        logger.warning(f"capture_axtree: aria_snapshot failed: {exc}")
        snapshot = ""

    snapshot_path.write_text(snapshot, encoding="utf-8")
    logger.debug(
        f"capture_axtree | step={step} | path={snapshot_path} | bytes={len(snapshot)}"
    )
    return snapshot_path


def _format_hint_block(hint: DriverFeedback | None) -> str:
    """Render the advisory Observer-signal block, or ``""`` when there is no hint.

    The block is ADVISORY ONLY and is inserted *before* the JSON action contract;
    its trailing blank line keeps everything from ``"Reply with a SINGLE JSON
    object"`` onward byte-identical to the no-hint prompt, so a hint can never
    perturb the action schema the Driver must produce.
    """
    if hint is None:
        return ""
    lines = [
        f"PRIOR AUDIT SIGNALS (advisory — from step {hint.from_step}, "
        f"top severity {hint.top_severity}):"
    ]
    for i, directive in enumerate(hint.directives, start=1):
        lines.append(f"  {i}. {directive}")
    lines.append(
        "These are advisory hints from parallel auditors. Never fabricate a target "
        "that is not present in the accessibility tree; if a hint references "
        "something not in the tree, ignore it."
    )
    return "\n".join(lines) + "\n\n"


def _build_prompt(
    ax_yaml: str,
    step: int,
    url: str,
    hint: DriverFeedback | None = None,
) -> str:
    """Compose the structured prompt for the Driver LLM.

    When *hint* is provided, a clearly-delimited advisory block of prior Observer
    audit signals is inserted between the accessibility tree and the JSON action
    contract. Defaults to ``None`` → backward-compatible with existing callers.
    """
    trimmed = ax_yaml[:AX_SNAPSHOT_MAX_BYTES]
    advisory = _format_hint_block(hint)
    return (
        "You are the DRIVER of a browser automation agent. Pick exactly ONE "
        "next action to make progress on the page.\n"
        f"Current URL: {url}\n"
        f"Step index (0-based): {step}\n\n"
        "Accessibility tree (YAML, truncated):\n"
        "---\n"
        f"{trimmed}\n"
        "---\n\n"
        f"{advisory}"
        "Reply with a SINGLE JSON object — no prose, no markdown — using "
        "exactly these keys:\n"
        '  "action_type": one of "click", "fill", "navigate", "assert", "noop"\n'
        '  "target_role": ARIA role of the target (e.g. "link", "button", '
        '"textbox"); use "" for navigate / noop\n'
        '  "target_name": the accessible name shown in the tree above; '
        'use "" for navigate / noop\n'
        '  "value": URL for navigate, text for fill, expected text for assert, '
        'else ""\n\n'
        "Prefer interacting with visible interactive elements (link, button, "
        "textbox). If nothing useful is interactable, reply with "
        '{"action_type":"noop","target_role":"","target_name":"","value":""}.'
    )


def _parse_action_payload(raw: str) -> dict[str, Any]:
    """Coerce an LLM response into a DriverAction-shaped dict.

    Tolerates fenced code blocks around the JSON object. Raises ``ValueError``
    if no JSON object can be located or required keys are missing.
    """
    candidate = raw.strip()
    if candidate.startswith("```"):
        # strip ```json fences
        lines = [
            ln for ln in candidate.splitlines() if not ln.strip().startswith("```")
        ]
        candidate = "\n".join(lines).strip()

    if not candidate.startswith("{"):
        first = candidate.find("{")
        last = candidate.rfind("}")
        if first == -1 or last == -1 or last <= first:
            raise ValueError(f"Driver LLM produced no JSON object: {raw!r}")
        candidate = candidate[first : last + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Driver LLM JSON parse failed: {exc}: {raw!r}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Driver LLM JSON was not an object: {parsed!r}")

    action_type = str(parsed.get("action_type", "")).strip().lower()
    if action_type not in _ALLOWED_ACTIONS:
        raise ValueError(
            f"Driver LLM returned disallowed action_type={action_type!r}; "
            f"allowed={_ALLOWED_ACTIONS}"
        )

    return {
        "action_type": action_type,
        "target_role": str(parsed.get("target_role", "")),
        "target_name": str(parsed.get("target_name", "")),
        "value": str(parsed.get("value", "")),
    }


async def plan_action(
    ax_path: Path,
    step: int,
    url: str,
    ollama_url: str = OLLAMA_URL,
    model: str = DRIVER_MODEL,
    hint: DriverFeedback | None = None,
) -> DriverAction:
    """Ask Ollama for the next :class:`DriverAction` given the saved AXTree.

    *hint* (default ``None``) carries the distilled Observer feedback from the
    previous step; when present it is rendered into the prompt as an advisory block.
    """
    ax_yaml = ax_path.read_text(encoding="utf-8") if ax_path.exists() else ""
    prompt = _build_prompt(ax_yaml, step, url, hint)

    # Route through OllamaAdapter (holds _inference_semaphore + VRAM guard) instead
    # of a direct aiohttp call to :11434. JSON mode is preserved via response_format.
    adapter = OllamaAdapter(model=model, base_url=ollama_url)
    try:
        raw_response = (
            await adapter.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        ).strip()
    except Exception as exc:
        raise ValueError(f"Driver plan_action: Ollama call failed: {exc}") from exc
    finally:
        await adapter.close()

    if not raw_response:
        raise ValueError("Driver plan_action: empty response from Ollama")

    fields = _parse_action_payload(raw_response)
    action = DriverAction(
        action_type=fields["action_type"],  # type: ignore[arg-type]
        target_role=fields["target_role"],
        target_name=fields["target_name"],
        value=fields["value"],
        ax_snapshot_path=ax_path,
    )
    logger.info(
        f"Driver plan_action | step={step} | "
        f"{action.action_type} role={action.target_role!r} "
        f"name={action.target_name!r}"
    )
    return action


async def execute_action(
    page: Page, action: DriverAction, bus: TraceBus
) -> dict[str, Any]:
    """Run *action* against *page* and broadcast a trace event.

    The trace event is published regardless of whether the underlying
    Playwright call succeeded — Observers should see the intent and the
    outcome.
    """
    outcome = "ok"
    error: str | None = None

    try:
        if action.action_type == "navigate":
            target = action.value or page.url
            await page.goto(target, wait_until="domcontentloaded", timeout=30_000)

        elif action.action_type == "click":
            locator = page.get_by_role(
                action.target_role,  # type: ignore[arg-type]
                name=action.target_name or None,
            ).first
            await locator.click(timeout=ACTION_TIMEOUT_MS)

        elif action.action_type == "fill":
            locator = page.get_by_role(
                action.target_role,  # type: ignore[arg-type]
                name=action.target_name or None,
            ).first
            await locator.fill(action.value, timeout=ACTION_TIMEOUT_MS)

        elif action.action_type == "assert":
            target_text = action.value or action.target_name
            if target_text:
                count = await page.get_by_text(target_text, exact=False).count()
                if count == 0:
                    outcome = "assertion_missing"

        elif action.action_type == "noop":
            outcome = "noop"

        else:
            outcome = "unknown_action"

    except Exception as exc:  # noqa: BLE001 — record then continue
        outcome = "error"
        error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            f"Driver execute_action | step error: {error} | action={action.model_dump()}"
        )

    payload = action.model_dump(mode="json")
    payload["outcome"] = outcome
    if error:
        payload["error"] = error
    payload["url"] = page.url

    event = TraceEvent(
        event_type="driver_action",
        payload=payload,
        timestamp_ms=_now_ms(),
    )
    await bus.publish(event)
    return payload


__all__ = [
    "ACTION_TIMEOUT_MS",
    "AGENT_STATE_DIR",
    "DRIVER_MODEL",
    "OLLAMA_URL",
    "PLAN_TIMEOUT_SEC",
    "capture_axtree",
    "execute_action",
    "plan_action",
]
