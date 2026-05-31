"""LangGraph coordinator wiring Driver + 3 Observers per Observer-Driver pattern.

The single ``step`` node implements the parallel execution pattern:

1. Spawn one ``asyncio.create_task`` per Observer **before** the Driver runs.
2. Run the Driver (capture AXTree → plan via Ollama → execute on Playwright).
3. Driver publishes the resulting :class:`TraceEvent` to the bus during step 2.
4. Set ``stop_event`` and ``asyncio.gather`` the Observer tasks.
5. Collect each Observer's reports into the state via the ``operator.add``
   reducer on :attr:`AgentState.observer_reports` — so concurrent Observers
   never trample one another's writes.

The router edge loops the graph through ``step`` until ``max_steps`` is reached
or ``halt`` is set, then runs ``finalize`` to log a structured summary.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger
from playwright.async_api import Page

from src.agents.observer_driver.driver_agent import (
    DRIVER_MODEL,
    OLLAMA_URL,
    capture_axtree,
    execute_action,
    plan_action,
)
from src.agents.observer_driver.observers import (
    AccessibilityObserver,
    PerformanceObserver,
    SecurityObserver,
)
from src.agents.observer_driver.state import AgentState
from src.agents.observer_driver.trace_bus import TraceBus, TraceEvent

OBSERVER_DRAIN_GRACE_SEC = 0.25


def make_thread_id(url: str) -> str:
    """Stable ``thread_id`` derived from *url* for the checkpointer."""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"observer_driver_{digest}"


def _coerce_state(state: AgentState | dict[str, Any]) -> AgentState:
    """Accept either a Pydantic instance or a raw dict (LangGraph quirk)."""
    if isinstance(state, AgentState):
        return state
    return AgentState(**state)


async def step_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Run one Driver action with all Observers consuming concurrently."""
    s = _coerce_state(state)
    cfg = (config or {}).get("configurable", {}) or {}
    page: Page = cfg["page"]
    ollama_url: str = cfg.get("ollama_url", OLLAMA_URL)
    model: str = cfg.get("model", DRIVER_MODEL)

    step_idx = s.current_step
    bus = TraceBus()
    observers = [
        AccessibilityObserver(bus.subscribe()),
        SecurityObserver(bus.subscribe()),
        PerformanceObserver(bus.subscribe()),
    ]
    stop_event = asyncio.Event()
    observer_tasks = [
        asyncio.create_task(obs.run(page, stop_event)) for obs in observers
    ]

    halt = False
    action = None
    payload: dict[str, Any] = {}
    try:
        ax_path = await capture_axtree(page, step_idx)
        action = await plan_action(
            ax_path,
            step_idx,
            s.url,
            ollama_url=ollama_url,
            model=model,
        )
        payload = await execute_action(page, action, bus)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Driver step {step_idx} failed: {exc}")
        halt = True
        await bus.publish(
            TraceEvent(
                event_type="driver_error",
                payload={"step": step_idx, "error": str(exc)},
                timestamp_ms=int(time.time() * 1000),
            )
        )
    finally:
        # Give observers a brief window to consume already-queued events
        # before signalling shutdown. ``wait_for`` inside the observer loop
        # picks them up promptly thanks to the 2-second poll cap.
        await asyncio.sleep(OBSERVER_DRAIN_GRACE_SEC)
        stop_event.set()

    observer_results = await asyncio.gather(*observer_tasks, return_exceptions=True)
    all_reports = []
    for result in observer_results:
        if isinstance(result, Exception):
            logger.warning(f"Observer task raised: {result}")
            continue
        all_reports.extend(result)

    update: dict[str, Any] = {
        "current_step": step_idx + 1,
        "observer_reports": all_reports,
    }
    if action is not None:
        update["driver_actions"] = [action]
        update["trace_events"] = [payload]
    if halt:
        update["halt"] = True
    return update


def route_after_step(state: AgentState | dict[str, Any]) -> str:
    """Conditional edge: loop back into ``step`` or proceed to ``finalize``."""
    s = _coerce_state(state)
    if s.halt:
        return "finalize"
    if s.current_step >= s.max_steps:
        return "finalize"
    return "step"


async def finalize_node(state: AgentState) -> dict[str, Any]:
    """Aggregate observer reports into a single logged summary."""
    s = _coerce_state(state)
    by_observer: dict[str, int] = {}
    severity_counts: dict[str, int] = {"info": 0, "warn": 0, "critical": 0}
    for report in s.observer_reports:
        by_observer[report.observer_name] = by_observer.get(report.observer_name, 0) + 1
        severity_counts[report.severity] = severity_counts.get(report.severity, 0) + 1

    logger.info(
        "Observer-Driver finalize | "
        f"actions={len(s.driver_actions)} "
        f"reports={len(s.observer_reports)} "
        f"by_observer={by_observer} "
        f"severities={severity_counts}"
    )
    return {}


def build_graph() -> Any:
    """Wire the StateGraph and return a compiled, checkpointed pipeline."""
    builder: StateGraph = StateGraph(AgentState)
    builder.add_node("step", step_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "step")
    builder.add_conditional_edges(
        "step",
        route_after_step,
        {"step": "step", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


__all__ = [
    "build_graph",
    "finalize_node",
    "make_thread_id",
    "route_after_step",
    "step_node",
]
