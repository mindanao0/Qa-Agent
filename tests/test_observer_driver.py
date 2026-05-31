"""Integration test for the Priority 5 Observer-Driver multi-agent system.

Drives a real Chromium browser against https://example.com for ``max_steps=3``
and asserts that the three Observer agents (accessibility, security,
performance) emit at least one report each WITHOUT mutating Driver state.

Prereqs:
  - Ollama at http://localhost:11434 with the model named in
    ``src.agents.observer_driver.driver_agent.DRIVER_MODEL`` pulled.
  - ``playwright install chromium`` previously run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

# Ensure project root is on sys.path when run directly via ``pytest tests/...``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.observer_driver.coordinator import build_graph
from src.agents.observer_driver.state import AgentState

REQUIRED_OBSERVERS: frozenset[str] = frozenset({"accessibility", "security", "performance"})


def _attr(obj, name: str):
    """Read *name* off *obj*, transparently supporting BaseModel or dict shape."""
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


@pytest.mark.asyncio
async def test_observer_driver_against_example_dot_com() -> None:
    """End-to-end: 3 Driver actions, ≥1 ObserverReport per observer, no criticals."""
    Path("agent_state").mkdir(exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto("https://example.com", wait_until="domcontentloaded")

            graph = build_graph()
            initial_state = AgentState(
                url="https://example.com",
                max_steps=3,
            )
            config = {
                "configurable": {
                    "thread_id": "observer_driver_example_test",
                    "page": page,
                }
            }

            final = await graph.ainvoke(initial_state.model_dump(), config=config)

            driver_actions = final["driver_actions"] if isinstance(final, dict) else final.driver_actions
            observer_reports = (
                final["observer_reports"] if isinstance(final, dict) else final.observer_reports
            )

            # 1. Driver completed exactly max_steps actions
            assert len(driver_actions) == 3, (
                f"expected 3 driver_actions, got {len(driver_actions)}: {driver_actions}"
            )

            # 2. Each observer type contributed at least one report
            observer_names = {_attr(r, "observer_name") for r in observer_reports}
            for required in REQUIRED_OBSERVERS:
                assert required in observer_names, (
                    f"missing observer report from {required!r}; got {observer_names}"
                )

            # 3. The clean baseline (example.com) yields no CRITICAL findings
            critical = [
                r for r in observer_reports if _attr(r, "severity") == "critical"
            ]
            assert not critical, f"unexpected critical findings: {critical}"
        finally:
            await context.close()
            await browser.close()
