"""LangGraph pipeline wiring planner → generator → executor → healer.

Includes the Deterministic Safety Boundary, the inline Playwright executor, and
the conditional routing that lets the graph re-plan, heal, or terminate. All
LLM calls go to a local Ollama instance per the project hard constraints.
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from playwright.async_api import (
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from rich.console import Console

from src.agents.generator_agent import generator_node
from src.agents.healer_agent import healer_node
from src.agents.planner_agent import planner_node
from src.core.contract_skill import ContractSkillRepository
from src.core.sfg_engine import SFGEngine
from src.core.state_schema import (
    ActionStep,
    AgentPipelineState,
    ContractSkillArtifact,
)

console = Console()

FORBIDDEN_KEYWORDS = (
    "delete",
    "drop",
    "transfer",
    "password",
    "admin",
    "sudo",
    "truncate",
)

OLLAMA_URL = "http://localhost:11434"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _is_forbidden(action_type: str, payload: str | None, locator_value: str) -> str | None:
    """Return the offending keyword if any forbidden keyword is present."""
    haystack = " ".join(
        [
            (action_type or "").lower(),
            (payload or "").lower(),
            (locator_value or "").lower(),
        ]
    )
    for kw in FORBIDDEN_KEYWORDS:
        if kw in haystack:
            return kw
    return None


async def _execute_step(page: Page, step: ActionStep) -> None:
    """Resolve a single ActionStep against a Playwright Page."""
    if step.action_type == "navigate":
        target = step.payload or step.locator_value
        await page.goto(target, wait_until="domcontentloaded", timeout=30_000)
        return

    if step.locator_strategy == "getByRole":
        locator = page.get_by_role(step.locator_value)  # type: ignore[arg-type]
    elif step.locator_strategy == "getByLabel":
        locator = page.get_by_label(step.locator_value)
    elif step.locator_strategy == "getByTestId":
        locator = page.get_by_test_id(step.locator_value)
    else:
        raise ValueError(f"Unsupported locator strategy: {step.locator_strategy}")

    if step.action_type == "click":
        await locator.click(timeout=15_000)
    elif step.action_type == "fill":
        await locator.fill(step.payload or "", timeout=15_000)
    elif step.action_type == "select":
        await locator.select_option(step.payload or "", timeout=15_000)
    elif step.action_type == "press":
        await locator.press(step.payload or "Enter", timeout=15_000)
    else:
        raise ValueError(f"Unsupported action type: {step.action_type}")


async def _verify_postconditions(
    page: Page, contract: ContractSkillArtifact
) -> tuple[bool, str]:
    """Best-effort verification against URL substrings and visible text."""
    for cond in contract.postconditions:
        if not cond:
            continue
        text_match = page.get_by_text(cond, exact=False).first
        try:
            if await text_match.count() > 0:
                continue
        except Exception:  # noqa: BLE001
            pass
        if cond in page.url:
            continue
        return False, cond
    return True, ""


async def execute_contract_node(state: AgentPipelineState, config: dict) -> dict:
    """LangGraph executor node — drives Playwright through the active contract."""
    cfg = (config or {}).get("configurable", {}) or {}
    sfg_engine: SFGEngine = cfg.get("sfg_engine") or SFGEngine(OLLAMA_URL)
    repo: ContractSkillRepository = cfg.get("contract_repo") or ContractSkillRepository()
    playwright_obj: Playwright | None = cfg.get("playwright")

    contract = state.get("active_contract")
    if contract is None:
        return {
            "execution_log": ["[EXECUTOR] No active contract → re-plan"],
            "mode": "explore",
        }

    start_url = state.get("current_url") or ""
    log: list[str] = []
    new_nodes = []
    failed_idx: int | None = None
    failure_log: str | None = None

    owns_playwright = playwright_obj is None
    pw = playwright_obj or await async_playwright().start()
    browser = None
    try:
        browser = await pw.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        if start_url:
            try:
                await page.goto(start_url, wait_until="domcontentloaded", timeout=30_000)
            except PlaywrightTimeoutError as exc:
                log.append(f"[EXECUTOR] Initial navigation timeout: {exc}")

        for step in contract.action_steps:
            offending = _is_forbidden(step.action_type, step.payload, step.locator_value)
            if offending:
                msg = (
                    f"SAFETY HALT: forbidden action detected: '{offending}' in step "
                    f"#{step.step_index} payload={step.payload!r} locator={step.locator_value!r}"
                )
                log.append(f"[EXECUTOR] {msg}")
                raise RuntimeError(msg)

            log.append(
                f"[EXECUTOR] step {step.step_index}: {step.action_type} via "
                f"{step.locator_strategy}('{step.locator_value}')"
            )
            try:
                await _execute_step(page, step)
            except Exception as exc:  # noqa: BLE001
                failed_idx = step.step_index
                failure_log = f"[EXECUTOR] step {step.step_index} failed: {exc}"
                log.append(failure_log)
                break

        if failed_idx is None:
            ok, missing = await _verify_postconditions(page, contract)
            if not ok:
                log.append(f"[EXECUTOR] postcondition not satisfied: '{missing}'")
                final_state = await sfg_engine.capture_gui_state(page)
                new_nodes.append(final_state)
                return {
                    "sfg_nodes": new_nodes,
                    "current_url": page.url,
                    "execution_log": log,
                    "mode": "explore",
                    "final_result": None,
                }

            final_state = await sfg_engine.capture_gui_state(page)
            new_nodes.append(final_state)
            repo.write_cache(start_url or page.url, contract.action_steps)
            log.append("[EXECUTOR] postconditions satisfied; cache written")
            return {
                "sfg_nodes": new_nodes,
                "current_url": page.url,
                "execution_log": log,
                "mode": "execute",
                "final_result": "SUCCESS",
            }

        snapshot_state = await sfg_engine.capture_gui_state(page)
        new_nodes.append(snapshot_state)
        return {
            "sfg_nodes": new_nodes,
            "current_url": page.url,
            "execution_log": log,
            "failed_step_index": failed_idx,
            "mode": "heal",
        }
    finally:
        if browser is not None:
            await browser.close()
        if owns_playwright:
            await pw.stop()


def route_after_generator(state: AgentPipelineState) -> str:
    """Pick the next node after generator or executor produce updates."""
    mode = state.get("mode")
    if mode == "heal":
        return "heal"
    if mode == "human_review":
        return "end"
    if state.get("final_result") == "SUCCESS":
        return "end"
    if mode == "execute" and state.get("active_contract") is not None:
        return "executor"
    return "planner"


def route_after_healer(state: AgentPipelineState) -> str:
    """Decide whether to retry healing, resume execution, or stop."""
    mode = state.get("mode")
    if mode == "human_review":
        return "human_review"
    if mode == "execute":
        return "executor"
    if (state.get("repair_attempts") or 0) < 3:
        return "healer"
    return "human_review"


def build_graph(
    sfg_engine: SFGEngine | None = None,
    repo: ContractSkillRepository | None = None,
) -> Any:
    """Wire the StateGraph and return a compiled checkpointed pipeline."""
    sfg_engine = sfg_engine or SFGEngine(OLLAMA_URL)
    repo = repo or ContractSkillRepository()

    builder: StateGraph = StateGraph(AgentPipelineState)
    builder.add_node("planner", planner_node)
    builder.add_node("generator", generator_node)
    builder.add_node("healer", healer_node)
    builder.add_node("executor", execute_contract_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "generator")
    builder.add_conditional_edges(
        "generator",
        route_after_generator,
        {
            "heal": "healer",
            "executor": "executor",
            "planner": "planner",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "executor",
        route_after_generator,
        {
            "heal": "healer",
            "executor": "executor",
            "planner": "planner",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "healer",
        route_after_healer,
        {
            "healer": "healer",
            "executor": "executor",
            "human_review": END,
        },
    )

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


async def run_pipeline(goal: str, start_url: str) -> str | None:
    """End-to-end async invocation that streams node outputs to the console."""
    sfg_engine = SFGEngine(OLLAMA_URL)
    repo = ContractSkillRepository()
    graph = build_graph(sfg_engine=sfg_engine, repo=repo)

    initial_state: AgentPipelineState = {
        "task_goal": goal,
        "current_url": start_url,
        "sfg_nodes": [],
        "sfg_edges": [],
        "active_contract": None,
        "execution_log": [],
        "failed_step_index": None,
        "repair_attempts": 0,
        "memory_summary": "",
        "mode": "explore",
        "final_result": None,
    }
    thread_id = f"skill-{uuid.uuid4()}"
    config: dict = {
        "configurable": {
            "thread_id": thread_id,
            "ollama_url": OLLAMA_URL,
            "sfg_engine": sfg_engine,
            "contract_repo": repo,
        }
    }

    final_state: dict[str, Any] = {}
    async for chunk in graph.astream(initial_state, config):
        console.print(f"[NODE OUTPUT] {chunk}")
        for node_update in chunk.values():
            if isinstance(node_update, dict):
                final_state.update(node_update)

    return final_state.get("final_result")


if __name__ == "__main__":
    asyncio.run(
        run_pipeline(
            goal="Navigate to login page and fill in test credentials",
            start_url="http://localhost:3000",
        )
    )
