"""Planner agent — emits a 3-5 step natural-language Strategy.

The planner is also responsible for noticing UI tarpits via the SFGEngine and
escalating the prompt accordingly so the downstream Generator can pick a
non-repeating action plan.
"""
from __future__ import annotations

import networkx as nx
from rich.console import Console

from src.core.sfg_engine import SFGEngine
from src.core.state_schema import AgentPipelineState
from src.llm.adapter import OllamaAdapter

console = Console()

PLANNER_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434"


async def planner_node(state: AgentPipelineState, config: dict) -> dict:
    """LangGraph node: build a Strategy prompt and call Ollama for a plan."""
    ollama_url = (
        (config or {}).get("configurable", {}).get("ollama_url") or OLLAMA_URL
    ).rstrip("/")

    sfg_engine: SFGEngine = (
        (config or {}).get("configurable", {}).get("sfg_engine") or SFGEngine(ollama_url)
    )

    sfg_nodes = state.get("sfg_nodes", []) or []
    recent_hashes = [n.state_hash for n in sfg_nodes[-3:]]
    edges = state.get("sfg_edges", []) or []
    graph: nx.DiGraph = sfg_engine.build_networkx_graph(sfg_nodes, edges)
    stagnant = sfg_engine.detect_stagnation(graph, recent_hashes)

    snapshot_blurb = ""
    if sfg_nodes:
        snapshot_blurb = sfg_nodes[-1].accessibility_snapshot[:2000]

    task_goal = state.get("task_goal", "")
    current_url = state.get("current_url", "")
    memory_summary = state.get("memory_summary", "")

    if stagnant:
        temperature = 0.9
        prompt = (
            "ESCAPE CURRENT UI TARPIT. Use browser_back or navigate to alternative "
            "path. Do NOT repeat previous actions.\n"
            f"Goal: {task_goal}\n"
            f"Current URL: {current_url}\n"
            f"Memory: {memory_summary}\n"
            f"Snapshot (trimmed): {snapshot_blurb}\n"
            "Generate a 3-5 step Strategy in plain English to break out of the "
            "loop and still make progress toward the goal."
        )
    else:
        temperature = 0.3
        prompt = (
            f"Goal: {task_goal}\n"
            f"Current URL: {current_url}\n"
            f"Memory: {memory_summary}\n"
            f"Snapshot (trimmed): {snapshot_blurb}\n"
            "Generate a 3-5 step Strategy in plain English to achieve the goal."
        )

    # Route through OllamaAdapter (holds _inference_semaphore + VRAM guard) instead
    # of a direct httpx call to :11434.
    adapter = OllamaAdapter(model=PLANNER_MODEL, base_url=ollama_url)
    try:
        strategy = (
            await adapter.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
        ).strip()
    except Exception as exc:  # noqa: BLE001
        console.log(f"[PLANNER] Ollama call failed: {exc}")
        strategy = "FALLBACK: navigate to the start URL and inspect the page."
    finally:
        await adapter.close()

    label = "ESCAPE" if stagnant else "PLAN"
    log_line = f"[PLANNER] {label} Strategy: {strategy[:100]}..."
    return {
        "execution_log": [log_line],
        "mode": "execute",
        "memory_summary": memory_summary,
    }


__all__ = ["planner_node"]
