# src/spa/spa_agent.py
"""
SPAAgent — LangGraph sub-graph combining SPA route tracking + Shadow DOM.

Nodes (in order):
  1. attach_trackers  — HydrationGuard + SPARouteTracker attach
  2. navigate         — page.goto(url), HydrationGuard.wait_stable()
  3. extract_ax       — AOMExtractor + ShadowDOMExtractor.extract()
                        + ShadowDOMExtractor.merge_into_ax()
  4. ground           — Grounder.ground() on merged AX
  5. generate_test    — Ollama (Semaphore(1), temp=0.2, format="json")
  6. judge            — 4-check CodeJudge (Sprint 6)
  7. flush_routes     — SPARouteTracker.flush() → record RouteEvents
  8. store_sfg        — SFGStore.upsert_node/edge with route transitions

Edges:
  START → attach_trackers → navigate → extract_ax → ground
        → generate_test → judge → flush_routes → store_sfg → END
  judge routes back to generate_test if needs_revision (max 2 retries)

Dependencies injected via config["configurable"]:
  page         — Playwright Page
  sfg_store    — SFGStore instance
  instructor   — InstructorClient instance (optional; created on demand)
"""
from __future__ import annotations

import asyncio
import datetime
import hashlib
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.codetest.generator import GeneratedTest
from src.codetest.judge import CodeJudge, JudgeResult
from src.contractskill.sfg import SFGEdge, SFGNode, SFGStore
from src.llm.instructor_client import InstructorClient, StructuredGenerationError
from src.perception.aom_extractor import AOMExtractor
from src.perception.grounder import Grounder
from src.shadow.extractor import ShadowDOMExtractor
from src.spa.hydration_guard import HydrationGuard
from src.spa.route_tracker import RouteEvent, SPARouteTracker

# Semaphore(1) serialises Ollama calls from this agent
_SPA_GENERATE_SEM = asyncio.Semaphore(1)


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────


class SPAAgentState(TypedDict):
    url: str
    ax_nodes: list[dict]
    shadow_nodes: list[dict]           # ShadowNode.model_dump() list
    merged_ax: list[dict]
    compact_pam_text: str
    route_events: list[dict]           # RouteEvent.model_dump() list
    generated_test_code: str
    judge_grade: str                   # "acceptable" | "needs_revision" | "reject"
    judge_feedback: str
    revision_count: int


# ─────────────────────────────────────────────────────────────────────────────
# Node helpers
# ─────────────────────────────────────────────────────────────────────────────


def _page(config: dict) -> Any:
    return config["configurable"]["page"]


def _sfg_store(config: dict) -> SFGStore:
    return config["configurable"]["sfg_store"]


def _instructor(config: dict) -> InstructorClient:
    client = config["configurable"].get("instructor")
    if client is None:
        client = InstructorClient()
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────


async def attach_trackers(state: SPAAgentState, config: dict) -> dict:
    page = _page(config)
    guard = HydrationGuard()
    tracker = SPARouteTracker()
    config["configurable"]["_guard"] = guard
    config["configurable"]["_tracker"] = tracker
    await tracker.attach(page)
    logger.debug("SPAAgent: trackers attached")
    return {}


async def navigate(state: SPAAgentState, config: dict) -> dict:
    page = _page(config)
    guard: HydrationGuard = config["configurable"]["_guard"]
    await page.goto(state["url"])
    await guard.wait_stable(page)
    logger.debug(f"SPAAgent: navigated to {state['url']}")
    return {}


async def extract_ax(state: SPAAgentState, config: dict) -> dict:
    page = _page(config)
    aom = AOMExtractor()
    shadow_ext = ShadowDOMExtractor()

    try:
        snapshot = await aom.extract(page)
        raw_ax: list[dict] = [snapshot.root_node.model_dump()] if snapshot.root_node else []
    except Exception as exc:
        logger.warning(f"SPAAgent.extract_ax: AOMExtractor failed ({exc!r}); using []")
        raw_ax = []

    shadow_nodes = await shadow_ext.extract(page)
    merged = shadow_ext.merge_into_ax(raw_ax, shadow_nodes)

    return {
        "ax_nodes": raw_ax,
        "shadow_nodes": [sn.model_dump() for sn in shadow_nodes],
        "merged_ax": merged,
    }


async def ground(state: SPAAgentState, config: dict) -> dict:
    page = _page(config)
    grounder = Grounder()
    compact_pam = await grounder.ground(page)
    return {"compact_pam_text": str(compact_pam)}


async def generate_test(state: SPAAgentState, config: dict) -> dict:
    instructor = _instructor(config)
    pam = state.get("compact_pam_text", "")
    url = state.get("url", "")
    feedback = state.get("judge_feedback", "")

    revision_note = f"\nPrevious feedback: {feedback}" if feedback else ""
    prompt = (
        f"Generate a pytest test for this web page.\n"
        f"URL: {url}\n"
        f"Page state (AX tree):\n{pam[:1500]}\n"
        f"{revision_note}\n"
        "Rules: function name must start with test_; at least one assert; "
        "no time.sleep(); no asyncio.sleep(); use absolute imports only."
    )

    func_id = hashlib.md5(url.encode()).hexdigest()[:8]

    async with _SPA_GENERATE_SEM:
        try:
            result: GeneratedTest = await instructor.create_structured(
                prompt=prompt,
                response_model=GeneratedTest,
                temperature=0.2,
            )
        except StructuredGenerationError as exc:
            logger.warning(f"SPAAgent.generate_test: generation failed ({exc!r}); using stub")
            result = GeneratedTest(
                test_id=f"spa_{func_id}",
                func_id=func_id,
                test_code=(
                    "def test_spa_stub():\n"
                    f"    # auto-stub: generation failed for {url}\n"
                    "    assert True\n"
                ),
                test_type="happy_path",
                metamorphic_relation=None,
            )

    return {"generated_test_code": result.test_code}


async def judge(state: SPAAgentState, config: dict) -> dict:
    code = state.get("generated_test_code", "")
    url = state.get("url", "")
    func_id = hashlib.md5(url.encode()).hexdigest()[:8]

    generated = GeneratedTest(
        test_id=f"spa_{func_id}_rev{state.get('revision_count', 0)}",
        func_id=func_id,
        test_code=code,
        test_type="happy_path",
        metamorphic_relation=None,
    )
    judge_obj = CodeJudge()
    try:
        result: JudgeResult = await judge_obj.judge(generated)
    finally:
        await judge_obj.close()

    return {
        "judge_grade": result.grade,
        "judge_feedback": result.feedback,
        "revision_count": state.get("revision_count", 0) + 1,
    }


async def flush_routes(state: SPAAgentState, config: dict) -> dict:
    page = _page(config)
    tracker: SPARouteTracker = config["configurable"].get("_tracker", SPARouteTracker())
    events = await tracker.flush(page)
    logger.debug(f"SPAAgent: flushed {len(events)} route events")
    return {"route_events": [e.model_dump() for e in events]}


async def store_sfg(state: SPAAgentState, config: dict) -> dict:
    page = _page(config)
    store = _sfg_store(config)

    url = state.get("url", "")
    node_id = hashlib.md5(url.encode()).hexdigest()

    try:
        page_title = await page.title()
    except Exception:
        page_title = ""

    pam_text = state.get("compact_pam_text", "")
    aom_hash = hashlib.md5(str(state.get("ax_nodes", [])).encode()).hexdigest()

    node = SFGNode(
        node_id=node_id,
        url=url,
        page_title=page_title,
        aom_hash=aom_hash,
        pam_content=pam_text[:2000],
        coverage_tags=[],
        outgoing_edges=[],
        discovered_at_iso=datetime.datetime.utcnow().isoformat(),
        visit_count=1,
    )
    store.upsert_node(node)

    for raw_evt in state.get("route_events", []):
        evt = RouteEvent(**raw_evt)
        target_id = hashlib.md5(evt.to_url.encode()).hexdigest()
        edge_id = hashlib.md5(f"{node_id}->{target_id}".encode()).hexdigest()
        edge = SFGEdge(
            edge_id=edge_id,
            source_node_id=node_id,
            target_node_id=target_id,
            action_type=evt.trigger,
            locator="",
            input_value="",
            safety_flag=True,
            replay_script="",
        )
        store.upsert_edge(edge)

    logger.debug(f"SPAAgent: stored SFG node {node_id[:8]}... + {len(state.get('route_events', []))} edges")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Conditional routing
# ─────────────────────────────────────────────────────────────────────────────


def _should_revise(state: SPAAgentState) -> str:
    if state.get("judge_grade") == "needs_revision" and state.get("revision_count", 0) < 2:
        return "generate_test"
    return "flush_routes"


# ─────────────────────────────────────────────────────────────────────────────
# Graph factory
# ─────────────────────────────────────────────────────────────────────────────


class SPAAgent:
    """LangGraph sub-graph: SPA route tracking + Shadow DOM extraction."""

    def compile(self):
        """Compile and return the LangGraph app."""
        graph = StateGraph(SPAAgentState)

        graph.add_node("attach_trackers", attach_trackers)
        graph.add_node("navigate", navigate)
        graph.add_node("extract_ax", extract_ax)
        graph.add_node("ground", ground)
        graph.add_node("generate_test", generate_test)
        graph.add_node("judge", judge)
        graph.add_node("flush_routes", flush_routes)
        graph.add_node("store_sfg", store_sfg)

        graph.add_edge(START, "attach_trackers")
        graph.add_edge("attach_trackers", "navigate")
        graph.add_edge("navigate", "extract_ax")
        graph.add_edge("extract_ax", "ground")
        graph.add_edge("ground", "generate_test")
        graph.add_edge("generate_test", "judge")
        graph.add_conditional_edges("judge", _should_revise)
        graph.add_edge("flush_routes", "store_sfg")
        graph.add_edge("store_sfg", END)

        return graph.compile()


__all__ = ["SPAAgent", "SPAAgentState"]
