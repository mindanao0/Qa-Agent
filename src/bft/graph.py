"""
LangGraph StateGraph for the Hybrid-BFT pipeline.

Topology::

    START
      │
      ▼
    router ── Send("generator_node", payload) × N ──▶ generator_node
                                                          │
                                                          ▼
                                                       voter_node
                                                          │
                                          ┌───────────────┴───────────────┐
                                          ▼                               ▼
                                    executor_node                    abort_node
                                          │                               │
                                          └────────────► END ◄────────────┘

The router emits ``N`` :class:`Send` commands dispatching one payload to
each generator slot — *never* invoking subgraphs imperatively, since that
would trip LangGraph's ``MULTIPLE_SUBGRAPHS`` namespace crash (treated as
Byzantine fault by KS-3).
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from loguru import logger

from src.config import F, N, NODE_CONFIGS

from .generator import run_generator_node
from .models import (
    BFTConsensusResult,
    BFTGeneratorInput,
    BFTGraphState,
    BFTQuorumFailure,
)
from .voter import certify_quorum


# ─────────────────────────────────────────────────────────────────────────────
# Node implementations
# ─────────────────────────────────────────────────────────────────────────────


def _router_dispatch(state: BFTGraphState) -> list[Send]:
    """Conditional-edge function: fan out N Sends to generator_node.

    This MUST be used with :func:`StateGraph.add_conditional_edges` so the
    returned list of :class:`Send` objects becomes the routing decision. Any
    imperative subgraph invocation would trigger MULTIPLE_SUBGRAPHS (KS-3).
    """
    task_prompt = state.get("task_prompt", "") or ""
    if not task_prompt:
        raise ValueError("router: task_prompt is empty — nothing to generate")

    if len(NODE_CONFIGS) != N:
        raise RuntimeError(
            f"router: NODE_CONFIGS length ({len(NODE_CONFIGS)}) does not match N={N}"
        )

    sends: list[Send] = []
    for cfg in NODE_CONFIGS:
        payload = BFTGeneratorInput(
            task_prompt=task_prompt,
            node_id=int(cfg["node_id"]),
            model=str(cfg["model"]),
            temperature=float(cfg["temperature"]),
            top_p=float(cfg["top_p"]),
        )
        sends.append(Send("generator_node", payload))
    logger.info("BFT router: dispatching {n} generator Sends", n=len(sends))
    return sends


async def router_node(state: BFTGraphState) -> dict[str, Any]:
    """Pass-through node — actual fan-out happens on the conditional edge.

    Exists so we have a stable named entry point that the router conditional
    edge can attach to.
    """
    return {}


async def voter_node(state: BFTGraphState) -> dict[str, Any]:
    """Run BFT quorum certification over collected generator outputs.

    On quorum failure (KS-1) the exception is caught and translated into an
    ``abort_reason`` on the state so the conditional edge can route to the
    abort terminal. This keeps the kill switch user-visible without crashing
    the graph runtime.
    """
    outputs = list(state.get("generator_outputs") or [])
    logger.info("BFT voter: received {n} generator outputs", n=len(outputs))

    try:
        consensus: BFTConsensusResult = await certify_quorum(outputs, f=F)
        return {"consensus_result": consensus, "abort_reason": None}
    except BFTQuorumFailure as exc:
        reason = str(exc)
        logger.error("BFT voter: KS-1 abort — {r}", r=reason)
        failed_consensus = BFTConsensusResult(
            certified_code="",
            winning_fingerprint="",
            quorum_count=0,
            total_nodes=len(outputs),
            divergence_map={},
            quorum_achieved=False,
        )
        return {"consensus_result": failed_consensus, "abort_reason": reason}


async def executor_node(state: BFTGraphState) -> dict[str, Any]:
    """Safety gate — print the certified code but DO NOT execute it.

    Hitting this node means consensus was reached. The spec requires this
    node to be reachable only when ``consensus_result.quorum_achieved`` is
    True; we double-check here defensively.
    """
    consensus = state.get("consensus_result")
    if not consensus or not consensus.quorum_achieved:
        # Defensive — the conditional edge should have routed to abort_node.
        return {
            "abort_reason": (
                state.get("abort_reason")
                or "executor_node reached without quorum_achieved"
            )
        }

    logger.info(
        "BFT executor: quorum achieved ({q}/{n}) — emitting certified code",
        q=consensus.quorum_count,
        n=consensus.total_nodes,
    )
    print("─" * 70)
    print(f"# Certified by BFT consensus ({consensus.quorum_count}/{consensus.total_nodes})")
    print(f"# Winning fingerprint: {consensus.winning_fingerprint[:120]}…")
    print("─" * 70)
    print(consensus.certified_code)
    print("─" * 70)
    return {}


async def abort_node(state: BFTGraphState) -> dict[str, Any]:
    """Terminal node for the failure path — surface the abort reason."""
    reason = state.get("abort_reason") or "unknown"
    logger.error("BFT abort: {r}", r=reason)
    print("─" * 70)
    print(f"BFT ABORT — {reason}")
    print("─" * 70)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edges
# ─────────────────────────────────────────────────────────────────────────────


def _post_voter_route(state: BFTGraphState) -> str:
    if state.get("abort_reason"):
        return "abort_node"
    consensus = state.get("consensus_result")
    if not consensus or not consensus.quorum_achieved:
        return "abort_node"
    return "executor_node"


# ─────────────────────────────────────────────────────────────────────────────
# Public builder
# ─────────────────────────────────────────────────────────────────────────────


async def build_bft_graph() -> Any:
    """Construct and compile the BFT LangGraph.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph application. Invoke with
        ``await graph.ainvoke({"task_prompt": "..."})``.
    """

    builder: StateGraph = StateGraph(BFTGraphState)

    builder.add_node("router", router_node)
    builder.add_node("generator_node", run_generator_node)
    builder.add_node("voter_node", voter_node)
    builder.add_node("executor_node", executor_node)
    builder.add_node("abort_node", abort_node)

    builder.add_edge(START, "router")

    # Dynamic fan-out: the router's conditional edge returns N Send objects.
    builder.add_conditional_edges(
        "router",
        _router_dispatch,
        ["generator_node"],
    )

    # All generator Sends converge on the voter.
    builder.add_edge("generator_node", "voter_node")

    # Voter dispatches to executor or abort based on consensus.
    builder.add_conditional_edges(
        "voter_node",
        _post_voter_route,
        {"executor_node": "executor_node", "abort_node": "abort_node"},
    )

    builder.add_edge("executor_node", END)
    builder.add_edge("abort_node", END)

    return builder.compile()


__all__: list[str] = ["build_bft_graph"]
