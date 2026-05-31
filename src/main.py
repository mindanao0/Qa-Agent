"""
CLI entrypoint for the Hybrid-BFT Deterministic AST Consensus Orchestrator.

Usage::

    python -m src.main "Write a Playwright test that opens https://example.com and asserts the title"

The orchestrator dispatches ``N`` heterogeneous LLM generators in parallel,
normalizes every output to a structural AST fingerprint, and certifies
consensus only when at least ``Q = 2f + 1`` fingerprints agree. The CLI
prints a JSON summary of the final :class:`BFTConsensusResult` to stdout.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from loguru import logger

from src.bft.graph import build_bft_graph
from src.bft.models import BFTConsensusResult
from src.config import F, N, Q


def _summary(state: dict[str, Any]) -> dict[str, Any]:
    """Serialize the final state into a JSON-ready summary."""
    consensus: BFTConsensusResult | None = state.get("consensus_result")
    outputs = list(state.get("generator_outputs") or [])
    consensus_dict: dict[str, Any]
    if consensus is None:
        consensus_dict = {"quorum_achieved": False}
    else:
        consensus_dict = consensus.model_dump()

    return {
        "config": {"N": N, "F": F, "Q": Q},
        "abort_reason": state.get("abort_reason"),
        "consensus_result": consensus_dict,
        "per_node": [
            {
                "node_id": o.node_id,
                "status": o.status,
                "fingerprint_present": o.ast_fingerprint is not None,
                "error_msg": o.error_msg,
            }
            for o in outputs
        ],
    }


async def main(task_prompt: str) -> int:
    """Build the graph, invoke it with ``task_prompt``, print summary JSON.

    Returns the process exit code: 0 on quorum achieved, 2 on abort.
    """
    if not task_prompt or not task_prompt.strip():
        logger.error("main: task_prompt is empty")
        return 64

    graph = await build_bft_graph()
    final_state: dict[str, Any] = await graph.ainvoke(
        {"task_prompt": task_prompt}
    )

    summary = _summary(final_state)
    print(json.dumps(summary, indent=2, default=str))

    consensus = final_state.get("consensus_result")
    if consensus is not None and consensus.quorum_achieved:
        return 0
    return 2


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "usage: python -m src.main \"<task prompt>\"",
            file=sys.stderr,
        )
        raise SystemExit(64)
    exit_code = asyncio.run(main(sys.argv[1]))
    raise SystemExit(exit_code)
