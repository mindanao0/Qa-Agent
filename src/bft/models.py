"""
Pydantic V2 schemas and LangGraph state model for the Hybrid-BFT
Deterministic AST Consensus Orchestrator.

The graph state uses ``Annotated[list, operator.add]`` as the reducer on
``generator_outputs`` so that parallel ``Send`` fan-outs append rather than
overwrite each other — this avoids LangGraph's ``InvalidUpdateError`` on
concurrent writes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions (kill switches)
# ─────────────────────────────────────────────────────────────────────────────


class BFTQuorumFailure(RuntimeError):
    """KS-1: Voter could not secure Q >= 2f+1 matching AST fingerprints."""


class BFTSecurityViolation(RuntimeError):
    """KS-2: ASTNormalizer detected an attempt to remap a PROTECTED_KEYWORD."""


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models — generator I/O and consensus result
# ─────────────────────────────────────────────────────────────────────────────


class BFTGeneratorInput(BaseModel):
    """Payload dispatched to a single generator node via ``Send``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_prompt: str = Field(..., description="Task description for code generation.")
    node_id: int = Field(..., ge=0, description="Index of the generator node (0..N-1).")
    model: str = Field(..., description="Ollama model tag.")
    temperature: float = Field(..., ge=0.0, le=2.0)
    top_p: float = Field(..., gt=0.0, le=1.0)


class BFTGeneratorOutput(BaseModel):
    """Structured output from a single generator node.

    On any unrecoverable failure (LLM error, retry exhaustion, normalization
    error) the node MUST still emit one of these with ``status="FAILED"`` so
    that LangGraph's fail-fast cancellation does not blank the run.
    """

    model_config = ConfigDict(extra="forbid")

    node_id: int = Field(..., ge=0)
    raw_code: str = Field(default="")
    ast_fingerprint: str | None = Field(default=None)
    status: Literal["OK", "FAILED"] = "OK"
    error_msg: str | None = Field(default=None)


class BFTConsensusResult(BaseModel):
    """Final certified output of the voter / quorum certifier."""

    model_config = ConfigDict(extra="forbid")

    certified_code: str = ""
    winning_fingerprint: str = ""
    quorum_count: int = 0
    total_nodes: int = 0
    divergence_map: dict[str, int] = Field(default_factory=dict)
    quorum_achieved: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph state — TypedDict with concurrent-safe reducer
# ─────────────────────────────────────────────────────────────────────────────


class BFTGraphState(TypedDict, total=False):
    """LangGraph state shared across the BFT pipeline.

    ``generator_outputs`` uses ``operator.add`` so each parallel ``Send`` from
    the router appends to the list instead of racing on a single write.
    """

    task_prompt: str
    generator_outputs: Annotated[list[BFTGeneratorOutput], operator.add]
    consensus_result: BFTConsensusResult | None
    abort_reason: str | None
