"""Pydantic V2 state definitions and TypedDict pipeline state for the SFG / ContractSkill stack.

All models are declared with strict=True / extra="forbid" so any drift in upstream
LLM output or pipeline plumbing fails loudly at validation time instead of being
silently coerced.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class GUIState(BaseModel):
    """Snapshot of an interactive GUI moment, keyed by a stable hash."""

    model_config = ConfigDict(strict=True, extra="forbid")

    state_hash: str
    url: str
    accessibility_snapshot: str
    timestamp: float


class SFGEdge(BaseModel):
    """An interaction that transitions one GUIState to another."""

    model_config = ConfigDict(strict=True, extra="forbid")

    source_hash: str
    target_hash: str
    aria_role: str
    accessible_name: str
    action_type: Literal["click", "fill", "select", "navigate", "press"]
    locator_strategy: Literal["getByRole", "getByLabel", "getByTestId"]


class ActionStep(BaseModel):
    """A single Playwright action inside a ContractSkill trajectory."""

    model_config = ConfigDict(strict=True, extra="forbid")

    step_index: int
    intent: str
    locator_strategy: str
    locator_value: str
    action_type: str
    payload: str | None = None


class RecoveryRule(BaseModel):
    """Trigger → patch operator mapping consulted by the healer."""

    model_config = ConfigDict(strict=True, extra="forbid")

    trigger_condition: Literal["NOT_FOUND", "INPUT_INVALID", "TIMEOUT", "STALE"]
    patch_operator: Literal["SelReplace", "PreInsert", "ArgCorrect", "PostInsert"]
    patch_payload: str


class ContractSkillArtifact(BaseModel):
    """Versioned, repairable contract describing a reusable skill."""

    model_config = ConfigDict(strict=True, extra="forbid")

    skill_id: str
    goal: str
    preconditions: list[str] = Field(default_factory=list)
    action_steps: list[ActionStep] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    recovery_rules: list[RecoveryRule] = Field(default_factory=list)
    version: int = 1
    status: Literal["active", "degraded", "patched"] = "active"


class AgentPipelineState(TypedDict, total=False):
    """LangGraph channel definition. Annotated reducers accumulate across nodes."""

    task_goal: str
    current_url: str
    sfg_nodes: Annotated[list[GUIState], operator.add]
    sfg_edges: Annotated[list[SFGEdge], operator.add]
    active_contract: ContractSkillArtifact | None
    execution_log: Annotated[list[str], operator.add]
    failed_step_index: int | None
    repair_attempts: int
    memory_summary: str
    mode: Literal["explore", "execute", "heal", "human_review"]
    final_result: str | None


__all__ = [
    "GUIState",
    "SFGEdge",
    "ActionStep",
    "RecoveryRule",
    "ContractSkillArtifact",
    "AgentPipelineState",
]
