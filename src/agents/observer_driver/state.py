"""Pydantic V2 state schemas for the Observer-Driver multi-agent system.

The :class:`AgentState` flows through the LangGraph coordinator. Driver and
Observer outputs are accumulated into list fields via LangGraph's ``Annotated``
reducer (``operator.add``), so concurrent branches can append without lock-step
coordination — the very property that lets Observers run in parallel with the
Driver without state-mutation conflicts.
"""
from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["info", "warn", "critical"]


class ObserverReport(BaseModel):
    """A single audit finding emitted by an Observer agent."""

    model_config = ConfigDict(extra="forbid")

    observer_name: str
    findings: list[str] = Field(default_factory=list)
    severity: Severity = "info"
    timestamp_ms: int


class DriverAction(BaseModel):
    """A single state-mutating action chosen by the Driver agent."""

    model_config = ConfigDict(extra="forbid")

    action_type: Literal["click", "fill", "navigate", "assert", "noop"]
    target_role: str = ""
    target_name: str = ""
    value: str = ""
    ax_snapshot_path: Path


class AgentState(BaseModel):
    """LangGraph channel state for the Observer-Driver graph.

    All list channels use ``operator.add`` so the Driver branch and each
    Observer branch can emit items concurrently without overwriting one
    another.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    url: str
    driver_actions: Annotated[list[DriverAction], operator.add] = Field(
        default_factory=list
    )
    observer_reports: Annotated[list[ObserverReport], operator.add] = Field(
        default_factory=list
    )
    trace_events: Annotated[list[dict], operator.add] = Field(default_factory=list)
    current_step: int = 0
    max_steps: int = 5
    halt: bool = False


__all__ = [
    "AgentState",
    "DriverAction",
    "ObserverReport",
    "Severity",
]
