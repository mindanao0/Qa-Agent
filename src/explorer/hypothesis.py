from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, model_validator

_HYPOTHESIS_ID_LENGTH = 12


class TestHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    goal: str
    start_url: str
    preconditions: list[str]
    steps: list[str]
    expected_outcome: str
    source_skill_id: str | None = None
    confidence: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _auto_hypothesis_id(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values
        if values.get("hypothesis_id") is None:
            goal = values.get("goal", "")
            start_url = values.get("start_url", "")
            values["hypothesis_id"] = hashlib.sha256(
                (goal + start_url).encode()
            ).hexdigest()[:_HYPOTHESIS_ID_LENGTH]
        return values


class ExplorationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id:               str
    start_url:            str
    sfg_nodes_found:      int
    hypotheses:           list[TestHypothesis]
    exploration_coverage: float
    skills_reused:        int
    generated_at:         float


__all__ = ["TestHypothesis", "ExplorationReport"]
