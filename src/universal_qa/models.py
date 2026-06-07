from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str
    type: Literal["functional", "accessibility", "security"]
    priority: Literal["high", "medium", "low"]
    preconditions: list[str] = Field(default_factory=list)
    steps: list[str]
    expected_outcome: str
    source_url: str


class StepTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    status: Literal["passed", "failed", "skipped"]
    detail: str
    error: str | None = None


class TestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_case: TestCase
    passed: bool
    steps_trace: list[StepTrace] = Field(default_factory=list)
    failure_reason: str | None = None
    screenshot_path: str | None = None
    duration_ms: int = 0


__all__ = ["TestCase", "StepTrace", "TestResult"]
