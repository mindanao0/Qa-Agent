# src/llm/schemas.py
"""
Canonical Pydantic V2 output schemas for LLM-generated structures.

This module is the preferred import target for the Instructor-based structured
output pipeline (src/llm/instructor_client.py).  Legacy callers that import
from src.llm.structured continue to work — structured.py re-exports these
schemas for backward compatibility.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TestStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_number: int
    description: str
    action: str
    expected_result: str
    role: str = "admin"
    preconditions: list[str] = Field(default_factory=list)

    @field_validator("action", mode="before")
    @classmethod
    def coerce_action_list_to_str(cls, v: object) -> str:
        # action is a command token (e.g. "fill+click"), so list items are joined
        # with "+" rather than " " to preserve the token-sequence semantics.
        if v is None:
            raise ValueError("action field must not be None")
        if isinstance(v, list):
            return "+".join(str(x) for x in v)
        return str(v)

    @field_validator("description", "expected_result", "role", mode="before")
    @classmethod
    def coerce_str_fields_list_to_str(cls, v: object) -> str:
        """Coerce list → str for string fields the LLM may erroneously return as lists.

        Uses " " join (space-separated prose) for description/expected_result/role,
        unlike action which uses "+" join for command-token semantics.
        """
        if v is None:
            raise ValueError("description/expected_result/role fields must not be None")
        if isinstance(v, list):
            return " ".join(str(x) for x in v)
        return str(v)


class TestPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    requirement_summary: str
    estimated_complexity: Literal["low", "medium", "high"]
    domain: str = Field(
        default="crud_operations",
        description="Detected application domain (key from DOMAIN_REGISTRY)",
    )
    domain_specific_notes: list[str] = Field(
        default_factory=list,
        description="Domain-specific business rules / pitfalls the test must respect",
    )
    steps: list[TestStep]
    rbac_scenarios: list[str] = Field(default_factory=list)
    edge_cases: list[str] = Field(default_factory=list)

    @field_validator("steps")
    @classmethod
    def steps_not_empty(cls, v: list[TestStep]) -> list[TestStep]:
        if not v:
            raise ValueError("TestPlan must contain at least one step")
        return v

    @field_validator("edge_cases", "rbac_scenarios", "domain_specific_notes", mode="before")
    @classmethod
    def normalize_string_list(cls, v: Any) -> list[str]:
        """Normalize list items that the LLM may return as dicts instead of strings."""
        if not isinstance(v, list):
            return []
        result: list[str] = []
        for item in v:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                for val in item.values():
                    if isinstance(val, str):
                        result.append(val)
                        break
                else:
                    result.append(str(item))
        return result


class PlaywrightScript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # reasoning MUST come before code — left-to-right generation improves quality
    reasoning: str = Field(
        description="Step-by-step reasoning about how to implement the test before writing code"
    )
    code: str = Field(description="Complete, runnable Playwright Python async test code")
    locators_used: list[str] = Field(
        default_factory=list,
        description="All locator strategies used (e.g. get_by_role, get_by_label)",
    )
    test_function_name: str = Field(
        default="test_generated",
        description="Name of the top-level async test function",
    )

    @field_validator("code")
    @classmethod
    def no_forbidden_patterns(cls, v: str) -> str:
        forbidden = [
            (r"page\.locator\s*\(\s*['\"]css=", "CSS selector via page.locator(css=)"),
            (r"page\.locator\s*\(\s*['\"]xpath=", "XPath selector via page.locator(xpath=)"),
            (r"page\.locator\s*\(\s*['\"]//", "bare XPath locator"),
            (r"page\.wait_for_timeout", "page.wait_for_timeout() (forbidden hardcoded wait)"),
            (r"asyncio\.sleep", "asyncio.sleep() (use Playwright auto-wait instead)"),
        ]
        violations: list[str] = []
        for pattern, label in forbidden:
            if re.search(pattern, v):
                violations.append(label)
        if violations:
            raise ValueError(f"Generated code contains forbidden patterns: {violations}")
        return v


class HealedLocator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # reasoning before answer (left-to-right generation)
    reasoning: str = Field(
        description="Explanation of why the original locator failed and how the healed one was chosen"
    )
    original: str
    healed: str
    confidence: float = Field(ge=0.0, le=1.0)
    method: Literal["fuzzy", "ai", "vlm"]

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        return round(v, 4)


class SyntheticQAExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(
        description="Natural language requirement for the test (user turn)"
    )
    input_context: str = Field(
        description="Additional context provided alongside the requirement"
    )
    output_code: str = Field(
        description="Complete Playwright Python test code satisfying the requirement"
    )
    domain: str = Field(
        description="Domain seed the example belongs to (e.g. hrm_login, payroll_calculation)"
    )
    scenario_type: Literal["happy_path", "negative", "rbac_boundary"] = "happy_path"

    def to_chatml(self, system_prompt: str) -> dict[str, Any]:
        user_content = self.instruction
        if self.input_context:
            user_content = f"{self.instruction}\n\nContext:\n{self.input_context}"
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": self.output_code},
            ]
        }


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approved: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rejection_reason: str | None = Field(default=None, max_length=300)
    issues_found: list[str] = Field(default_factory=list, max_length=5)


__all__ = [
    "TestStep",
    "TestPlan",
    "PlaywrightScript",
    "HealedLocator",
    "SyntheticQAExample",
    "JudgeVerdict",
]
