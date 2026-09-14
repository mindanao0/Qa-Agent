"""
CodeJudge — Sprint 6.

4-check LLM-as-Judge for generated pytest code. Exactly 4 checks.
Checks 1-3 are deterministic string checks (no LLM).
Check 4 calls LLM only for metamorphic tests.
"""
from __future__ import annotations

import re
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.codetest.generator import GeneratedTest
from src.llm.instructor_client import InstructorClient

# A MetamorphicCheckResult is one bool + a short reason string — never needs
# more than a couple hundred tokens. Capped for the same reason as
# generator.py's _MAX_TEST_TOKENS: an uncapped call can run unbounded on a
# CPU-only runner if no natural stop token is emitted.
_MAX_JUDGE_TOKENS = 256


class MetamorphicCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    valid: bool
    reason: str


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grade: Literal["acceptable", "needs_revision", "reject"]
    feedback: str


class CodeJudge:
    """4-check quality judge for generated pytest code."""

    def __init__(self) -> None:
        self._client = InstructorClient()

    async def judge(self, test: GeneratedTest) -> JudgeResult:
        code = test.test_code
        issues: list[str] = []

        # Check 1: has_assert
        if not re.search(r"\bassert\b", code):
            issues.append("missing assert statement")

        # Check 2: no_hardcoded_sleep
        if re.search(r"\btime\.sleep\s*\(|\basyncio\.sleep\s*\(", code):
            issues.append("contains time.sleep() call")

        # Check 3: valid_pytest_sig — at least one def starting with test_ is required
        funcs = re.findall(r"\bdef\s+(\w+)\s*\(", code)
        if not funcs or not any(name.startswith("test_") for name in funcs):
            issues.append("function name must start with test_")

        if issues:
            return JudgeResult(grade="reject", feedback="; ".join(issues))

        # Pattern B: async def test_ without @pytest.mark.asyncio → needs_revision
        if re.search(r"\basync\s+def\s+test_", code) and "@pytest.mark.asyncio" not in code:
            return JudgeResult(
                grade="needs_revision",
                feedback="async test requires @pytest.mark.asyncio decorator",
            )

        # Check 4: metamorphic_valid
        if test.test_type == "metamorphic" and (test.metamorphic_relation or "").strip():
            relation = test.metamorphic_relation or ""
            # Static check: "always"/"never" without a quantified bound is too vague to be falsifiable
            if re.search(r"\b(always|never)\b", relation, re.IGNORECASE):
                if not re.search(
                    r"\bfor\s+(all|positive|non|any|each|every)\b", relation, re.IGNORECASE
                ):
                    return JudgeResult(
                        grade="needs_revision",
                        feedback="metamorphic relation too vague: 'always'/'never' without quantified bound",
                    )
            result = await self._check_metamorphic(relation)
            if not result.valid:
                return JudgeResult(
                    grade="needs_revision",
                    feedback=f"metamorphic relation invalid: {result.reason}",
                )

        return JudgeResult(grade="acceptable", feedback="")

    async def _check_metamorphic(self, relation: str) -> MetamorphicCheckResult:
        prompt = (
            f"Metamorphic relation: {relation}\n"
            "Is this relation logically sound and testable? "
            "A valid relation must describe a concrete, verifiable input-output "
            "relationship (e.g. 'f(x+1) > f(x) for all positive x'). "
            "Respond with valid=true if sound, valid=false if not, plus a brief reason."
        )
        try:
            return await self._client.create_structured(
                prompt=prompt,
                response_model=MetamorphicCheckResult,
                temperature=0.0,
                max_tokens=_MAX_JUDGE_TOKENS,
            )
        except Exception as exc:
            logger.warning(f"CodeJudge._check_metamorphic: error ({type(exc).__name__}): {exc!r}; failing open")
            return MetamorphicCheckResult(valid=True, reason="judge_unavailable")

    async def close(self) -> None:
        await self._client.close()


__all__ = ["CodeJudge", "JudgeResult", "MetamorphicCheckResult"]
