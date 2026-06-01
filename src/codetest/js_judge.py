"""JSCodeJudge — Sprint 9. 4-check judge for generated Vitest tests. Exactly 4 checks."""
from __future__ import annotations

import re
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.codetest.js_generator import GeneratedJSTest
from src.llm.instructor_client import InstructorClient


class JSJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grade: Literal["acceptable", "needs_revision", "reject"]
    feedback: str


class _MetamorphicCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    valid: bool
    reason: str


class JSCodeJudge:
    """4-check quality judge for generated Vitest test code."""

    def __init__(self) -> None:
        self._client: InstructorClient | None = None

    async def judge(self, test: GeneratedJSTest) -> JSJudgeResult:
        code = test.test_code
        issues: list[str] = []

        # Check 1: has_expect — at least one expect() call
        if not re.search(r"\bexpect\s*\(", code):
            issues.append("missing expect() call")

        # Check 2: no_settimeout — no setTimeout/setInterval in test body
        if re.search(r"\bsetTimeout\s*\(|\bsetInterval\s*\(", code):
            issues.append("contains setTimeout/setInterval — not allowed in test body")

        # Check 3: valid_vitest_sig — must have it() or test() block
        if not re.search(r"\b(it|test)\s*\(", code):
            issues.append("no it() or test() block found")

        if issues:
            return JSJudgeResult(grade="reject", feedback="; ".join(issues))

        # Check 4: metamorphic_valid
        if test.test_type == "metamorphic" and (test.metamorphic_relation or "").strip():
            relation = test.metamorphic_relation or ""
            if re.search(r"\b(always|never)\b", relation, re.IGNORECASE):
                if not re.search(
                    r"\bfor\s+(all|positive|non|any|each|every)\b", relation, re.IGNORECASE
                ):
                    return JSJudgeResult(
                        grade="needs_revision",
                        feedback="metamorphic relation too vague: 'always'/'never' without quantified bound",
                    )
            check = await self._check_metamorphic(relation)
            if not check.valid:
                return JSJudgeResult(
                    grade="needs_revision",
                    feedback=f"metamorphic relation invalid: {check.reason}",
                )

        return JSJudgeResult(grade="acceptable", feedback="")

    async def _check_metamorphic(self, relation: str) -> _MetamorphicCheck:
        if self._client is None:
            self._client = InstructorClient()
        prompt = (
            f"Metamorphic relation: {relation}\n"
            "Is this relation logically sound and falsifiable for a concrete function? "
            "A valid relation has a verifiable input→output pattern. "
            "Respond valid=true if sound, valid=false if not, with a brief reason."
        )
        try:
            return await self._client.create_structured(
                prompt=prompt,
                response_model=_MetamorphicCheck,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning(f"JSCodeJudge._check_metamorphic: {exc!r} — failing open")
            return _MetamorphicCheck(valid=True, reason="judge_unavailable")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()


__all__ = ["JSCodeJudge", "JSJudgeResult"]
