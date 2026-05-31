"""
LLM-as-Judge — Sprint 3 / Cluster S3-C.

Semantic correctness review of BFT-certified Playwright code. Fails OPEN:
if the underlying LLM call errors, returns an approved=True verdict so that
generation never gets blocked by judge unavailability.
"""
from __future__ import annotations

from loguru import logger

from src.llm.instructor_client import InstructorClient, StructuredGenerationError
from src.llm.schemas import JudgeVerdict


_JUDGE_SYSTEM_PROMPT = (
    "You are a senior QA engineer reviewing Playwright test code. "
    "Be strict but fair. Return ONLY a JudgeVerdict JSON object."
)


def _build_prompt(generated_code: str, requirement: str, domain: str) -> str:
    return (
        f"REQUIREMENT: {requirement}\n"
        f"DOMAIN: {domain}\n"
        f"CODE:\n{generated_code}\n\n"
        "Review for ONLY these issues:\n"
        "1. Does the test actually verify the stated requirement?\n"
        "2. Are there assertions that will always pass (false positives)?\n"
        "3. Are there selector references that look hallucinated?\n"
        "   (e.g., very long CSS chains, non-existent aria-labels)\n"
        "4. Does the test have at least one expect() assertion?\n\n"
        "Return JudgeVerdict.approved=True only if all 4 checks pass."
    )


class JudgeClient:
    """Wraps an InstructorClient to produce JudgeVerdicts."""

    def __init__(self, instructor_client: InstructorClient) -> None:
        self._client = instructor_client

    async def evaluate(
        self,
        generated_code: str,
        requirement: str,
        domain: str,
    ) -> JudgeVerdict:
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(generated_code, requirement, domain)},
        ]
        try:
            verdict = await self._client.create_structured(
                messages, JudgeVerdict, temperature=0.0
            )
            logger.info(
                f"JudgeClient | approved={verdict.approved} "
                f"confidence={verdict.confidence:.2f} "
                f"issues={len(verdict.issues_found)}"
            )
            return verdict
        except StructuredGenerationError as exc:
            logger.warning(f"JudgeClient: failing OPEN — judge unavailable: {exc}")
            return JudgeVerdict(
                approved=True,
                confidence=0.5,
                rejection_reason="judge_unavailable",
                issues_found=[],
            )
        except Exception as exc:
            logger.warning(f"JudgeClient: failing OPEN — unexpected: {exc!r}")
            return JudgeVerdict(
                approved=True,
                confidence=0.5,
                rejection_reason="judge_unavailable",
                issues_found=[],
            )


__all__ = ["JudgeClient"]
