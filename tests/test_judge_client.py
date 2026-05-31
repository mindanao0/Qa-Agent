"""
Unit tests for JudgeClient (Sprint 3 / Cluster S3-C).

These tests mock InstructorClient.create_structured — they do NOT hit Ollama.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from src.llm.instructor_client import StructuredGenerationError
from src.llm.judge_client import JudgeClient
from src.llm.schemas import JudgeVerdict


_SAMPLE_CODE = (
    "def test_x(page):\n"
    "    page.goto('/')\n"
    "    expect(page).to_have_url('/')\n"
)


@pytest.mark.asyncio
async def test_judge_approves_clean_code() -> None:
    mock_client = AsyncMock()
    mock_client.create_structured = AsyncMock(
        return_value=JudgeVerdict(
            approved=True,
            confidence=0.9,
            rejection_reason=None,
            issues_found=[],
        )
    )
    judge = JudgeClient(mock_client)

    verdict = await judge.evaluate(
        generated_code=_SAMPLE_CODE,
        requirement="test foo",
        domain="auth",
    )

    assert verdict.approved is True
    assert verdict.confidence == 0.9
    assert verdict.rejection_reason is None
    assert verdict.issues_found == []
    mock_client.create_structured.assert_awaited_once()


@pytest.mark.asyncio
async def test_judge_rejects_hallucinated_selector() -> None:
    mock_client = AsyncMock()
    mock_client.create_structured = AsyncMock(
        return_value=JudgeVerdict(
            approved=False,
            confidence=0.85,
            rejection_reason="hallucinated selector: aria-label 'NonExistent' not in DOM",
            issues_found=["hallucinated_selector"],
        )
    )
    judge = JudgeClient(mock_client)

    verdict = await judge.evaluate(
        generated_code=_SAMPLE_CODE,
        requirement="click the non-existent button",
        domain="general",
    )

    assert verdict.approved is False
    assert verdict.rejection_reason is not None
    assert "hallucinated" in verdict.rejection_reason
    assert verdict.issues_found == ["hallucinated_selector"]


@pytest.mark.asyncio
async def test_judge_fails_open_on_structured_error() -> None:
    mock_client = AsyncMock()
    mock_client.create_structured = AsyncMock(
        side_effect=StructuredGenerationError("hash", RuntimeError("boom"))
    )
    judge = JudgeClient(mock_client)

    verdict = await judge.evaluate(
        generated_code=_SAMPLE_CODE,
        requirement="anything",
        domain="general",
    )

    assert verdict.approved is True
    assert verdict.rejection_reason == "judge_unavailable"
    assert verdict.confidence == 0.5


def test_judge_verdict_schema_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict(approved=True, confidence=0.9, foo="bar")  # type: ignore[call-arg]
