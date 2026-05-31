"""Tests for the Sprint 2 Cluster S2-D planner memory injection.

Verify PlannerAgent only injects a "## Memory-Based Strategy" block when the
EpisodicStore returns a populated LocatorPolicy.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.planner import PlannerAgent
from src.llm.schemas import TestPlan, TestStep
from src.memory.episodic_store import LocatorPolicy


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _stub_plan() -> TestPlan:
    return TestPlan(
        title="Login flow",
        requirement_summary="Verify user login",
        estimated_complexity="low",
        domain="authentication",
        steps=[
            TestStep(
                step_number=1,
                description="Navigate to login page",
                action="navigate",
                expected_result="Login form visible",
                role="user",
            )
        ],
    )


def _make_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.base_url = "http://localhost:11434"
    adapter.model = "stub"
    adapter.generate = AsyncMock(return_value="{}")
    return adapter


def _make_instructor_capture() -> tuple[MagicMock, list]:
    """Return (instructor_client, captured_messages_list).

    The instructor stub records every messages list passed to
    create_structured(...) so tests can assert on prompt content.
    """
    captured: list = []
    client = MagicMock()

    async def _create_structured(messages, response_model, temperature=0.0):
        captured.append(messages)
        return _stub_plan()

    client.create_structured = AsyncMock(side_effect=_create_structured)
    return client, captured


def _flatten_messages(messages_list: list) -> str:
    """Concatenate all message contents into one string for substring checks."""
    return "\n".join(m.get("content", "") for m in messages_list)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_injection_when_store_is_none(monkeypatch):
    """With episodic_store=None, the planner must never invoke store methods."""
    adapter = _make_adapter()
    instructor, captured = _make_instructor_capture()

    # Skip domain auto-detection (no live Ollama)
    async def _stub_detect_domain(**kwargs):
        return "authentication"

    monkeypatch.setattr("src.agents.planner.detect_domain", _stub_detect_domain)

    planner = PlannerAgent(
        adapter=adapter,
        retriever=None,
        instructor_client=instructor,
        episodic_store=None,
    )

    plan = await planner.plan(
        requirement="Login as user",
        url="https://example.com/login",
        role="user",
        domain="authentication",
    )

    assert plan.title == "Login flow"
    assert captured, "create_structured should have been called once"
    text = _flatten_messages(captured[0])
    assert "## Memory-Based Strategy" not in text


@pytest.mark.asyncio
async def test_no_injection_when_policy_is_empty(monkeypatch):
    """Empty policy → memory block must NOT appear in the prompt."""
    adapter = _make_adapter()
    instructor, captured = _make_instructor_capture()

    async def _stub_detect_domain(**kwargs):
        return "authentication"

    monkeypatch.setattr("src.agents.planner.detect_domain", _stub_detect_domain)

    empty_policy = LocatorPolicy(
        domain="authentication",
        failure_signature="0000000000000000",
        preferred_strategies=[],
        avoid_strategies=[],
    )
    store = MagicMock()
    store.evolve_locator_policy = AsyncMock(return_value=empty_policy)

    planner = PlannerAgent(
        adapter=adapter,
        retriever=None,
        instructor_client=instructor,
        episodic_store=store,
    )

    await planner.plan(
        requirement="Login as user",
        url="https://example.com/login",
        role="user",
        domain="authentication",
    )

    store.evolve_locator_policy.assert_awaited_once()
    text = _flatten_messages(captured[0])
    assert "## Memory-Based Strategy" not in text


@pytest.mark.asyncio
async def test_injection_when_policy_has_strategies(monkeypatch):
    """Populated policy → memory block must appear with preferred entries."""
    adapter = _make_adapter()
    instructor, captured = _make_instructor_capture()

    async def _stub_detect_domain(**kwargs):
        return "authentication"

    monkeypatch.setattr("src.agents.planner.detect_domain", _stub_detect_domain)

    populated_policy = LocatorPolicy(
        domain="authentication",
        failure_signature="0000000000000000",
        preferred_strategies=[
            'page.get_by_role("button", name="Login")',
            'page.get_by_test_id("login-btn")',
        ],
        avoid_strategies=['page.get_by_text("Submit")'],
    )
    store = MagicMock()
    store.evolve_locator_policy = AsyncMock(return_value=populated_policy)

    planner = PlannerAgent(
        adapter=adapter,
        retriever=None,
        instructor_client=instructor,
        episodic_store=store,
    )

    await planner.plan(
        requirement="Login as user",
        url="https://example.com/login",
        role="user",
        domain="authentication",
    )

    store.evolve_locator_policy.assert_awaited_once()
    text = _flatten_messages(captured[0])
    assert "## Memory-Based Strategy" in text
    assert "Preferred (proven):" in text
    assert 'page.get_by_role("button", name="Login")' in text
    assert "Avoid (known-fail):" in text
