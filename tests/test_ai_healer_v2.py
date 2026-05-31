"""Tests for the Sprint 2 Cluster S2-D AIHealer class.

These tests exercise the 3-attempt pipeline (Fuzzy → AOM → Memory →
Quarantine) using AsyncMock/Mock so no live browser or Ollama is required.
"""
from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.healing.ai_healer import (
    ActionContext,
    AIHealer,
    HealResult,
)
from src.healing.fuzzy_matcher import FuzzyCandidate
from src.memory.episodic_store import HealedExperience, MemoryHit


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_action_context() -> ActionContext:
    return ActionContext(
        action_type="click",
        domain="example.com",
        page_url="https://example.com/login",
        element_description="Login button",
    )


def _make_store(*, retrieve_hits=None) -> MagicMock:
    """Build an EpisodicStore mock with the async methods we use."""
    store = MagicMock()
    store._embed = AsyncMock(return_value=[0.0] * 768)
    store._embedding_fn = AsyncMock(return_value=[0.0] * 768)
    store.save_healed_experience = AsyncMock(return_value="memory-id-1")
    store.save_post_mortem = AsyncMock(return_value="pm-id-1")
    store.retrieve_for_planning = AsyncMock(return_value=retrieve_hits or [])
    return store


def _make_fuzzy(candidates=None) -> MagicMock:
    fm = MagicMock()
    fm.find_candidates = MagicMock(return_value=candidates or [])
    return fm


def _make_aom_extractor(raise_exc: bool = True, snapshot=None) -> MagicMock:
    ext = MagicMock()
    if raise_exc:
        ext.extract = AsyncMock(side_effect=RuntimeError("no page"))
    else:
        ext.extract = AsyncMock(return_value=snapshot)
    return ext


def _make_page() -> MagicMock:
    page = MagicMock()
    page.url = "https://example.com/login"
    return page


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_failure_signature_is_deterministic():
    sig1 = AIHealer._compute_failure_signature(
        "page.get_by_role('button', name='Login')",
        "https://example.com/login?foo=bar",
        "click",
    )
    sig2 = AIHealer._compute_failure_signature(
        "page.get_by_role('button', name='Login')",
        "https://example.com/login?foo=bar",
        "click",
    )
    assert sig1 == sig2
    assert len(sig1) == 16
    # Different inputs → different hash
    sig3 = AIHealer._compute_failure_signature(
        "page.get_by_role('button', name='Login')",
        "https://example.com/login",
        "fill",
    )
    assert sig3 != sig1


@pytest.mark.asyncio
async def test_fuzzy_hit_returns_correct_strategy(monkeypatch):
    """A HIGH-tier fuzzy candidate that resolves → FUZZY_HIT result."""
    healed_locator = 'page.get_by_role("button", name="Login")'
    candidate = FuzzyCandidate(
        candidate_locator=healed_locator,
        similarity_score=0.95,
        source_file="locators/locators.json",
        last_used_iso="2026-05-20T00:00:00+00:00",
        confidence_tier="HIGH",
    )

    fm = _make_fuzzy([candidate])
    store = _make_store()
    aom = _make_aom_extractor(raise_exc=True)

    healer = AIHealer(
        fuzzy_matcher=fm,
        aom_extractor=aom,
        episodic_store=store,
        session_id="sess-1",
    )

    async def _ok(self, page, locator_str):
        return True

    # Force _try_locator to succeed
    monkeypatch.setattr(AIHealer, "_try_locator", staticmethod(lambda page, ls: _async_true()))

    result = await healer.heal(_make_page(), "page.get_by_role('button', name='Login')", _make_action_context())

    assert isinstance(result, HealResult)
    assert result.strategy == "FUZZY_HIT"
    assert result.locator == healed_locator
    assert result.attempt_count == 1
    assert result.latency_ms >= 0
    assert len(result.failure_signature) == 16
    # save_healed_experience should have been invoked
    store.save_healed_experience.assert_awaited_once()


async def _async_true() -> bool:
    return True


async def _async_false() -> bool:
    return False


@pytest.mark.asyncio
async def test_quarantine_when_all_attempts_fail(monkeypatch):
    """No fuzzy candidates, AOM raises, no memories → QUARANTINE."""
    fm = _make_fuzzy([])  # no candidates
    store = _make_store(retrieve_hits=[])
    aom = _make_aom_extractor(raise_exc=True)

    healer = AIHealer(
        fuzzy_matcher=fm,
        aom_extractor=aom,
        episodic_store=store,
        session_id="sess-1",
    )

    # _try_locator never invoked because no candidates exist, but keep it safe.
    monkeypatch.setattr(AIHealer, "_try_locator", staticmethod(lambda page, ls: _async_false()))

    result = await healer.heal(
        _make_page(),
        "page.get_by_role('button', name='Login')",
        _make_action_context(),
    )

    assert result.strategy == "QUARANTINE"
    assert result.locator is None
    assert result.attempt_count == 3
    # Post-mortem must have been saved
    store.save_post_mortem.assert_awaited_once()
    store.save_healed_experience.assert_not_called()


@pytest.mark.asyncio
async def test_memory_hit_writes_to_store(monkeypatch):
    """On FUZZY_HIT path, verify _save_heal builds a correct HealedExperience."""
    candidate_locator = 'page.get_by_role("button", name="Login")'
    candidate = FuzzyCandidate(
        candidate_locator=candidate_locator,
        similarity_score=0.95,
        source_file="locators/locators.json",
        last_used_iso="2026-05-20T00:00:00+00:00",
        confidence_tier="HIGH",
    )

    fm = _make_fuzzy([candidate])
    store = _make_store()
    aom = _make_aom_extractor(raise_exc=True)

    healer = AIHealer(
        fuzzy_matcher=fm,
        aom_extractor=aom,
        episodic_store=store,
        session_id="sess-42",
    )

    monkeypatch.setattr(AIHealer, "_try_locator", staticmethod(lambda page, ls: _async_true()))

    failed = "page.get_by_role('button', name='Login')"
    await healer.heal(_make_page(), failed, _make_action_context())

    # Inspect the HealedExperience passed to save_healed_experience
    store.save_healed_experience.assert_awaited_once()
    args, _kwargs = store.save_healed_experience.call_args
    exp = args[0]
    assert isinstance(exp, HealedExperience)
    assert exp.session_id == "sess-42"
    assert exp.domain == "example.com"
    assert exp.bad_strategy == failed
    assert exp.winning_strategy == candidate_locator
    assert exp.root_cause == "fuzzy_match"
    assert exp.confidence == 0.7
    assert exp.impact_score == 0.5
    assert len(exp.vector) == 768
    assert len(exp.failure_signature) == 16


@pytest.mark.asyncio
async def test_memory_hit_returns_memory_strategy(monkeypatch):
    """When fuzzy/AOM fail but memory returns a high-conf healed_experience → MEMORY_HIT."""
    hit_locator = 'page.get_by_role("link", name="Profile")'
    fm = _make_fuzzy([])
    aom = _make_aom_extractor(raise_exc=True)
    store = _make_store(
        retrieve_hits=[
            MemoryHit(
                memory_type="healed_experience",
                failure_signature="abc1234567890abc",
                strategy=hit_locator,
                root_cause="memory_replay",
                confidence=0.85,
                impact_score=0.9,
            )
        ]
    )

    healer = AIHealer(
        fuzzy_matcher=fm,
        aom_extractor=aom,
        episodic_store=store,
        session_id="sess-1",
    )

    monkeypatch.setattr(AIHealer, "_try_locator", staticmethod(lambda page, ls: _async_true()))

    result = await healer.heal(
        _make_page(),
        "page.get_by_role('link', name='Profile')",
        _make_action_context(),
    )

    assert result.strategy == "MEMORY_HIT"
    assert result.locator == hit_locator
    assert result.attempt_count == 3
    store.save_healed_experience.assert_awaited_once()
